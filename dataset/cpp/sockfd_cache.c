/*
 * Copyright (C) 2012-2013 Taobao Inc.
 *
 * Liu Yuan <namei.unix@gmail.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License version
 * 2 as published by the Free Software Foundation.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * The sockfd cache provides us long TCP connections connected to the nodes
 * in the cluster to accelerator the data transfer, which has the following
 * characteristics:
 *    0 dynamically allocated/deallocated at node granularity.
 *    1 cached fds are multiplexed by all threads.
 *    2 each session (for e.g, forward_write_obj_req) can grab one fd at a time.
 *    3 if there isn't any FD available from cache, use normal connect_to() and
 *      close() internally.
 *    4 FD are named by IP:PORT uniquely, hence no need of resetting at
 *      membership change.
 *    5 the total number of FDs is scalable to massive nodes.
 *    6 total 3 APIs: sheep_{get,put,del}_sockfd().
 *    7 support dual connections to a single node.
 */

#include <pthread.h>

#include "sockfd_cache.h"
#include "work.h"
#include "rbtree.h"
#include "util.h"
#include "sheep.h"
#include "list.h"

#define TRACEPOINT_DEFINE
#include "sockfd_cache_tp.h"
#define MONITOR_INTERVAL 5

struct nid_test_work {
	struct list_node w_list;
	struct node_id nid;
};
struct list_head to_connect_list;
struct sockfd_cache {
	struct rb_root root;
	struct sd_rw_lock lock;
	int count;
};

static struct sockfd_cache sockfd_cache = {
	.root = RB_ROOT,
	.lock = SD_RW_LOCK_INITIALIZER,
};

/*
 * Suppose request size from Guest is 512k, then 4M / 512k = 8, so at
 * most 8 requests can be issued to the same sheep object. Based on this
 * assumption, '8' would be efficient for servers that only host 2~4
 * Guests.
 *
 * This fd count will be dynamically grown when the idx reaches watermark which
 * is calculated by FDS_WATERMARK
 */
#define FDS_WATERMARK(x) ((x) * 3 / 4)
#define DEFAULT_FDS_COUNT	8

/* How many FDs we cache for one node */
static int fds_count = DEFAULT_FDS_COUNT;

struct sockfd_cache_fd {
	int fd;
	uatomic_bool in_use;
};

struct sockfd_cache_entry {
	struct rb_node rb;
	struct node_id nid;
	struct sockfd_cache_fd *fds_io;
	struct sockfd_cache_fd *fds_nio;
	enum channel_status channel_status;
};

static int sockfd_cache_cmp(const struct sockfd_cache_entry *a,
			    const struct sockfd_cache_entry *b)
{
	return node_id_cmp(&a->nid, &b->nid);
}

static struct sockfd_cache_entry *
sockfd_cache_insert(struct sockfd_cache_entry *new)
{
	return rb_insert(&sockfd_cache.root, new, rb, sockfd_cache_cmp);
}

static struct sockfd_cache_entry *sockfd_cache_search(const struct node_id *nid)
{
	struct sockfd_cache_entry key = { .nid = *nid };

	return rb_search(&sockfd_cache.root, &key, rb, sockfd_cache_cmp);
}

static inline int get_free_slot(struct sockfd_cache_entry *entry, bool *isIO)
{
	int idx = -1, i;

	if (entry->nid.io_port && entry->channel_status == IO) {
		for (i = 0; i < fds_count; i++) {
			if (entry->fds_io[i].fd == -1 ||
				!uatomic_set_true(&entry->fds_io[i].in_use))
				continue;
			idx = i;
			*isIO = true;
			goto out;
		}
	}
	for (i = 0; i < fds_count; i++) {
		if (entry->fds_nio[i].fd == -1 ||
			!uatomic_set_true(&entry->fds_nio[i].in_use))
			continue;
		/*let caller know if this is an IO or NonIO port*/
		idx = i;
		*isIO = false;
		goto out;
	}
out:
	return idx;
}

/*
 * Grab a free slot of the node and inc the refcount of the slot
 *
 * If no free slot available, this typically means we should use short FD.
 */
static struct sockfd_cache_entry *sockfd_cache_grab(const struct node_id *nid,
						    int *ret_idx, bool *isIO)
{
	struct sockfd_cache_entry *entry;

	sd_read_lock(&sockfd_cache.lock);
	entry = sockfd_cache_search(nid);
	if (!entry) {
		sd_debug("failed node %s", addr_to_str(nid->addr, nid->port));
		goto out;
	}

	*ret_idx = get_free_slot(entry, isIO);
	if (*ret_idx == -1)
		entry = NULL;
out:
	sd_rw_unlock(&sockfd_cache.lock);
	return entry;
}

static inline bool slots_all_free(struct sockfd_cache_entry *entry)
{
	int i;
	for (i = 0; i < fds_count; i++) {
		if (uatomic_is_true(&entry->fds_io[i].in_use) ||
			uatomic_is_true(&entry->fds_nio[i].in_use))
			return false;
	}
	return true;
}

static inline void destroy_all_slots(struct sockfd_cache_entry *entry)
{
	int i;
	for (i = 0; i < fds_count; i++)
		if (entry->fds_io[i].fd != -1) {
			close(entry->fds_io[i].fd);
			entry->fds_io[i].fd = -1;
			close(entry->fds_nio[i].fd);
			entry->fds_nio[i].fd = -1;
		}
}

static void free_cache_entry(struct sockfd_cache_entry *entry)
{
	free(entry->fds_io);
	entry->fds_io = 0;
	free(entry->fds_nio);
	entry->fds_nio = 0;
	free(entry);
}

/*
 * Destroy all the Cached FDs of the node
 *
 * We don't proceed if some other node grab one FD of the node. In this case,
 * the victim node will finally find itself talking to a dead node and call
 * sockfd_cache_del() to delete this node from the cache.
 */
static bool sockfd_cache_destroy(const struct node_id *nid)
{
	struct sockfd_cache_entry *entry;

	sd_write_lock(&sockfd_cache.lock);
	entry = sockfd_cache_search(nid);
	if (!entry) {
		sd_debug("It is already destroyed");
		goto false_out;
	}

	if (!slots_all_free(entry)) {
		sd_debug("Some victim still holds it");
		goto false_out;
	}

	rb_erase(&entry->rb, &sockfd_cache.root);

	destroy_all_slots(entry);
	free_cache_entry(entry);
	sd_rw_unlock(&sockfd_cache.lock);

	return true;
false_out:
	sd_rw_unlock(&sockfd_cache.lock);
	return false;
}

static void sockfd_cache_add_nolock(const struct node_id *nid)
{
	struct sockfd_cache_entry *new = xmalloc(sizeof(*new));
	int i;

	new->fds_io = xzalloc(sizeof(struct sockfd_cache_fd) * fds_count);
	new->fds_nio = xzalloc(sizeof(struct sockfd_cache_fd) * fds_count);
	for (i = 0; i < fds_count; i++) {
		new->fds_io[i].fd = -1;
		new->fds_nio[i].fd = -1;
	}

	memcpy(&new->nid, nid, sizeof(struct node_id));
	new->channel_status = nid->io_port ? IO : NonIO;
	if (sockfd_cache_insert(new)) {
		free_cache_entry(new);
		return;
	}
	sockfd_cache.count++;

	tracepoint(sockfd_cache, new_sockfd_entry, new, fds_count);
}

/* Add group of nodes to the cache */
void sockfd_cache_add_group(const struct rb_root *nroot)
{
	struct sd_node *n;

	sd_write_lock(&sockfd_cache.lock);
	rb_for_each_entry(n, nroot, rb) {
		sockfd_cache_add_nolock(&n->nid);
	}
	sd_rw_unlock(&sockfd_cache.lock);
}

/* Add one node to the cache means we can do caching tricks on this node */
void sockfd_cache_add(const struct node_id *nid)
{
	struct sockfd_cache_entry *new;
	int n, i;

	sd_write_lock(&sockfd_cache.lock);
	new = xmalloc(sizeof(*new));
	new->fds_io = xzalloc(sizeof(struct sockfd_cache_fd) * fds_count);
	new->fds_nio = xzalloc(sizeof(struct sockfd_cache_fd) * fds_count);
	for (i = 0; i < fds_count; i++) {
		new->fds_io[i].fd = -1;
		new->fds_nio[i].fd = -1;
	}

	memcpy(&new->nid, nid, sizeof(struct node_id));
	new->channel_status = nid->io_port ? IO : NonIO;
	if (sockfd_cache_insert(new)) {
		free_cache_entry(new);
		sd_rw_unlock(&sockfd_cache.lock);
		return;
	}
	sd_rw_unlock(&sockfd_cache.lock);
	n = uatomic_add_return(&sockfd_cache.count, 1);
	sd_debug("%s, count %d", addr_to_str(nid->addr, nid->port), n);

	tracepoint(sockfd_cache, new_sockfd_entry, new, fds_count);
}

static uatomic_bool fds_in_grow;
static int fds_high_watermark = FDS_WATERMARK(DEFAULT_FDS_COUNT);

static struct work_queue *grow_wq;

static void do_grow_fds(struct work *work)
{
	struct sockfd_cache_entry *entry;
	int old_fds_count, new_fds_count, new_size, i;

	sd_debug("%d", fds_count);
	sd_write_lock(&sockfd_cache.lock);
	old_fds_count = fds_count;
	new_fds_count = fds_count * 2;
	new_size = sizeof(struct sockfd_cache_fd) * fds_count * 2;
	rb_for_each_entry(entry, &sockfd_cache.root, rb) {
		entry->fds_io = xrealloc(entry->fds_io, new_size);
		entry->fds_nio = xrealloc(entry->fds_nio, new_size);
		for (i = old_fds_count; i < new_fds_count; i++) {
			entry->fds_io[i].fd = -1;
			uatomic_set_false(&entry->fds_io[i].in_use);
			entry->fds_nio[i].fd = -1;
			uatomic_set_false(&entry->fds_nio[i].in_use);
		}
	}

	fds_count *= 2;
	fds_high_watermark = FDS_WATERMARK(fds_count);
	sd_rw_unlock(&sockfd_cache.lock);

	tracepoint(sockfd_cache, grow_fd_count, new_fds_count);
}

static void grow_fds_done(struct work *work)
{
	sd_debug("fd count has been grown into %d", fds_count);
	uatomic_set_false(&fds_in_grow);
	free(work);
}

static inline void check_idx(int idx)
{
	struct work *w;

	if (idx <= fds_high_watermark)
		return;
	if (!uatomic_set_true(&fds_in_grow))
		return;

	w = xmalloc(sizeof(*w));
	w->fn = do_grow_fds;
	w->done = grow_fds_done;
	queue_work(grow_wq, w);
}

/* Add the node back if it is still alive */
static inline int revalidate_node(const struct node_id *nid)
{
	bool use_io = nid->io_port ? true : false;
	int fd;

	if (use_io) {
		fd = connect_to_addr(nid->io_addr, nid->io_port);
		if (fd >= 0)
			goto alive;
	}
	fd = connect_to_addr(nid->addr, nid->port);
	if (fd < 0)
		return false;
alive:
	close(fd);
	sockfd_cache_add(nid);
	return true;
}

static void prepare_conns(struct sockfd_cache_entry *entry, bool first_only)
{
	int idx;
	struct node_id *nid = &entry->nid;
	if (entry->channel_status == IO) {
		for (idx = 0; idx < fds_count; idx++) {
			/*
			 * IO channel recovered or fds_count increased,
			 * connect fds
			 */
			if (entry->fds_io[idx].fd != -1)
				continue;
			int fd = connect_to_addr(nid->io_addr, nid->io_port);
			if (fd >= 0) {
				entry->fds_io[idx].fd = fd;
				if (first_only) {
					break;
				}
			} else {
				/*
				 * if any thread close and the fd to -1, we can
				 * find it here and retest to confirm IO
				 * channel is down
				 */
				entry->channel_status = NonIO;
				struct nid_test_work *work =
					xmalloc(sizeof(struct nid_test_work));
				memcpy(&work->nid, &entry->nid,
						sizeof(struct node_id));
				list_add_tail(&work->w_list, &to_connect_list);
				sd_err("fallback to non-io connection");
				break; /*clear fds next round*/
			}
		}
	} else {
		for (idx = 0; idx < fds_count; idx++) {
			if (entry->fds_io[idx].fd != -1 &&
				uatomic_set_true(&entry->fds_io[idx].in_use)) {
				close(entry->fds_io[idx].fd);
				entry->fds_io[idx].fd = -1;
				uatomic_set_false(&entry->fds_io[idx].in_use);
			}
		}
	}

	for (idx = 0; idx < fds_count; idx++) {
		if (entry->fds_nio[idx].fd != -1)
			continue;
		int fd = connect_to_addr(nid->addr, nid->port);
		if (fd >= 0) {
			entry->fds_nio[idx].fd = fd;
			if (first_only) {
				break;
			}
		} else {
			sd_err("Can not connect to %s through NonIO channel!",
				addr_to_str(nid->addr, nid->port));
		}
	}
}

/* Try to create/get cached IO connection. If failed, fallback to non-IO one */
static struct sockfd *sockfd_cache_get_long(const struct node_id *nid)
{
	struct sockfd_cache_entry *entry;
	struct sockfd *sfd;
	bool isIO = true;
#ifndef HAVE_ACCELIO
	bool use_io = nid->io_port ? true : false;
	const uint8_t *addr = use_io ? nid->io_addr : nid->addr;
	int fd, idx = -1, port = use_io ? nid->io_port : nid->port;
#else
	bool use_io = false;
	const uint8_t *addr = nid->addr;
	int fd, idx = -1, port = nid->port;
#endif
grab:
	entry = sockfd_cache_grab(nid, &idx, &isIO);
	if (!entry) {
		/*
		 * The node is deleted, but someone asks us to grab it.
		 * The nid is not in the sockfd cache but probably it might be
		 * still alive due to broken network connection or was just too
		 * busy to serve any request that makes other nodes deleted it
		 * from the sockfd cache. In such cases, we need to add it back.
		 */
		if (!revalidate_node(nid))
			return NULL;
		/*
		 * When cache entry was newly added by revalidate_node()
		 * function, all fds are -1, so any call to sockfd_cache_grab()
		 * will return NULL for first time. we need to establish at
		 * least one fd for current grab. Most likely, this step is
		 * required for those cache entries created by external
		 * commands such as dog.
		 */
		entry = sockfd_cache_search(nid);
		sd_write_lock(&sockfd_cache.lock);
		prepare_conns(entry, true);
		sd_rw_unlock(&sockfd_cache.lock);

		goto grab;
	}

	check_idx(idx);
	if (!isIO)
		fd = entry->fds_nio[idx].fd;
	else
		fd = entry->fds_io[idx].fd;

	if (fd == -1)
		return NULL;

	sd_debug("%s, idx %d", addr_to_str(addr, port), idx);

	sfd = xmalloc(sizeof(*sfd));
	sfd->fd = fd;
	sfd->idx = idx;
	sfd->isIO = isIO; /*Need to know which fds array we are using*/

	tracepoint(sockfd_cache, cache_get, 0);

	return sfd;
}

static void sockfd_cache_put_long(const struct node_id *nid, int idx, bool isIO)
{
	bool use_io = nid->io_port ? true : false;
	const uint8_t *addr = use_io ? nid->io_addr : nid->addr;
	int port = use_io ? nid->io_port : nid->port;
	struct sockfd_cache_entry *entry;

	sd_debug("%s idx %d", addr_to_str(addr, port), idx);

	sd_read_lock(&sockfd_cache.lock);
	entry = sockfd_cache_search(nid);
	if (!entry) {
		sd_rw_unlock(&sockfd_cache.lock);
		return;
	}
	if (!isIO)
		uatomic_set_false(&entry->fds_nio[idx].in_use);
	else
		uatomic_set_false(&entry->fds_io[idx].in_use);

	sd_rw_unlock(&sockfd_cache.lock);
}

static void sockfd_cache_close(const struct node_id *nid, int idx, bool isIO)
{
	bool use_io = nid->io_port ? true : false;
	const uint8_t *addr = use_io ? nid->io_addr : nid->addr;
	int port = use_io ? nid->io_port : nid->port;
	struct sockfd_cache_entry *entry;

	sd_debug("%s idx %d", addr_to_str(addr, port), idx);

	sd_write_lock(&sockfd_cache.lock);
	entry = sockfd_cache_search(nid);
	if (!entry) {
		sd_rw_unlock(&sockfd_cache.lock);
		return;
	}
	if (!isIO) {
		close(entry->fds_nio[idx].fd);
		entry->fds_nio[idx].fd = -1;
		uatomic_set_false(&entry->fds_nio[idx].in_use);
	} else {
		close(entry->fds_io[idx].fd);
		entry->fds_io[idx].fd = -1;
		uatomic_set_false(&entry->fds_io[idx].in_use);
	}
	sd_rw_unlock(&sockfd_cache.lock);
}

/*
 * Create work queue for growing fds.
 * Before this function called, growing cannot be done.
 */
int sockfd_init(void)
{
	grow_wq = create_ordered_work_queue("sockfd_grow");

	if (!grow_wq) {
		sd_err("error at creating workqueue for sockfd growth");
		return -1;
	}

	return 0;
}

/*
 * Return a sockfd connected to the node to the caller
 *
 * Try to get a 'long' FD as best, which is cached and never closed. If no FD
 * available, we return a 'short' FD which is supposed to be closed by
 * sockfd_cache_put().
 *
 * ret_idx is opaque to the caller, -1 indicates it is a short FD.
 */
struct sockfd *sockfd_cache_get(const struct node_id *nid)
{
	struct sockfd *sfd;
	int fd;

	sfd = sockfd_cache_get_long(nid);
	if (sfd)
		return sfd;

	/* Fallback on a non-io connection that is to be closed shortly */
	fd = connect_to_addr(nid->addr, nid->port);
	if (fd < 0)
		return NULL;

	sfd = xmalloc(sizeof(*sfd));
	sfd->idx = -1;
	sfd->fd = fd;
	sfd->isIO = false;
	sd_debug("%d", fd);
	return sfd;
}

/*
 * Release a sockfd connected to the node, which is acquired from
 * sockfd_cache_get()
 *
 * If it is a long FD, just decrease the refcount to make it available again.
 * If it is a short FD, close it.
 */
void sockfd_cache_put(const struct node_id *nid, struct sockfd *sfd)
{
	if (sfd->idx == -1) {
		assert(!isIO);
		sd_debug("%d", sfd->fd);
		close(sfd->fd);
		free(sfd);

		tracepoint(sockfd_cache, cache_put, 0);
		return;
	}

	sockfd_cache_put_long(nid, sfd->idx, sfd->isIO);
	free(sfd);

	tracepoint(sockfd_cache, cache_put, 1);
}

/* Delete all sockfd connected to the node, when node is crashed. */
void sockfd_cache_del_node(const struct node_id *nid)
{
	int n;

	if (!sockfd_cache_destroy(nid))
		return;

	n = uatomic_sub_return(&sockfd_cache.count, 1);
	sd_debug("%s, count %d", addr_to_str(nid->addr, nid->port), n);
}

/*
 * Delete a sockfd connected to the node.
 *
 * If it is a long FD, de-refcount it and try to destroy all the cached FDs of
 * this node in the cache.
 * If it is a short FD, just close it.
 */
void sockfd_cache_del(const struct node_id *nid, struct sockfd *sfd)
{
	if (sfd->idx == -1) {
		assert(!isIO);
		sd_debug("%d", sfd->fd);
		close(sfd->fd);
		free(sfd);
		return;
	}

	sockfd_cache_close(nid, sfd->idx, sfd->isIO);
	sockfd_cache_del_node(nid);
	free(sfd);
}

void init_to_connect_list()
{
	INIT_LIST_HEAD(&to_connect_list);
}

static void *monitor_sd_node_connectivity(void *ignored)
{
	int err;

	sd_info("node connectivity monitor main loop");
	init_to_connect_list();

	for (;;) {
		struct sockfd_cache_entry *entry;

		if (list_empty(&to_connect_list))
			sleep(MONITOR_INTERVAL);
		else {
			struct nid_test_work *work =
				list_first_entry(&to_connect_list,
					struct nid_test_work, w_list);
			struct node_id *nid = &work->nid;

			if (nid->io_port) {
				int fd = connect_to_addr(nid->io_addr,
					       nid->io_port);
				sd_write_lock(&sockfd_cache.lock);
				entry = sockfd_cache_search(nid);
				if (entry) {
					if (fd > 0) {
						close(fd);
						entry->channel_status = IO;
						list_del(&work->w_list);
						free(work);
					} else { /*still can not connect*/
						list_del(&work->w_list);
						list_add_tail(&work->w_list,
							&to_connect_list);
					}
				} else {
					list_del(&work->w_list);
					free(work);
					sd_err("entry for node %s not exists",
						addr_to_str(nid->addr,
							nid->port));
					if (fd > 0)
						close(fd);
				}
				sd_rw_unlock(&sockfd_cache.lock);
			}
		}
		sd_write_lock(&sockfd_cache.lock);
		rb_for_each_entry(entry, &sockfd_cache.root, rb) {
			if (entry->fds_io)
				prepare_conns(entry, false);
		}
		sd_rw_unlock(&sockfd_cache.lock);
	}

	err = pthread_detach(pthread_self());
	if (err)
		sd_err("%s", strerror(err));
	pthread_exit(NULL);
}

int start_node_connectivity_monitor(void)
{
	sd_thread_t t;
	int err;
	err = sd_thread_create("monio", &t, monitor_sd_node_connectivity, NULL);
	if (err) {
		sd_err("%s", strerror(err));
		return -1;
	}
	return 0;
}
