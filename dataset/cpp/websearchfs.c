/*
 * Copyright 2004-2010, François Revol, <revol@free.fr>.
 * Distributed under the terms of the MIT License.
 */

/*
 * websearchfs - a bookmark-populated virtual filesystem using DuckDuckGo results.
 */

#define _BUILDING_fs 1

#include <sys/param.h>
#include <sys/stat.h>
#include <malloc.h>
#include <KernelExport.h>
//#include <NodeMonitor.h>
#include <stddef.h>
#include <stdio.h>
#include <signal.h>
#include <string.h>
#include <fs_query.h>
#include "query.h"
#include "websearchfs.h"
#include "vnidpool.h"
#include "duckduckgo_request.h"
#include "settings.h"

/* just publish fake entries; for debugging */
//#define NO_SEARCH

#define PFS "websearchfs: "

#define TRACE_WEBSEARCHFS
#ifdef TRACE_WEVSEARCHFS
#	define TRACE(x...) fprintf(stderr, PFS x)
#else
#	define TRACE(x...)
#endif


/* needed to get /bin/df tell the mountpoint... */
#define ALLOW_DIR_OPEN

int32 refcount = 0;


extern struct attr_entry root_folder_attrs[];
extern struct attr_entry folders_attrs[];
extern struct attr_entry bookmark_attrs[];
extern struct attr_entry fake_bookmark_attrs[]; /* for debugging */
extern struct attr_entry template_1_attrs[];
extern struct attr_entry text_attrs[];
extern struct attr_entry mailto_me_bookmark_attrs[];

extern char *readmestr;

static fs_volume_ops sWebSearchFSVolumeOps;
static fs_vnode_ops sWebSearchFSVnodeOps;


static int websearchfs_create_gen(fs_volume *_volume, fs_node *dir, const char *name, int omode, int perms, ino_t *vnid, fs_node **node, struct attr_entry *iattrs, bool mkdir, bool uniq);
static int websearchfs_free_vnode(fs_volume *_volume, fs_node *node);

static void fill_default_stat(struct stat *st, nspace_id nsid, ino_t vnid, mode_t mode)
{
	time_t tm = time(NULL);
	st->st_dev = nsid;
	st->st_ino = vnid;
	st->st_mode = mode;
	st->st_nlink = 1;
	st->st_uid = 0;
	st->st_gid = 0;
	st->st_size = 0LL;
	st->st_blksize = 1024;
	st->st_atime = tm;
	st->st_mtime = tm;
	st->st_ctime = tm;
	st->st_crtime = tm;
}

/**	Publishes some entries in the root vnode: a query template, the readme file, and a People file of the author.
 */
static int websearchfs_publish_static_entries(fs_volume *_volume)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	status_t err = B_OK;
	fs_node *dir = ns->root;
	fs_node *n;// *dummy;
	//char ename[WEBSEARCHFS_NAME_LEN];
	//char *p;
	//int i;
	TRACE("websearchfs_publish_static_entries(%" B_PRId32 ")\n", ns->nsid);
	if (!ns || !dir)
		return EINVAL;

	err = websearchfs_create_gen(_volume, dir, "Search the Web", 0, 0444, NULL, &n, template_1_attrs, false, true);
	if (err)
		return err;
	n->is_perm = 1;

	err = websearchfs_create_gen(_volume, dir, "README", 0, 0444, NULL, &n, text_attrs, false, true);
	if (err)
		return err;
	n->is_perm = 1;
	n->data = readmestr;
	n->data_size = strlen(n->data);// + 1;

	err = websearchfs_create_gen(_volume, dir, "Author", 0, 0444, NULL, &n, mailto_me_bookmark_attrs, false, true);
	if (err)
		return err;
	n->is_perm = 1;

	return B_OK;

/*
err:
	TRACE("push_result_to_query: error 0x%08"B_PRIx32"\n", err);
	return err;
*/
}

static status_t websearchfs_mount(fs_volume *_vol, const char *devname, uint32 flags,
		const char *parms, ino_t *vnid)
{
	fs_nspace *ns;
	fs_node *root;
	int err;
	TRACE("mount(%p, %s, 0x%08" B_PRIx32 ", %s, , )\n", _vol, devname, flags, parms);

	/* only allow a single mount */
	if (atomic_add(&refcount, 1))
		return EALREADY;

	err = load_settings();

	ns = malloc(sizeof(fs_nspace));
	if (!ns)
		return B_NO_MEMORY;
	memset(ns, 0, sizeof(fs_nspace));
	ns->nsid = _vol->id;

	err = vnidpool_alloc(&ns->vnids, MAX_VNIDS);
	if (err < 0)
		return err;
	err = vnidpool_get(ns->vnids, &ns->rootid);
	if (err < 0)
		return err;
	atomic_add(&ns->nodecount, 1);

	new_lock(&(ns->l), "websearchfs main lock");

	ns->nodes = NULL;

	/* create root dir */
	err = B_NO_MEMORY;
	root = malloc(sizeof(fs_node));
	ns->root = root;
	if (root) {
		memset(root, 0, sizeof(fs_node));
		strcpy(root->name, ".");
		root->is_perm = 1;
		root->vnid = ns->rootid;
		fill_default_stat(&root->st, ns->nsid, ns->rootid, 0777 | S_IFDIR);
		root->attrs_indirect = root_folder_attrs;
		new_lock(&(root->l), "websearchfs root dir");
		TRACE("mount: root->l @ %p\n", &root->l);

		_vol->private_volume = ns;
		_vol->ops = &sWebSearchFSVolumeOps;
		*vnid = ns->rootid;
		ns->nodes = root; // sll_insert
		err = publish_vnode(_vol, *vnid, root, &sWebSearchFSVnodeOps, S_IFDIR, 0);
		if (err == B_OK) {
			websearchfs_publish_static_entries(_vol);
			TRACE("mount() OK, nspace@ %p, id %" B_PRId32 ", root@ %p, id %" B_PRId64 "\n", ns, ns->nsid, root, ns->rootid);
			return B_OK;
		}
		free_lock(&root->l);
		free(root);
	}
	free_lock(&ns->l);
	free(ns);
	atomic_add(&refcount, -1);
	return err;
}

static status_t websearchfs_unmount(fs_volume *_volume)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	status_t err;
	struct fs_node *node;
	TRACE("unmount(%" B_PRId32 ")\n", ns->nsid);
	err = LOCK(&ns->l);
	if (err)
		return err;
	/* anything in still in use ? */
	for (node = ns->nodes; node; node = ns->nodes) {
		ns->nodes = node->nlnext; /* better cache that before we free node */
		websearchfs_free_vnode(_volume, node);
	}

	// Unlike in BeOS, we need to put the reference to our root node ourselves
	put_vnode(_volume, ns->rootid);

	free_lock(&ns->l);
	vnidpool_free(ns->vnids);
	free(ns);

	atomic_add(&refcount, -1);

	return B_OK;
}

static int compare_fs_node_by_vnid(fs_node *node, ino_t *id)
{
	return !(node->vnid == *id);
}

static int websearchfs_free_vnode(fs_volume *_volume, fs_node *node)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	TRACE("%s(%" B_PRId32 ", %" B_PRId64 ")\n", __FUNCTION__, ns->nsid, node->vnid);
	free_lock(&node->l);
	atomic_add(&ns->nodecount, -1);
	vnidpool_put(ns->vnids, node->vnid);
	if (node->request)
		duckduckgo_request_free(node->request);
	free(node->result);
	free(node);
	return 0;
}

static status_t websearchfs_remove_vnode(fs_volume *_volume, fs_vnode *_node, bool reenter)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	status_t err = B_OK;
	TRACE("%s(%" B_PRId32 ", %" B_PRId64 ", %s)\n", __FUNCTION__, ns->nsid, node->vnid, reenter?"r":"!r");
	if (!reenter)
		err = LOCK(&ns->l);
	if (err)
		return err;
	if (node->vnid == ns->rootid) {
		TRACE("asked to remove the root node!!\n");
	}
TRACE("SLL_REMOVE(ns->nodes %p, nlnext, %p)\n", ns->nodes, node);
	//LOCK(&node->l);
	err = SLL_REMOVE(ns->nodes, nlnext, node);
	/* query dirs must be removed from the query list too */
TRACE("SLL_REMOVE(ns->queries %p, qnext, %p)\n", ns->nodes, node);
	err = SLL_REMOVE(ns->queries, qnext, node);
	if (node->parent) {
		LOCK(&node->parent->l);
TRACE("SLL_REMOVE(node->parent->children %p, next, %p)\n", node->parent->children, node);
		SLL_REMOVE(node->parent->children, next, node);
		UNLOCK(&node->parent->l);
	}
	websearchfs_free_vnode(_volume, node);
	if (!reenter)
		UNLOCK(&ns->l);
	return err;
}

static status_t websearchfs_read_vnode(fs_volume *_volume, ino_t vnid, fs_vnode *_node, int* _type, uint32* _flags, bool reenter)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *n;
	status_t err = B_OK;
	TRACE("%s(%" B_PRId32 ", %" B_PRId64 ", %s)\n", __FUNCTION__, _volume->id, vnid, reenter?"r":"!r");
	if (!reenter)
		err = LOCK(&ns->l);
	if (err)
		return err;
	n = (fs_node *)SLL_FIND(ns->nodes, nlnext, (sll_compare_func)compare_fs_node_by_vnid, (void *)&vnid);
	if (n) {
		_node->private_node = n;
		_node->ops = &sWebSearchFSVnodeOps;
		*_type = n->st.st_mode & ~S_IUMSK; /*XXX: S_IFMT ?*/
		*_flags = 0;

	} else
		err = ENOENT;
	if (!reenter)
		UNLOCK(&ns->l);
	return err;
}

static status_t websearchfs_release_vnode(fs_volume *_volume, fs_vnode *_node, bool reenter)
{
	fs_node *node = (fs_node *)_node->private_node;
	TRACE("%s(%" B_PRId32 ", %" B_PRId64 ", %s)\n", __FUNCTION__, _volume->id, node->vnid, reenter?"r":"!r");
	return B_OK;
}

static int compare_fs_node_by_name(fs_node *node, char *name)
{
	//return memcmp(node->name, name, WEBSEARCHFS__NAME_LEN);
	//TRACE("find_by_name: '%s' <> '%s'\n", node->name, name);
	return strncmp(node->name, name, WEBSEARCHFS_NAME_LEN);
}

static status_t websearchfs_get_vnode_name(fs_volume *_volume, fs_vnode *_node, char *buffer, size_t len)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;

	TRACE("get_vnode_name(%" B_PRId32 ", %" B_PRId64 ", )\n", ns->nsid, (int64)(node?node->vnid:-1));
	strlcpy(buffer, node->name, MIN(WEBSEARCHFS_NAME_LEN, len));
	return B_OK;
}


static status_t websearchfs_walk(fs_volume *_volume, fs_vnode *_base, const char *file, ino_t *vnid)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *base = _base->private_node;
	fs_node *n, *dummy;
	status_t err = B_OK;
	TRACE("walk(%" B_PRId32 ", %" B_PRId64 ", %s)\n", ns->nsid, (int64)(base?base->vnid:-1), file);
	err = LOCK(&base->l);
	if (err)
		return err;
	if (!file) {
		err = EINVAL;
	} else if (!strcmp(file, "..")) {
		if (base && base->parent) {
			*vnid = base->parent->vnid; // XXX: LOCK(&base->l) ?
			//*type = S_IFDIR;
		} else
			err = EINVAL;
	} else if (!strcmp(file, ".")) { /* root dir */
		if (base) { // XXX: LOCK(&base->l) ?
			*vnid = base->vnid;
			//*type = S_IFDIR;
		} else
			err = EINVAL;
	} else if (base) { /* child of dir */
		n = (fs_node *)SLL_FIND(base->children, next,
								(sll_compare_func)compare_fs_node_by_name, (void *)file);
		if (n) {
			*vnid = n->vnid;
			//*type = n->st.st_type & ~S_IUMSK; /*XXX: S_IFMT ?*/
		} else
			err = ENOENT;
	} else
		err = ENOENT;
	if (err == B_OK) {
		if (get_vnode(_volume, *vnid, (void **)&dummy) != B_OK) /* inc ref count */
			err = EINVAL;
	}
	UNLOCK(&base->l);
	TRACE("walk() -> error 0x%08" B_PRIx32 "\n", err);
	return err;
}

static status_t websearchfs_opendir(fs_volume *_volume, fs_vnode *_node, void **cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	status_t err = B_OK;
	fs_dir_cookie *c;
	TRACE("opendir(%" B_PRId32 ", %" B_PRId64 ")\n", ns->nsid, node->vnid);
	if (!node)
		return EINVAL;
	if (!S_ISDIR(node->st.st_mode))
		return B_NOT_A_DIRECTORY;
	err = LOCK(&node->l);
	if (err)
		return err;
	c = malloc(sizeof(fs_dir_cookie));
	if (c) {
		memset(c, 0, sizeof(fs_dir_cookie));
		c->omode = O_RDONLY;
		c->type = S_IFDIR;
		c->node = node;
		c->dir_current = 0;
		*cookie = (void *)c;
		SLL_INSERT(node->opened, next, c);
		UNLOCK(&node->l);
		return B_OK;
	} else
		err = B_NO_MEMORY;
	UNLOCK(&node->l);
	return err;
}

static status_t websearchfs_closedir(fs_volume *_volume, fs_vnode *_node, void *_cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_dir_cookie *cookie = (fs_dir_cookie *)_cookie;
	status_t err = B_OK;
//	node = cookie->node; // work around VFS bug
	TRACE("closedir(%" B_PRId32 ", %" B_PRId64 ", %p)\n", ns->nsid, node->vnid, cookie);
	err = LOCK(&node->l);
	if (err)
		return err;

	SLL_REMOVE(node->opened, next, cookie);
	UNLOCK(&node->l);

	return err;
}

static status_t websearchfs_rewinddir(fs_volume *_volume, fs_vnode *_node, void *_cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_dir_cookie *cookie = (fs_dir_cookie *)_cookie;
	TRACE("rewinddir(%" B_PRId32 ", %" B_PRId64 ")\n", ns->nsid, node->vnid);
	cookie->dir_current = 0;
	return B_OK;
}

static status_t websearchfs_readdir(fs_volume *_volume, fs_vnode *_node, void *_cookie,
	struct dirent *buf, size_t bufsize, uint32 *num)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_dir_cookie *cookie = (fs_dir_cookie *)_cookie;
	fs_node *n = NULL;
	fs_node *parent = node->parent;
	int index;
	TRACE("readdir(%" B_PRId32 ", %" B_PRId64 ") @ %d\n", ns->nsid, node->vnid,
		cookie->dir_current);
	if (!node || !cookie || !num || !*num || !buf
		|| (bufsize < (sizeof(dirent_t) + WEBSEARCHFS_NAME_LEN)))
		return EINVAL;
	LOCK(&node->l);
	if (cookie->dir_current == 0) { /* .. */
		TRACE("readdir: giving ..\n");
		/* the VFS will correct that anyway */
		buf->d_dev = ns->nsid;
		buf->d_pdev = ns->nsid;
		buf->d_ino = parent?parent->vnid:ns->rootid;
		buf->d_pino = (parent && parent->parent)?parent->parent->vnid:ns->rootid;
		strcpy(buf->d_name, "..");
		buf->d_reclen = offsetof(struct dirent, d_name)+strlen(buf->d_name)+1;
		cookie->dir_current++;
		*num = 1;
	} else if (cookie->dir_current == 1) { /* . */
		TRACE("readdir: giving .\n");
		/* the VFS will correct that anyway */
		buf->d_dev = ns->nsid;
		buf->d_pdev = ns->nsid;
		buf->d_ino = node->vnid;
		buf->d_pino = parent?parent->vnid:ns->rootid;
		strcpy(buf->d_name, ".");
		buf->d_reclen = offsetof(struct dirent, d_name)+strlen(buf->d_name)+1;
		cookie->dir_current++;
		*num = 1;
	} else {
		index = cookie->dir_current-2;
		for (n = node->children; n && index; n = n->next, index--); //XXX: care about n->hidden || n->deleted
		if (n) {
			TRACE("readdir: giving ino %" B_PRId64 ", %s\n", n->vnid, n->name);
			buf->d_dev = ns->nsid;
			buf->d_pdev = ns->nsid;
			buf->d_ino = n->vnid;
			buf->d_pino = node->vnid;
			strcpy(buf->d_name, n->name);
			buf->d_reclen = offsetof(struct dirent, d_name)+strlen(buf->d_name)+1;
			cookie->dir_current++;
			*num = 1;
		} else {
			*num = 0;
		}
	}
	UNLOCK(&node->l);
	return B_OK;
}

static status_t websearchfs_free_dircookie(fs_volume *_volume, fs_vnode *_node, void *_cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_dir_cookie *cookie = (fs_dir_cookie *)_cookie;
	status_t err = B_OK;
//	node = cookie->node; // work around VFS bug
	TRACE("freedircookie(%" B_PRId32 ", %" B_PRId64 ", %p)\n", ns->nsid, node?node->vnid:(int64)0, (void *)cookie);
	err = LOCK(&node->l);
	if (err)
		return err;
	err = SLL_REMOVE(node->opened, next, cookie); /* just to make sure */
	UNLOCK(&node->l);
	free(cookie);
	return B_OK;
}

static status_t websearchfs_rstat(fs_volume *_volume, fs_vnode *_node, struct stat *st)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	status_t err = B_OK;
	if (!node || !st)
		return EINVAL;
	err = LOCK(&node->l);
	if (err)
		return err;
	memcpy(st, &node->st, sizeof(struct stat));
	st->st_dev = ns->nsid;
	st->st_ino = node->vnid;
	if (node->data_size)
		st->st_size = node->data_size;
	//st->st_size = 0LL;
	UNLOCK(&node->l);
	return err;
}

static status_t websearchfs_rfsstat(fs_volume *_volume, struct fs_info *info)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	info->block_size = 1024; // websearchfs_BUFF_SIZE;
	info->io_size = 1024; // WEBSEARCHFS_BUFF_SIZE;
	info->total_blocks=0;
	info->free_blocks=0;
	info->total_nodes=MAX_VNIDS;
	info->free_nodes=ns->nodecount;
	info->dev=ns->nsid;
	info->root=ns->rootid;
	info->flags=/*B_FS_IS_SHARED|*/B_FS_IS_PERSISTENT|B_FS_HAS_MIME|B_FS_HAS_ATTR|B_FS_HAS_QUERY;
	strcpy (info->device_name, "");
	strcpy (info->volume_name, "Web Search");
	strcpy (info->fsh_name, WEBSEARCHFS_NAME);
	return B_OK;
}

static status_t websearchfs_open(fs_volume *_volume, fs_vnode *_node, int omode, void **cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	status_t err = B_OK;
	//fs_node *dummy;
	fs_file_cookie *fc;
	TRACE("open(%" B_PRId32 ", %" B_PRId64 ", 0x%x)\n", ns->nsid, node->vnid, omode);
	if (!node || !cookie)
		return EINVAL;

//	err = LOCK(&ns->l);
//	if (err)
//		return err;
	err = LOCK(&node->l);
	if (err)
		goto err_n_l;
	err = EEXIST;
#ifndef ALLOW_DIR_OPEN
	err = EINVAL;//EISDIR;
	if (S_ISDIR(node->st.st_mode))
		goto err_malloc;
#endif
	err = B_NO_MEMORY;
	fc = malloc(sizeof(fs_file_cookie));
	if (!fc)
		goto err_malloc;
	memset(fc, 0, sizeof(fs_file_cookie));
	fc->node = node;
	fc->omode = omode;
	fc->type = S_IFREG;
	err = SLL_INSERT(node->opened, next, fc);
	if (err)
		goto err_linsert;
/*	err = get_vnode(ns->nsid, node->vnid, &dummy);
	if (err)
		goto err_getvn;*/
	//*vnid = node->vnid;
	*cookie = (void *)fc;
	err = B_OK;
	goto all_ok;
//err_:
//	put_vnode(ns->nsid, node->nsid);
//err_getvn:
//	SLL_REMOVE(node->opened, next, fc);
err_linsert:
	free(fc);
err_malloc:
all_ok:
	UNLOCK(&node->l);
err_n_l:
//	UNLOCK(&ns->l);
	return err;
}

static status_t websearchfs_close(fs_volume *_volume, fs_vnode *_node, void *_cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_file_cookie *cookie = (fs_file_cookie *)_cookie;
	status_t err;
	TRACE("close(%" B_PRId32 ", %" B_PRId64 ")\n", ns->nsid, node->vnid);
	if (!ns || !node || !cookie)
		return EINVAL;
	err = LOCK(&node->l);
	if (err)
		return err;
	SLL_REMOVE(node->opened, next, cookie);

//all_ok:
//err_n_l:
	UNLOCK(&node->l);
	return err;
}

static status_t websearchfs_free_cookie(fs_volume *_volume, fs_vnode *_node, void *_cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_file_cookie *cookie = (fs_file_cookie *)_cookie;
	status_t err = B_OK;
	TRACE("freecookie(%" B_PRId32 ", %" B_PRId64 ")\n", ns->nsid, node->vnid);
	err = LOCK(&node->l);
	if (err)
		return err;
	err = SLL_REMOVE(node->opened, next, cookie); /* just to make sure */
//	if (err)
//		goto err_n_l;
	if (/*!node->is_perm &&*/ false) { /* not yet */
		err = remove_vnode(_volume, node->vnid);
		ns->root->st.st_mtime = time(NULL);
#if 0
		notify_listener(B_ENTRY_REMOVED, ns->nsid, ns->rootid, 0LL, node->vnid, NULL);
		notify_listener(B_STAT_CHANGED, ns->nsid, 0LL, 0LL, ns->rootid, NULL);
#endif
	}
	UNLOCK(&node->l);
	free(cookie);
//	err = B_OK;
//err_n_l:
	return err;
}

static status_t websearchfs_read(fs_volume *_volume, fs_vnode *_node, void *_cookie, off_t pos, void *buf, size_t *len)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	status_t err = B_OK;
	TRACE("read(%" B_PRId32 ", %" B_PRId64 ", %jd, %zu)\n", ns->nsid, node->vnid, pos, *len);
	if (pos < 0 || (size_t)pos > node->data_size)
		err = EFPOS;
	if (err || node->data_size == 0 || !node->data) {
		*len = 0;
		return err;
	}
	*len = MIN(*len, node->data_size - (long)pos);
	memcpy(buf, ((char *)node->data) + pos, *len);
	return B_OK;
}

static status_t websearchfs_write(fs_volume *_volume, fs_vnode *_node, void *_cookie, off_t pos, const void *buf, size_t *len)
{
	fs_node *node = (fs_node *)_node->private_node;
	TRACE("write(%" B_PRId32 ", %" B_PRId64 ", %jd, %zu)\n", _volume->id, node->vnid, pos, *len);
	*len = 0;
	return ENOSYS;
}

static status_t websearchfs_wstat(fs_volume *_volume, fs_vnode *_node, const struct stat *st, uint32 mask)
{
	fs_node *node = (fs_node *)_node->private_node;
	TRACE("wstat(%" B_PRId32 ", %" B_PRId64 ", , 0x%08" B_PRIx32 ")\n", _volume->id, node->vnid, mask);
	return ENOSYS;
}

static status_t websearchfs_wfsstat(fs_volume *_volume, const struct fs_info *info, uint32 mask)
{
	TRACE("wfsstat(%" B_PRId32 ", , 0x%08" B_PRIx32 ")\n", _volume->id, mask);
	return ENOSYS;
}

/* this one returns the created fs_node to caller (for use by query engine) */
/**
 * @param dir the dir's fs_node we mkdir in
 * @param name name to mkdir (basename is uniq is set)
 * @param perms create with those permissions
 * @param node make this point to the fs_node if !NULL
 * @param iattr indirect attributes to set if desired (must be statically allocated)
 * @param mkdir create a directory instead of a file
 * @param uniq choose an unique name, appending a number if required
 */
static int websearchfs_create_gen(fs_volume *_volume, fs_node *dir, const char *name, int omode, int perms, ino_t *vnid, fs_node **node, struct attr_entry *iattrs, bool mkdir, bool uniq)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	//fs_node *dir = (fs_node *)_dir->private_node;
	char newname[WEBSEARCHFS_NAME_LEN];
	status_t err;
	fs_node *n;
	int i;
	TRACE("create_gen(%" B_PRId32 ", %" B_PRId64 ", '%s', 0x%08x, %c, %c)\n", ns->nsid, dir->vnid, name, omode, mkdir?'t':'f', uniq?'t':'f');

	if (strlen(name) > WEBSEARCHFS_NAME_LEN-1)
		return ENAMETOOLONG;
	err = LOCK(&dir->l);
	if (err < 0)
		return err;
	err = ENOTDIR;
	if (!S_ISDIR(dir->st.st_mode))
		goto err_l;
	n = (fs_node *)SLL_FIND(dir->children, next,
							(sll_compare_func)compare_fs_node_by_name, (void *)name);
	err = EEXIST;
	if (n && (omode & O_EXCL) && !uniq) /* already existing entry in there! */
		goto err_l;

	strncpy(newname, name, WEBSEARCHFS_NAME_LEN);
	newname[WEBSEARCHFS_NAME_LEN-1] = '\0';

	for (i = 1; uniq && n && i < 5000; i++) { /* uniquify the name */
		//sprintf("%"#(WEBSEARCHFS_NAME_LEN-8)"s %05d", name, i);
		strncpy(newname, name, 56);
		newname[56] = '\0';
		sprintf(newname+strlen(newname), " %05d", i);
		n = (fs_node *)SLL_FIND(dir->children, next,
								(sll_compare_func)compare_fs_node_by_name, (void *)newname);
	}
	if (n && (uniq || mkdir)) /* still there! */
		goto err_l;
	name = newname;

	if (n) { /* already exists, so return it */
		if (node)
			*node = n;
		if (vnid)
			*vnid = n->vnid;
		err = B_OK;
		goto done;
	}
	err = ENOMEM;
	n = malloc(sizeof(fs_node));
	if (!n)
		goto err_l;
	memset(n, 0, sizeof(fs_node));
	err = vnidpool_get(ns->vnids, &n->vnid);
	if (err < B_OK)
		goto err_m;
	atomic_add(&ns->nodecount, 1);
	strcpy(n->name, name);
	//n->is_perm = 1;
	fill_default_stat(&n->st, ns->nsid, n->vnid, (perms & ~S_IFMT) | (mkdir?S_IFDIR:S_IFREG));

	new_lock(&(n->l), mkdir?"websearchfs dir":"websearchfs file");

	err = LOCK(&ns->l);
	if (err)
		goto err_nl;
	err = SLL_INSERT(ns->nodes, nlnext, n);
	if (err)
		goto err_lns;
	/* _TAIL so they are in order */
	err = SLL_INSERT(dir->children, next, n);
	if (err)
		goto err_insnl;
//	err = new_vnode(ns->nsid, n->vnid, n);
//	if (err)
//		goto err_ins;
	n->parent = dir;
	dir->st.st_nlink++;
	UNLOCK(&ns->l);
	n->attrs_indirect = iattrs;
	notify_entry_created(ns->nsid, dir->vnid, name, n->vnid);
	/* dosfs doesn't do that one but I believe it should */
	notify_stat_changed(B_STAT_CHANGED, -1, ns->nsid, -1);
	/* give node to caller if it wants it */
	if (node)
		*node = n;
	if (vnid)
		*vnid = n->vnid;
	goto done;

err_insnl:
	SLL_REMOVE(ns->nodes, nlnext, n);
err_lns:
	UNLOCK(&ns->l);
err_nl:
	free_lock(&n->l);
	atomic_add(&ns->nodecount, -1);
err_m:
	free(n);
err_l:
done:
	UNLOCK(&dir->l);
	return err;
}

static status_t websearchfs_create(fs_volume *_volume, fs_vnode *_dir, const char *name, int omode, int perms, void **cookie, ino_t *vnid)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *dir = (fs_node *)_dir->private_node;
	status_t err;
	fs_node *n;
	struct fs_vnode child = { NULL, &sWebSearchFSVnodeOps };
	TRACE("create(%" B_PRId32 ", %" B_PRId64 ", '%s', 0x%08x)\n", ns->nsid, dir->vnid, name, omode);
	/* don't let ppl mess our fs up */
	return ENOSYS;

	err = websearchfs_create_gen(_volume, dir, name, omode, perms, vnid, &n, NULL, false, false);
	if (err)
		return err;

	child.private_node = (void *)n;
	err = websearchfs_open(_volume, &child, omode, cookie);
	return err;
}

static int websearchfs_unlink_gen(fs_volume *_volume, fs_node *dir, const char *name)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	status_t err;
	fs_node *n;
	TRACE("unlink(%" B_PRId32 ", %" B_PRId64 ", %s)\n", ns->nsid, dir->vnid, name);
	//dprintf(PFS"unlink(%" B_PRId32 ", %" B_PRId64 ", %s)\n", ns->nsid, dir->vnid, name);
	err = LOCK(&dir->l);
	if (err)
		return err;
	err = ENOENT;
	/* no need to check for S_ISDIR */
	n = (fs_node *)SLL_FIND(dir->children, next,
							(sll_compare_func)compare_fs_node_by_name, (void *)name);
	if (n) {
		if (n->children)
			err = ENOTEMPTY;
		else if (n->is_perm)
			err = EROFS;
		//else if (S_ISDIR(n->st.st_mode))
		//	err = EISDIR;
		else if (n->vnid == ns->rootid)
			err = EACCES;
		else {
			SLL_REMOVE(dir->children, next, n);
			notify_entry_removed(ns->nsid, dir->vnid, name, n->vnid);
			//notify_listener(B_STAT_CHANGED, ns->nsid, 0LL, 0LL, dir->vnid, NULL);
			remove_vnode(_volume, n->vnid);
			err = B_OK;
		}
	}
	UNLOCK(&dir->l);
	return err;
}

static status_t websearchfs_unlink(fs_volume *_volume, fs_vnode *_dir, const char *name)
{
	//fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	//fs_node *dir = (fs_node *)_dir->private_node;
	return websearchfs_unlink_gen(_volume, (fs_node *)_dir->private_node, name);
}

static status_t websearchfs_rmdir(fs_volume *_volume, fs_vnode *_dir, const char *name)
{
	//fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *dir = (fs_node *)_dir->private_node;
	TRACE("rmdir(%" B_PRId32 ", %" B_PRId64 ", %s)\n", _volume->id, dir->vnid, name);
	return websearchfs_unlink(_volume, _dir, name);
}

static int websearchfs_unlink_node_rec(fs_volume *_volume, fs_node *node)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	status_t err;
	fs_node *n;
	TRACE("websearchfs_unlink_node_rec(%" B_PRId32 ", %" B_PRId64 ":%s)\n", ns->nsid, node->vnid, node->name);
	if (!ns || !node)
		return EINVAL;
	// kill_request();
	LOCK(&node->l);
	while (1) {
		n = node->children;
		if (!n)
			break;
		UNLOCK(&node->l);
		err = websearchfs_unlink_node_rec(_volume, n);
		LOCK(&node->l);
	}
	UNLOCK(&node->l);
	err = websearchfs_unlink_gen(_volume, node->parent, node->name);
	return err;
}

static status_t websearchfs_access(fs_volume *_volume, fs_vnode *_node, int mode)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	TRACE("access(%" B_PRId32 ", %" B_PRId64 ", 0x%x)\n", ns->nsid, node->vnid, mode);
	return B_OK;
}


static status_t websearchfs_mkdir(fs_volume *_volume, fs_vnode *_dir, const char *name, int perms)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *dir = (fs_node *)_dir->private_node;
	TRACE("mkdir(%" B_PRId32 ", %" B_PRId64 ", '%s', 0x%08x)\n", ns->nsid, dir->vnid, name, perms);
	return websearchfs_create_gen(_volume, dir, name, O_EXCL, perms, NULL, NULL, folders_attrs, true, false);
}

/* attr stuff */

static status_t websearchfs_open_attrdir(fs_volume *_volume, fs_vnode *_node, void **cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	status_t err = B_OK;
	fs_attr_dir_cookie *c;
	TRACE("open_attrdir(%" B_PRId32 ", %" B_PRId64 ")\n", ns->nsid, node->vnid);
	if (!node)
		return EINVAL;
	err = LOCK(&node->l);
	if (err)
		return err;
	c = malloc(sizeof(fs_attr_dir_cookie));
	if (c) {
		memset(c, 0, sizeof(fs_attr_dir_cookie));
		c->omode = O_RDONLY;
		c->type = S_ATTR_DIR;
		c->node = node;
		c->dir_current = 0;
		*cookie = (void *)c;
		SLL_INSERT(node->opened, next, c);
		UNLOCK(&node->l);
		return B_OK;
	} else
		err = B_NO_MEMORY;
	UNLOCK(&node->l);
	return err;
}

static status_t websearchfs_close_attrdir(fs_volume *_volume, fs_vnode *_node, void *_cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_attr_dir_cookie *cookie = (fs_attr_dir_cookie *)_cookie;
	status_t err = B_OK;
	TRACE("close_attrdir(%" B_PRId32 ", %" B_PRId64 ")\n", ns->nsid, node->vnid);
	err = LOCK(&node->l);
	if (err)
		return err;
	SLL_REMOVE(node->opened, next, cookie);
	UNLOCK(&node->l);
	return err;
}

static status_t websearchfs_free_attrdircookie(fs_volume *_volume, fs_vnode *_node, void *_cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_attr_dir_cookie *cookie = (fs_attr_dir_cookie *)_cookie;
	status_t err = B_OK;
	TRACE("free_attrdircookie(%" B_PRId32 ", %" B_PRId64 ")\n", ns->nsid, node->vnid);
	err = LOCK(&node->l);
	if (err)
		return err;
	SLL_REMOVE(node->opened, next, cookie); /* just to make sure */
	UNLOCK(&node->l);
	free(cookie);
	return B_OK;
}

static status_t websearchfs_rewind_attrdir(fs_volume *_volume, fs_vnode *_node, void *_cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_attr_dir_cookie *cookie = (fs_attr_dir_cookie *)_cookie;
	TRACE("rewind_attrdir(%" B_PRId32 ", %" B_PRId64 ")\n", ns->nsid, node->vnid);
	cookie->dir_current = 0;
	return B_OK;
}

static status_t websearchfs_read_attrdir(fs_volume *_volume, fs_vnode *_node, void *_cookie,
	struct dirent *buf, size_t bufsize, uint32 *num)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_file_cookie *cookie = (fs_file_cookie *)_cookie;
	//fs_node *n = NULL;
	//fs_node *parent = node->parent;
	attr_entry *ae = NULL;
	int i;
	int count_indirect;
	TRACE("read_attrdir(%" B_PRId32 ", %" B_PRId64 ") @ %d\n", ns->nsid, node->vnid,
		cookie->dir_current);
	if (!node || !cookie || !num || !*num || !buf
		|| (bufsize < (sizeof(dirent_t) + WEBSEARCHFS_NAME_LEN)))
		return EINVAL;
	LOCK(&node->l);
	for (i = 0, count_indirect = 0; node->attrs_indirect && !ae && node->attrs_indirect[i].name;
		i++, count_indirect++) {
		if (i == cookie->dir_current)
			ae = &node->attrs_indirect[i];
	}
	for (i = 0; !ae && i < 10 && node->attrs[i].name; i++) {
		if (i + count_indirect == cookie->dir_current)
			ae = &node->attrs[i];
	}

	if (ae) {
		TRACE("read_attrdir: giving %s\n", ae->name);
		buf->d_dev = ns->nsid;
		buf->d_pdev = ns->nsid;
		buf->d_ino = node->vnid;
		buf->d_pino = node->parent?node->parent->vnid:ns->rootid;
		strcpy(buf->d_name, ae->name);
		buf->d_reclen = offsetof(struct dirent, d_name)+strlen(buf->d_name)+1;
		cookie->dir_current++;
		*num = 1;
	} else
		*num = 0;

	UNLOCK(&node->l);
	return B_OK;
}

/* Haiku and BeOs differ in the way the handle attributes at the vfs layer.
   BeOS uses atomic calls on the vnode,
   Haiku retains the open/close/read/write semantics for attributes (loosing atomicity).
   Here we don't care much though, open is used for both to factorize attribute lookup. <- TODO
   _h suffixed funcs are for Haiku API, _b are for BeOS.
 */

/* for Haiku, but also used by BeOS calls to factorize code */
static status_t websearchfs_open_attr_h(fs_volume *_volume, fs_vnode *_node, const char *name, int omode, void **cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	status_t err = B_OK;
	//fs_node *dummy;
	fs_file_cookie *fc;
	attr_entry *ae = NULL;
	int i;
	TRACE("open_attr(%" B_PRId32 ", %" B_PRId64 ", %s, 0x%x)\n", ns->nsid, node->vnid, name, omode);
	if (!node || !name || !cookie)
		return EINVAL;

	err = LOCK(&node->l);
	if (err)
		goto err_n_l;

	/* lookup attribute */
	for (i = 0; node->attrs_indirect && !ae && node->attrs_indirect[i].name; i++)
		if (!strcmp(name, node->attrs_indirect[i].name))
			ae = &node->attrs_indirect[i];
	for (i = 0; !ae && i < 10 && node->attrs[i].name; i++)
		if (!strcmp(name, node->attrs[i].name))
			ae = &node->attrs[i];

	/* should check omode */
	err = ENOENT;
	if (!ae)
		goto err_malloc;
	err = EEXIST;

	err = B_NO_MEMORY;
	fc = malloc(sizeof(fs_file_cookie));
	if (!fc)
		goto err_malloc;
	memset(fc, 0, sizeof(fs_file_cookie));
	fc->node = node;
	fc->omode = omode;
	fc->type = S_ATTR;
	fc->attr = ae;
	err = SLL_INSERT(node->opened, next, fc);
	if (err)
		goto err_linsert;

	*cookie = (void *)fc;
	err = B_OK;
	goto all_ok;
//err_:
//	put_vnode(ns->nsid, node->nsid);
//err_getvn:
//	SLL_REMOVE(node->opened, next, fc);
err_linsert:
	free(fc);
err_malloc:
all_ok:
	UNLOCK(&node->l);
err_n_l:
//	UNLOCK(&ns->l);
	return err;
}

static status_t websearchfs_close_attr_h(fs_volume *_volume, fs_vnode *_node, void *_cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_file_cookie *cookie = (fs_file_cookie *)_cookie;
	status_t err;
	TRACE("close_attr(%" B_PRId32 ", %" B_PRId64 ":%s)\n", ns->nsid, node->vnid,
		cookie->attr ? cookie->attr->name : "?");
	if (!ns || !node || !cookie)
		return EINVAL;
	err = LOCK(&node->l);
	if (err)
		return err;
	SLL_REMOVE(node->opened, next, cookie);

//all_ok:
//err_n_l:
	UNLOCK(&node->l);
	return err;
}

static status_t websearchfs_free_attr_cookie_h(fs_volume *_volume, fs_vnode *_node, void *_cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_file_cookie *cookie = (fs_file_cookie *)_cookie;
	status_t err = B_OK;
	TRACE("free_attrcookie(%" B_PRId32 ", %" B_PRId64 ":%s)\n", ns->nsid, node->vnid,
		cookie->attr ? cookie->attr->name : "?");
	err = LOCK(&node->l);
	if (err)
		return err;
	err = SLL_REMOVE(node->opened, next, cookie); /* just to make sure */
//	if (err)
//		goto err_n_l;
	UNLOCK(&node->l);
	free(cookie);
//	err = B_OK;
//err_n_l:
	return err;
}

static status_t websearchfs_read_attr_stat(fs_volume *_volume, fs_vnode *_node, void *_cookie,
	struct stat *st)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_file_cookie *cookie = (fs_file_cookie *)_cookie;
	status_t err = B_OK;
	attr_entry *ae = cookie->attr;
	TRACE("stat_attr(%" B_PRId32 ", %" B_PRId64 ":%s)\n", ns->nsid, node->vnid, ae->name);
	if (!node || !st || !cookie || !cookie->attr)
		return EINVAL;
	memcpy(st, &node->st, sizeof(struct stat));
	st->st_type = ae->type;
	st->st_size = ae->size;
	err = B_OK;
	return err;
}

static status_t websearchfs_read_attr(fs_volume *_volume, fs_vnode *_node, void *_cookie,
	off_t pos, void *buf, size_t *len)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *node = (fs_node *)_node->private_node;
	fs_file_cookie *cookie = (fs_file_cookie *)_cookie;
	status_t err = B_OK;
	attr_entry *ae = cookie->attr;
	TRACE("read_attr(%" B_PRId32 ", %" B_PRId64 ":%s)\n", ns->nsid, node->vnid, ae->name);
	if (!node || !cookie || !len || !*len)
		return EINVAL;

	err = LOCK(&node->l);

	if (ae && (pos < 0 || (size_t)pos < ae->size)) {
		memcpy(buf, (char *)ae->value + pos, MIN(*len, ae->size-pos));
		*len = MIN(*len, ae->size-pos);
		err = B_OK;
	} else {
		*len = 0;
		err = ENOENT;
	}

	UNLOCK(&node->l);
	return err;
}


/* query stuff */

static int compare_fs_node_by_recent_query_string(fs_node *node, char *query)
{
	time_t tm = time(NULL);
	//return memcmp(node->name, name, WEBSEARCHFS_NAME_LEN);
	TRACE("find_by_recent_query_string: '%s' <> '%s'\n", \
			node->request?node->request->query_string:NULL, query);
	if (!node->request || !node->request->query_string)
		return -1;
	/* reject if older than 5 min */
	if (node->st.st_crtime + 60 * 5 < tm)
		return -1;
	return strcmp(node->request->query_string, query);
}

static status_t websearchfs_open_query(fs_volume *_volume, const char *query, uint32 flags,
					port_id port, uint32 token, void **cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	status_t err = B_OK;
	fs_query_cookie *c;
	fs_node *qn, *dummy;
	const char *p;
	char *q;
	char *qstring = NULL;
	char qname[WEBSEARCHFS_NAME_LEN];
	bool accepted = true;
	bool reused = false;
	//int i;
	TRACE("open_query(%" B_PRId32 ", '%s', 0x%08" B_PRIx32 ", %" B_PRId32 ", %" B_PRId32 ")\n",
		ns->nsid, query, flags, port, token);
//	if (flags & B_LIVE_QUERY)
//		return ENOSYS; /* no live query yet, they are live enough anyway */
	//return ENOSYS;
	if (!query || !cookie)
		return EINVAL;

	// filter out queries that aren't for us, we don't want to trigger DuckDuckGo searches when
	// apps check for mails, ... :)

	err = B_NO_MEMORY;
	c = malloc(sizeof(fs_query_cookie));
	if (!c)
		return err;
	memset(c, 0, sizeof(fs_query_cookie));
	c->omode = O_RDONLY;
	c->type = S_IFQUERY;
	c->dir_current = 0;

	err = ENOSYS;
	if (strncmp(query, "((name==\"*", 10))
		accepted = false;
	else {
		qstring = query_unescape_string(query + 10, &p, '"');
		if (!qstring)
			accepted = false;
		else if (!p)
			accepted = false;
		else if (strcmp(p, "\")&&(BEOS:TYPE==\"application/x-vnd.Be-bookmark\"))"))
			accepted = false;
		else {
			//if (qstring[0] == '*')
			//	strcpy(qstring+1, qstring);
			//if (qstring[strlen(qstring)-1] == '*')
			//	qstring[strlen(qstring)-1] = '\0';
			if (!query_strip_bracketed_Cc(qstring))
				goto err_qs;
		}
	}

	if (!accepted) {
		free(qstring);
		/* return an empty cookie */
		*cookie = (void *)c;
		return B_OK;
	}
	TRACE("open_query: QUERY: '%s'\n", qstring);
	/* reuse query if it's not too old */
	LOCK(&ns->l);
	qn = SLL_FIND(ns->queries, qnext,
				(sll_compare_func)compare_fs_node_by_recent_query_string, (void *)qstring);
	UNLOCK(&ns->l);
	reused = (qn != NULL);
	if (reused) {
		TRACE("open_query: reusing %" B_PRId32 ":%" B_PRId64 "\n", ns->nsid, qn->vnid);
		err = get_vnode(_volume, qn->vnid, (void **)&dummy); /* inc ref count */
		if (err)
			goto err_mkdir;
		/* wait for the query to complete */
		while (!qn->qcompleted)
			snooze(10000);
		goto reuse;
	}

	/* stripped name for folder */
	strncpy(qname, qstring, WEBSEARCHFS_NAME_LEN);
	qname[WEBSEARCHFS_NAME_LEN-1] = '\0';

	/* strip out slashes */
	q = qname;
	while ((q = strchr(q, '/')))
		strcpy(q, q + 1);

	/* should get/put_vnode(ns->root); around that I think... */
	err = websearchfs_create_gen(_volume, ns->root, qname, 0, 0755, NULL, &qn, folders_attrs, true, true);
	if (err)
		goto err_qs;

	err = get_vnode(_volume, qn->vnid, (void **)&dummy); /* inc ref count */
	if (err)
		goto err_mkdir;

//#ifndef NO_SEARCH

	/* let's ask DuckDuckGo */
	err = duckduckgo_request_open(qstring, _volume, qn, &qn->request);
	if (err)
		goto err_gn;

	TRACE("open_query: request_open done\n");
#ifndef NO_SEARCH
	err = duckduckgo_request_process(qn->request);
	if (err)
		goto err_gro;
	TRACE("open_query: request_process done\n");

#else
	/* fake entries */
	for (i = 0; i < 10; i++) {
		err = websearchfs_create_gen(_volume, qn, "B", 0, 0644, NULL, &n, fake_bookmark_attrs, false, true);
		/* fake that to test sorting */
		*(int32 *)&n->attrs[1].value = i + 1; // hack
		n->attrs[0].type = 'LONG';
		n->attrs[0].value = &n->attrs[1].value;
		n->attrs[0].size = sizeof(int32);
		n->attrs[0].name = "WEBSEARCH:order";
		notify_attribute_changed(ns->nsid, -1, n->vnid, n->attrs[0].name, B_ATTR_CHANGED);
		if (err)
			goto err_gn;
	}
#endif /*NO_SEARCH*/
	//
	//err = duckduckgo_request_close(q->request);

	LOCK(&ns->l);
	SLL_INSERT(ns->queries, qnext, qn);
	UNLOCK(&ns->l);
reuse:
	/* put the chocolate on the cookie */
	c->node = qn;
	LOCK(&qn->l);
	SLL_INSERT(qn->opened, next, c);
	UNLOCK(&qn->l);
	qn->qcompleted = 1; /* tell other cookies we're done */
	*cookie = (void *)c;
	free(qstring);
	return B_OK;

//err_grp:
err_gro:
	if (qn->request)
		duckduckgo_request_close(qn->request);
err_gn:
	put_vnode(_volume, qn->vnid);
err_mkdir:
	if (!reused)
		websearchfs_unlink_gen(_volume, ns->root, qn->name);
err_qs:
	free(qstring);
//err_m:
	free(c);
	TRACE("open_query: error 0x%08" B_PRIx32 "\n", err);
	return err;
}

static status_t websearchfs_close_query(fs_volume *_volume, void *_cookie)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_query_cookie *cookie = (fs_query_cookie *)_cookie;
	status_t err;
	fs_node *q;
	TRACE("close_query(%" B_PRId32 ", %" B_PRId64 ")\n", ns->nsid,
		cookie->node ? cookie->node->vnid : (int64)0);
	//return ENOSYS;
	q = cookie->node;
	if (!q)
		return B_OK;
	// kill_request();
	LOCK(&q->l);
	SLL_REMOVE(q->opened, next, cookie);
	if (q->request /*&& !q->opened*/) {
		err = duckduckgo_request_close(q->request);
	}
	UNLOCK(&q->l);
	/* if last cookie on the query and sync_unlink, trash all */
	if (sync_unlink_queries && !q->opened)
		err = websearchfs_unlink_node_rec(_volume, q);
	err = put_vnode(_volume, q->vnid);
	return err;
}

#ifdef __HAIKU__
/* protos are different... */
static status_t websearchfs_free_query_cookie(fs_volume *_volume, void *_cookie)
{
	fs_query_cookie *cookie = (fs_query_cookie *)_cookie;
	status_t err = B_OK;
	fs_node *q;
	TRACE("free_query_cookie(%" B_PRId32 ")\n", _volume->id);
	q = cookie->node;
	if (!q)
		goto no_node;
	err = LOCK(&q->l);
	if (err)
		return err;
	err = SLL_REMOVE(q->opened, next, cookie); /* just to make sure */
	if (q->request /*&& !q->opened*/) {
		err = duckduckgo_request_close(q->request);
	}
//	if (err)
//		goto err_n_l;
	UNLOCK(&q->l);
no_node:
	free(cookie);
	return B_OK;
}
#endif

static status_t websearchfs_read_query(fs_volume *_volume, void *_cookie, struct dirent *buf,
	size_t bufsize, uint32 *num)
{
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_query_cookie *cookie = (fs_query_cookie *)_cookie;
	fs_node *n = NULL;
	fs_node *node = cookie->node;
	int index;
	TRACE("read_query(%" B_PRId32 ", %" B_PRId64 ") @ %d\n", ns->nsid, node ? node->vnid : (int64)0,
		cookie->dir_current);
	if (!cookie || !num || !*num || !buf || (bufsize < (sizeof(dirent_t) + WEBSEARCHFS_NAME_LEN)))
		return EINVAL;
	if (!node) {
		/* a query we don't care about, just return no entries to please apps */
		*num = 0;
		return B_OK;
	}
	//return ENOSYS;
	LOCK(&node->l);
	index = cookie->dir_current;
	for (n = node->children; n && index; n = n->next, index--);
	if (n) {
		TRACE("read_query: giving ino %" PRId64 ", %s\n", n->vnid, n->name);
		buf->d_dev = ns->nsid;
		buf->d_pdev = ns->nsid;
		buf->d_ino = n->vnid;
		buf->d_pino = node->vnid;
		strcpy(buf->d_name, n->name);
		buf->d_reclen = offsetof(struct dirent, d_name)+strlen(buf->d_name)+1;
		cookie->dir_current++;
		*num = 1;
	} else {
		*num = 0;
	}
	UNLOCK(&node->l);
	return B_OK;
}

int websearchfs_push_result_to_query(struct duckduckgo_request *request,
	struct duckduckgo_result *result)
{
	status_t err = B_OK;
	fs_volume *_volume = request->volume;
	fs_nspace *ns = (fs_nspace *)_volume->private_volume;
	fs_node *qn = request->query_node;
	fs_node *n;
	char ename[WEBSEARCHFS_NAME_LEN];
	char *p;
	int i;
	TRACE("push_result_to_query(%" B_PRId32 ", %" B_PRId64 ", %ld:'%s')\n", ns->nsid, qn->vnid, result->id, result->name);
	//dprintf(PFS"push_result_to_query(%" B_PRId32 ", %" B_PRId64 ", %ld:'%s')\n", ns->nsid, qn->vnid, result->id, result->name);
	//return ENOSYS;
	if (!ns || !qn)
		return EINVAL;

	// filter out queries that aren't for us, we don't want to trigger DuckDuckGo searches when
	// apps check for mails, ... :)

	/* stripped name for folder */
	strncpy(ename, result->name, WEBSEARCHFS_NAME_LEN);
	ename[WEBSEARCHFS_NAME_LEN-1] = '\0';
	/* strip out slashes */
	p = ename;
	while ((p = strchr(p, '/')))
		*p++ = '_';

	err = websearchfs_create_gen(_volume, qn, ename, 0, 0644, NULL, &n, bookmark_attrs, false, true);
	if (err)
		return err;
	LOCK(&n->l);
	n->result = result;
	i = 0;
	n->attrs[i].type = 'CSTR';
	n->attrs[i].value = result->name;
	n->attrs[i].size = strlen(result->name)+1;
	n->attrs[i].name = "META:title";
	notify_attribute_changed(ns->nsid, -1, n->vnid, n->attrs[i].name, B_ATTR_CREATED);
	i++;
	n->attrs[i].type = 'CSTR';
	n->attrs[i].value = result->url;
	n->attrs[i].size = strlen(result->url)+1;
	n->attrs[i].name = "META:url";
	notify_attribute_changed(ns->nsid, -1, n->vnid, n->attrs[i].name, B_ATTR_CREATED);
	i++;
	n->attrs[i].type = 'CSTR';
	n->attrs[i].value = request->query_string;
	n->attrs[i].size = strlen(request->query_string)+1;
	n->attrs[i].name = "META:keyw";
	notify_attribute_changed(ns->nsid, -1, n->vnid, n->attrs[i].name, B_ATTR_CREATED);
	i++;
	n->attrs[i].type = 'LONG';
	n->attrs[i].value = &result->id;
	n->attrs[i].size = sizeof(int32);
	n->attrs[i].name = "WEBSEARCH:order";
	notify_attribute_changed(ns->nsid, -1, n->vnid, n->attrs[i].name, B_ATTR_CREATED);
	i++;
	if (result->snipset[0]) {
		n->attrs[i].type = 'CSTR';
		n->attrs[i].value = result->snipset;
		n->attrs[i].size = strlen(result->snipset)+1;
		n->attrs[i].name = "WEBSEARCH:excerpt";
		notify_attribute_changed(ns->nsid, -1, n->vnid, n->attrs[i].name, B_ATTR_CREATED);
		i++;
	}
	if (result->cache_url[0]) {
		n->attrs[i].type = 'CSTR';
		n->attrs[i].value = result->cache_url;
		n->attrs[i].size = strlen(result->cache_url)+1;
		n->attrs[i].name = "WEBSEARCH:cache_url";
		notify_attribute_changed(ns->nsid, -1, n->vnid, n->attrs[i].name, B_ATTR_CREATED);
		i++;
	}
	if (result->similar_url[0]) {
		n->attrs[i].type = 'CSTR';
		n->attrs[i].value = result->similar_url;
		n->attrs[i].size = strlen(result->similar_url)+1;
		n->attrs[i].name = "WEBSEARCH:similar_url";
		notify_attribute_changed(ns->nsid, -1, n->vnid, n->attrs[i].name, B_ATTR_CREATED);
		i++;
	}
	UNLOCK(&n->l);
	return B_OK;

	TRACE("push_result_to_query: error 0x%08" B_PRIx32 "\n", err);
	return err;
}

//	#pragma mark -

static status_t
websearchfs_std_ops(int32 op, ...)
{
	switch (op) {
		case B_MODULE_INIT:
			TRACE("std_ops(INIT)\n");
			return B_OK;
		case B_MODULE_UNINIT:
			TRACE("std_ops(UNINIT)\n");
			return B_OK;
		default:
			return B_ERROR;
	}
}


static fs_volume_ops sWebSearchFSVolumeOps = {
	&websearchfs_unmount,
	&websearchfs_rfsstat,
	&websearchfs_wfsstat,
	NULL,			// no sync!
	&websearchfs_read_vnode,

	/* index directory & index operations */
	NULL,	// &websearchfs_open_index_dir
	NULL,	// &websearchfs_close_index_dir
	NULL,	// &websearchfs_free_index_dir_cookie
	NULL,	// &websearchfs_read_index_dir
	NULL,	// &websearchfs_rewind_index_dir

	NULL,	// &websearchfs_create_index
	NULL,	// &websearchfs_remove_index
	NULL,	// &websearchfs_stat_index

	/* query operations */
	&websearchfs_open_query,
	&websearchfs_close_query,
	&websearchfs_free_query_cookie,
	&websearchfs_read_query,
	NULL,	// &websearchfs_rewind_query,
};


static fs_vnode_ops sWebSearchFSVnodeOps = {
	/* vnode operations */
	&websearchfs_walk,
	&websearchfs_get_vnode_name, //NULL, // fs_get_vnode_name
	&websearchfs_release_vnode,
	&websearchfs_remove_vnode,

	/* VM file access */
	NULL, 	// &websearchfs_can_page
	NULL,	// &websearchfs_read_pages
	NULL, 	// &websearchfs_write_pages

	NULL,	// io()
	NULL,	// cancel_io()

	NULL,	// &websearchfs_get_file_map,

	NULL, 	// &websearchfs_ioctl
	NULL,	// &websearchfs_setflags,
	NULL,	// &websearchfs_select
	NULL,	// &websearchfs_deselect
	NULL, 	// &websearchfs_fsync

	NULL,	// &websearchfs_readlink,
	NULL,	// &websearchfs_symlink,

	NULL,	// &websearchfs_link,
	&websearchfs_unlink,
	NULL,	// &websearchfs_rename,

	&websearchfs_access,
	&websearchfs_rstat,
	&websearchfs_wstat,
	NULL,	// fs_preallocate

	/* file operations */
	&websearchfs_create,
	&websearchfs_open,
	&websearchfs_close,
	&websearchfs_free_cookie,
	&websearchfs_read,
	&websearchfs_write,

	/* directory operations */
	&websearchfs_mkdir,
	&websearchfs_rmdir,
	&websearchfs_opendir,
	&websearchfs_closedir,
	&websearchfs_free_dircookie,
	&websearchfs_readdir,
	&websearchfs_rewinddir,

	/* attribute directory operations */
	&websearchfs_open_attrdir,
	&websearchfs_close_attrdir,
	&websearchfs_free_attrdircookie,
	&websearchfs_read_attrdir,
	&websearchfs_rewind_attrdir,

	/* attribute operations */
	NULL,	// &websearchfs_create_attr
	&websearchfs_open_attr_h,
	&websearchfs_close_attr_h,
	&websearchfs_free_attr_cookie_h,
	&websearchfs_read_attr,
	NULL,	// &websearchfs_write_attr_h,

	&websearchfs_read_attr_stat,
	NULL,	// &websearchfs_write_attr_stat
	NULL,	// &websearchfs_rename_attr
	NULL,	// &websearchfs_remove_attr
};

file_system_module_info sWebSearchFSModule = {
	{
		"file_systems/websearchfs" B_CURRENT_FS_API_VERSION,
		0,
		websearchfs_std_ops,
	},

	"websearchfs",					// short_name
	WEBSEARCHFS_PRETTY_NAME,		// pretty_name
	0,//B_DISK_SYSTEM_SUPPORTS_WRITING, // DDM flags

	// scanning
	NULL,	// fs_identify_partition,
	NULL,	// fs_scan_partition,
	NULL,	// fs_free_identify_partition_cookie,
	NULL,	// free_partition_content_cookie()

	&websearchfs_mount,
};

module_info *modules[] = {
	(module_info *)&sWebSearchFSModule,
	NULL,
};


