/* Copyright (c) 2014 The Regents of the University of California
 * Barret Rhoden <brho@cs.berkeley.edu>
 * See LICENSE for details.
 *
 * #s (srv) - a chan sharing service.  This was originally based off the Inferno
 * #s, but it's been completely rewritten to act like what I remember the plan9
 * one to be like.
 *
 *
 * I tried a style where we hang reference counted objects off c->aux, specific
 * to each chan.  Instead of looking up via qid.path, we just look at the c->aux
 * for our struct.  I originally tried having those be reference counted
 * structs, but that fails for a bunch of reasons.  Without them being reference
 * counted, we're really just using c->aux as if it was qid.path.
 *
 * We can't hang an external reference to an item off c->aux, and have that item
 * change as we gen (but we can use it as a weak ref, uncounted ref).  The main
 * thing is that devclone makes a 'half-chan' with a copy of c->aux.  This chan
 * may or may not be closed later.  If we transfer refs via a gen, we first
 * assumed we had a ref in the first place (devclone doesn't incref our srv),
 * and then we might not close.  This ends up decreffing top_dir too much, and
 * giving it's refs to some other file in the walk. */

#include <slab.h>
#include <kmalloc.h>
#include <kref.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <error.h>
#include <cpio.h>
#include <pmap.h>
#include <smp.h>
#include <net/ip.h>
#include <sys/queue.h>

struct dev srvdevtab;

static char *devname(void)
{
	return srvdevtab.name;
}

#define Qtopdir			1
#define Qsrvfile		2

struct srvfile {
	TAILQ_ENTRY(srvfile) link;
	char *name;
	struct chan *chan;
	struct kref ref;	/* +1 for existing on create, -1 on remove */
	char *user;
	uint32_t perm;
	atomic_t opens;		/* used for exclusive open checks */
};

struct srvfile *top_dir;
TAILQ_HEAD(srvfilelist, srvfile) srvfiles = TAILQ_HEAD_INITIALIZER(srvfiles);
/* the lock protects the list and its members.  we don't incref from a list ref
 * without the lock. (if you're on the list, we can grab a ref). */
spinlock_t srvlock = SPINLOCK_INITIALIZER;

atomic_t nr_srvs = 0;		/* debugging - concerned about leaking mem */

/* Given a pointer (internal ref), we attempt to get a kref */
static bool grab_ref(struct srvfile *srv)
{
	bool ret = FALSE;
	struct srvfile *srv_i;

	spin_lock(&srvlock);
	TAILQ_FOREACH(srv_i, &srvfiles, link) {
		if (srv_i == srv) {
			ret = kref_get_not_zero(&srv_i->ref, 1);
			break;
		}
	}
	spin_unlock(&srvlock);
	return ret;
}

static void srv_release(struct kref *kref)
{
	struct srvfile *srv = container_of(kref, struct srvfile, ref);

	kfree(srv->user);
	kfree(srv->name);
	if (srv->chan)
		cclose(srv->chan);
	kfree(srv);
	atomic_dec(&nr_srvs);
}

static int srvgen(struct chan *c, char *name, struct dirtab *tab,
				  int ntab, int s, struct dir *dp)
{
	struct srvfile *prev, *next;
	struct qid q;

	if (s == DEVDOTDOT) {
		/* changing whatever c->aux was to be topdir */
		mkqid(&q, Qtopdir, 0, QTDIR);
		devdir(c, q, devname(), 0, eve.name, 0555, dp);
		return 1;
	}
	spin_lock(&srvlock);
	TAILQ_FOREACH(next, &srvfiles, link) {
		/* come in with s == 0 on the first run */
		if (s-- == 0)
			break;
	}
	if (!next) {
		spin_unlock(&srvlock);
		return -1;
	}
	/* update c to point to our new srvfile.  this keeps the chan and its
	 * srv in sync with what we're genning. */
	c->aux = next;	/* uncounted ref */
	mkqid(&q, Qsrvfile, 0, QTFILE);
	/* once we release the lock, next could disappear, including next->name
	 */
	strlcpy(get_cur_genbuf(), next->name, GENBUF_SZ);
	devdir(c, q, get_cur_genbuf(), 1 /* length */ , next->user, next->perm,
	       dp);
	spin_unlock(&srvlock);
	return 1;
}

static void __srvinit(void)
{
	top_dir = kzmalloc(sizeof(struct srvfile), MEM_WAIT);
	/* kstrdup, just in case we free this later */
	kstrdup(&top_dir->name, "srv");
	kstrdup(&top_dir->user, current ? current->user.name : "eve");
	top_dir->perm = DMDIR | 0770;
	/* +1 for existing, should never decref this */
	kref_init(&top_dir->ref, fake_release, 1);
	atomic_set(&top_dir->opens, 0);
}

static void srvinit(void)
{
	run_once(__srvinit());
}

static struct chan *srvattach(char *spec)
{
	/* the inferno attach was pretty complicated, but
	 * we're not sure that complexity is needed. */
	struct chan *c = devattach(devname(), spec);

	mkqid(&c->qid, Qtopdir, 0, QTDIR);
	/* c->aux is an uncounted ref */
	c->aux = top_dir;
	return c;
}

static struct walkqid *srvwalk(struct chan *c, struct chan *nc, char **name,
							   unsigned int nname)
{
	return devwalk(c, nc, name, nname, 0, 0, srvgen);
}

static size_t srvstat(struct chan *c, uint8_t *db, size_t n)
{
	return devstat(c, db, n, 0, 0, srvgen);
}

char *srvname(struct chan *c)
{
	struct srvfile *srv_i;
	char *s;

	spin_lock(&srvlock);
	TAILQ_FOREACH(srv_i, &srvfiles, link) {
		if(srv_i->chan == c){
			int len = 3 + strlen(srv_i->name) + 1;
			s = kzmalloc(len, 0);
			snprintf(s, len, "#s/%s", srv_i->name);
			spin_unlock(&srvlock);
			return s;
		}
	}
	spin_unlock(&srvlock);
	return NULL;
}

static struct chan *srvopen(struct chan *c, int omode)
{
	ERRSTACK(1);
	struct srvfile *srv;

	/* used as an error checker in plan9, does little now */
	openmode(omode);
	if (c->qid.type & QTDIR) {
		if (omode & O_WRITE)
			error(EISDIR, ERROR_FIXME);
		c->mode = openmode(omode);
		c->flag |= COPEN;
		c->offset = 0;
		return c;
	}
	srv = c->aux;
	if (!grab_ref(srv))
		error(EFAIL, "Unable to open srv file, concurrent removal");
	if (waserror()) {
		kref_put(&srv->ref);
		nexterror();
	}
	devpermcheck(srv->user, srv->perm, omode);
	if ((srv->perm & DMEXCL) && atomic_read(&srv->opens))
		error(EBUSY, ERROR_FIXME);
	/* srv->chan is write-once, so we don't need to sync. */
	if (!srv->chan)
		error(EFAIL, "srv file has no chan yet");
	/* this is more than just the ref - 1, since there will be refs in
	 * flight as gens work their way through the list */
	atomic_inc(&srv->opens);
	/* the magic of srv: open c, get c->srv->chan back */
	cclose(c);
	c = srv->chan;
	chan_incref(c);
	poperror();
	kref_put(&srv->ref);
	return c;
}

static void srvcreate(struct chan *c, char *name, int omode, uint32_t perm,
                      char *ext)
{
	struct srvfile *srv;

	if (perm & DMSYMLINK)
		error(EINVAL, "#%s doesn't support symlinks", devname());
	srv = kzmalloc(sizeof(struct srvfile), MEM_WAIT);
	kstrdup(&srv->name, name);
	kstrdup(&srv->user, current ? current->user.name : "eve");
	srv->perm = 0770;	/* TODO need some security thoughts */
	atomic_set(&srv->opens, 1);	/* we return it opened */
	mkqid(&c->qid, Qsrvfile, 0, QTFILE);
	c->aux = srv;
	c->mode = openmode(omode);
	/* one ref for being on the list */
	kref_init(&srv->ref, srv_release, 1);
	spin_lock(&srvlock);
	TAILQ_INSERT_TAIL(&srvfiles, srv, link);
	spin_unlock(&srvlock);
	atomic_inc(&nr_srvs);
}

static void srvremove(struct chan *c)
{
	struct srvfile *srv_i, *temp;

	spin_lock(&srvlock);
	TAILQ_FOREACH_SAFE(srv_i, &srvfiles, link, temp) {
		if (srv_i == c->aux) {
			TAILQ_REMOVE(&srvfiles, srv_i, link);
			break;
		}
	}
	spin_unlock(&srvlock);
	if (srv_i)
		kref_put(&srv_i->ref);	/* dropping ref from the list */
}

static void srvclose(struct chan *c)
{
	struct srvfile *srv = c->aux;

	if (!grab_ref(srv))
		return;
	atomic_dec(&srv->opens);
	kref_put(&srv->ref);
	if (c->flag & O_REMCLO)
		srvremove(c);
}

/* N.B. srvopen gives the chan back. The only 'reading' we do
 * in srv is of the top level directory.
 */
static size_t srvread(struct chan *c, void *va, size_t count, off64_t offset)
{
	return devdirread(c, va, count, 0, 0, srvgen);
}

static size_t srvwrite(struct chan *c, void *va, size_t count, off64_t offset)
{
	ERRSTACK(2);
	struct srvfile *srv;
	struct chan *new_chan;
	char *kbuf = 0;
	int fd;

	if (c->qid.type & QTDIR)
		error(EPERM, ERROR_FIXME);
	srv = c->aux;
	if (!grab_ref(srv))
		error(EFAIL, "Unable to write srv file, concurrent removal");
	if (waserror()) {
		kref_put(&srv->ref);
		nexterror();
	}
	if (srv->chan)
		error(EFAIL, "srv file already has a stored chan!");
	if (waserror()) {
		kfree(kbuf);
		nexterror();
	}
	kbuf = kmalloc(count + 1, MEM_WAIT);
	strlcpy(kbuf, va, count + 1);
	fd = strtoul(kbuf, 0, 10);
	/* the magic of srv: srv stores the chan corresponding to the fd.  -1
	 * for mode, so we just get the chan with no checks (RDWR would work
	 * too). */
	new_chan = fdtochan(&current->open_files, fd, -1, FALSE, TRUE);
	/* fdtochan already increffed for us */
	if (!__sync_bool_compare_and_swap(&srv->chan, 0, new_chan)) {
		cclose(new_chan);
		error(EFAIL, "srv file already has a stored chan!");
	}
	poperror();
	kfree(kbuf);
	poperror();
	kref_put(&srv->ref);
	return count;
}

struct dev srvdevtab __devtab = {
	.name = "srv",

	.reset = devreset,
	.init = srvinit,
	.shutdown = devshutdown,
	.attach = srvattach,
	.walk = srvwalk,
	.stat = srvstat,
	.open = srvopen,
	.create = srvcreate,
	.close = srvclose,
	.read = srvread,
	.bread = devbread,
	.write = srvwrite,
	.bwrite = devbwrite,
	.remove = srvremove,
	.wstat = devwstat,
	.power = devpower,
	.chaninfo = devchaninfo,
};
