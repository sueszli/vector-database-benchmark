/*
 * CDDL HEADER START
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can obtain a copy of the license at usr/src/OPENSOLARIS.LICENSE
 * or http://www.opensolaris.org/os/licensing.
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL HEADER in each
 * file and include the License file at usr/src/OPENSOLARIS.LICENSE.
 * If applicable, add the following below this CDDL HEADER, with the
 * fields enclosed by brackets "[]" replaced with your own identifying
 * information: Portions Copyright [yyyy] [name of copyright owner]
 *
 * CDDL HEADER END
 */
/*
 * Copyright 2009 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 * Copyright (c) 2017 Joyent, Inc.
 */
/*
 * Copyright (c) 2016 by Delphix. All rights reserved.
 */

/*
 * Datalink management routines.
 */

#include <sys/types.h>
#include <sys/door.h>
#include <sys/zone.h>
#include <sys/modctl.h>
#include <sys/file.h>
#include <sys/modhash.h>
#include <sys/kstat.h>
#include <sys/vnode.h>
#include <sys/cmn_err.h>
#include <sys/softmac.h>
#include <sys/dls.h>
#include <sys/dls_impl.h>
#include <sys/stropts.h>
#include <sys/netstack.h>
#include <inet/iptun/iptun_impl.h>

/*
 * This vanity name management module is treated as part of the GLD framework
 * and we don't hold any GLD framework lock across a call to any mac
 * function that needs to acquire the mac perimeter. The hierarchy is
 * mac perimeter -> framework locks
 */

typedef struct dls_stack {
	zoneid_t	dlss_zoneid;
} dls_stack_t;

static kmem_cache_t	*i_dls_devnet_cachep;
static kmutex_t		i_dls_mgmt_lock;
static krwlock_t	i_dls_devnet_lock;
static mod_hash_t	*i_dls_devnet_id_hash;
static mod_hash_t	*i_dls_devnet_hash;

boolean_t		devnet_need_rebuild;

#define	VLAN_HASHSZ	67	/* prime */

/*
 * The following macros take a link name without the trailing PPA as input.
 * Opening a /dev/net node with one of these names causes a tunnel link to be
 * implicitly created in dls_devnet_hold_by_name() for backward compatibility
 * with Solaris 10 and prior.
 */
#define	IS_IPV4_TUN(name)	(strcmp((name), "ip.tun") == 0)
#define	IS_IPV6_TUN(name)	(strcmp((name), "ip6.tun") == 0)
#define	IS_6TO4_TUN(name)	(strcmp((name), "ip.6to4tun") == 0)
#define	IS_IPTUN_LINK(name)	(					\
    IS_IPV4_TUN(name) || IS_IPV6_TUN(name) || IS_6TO4_TUN(name))

/* Upcall door handle */
static door_handle_t	dls_mgmt_dh = NULL;

/* dls_devnet_t dd_flags */
#define	DD_CONDEMNED		0x1
#define	DD_IMPLICIT_IPTUN	0x2 /* Implicitly-created ip*.*tun* tunnel */
#define	DD_INITIALIZING		0x4

/*
 * If the link is marked as initializing or condemned then it should
 * not be visible outside of the DLS framework.
 */
#define	DD_NOT_VISIBLE(flags)	(					\
	(flags & (DD_CONDEMNED | DD_INITIALIZING)) != 0)

/*
 * This structure is used to keep the <linkid, macname> mapping.
 * This structure itself is not protected by the mac perimeter, but is
 * protected by the dd_mutex and i_dls_devnet_lock. Thus most of the
 * functions manipulating this structure such as dls_devnet_set/unset etc.
 * may be called while not holding the mac perimeter.
 */
typedef struct dls_devnet_s {
	datalink_id_t	dd_linkid;
	char		dd_linkname[MAXLINKNAMELEN];
	char		dd_mac[MAXNAMELEN];
	kstat_t		*dd_ksp;	/* kstat in owner_zid */
	kstat_t		*dd_zone_ksp;	/* in dd_zid if != owner_zid */
	uint32_t	dd_ref;
	kmutex_t	dd_mutex;
	kcondvar_t	dd_cv;
	uint32_t	dd_tref;
	uint_t		dd_flags;
	zoneid_t	dd_owner_zid;	/* zone where node was created */
	zoneid_t	dd_zid;		/* current zone */
	boolean_t	dd_prop_loaded;
	taskqid_t	dd_prop_taskid;
	boolean_t	dd_transient;	/* link goes away when zone does */
} dls_devnet_t;

static int i_dls_devnet_create_iptun(const char *, const char *,
    datalink_id_t *);
static int i_dls_devnet_destroy_iptun(datalink_id_t);
static int i_dls_devnet_setzid(dls_devnet_t *, zoneid_t, boolean_t, boolean_t);
static int dls_devnet_unset(mac_handle_t, datalink_id_t *, boolean_t);

/*ARGSUSED*/
static int
i_dls_devnet_constructor(void *buf, void *arg, int kmflag)
{
	dls_devnet_t	*ddp = buf;

	bzero(buf, sizeof (dls_devnet_t));
	mutex_init(&ddp->dd_mutex, NULL, MUTEX_DEFAULT, NULL);
	cv_init(&ddp->dd_cv, NULL, CV_DEFAULT, NULL);
	return (0);
}

/*ARGSUSED*/
static void
i_dls_devnet_destructor(void *buf, void *arg)
{
	dls_devnet_t	*ddp = buf;

	VERIFY(ddp->dd_ksp == NULL);
	VERIFY(ddp->dd_ref == 0);
	VERIFY(ddp->dd_tref == 0);
	mutex_destroy(&ddp->dd_mutex);
	cv_destroy(&ddp->dd_cv);
}

/* ARGSUSED */
static int
dls_zone_remove(datalink_id_t linkid, void *arg)
{
	dls_devnet_t *ddp;

	if (dls_devnet_hold_tmp(linkid, &ddp) == 0) {
		/*
		 * Don't bother moving transient links back to the global zone
		 * since we will simply delete them in dls_devnet_unset.
		 */
		if (!ddp->dd_transient)
			(void) dls_devnet_setzid(ddp, GLOBAL_ZONEID);
		dls_devnet_rele_tmp(ddp);
	}
	return (0);
}

/* ARGSUSED */
static void *
dls_stack_init(netstackid_t stackid, netstack_t *ns)
{
	dls_stack_t *dlss;

	dlss = kmem_zalloc(sizeof (*dlss), KM_SLEEP);
	dlss->dlss_zoneid = netstackid_to_zoneid(stackid);
	return (dlss);
}

/* ARGSUSED */
static void
dls_stack_shutdown(netstackid_t stackid, void *arg)
{
	dls_stack_t	*dlss = (dls_stack_t *)arg;

	/* Move remaining datalinks in this zone back to the global zone. */
	(void) zone_datalink_walk(dlss->dlss_zoneid, dls_zone_remove, NULL);
}

/* ARGSUSED */
static void
dls_stack_fini(netstackid_t stackid, void *arg)
{
	dls_stack_t	*dlss = (dls_stack_t *)arg;

	kmem_free(dlss, sizeof (*dlss));
}

/*
 * Module initialization and finalization functions.
 */
void
dls_mgmt_init(void)
{
	mutex_init(&i_dls_mgmt_lock, NULL, MUTEX_DEFAULT, NULL);
	rw_init(&i_dls_devnet_lock, NULL, RW_DEFAULT, NULL);

	/*
	 * Create a kmem_cache of dls_devnet_t structures.
	 */
	i_dls_devnet_cachep = kmem_cache_create("dls_devnet_cache",
	    sizeof (dls_devnet_t), 0, i_dls_devnet_constructor,
	    i_dls_devnet_destructor, NULL, NULL, NULL, 0);
	ASSERT(i_dls_devnet_cachep != NULL);

	/*
	 * Create a hash table, keyed by dd_linkid, of dls_devnet_t.
	 */
	i_dls_devnet_id_hash = mod_hash_create_idhash("dls_devnet_id_hash",
	    VLAN_HASHSZ, mod_hash_null_valdtor);

	/*
	 * Create a hash table, keyed by dd_mac
	 */
	i_dls_devnet_hash = mod_hash_create_extended("dls_devnet_hash",
	    VLAN_HASHSZ, mod_hash_null_keydtor, mod_hash_null_valdtor,
	    mod_hash_bystr, NULL, mod_hash_strkey_cmp, KM_SLEEP);

	devnet_need_rebuild = B_FALSE;

	netstack_register(NS_DLS, dls_stack_init, dls_stack_shutdown,
	    dls_stack_fini);
}

void
dls_mgmt_fini(void)
{
	netstack_unregister(NS_DLS);
	mod_hash_destroy_hash(i_dls_devnet_hash);
	mod_hash_destroy_hash(i_dls_devnet_id_hash);
	kmem_cache_destroy(i_dls_devnet_cachep);
	rw_destroy(&i_dls_devnet_lock);
	mutex_destroy(&i_dls_mgmt_lock);
}

int
dls_mgmt_door_set(boolean_t start)
{
	int	err;

	/* handle daemon restart */
	mutex_enter(&i_dls_mgmt_lock);
	if (dls_mgmt_dh != NULL) {
		door_ki_rele(dls_mgmt_dh);
		dls_mgmt_dh = NULL;
	}

	if (start && ((err = door_ki_open(DLMGMT_DOOR, &dls_mgmt_dh)) != 0)) {
		mutex_exit(&i_dls_mgmt_lock);
		return (err);
	}

	mutex_exit(&i_dls_mgmt_lock);

	/*
	 * Create and associate <link name, linkid> mapping for network devices
	 * which are already attached before the daemon is started.
	 */
	if (start)
		softmac_recreate();
	return (0);
}

static boolean_t
i_dls_mgmt_door_revoked(door_handle_t dh)
{
	struct door_info info;
	extern int sys_shutdown;

	ASSERT(dh != NULL);

	if (sys_shutdown) {
		cmn_err(CE_NOTE, "dls_mgmt_door: shutdown observed\n");
		return (B_TRUE);
	}

	if (door_ki_info(dh, &info) != 0)
		return (B_TRUE);

	return ((info.di_attributes & DOOR_REVOKED) != 0);
}

/*
 * Upcall to the datalink management daemon (dlmgmtd).
 */
static int
i_dls_mgmt_upcall(void *arg, size_t asize, void *rbuf, size_t rsize)
{
	door_arg_t			darg, save_arg;
	door_handle_t			dh;
	int				err;
	int				retry = 0;

#define	MAXRETRYNUM	3

	ASSERT(arg);
	darg.data_ptr = arg;
	darg.data_size = asize;
	darg.desc_ptr = NULL;
	darg.desc_num = 0;
	darg.rbuf = rbuf;
	darg.rsize = rsize;
	save_arg = darg;

retry:
	mutex_enter(&i_dls_mgmt_lock);
	dh = dls_mgmt_dh;
	if ((dh == NULL) || i_dls_mgmt_door_revoked(dh)) {
		mutex_exit(&i_dls_mgmt_lock);
		return (EBADF);
	}
	door_ki_hold(dh);
	mutex_exit(&i_dls_mgmt_lock);

	for (;;) {
		retry++;
		if ((err = door_ki_upcall_limited(dh, &darg, zone_kcred(),
		    SIZE_MAX, 0)) == 0)
			break;

		/*
		 * handle door call errors
		 */
		darg = save_arg;
		switch (err) {
		case EINTR:
			/*
			 * If the operation which caused this door upcall gets
			 * interrupted, return directly.
			 */
			goto done;
		case EAGAIN:
			/*
			 * Repeat upcall if the maximum attempt limit has not
			 * been reached.
			 */
			if (retry < MAXRETRYNUM) {
				delay(2 * hz);
				break;
			}
			cmn_err(CE_WARN, "dls: dlmgmtd fatal error %d\n", err);
			goto done;
		default:
			/* A fatal door error */
			if (i_dls_mgmt_door_revoked(dh)) {
				cmn_err(CE_NOTE,
				    "dls: dlmgmtd door service revoked\n");

				if (retry < MAXRETRYNUM) {
					door_ki_rele(dh);
					goto retry;
				}
			}
			cmn_err(CE_WARN, "dls: dlmgmtd fatal error %d\n", err);
			goto done;
		}
	}

	if (darg.rbuf != rbuf) {
		/*
		 * The size of the input rbuf was not big enough, so the
		 * upcall allocated the rbuf itself.  If this happens, assume
		 * that this was an invalid door call request.
		 */
		kmem_free(darg.rbuf, darg.rsize);
		err = ENOSPC;
		goto done;
	}

	if (darg.rsize != rsize) {
		err = EINVAL;
		goto done;
	}

	err = ((dlmgmt_retval_t *)rbuf)->lr_err;

done:
	door_ki_rele(dh);
	return (err);
}

/*
 * Request the datalink management daemon to create a link with the attributes
 * below.  Upon success, zero is returned and linkidp contains the linkid for
 * the new link; otherwise, an errno is returned.
 *
 *     - dev		physical dev_t.  required for all physical links,
 *		        including GLDv3 links.  It will be used to force the
 *		        attachment of a physical device, hence the
 *		        registration of its mac
 *     - class		datalink class
 *     - media type	media type; DL_OTHER means unknown
 *     - persist	whether to persist the datalink
 */
int
dls_mgmt_create(const char *devname, dev_t dev, datalink_class_t class,
    uint32_t media, boolean_t persist, datalink_id_t *linkidp)
{
	dlmgmt_upcall_arg_create_t	create;
	dlmgmt_create_retval_t		retval;
	int				err;

	create.ld_cmd = DLMGMT_CMD_DLS_CREATE;
	create.ld_class = class;
	create.ld_media = media;
	create.ld_phymaj = getmajor(dev);
	create.ld_phyinst = getminor(dev);
	create.ld_persist = persist;
	if (strlcpy(create.ld_devname, devname, sizeof (create.ld_devname)) >=
	    sizeof (create.ld_devname))
		return (EINVAL);

	if ((err = i_dls_mgmt_upcall(&create, sizeof (create), &retval,
	    sizeof (retval))) == 0) {
		*linkidp = retval.lr_linkid;
	}
	return (err);
}

/*
 * Request the datalink management daemon to destroy the specified link.
 * Returns zero upon success, or an errno upon failure.
 */
int
dls_mgmt_destroy(datalink_id_t linkid, boolean_t persist)
{
	dlmgmt_upcall_arg_destroy_t	destroy;
	dlmgmt_destroy_retval_t		retval;

	destroy.ld_cmd = DLMGMT_CMD_DLS_DESTROY;
	destroy.ld_linkid = linkid;
	destroy.ld_persist = persist;

	return (i_dls_mgmt_upcall(&destroy, sizeof (destroy),
	    &retval, sizeof (retval)));
}

/*
 * Request the datalink management daemon to verify/update the information
 * for a physical link.  Upon success, get its linkid.
 *
 *     - media type	media type
 *     - novanity	whether this physical datalink supports vanity naming.
 *			physical links that do not use the GLDv3 MAC plugin
 *			cannot suport vanity naming
 *
 * This function could fail with ENOENT or EEXIST.  Two cases return EEXIST:
 *
 * 1. A link with devname already exists, but the media type does not match.
 *    In this case, mediap will bee set to the media type of the existing link.
 * 2. A link with devname already exists, but its link name does not match
 *    the device name, although this link does not support vanity naming.
 */
int
dls_mgmt_update(const char *devname, uint32_t media, boolean_t novanity,
    uint32_t *mediap, datalink_id_t *linkidp)
{
	dlmgmt_upcall_arg_update_t	update;
	dlmgmt_update_retval_t		retval;
	int				err;

	update.ld_cmd = DLMGMT_CMD_DLS_UPDATE;

	if (strlcpy(update.ld_devname, devname, sizeof (update.ld_devname)) >=
	    sizeof (update.ld_devname))
		return (EINVAL);

	update.ld_media = media;
	update.ld_novanity = novanity;

	if ((err = i_dls_mgmt_upcall(&update, sizeof (update), &retval,
	    sizeof (retval))) == EEXIST) {
		*linkidp = retval.lr_linkid;
		*mediap = retval.lr_media;
	} else if (err == 0) {
		*linkidp = retval.lr_linkid;
	}

	return (err);
}

/*
 * Request the datalink management daemon to get the information for a link.
 * Returns zero upon success, or an errno upon failure.
 *
 * Only fills in information for argument pointers that are non-NULL.
 * Note that the link argument is expected to be MAXLINKNAMELEN bytes.
 */
int
dls_mgmt_get_linkinfo(datalink_id_t linkid, char *link,
    datalink_class_t *classp, uint32_t *mediap, uint32_t *flagsp)
{
	dlmgmt_door_getname_t	getname;
	dlmgmt_getname_retval_t	retval;
	int			err, len;

	getname.ld_cmd = DLMGMT_CMD_GETNAME;
	getname.ld_linkid = linkid;

	if ((err = i_dls_mgmt_upcall(&getname, sizeof (getname), &retval,
	    sizeof (retval))) != 0) {
		return (err);
	}

	len = strlen(retval.lr_link);
	if (len <= 1 || len >= MAXLINKNAMELEN)
		return (EINVAL);

	if (link != NULL)
		(void) strlcpy(link, retval.lr_link, MAXLINKNAMELEN);
	if (classp != NULL)
		*classp = retval.lr_class;
	if (mediap != NULL)
		*mediap = retval.lr_media;
	if (flagsp != NULL)
		*flagsp = retval.lr_flags;
	return (0);
}

/*
 * Request the datalink management daemon to get the linkid for a link.
 * Returns a non-zero error code on failure.  The linkid argument is only
 * set on success (when zero is returned.)
 */
int
dls_mgmt_get_linkid(const char *link, datalink_id_t *linkid)
{
	dlmgmt_door_getlinkid_t		getlinkid;
	dlmgmt_getlinkid_retval_t	retval;
	int				err;

	getlinkid.ld_cmd = DLMGMT_CMD_GETLINKID;
	(void) strlcpy(getlinkid.ld_link, link, MAXLINKNAMELEN);

	if ((err = i_dls_mgmt_upcall(&getlinkid, sizeof (getlinkid), &retval,
	    sizeof (retval))) == 0) {
		*linkid = retval.lr_linkid;
	}
	return (err);
}

datalink_id_t
dls_mgmt_get_next(datalink_id_t linkid, datalink_class_t class,
    datalink_media_t dmedia, uint32_t flags)
{
	dlmgmt_door_getnext_t	getnext;
	dlmgmt_getnext_retval_t	retval;

	getnext.ld_cmd = DLMGMT_CMD_GETNEXT;
	getnext.ld_class = class;
	getnext.ld_dmedia = dmedia;
	getnext.ld_flags = flags;
	getnext.ld_linkid = linkid;

	if (i_dls_mgmt_upcall(&getnext, sizeof (getnext), &retval,
	    sizeof (retval)) != 0) {
		return (DATALINK_INVALID_LINKID);
	}

	return (retval.lr_linkid);
}

static int
i_dls_mgmt_get_linkattr(const datalink_id_t linkid, const char *attr,
    void *attrval, size_t *attrszp)
{
	dlmgmt_upcall_arg_getattr_t	getattr;
	dlmgmt_getattr_retval_t		retval;
	int				err;

	getattr.ld_cmd = DLMGMT_CMD_DLS_GETATTR;
	getattr.ld_linkid = linkid;
	(void) strlcpy(getattr.ld_attr, attr, MAXLINKATTRLEN);

	if ((err = i_dls_mgmt_upcall(&getattr, sizeof (getattr), &retval,
	    sizeof (retval))) == 0) {
		if (*attrszp < retval.lr_attrsz)
			return (EINVAL);
		*attrszp = retval.lr_attrsz;
		bcopy(retval.lr_attrval, attrval, retval.lr_attrsz);
	}

	return (err);
}

/*
 * Note that this function can only get devp successfully for non-VLAN link.
 */
int
dls_mgmt_get_phydev(datalink_id_t linkid, dev_t *devp)
{
	uint64_t	maj, inst;
	size_t		attrsz = sizeof (uint64_t);

	if (i_dls_mgmt_get_linkattr(linkid, FPHYMAJ, &maj, &attrsz) != 0 ||
	    attrsz != sizeof (uint64_t) ||
	    i_dls_mgmt_get_linkattr(linkid, FPHYINST, &inst, &attrsz) != 0 ||
	    attrsz != sizeof (uint64_t)) {
		return (EINVAL);
	}

	*devp = makedevice((major_t)maj, (minor_t)inst);
	return (0);
}

/*
 * Request the datalink management daemon to push in
 * all properties associated with the link.
 * Returns a non-zero error code on failure.
 */
int
dls_mgmt_linkprop_init(datalink_id_t linkid)
{
	dlmgmt_door_linkprop_init_t	li;
	dlmgmt_linkprop_init_retval_t	retval;
	int				err;

	li.ld_cmd = DLMGMT_CMD_LINKPROP_INIT;
	li.ld_linkid = linkid;

	err = i_dls_mgmt_upcall(&li, sizeof (li), &retval, sizeof (retval));
	return (err);
}

static void
dls_devnet_prop_task(void *arg)
{
	dls_devnet_t		*ddp = arg;

	(void) dls_mgmt_linkprop_init(ddp->dd_linkid);

	mutex_enter(&ddp->dd_mutex);
	ddp->dd_prop_loaded = B_TRUE;
	ddp->dd_prop_taskid = 0;
	cv_broadcast(&ddp->dd_cv);
	mutex_exit(&ddp->dd_mutex);
}

/*
 * Ensure property loading task is completed.
 */
void
dls_devnet_prop_task_wait(dls_dl_handle_t ddp)
{
	mutex_enter(&ddp->dd_mutex);
	while (ddp->dd_prop_taskid != 0)
		cv_wait(&ddp->dd_cv, &ddp->dd_mutex);
	mutex_exit(&ddp->dd_mutex);
}

void
dls_devnet_rele_tmp(dls_dl_handle_t dlh)
{
	dls_devnet_t		*ddp = dlh;

	mutex_enter(&ddp->dd_mutex);
	ASSERT(ddp->dd_tref != 0);
	if (--ddp->dd_tref == 0)
		cv_signal(&ddp->dd_cv);
	mutex_exit(&ddp->dd_mutex);
}

int
dls_devnet_hold_link(datalink_id_t linkid, dls_dl_handle_t *ddhp,
    dls_link_t **dlpp)
{
	dls_dl_handle_t	dlh;
	dls_link_t	*dlp;
	int		err;

	if ((err = dls_devnet_hold_tmp(linkid, &dlh)) != 0)
		return (err);

	if ((err = dls_link_hold(dls_devnet_mac(dlh), &dlp)) != 0) {
		dls_devnet_rele_tmp(dlh);
		return (err);
	}

	ASSERT(MAC_PERIM_HELD(dlp->dl_mh));

	*ddhp = dlh;
	*dlpp = dlp;
	return (0);
}

void
dls_devnet_rele_link(dls_dl_handle_t dlh, dls_link_t *dlp)
{
	ASSERT(MAC_PERIM_HELD(dlp->dl_mh));

	dls_link_rele(dlp);
	dls_devnet_rele_tmp(dlh);
}

/*
 * "link" kstats related functions.
 */

/*
 * Query the "link" kstats.
 *
 * We may be called from the kstat subsystem in an arbitrary context.
 * If the caller is the stack, the context could be an upcall data
 * thread. Hence we can't acquire the mac perimeter in this function
 * for fear of deadlock.
 */
static int
dls_devnet_stat_update(kstat_t *ksp, int rw)
{
	datalink_id_t	linkid = (datalink_id_t)(uintptr_t)ksp->ks_private;
	dls_devnet_t	*ddp;
	dls_link_t	*dlp;
	int		err;

	if ((err = dls_devnet_hold_tmp(linkid, &ddp)) != 0) {
		return (err);
	}

	/*
	 * If a device detach happens at this time, it will block in
	 * dls_devnet_unset since the dd_tref has been bumped in
	 * dls_devnet_hold_tmp(). So the access to 'dlp' is safe even though
	 * we don't hold the mac perimeter.
	 */
	if (mod_hash_find(i_dls_link_hash, (mod_hash_key_t)ddp->dd_mac,
	    (mod_hash_val_t *)&dlp) != 0) {
		dls_devnet_rele_tmp(ddp);
		return (ENOENT);
	}

	err = dls_stat_update(ksp, dlp, rw);

	dls_devnet_rele_tmp(ddp);
	return (err);
}

/*
 * Create the "link" kstats.
 */
static void
dls_devnet_stat_create(dls_devnet_t *ddp, zoneid_t zoneid)
{
	kstat_t	*ksp;

	if (dls_stat_create("link", 0, ddp->dd_linkname, zoneid,
	    dls_devnet_stat_update, (void *)(uintptr_t)ddp->dd_linkid,
	    &ksp) == 0) {
		ASSERT(ksp != NULL);
		if (zoneid == ddp->dd_owner_zid) {
			ASSERT(ddp->dd_ksp == NULL);
			ddp->dd_ksp = ksp;
		} else {
			ASSERT(ddp->dd_zone_ksp == NULL);
			ddp->dd_zone_ksp = ksp;
		}
	}
}

/*
 * Destroy the "link" kstats.
 */
static void
dls_devnet_stat_destroy(dls_devnet_t *ddp, zoneid_t zoneid)
{
	if (zoneid == ddp->dd_owner_zid) {
		if (ddp->dd_ksp != NULL) {
			kstat_delete(ddp->dd_ksp);
			ddp->dd_ksp = NULL;
		}
	} else {
		if (ddp->dd_zone_ksp != NULL) {
			kstat_delete(ddp->dd_zone_ksp);
			ddp->dd_zone_ksp = NULL;
		}
	}
}

/*
 * The link has been renamed. Destroy the old non-legacy kstats ("link kstats")
 * and create the new set using the new name.
 */
static void
dls_devnet_stat_rename(dls_devnet_t *ddp)
{
	if (ddp->dd_ksp != NULL) {
		kstat_delete(ddp->dd_ksp);
		ddp->dd_ksp = NULL;
	}
	/* We can't rename a link while it's assigned to a non-global zone. */
	ASSERT(ddp->dd_zone_ksp == NULL);
	dls_devnet_stat_create(ddp, ddp->dd_owner_zid);
}

/*
 * Associate the linkid with the link identified by macname. If this
 * is called on behalf of a physical link then linkid may be
 * DATALINK_INVALID_LINKID. Otherwise, if called on behalf of a
 * virtual link, linkid must have a value.
 */
static int
dls_devnet_set(mac_handle_t mh, datalink_id_t linkid, zoneid_t zoneid,
    dls_devnet_t **ddpp)
{
	const char		*macname = mac_name(mh);
	dls_devnet_t		*ddp = NULL;
	datalink_class_t	class;
	int			err;
	boolean_t		stat_create = B_FALSE;
	char			linkname[MAXLINKNAMELEN];

	rw_enter(&i_dls_devnet_lock, RW_WRITER);

	/*
	 * Don't allow callers to set a link name with a linkid that already
	 * has a name association (that's what rename is for).
	 */
	if (linkid != DATALINK_INVALID_LINKID) {
		if (mod_hash_find(i_dls_devnet_id_hash,
		    (mod_hash_key_t)(uintptr_t)linkid,
		    (mod_hash_val_t *)&ddp) == 0) {
			err = EEXIST;
			goto done;
		}
		if ((err = dls_mgmt_get_linkinfo(linkid, linkname, &class,
		    NULL, NULL)) != 0)
			goto done;
	}

	if ((err = mod_hash_find(i_dls_devnet_hash,
	    (mod_hash_key_t)macname, (mod_hash_val_t *)&ddp)) == 0) {
		if (ddp->dd_linkid != DATALINK_INVALID_LINKID) {
			err = EEXIST;
			goto done;
		}

		/*
		 * If we arrive here we know we are attempting to set
		 * the linkid on a physical link. A virtual link
		 * should never arrive here because it should never
		 * call this function without a linkid. Virtual links
		 * are created through dlgmtmd and thus we know
		 * dlmgmtd is alive to assign it a linkid (search for
		 * uses of dladm_create_datalink_id() to prove this to
		 * yourself); we don't have the same guarantee for a
		 * physical link which may perform an upcall for a
		 * linkid while dlmgmtd is down but will continue
		 * creating a devnet without the linkid (see
		 * softmac_create_datalink() to see how physical link
		 * creation works). That is why there is no entry in
		 * the id hash but there is one in the macname hash --
		 * softmac couldn't acquire a linkid the first time it
		 * called this function.
		 *
		 * Because of the check above, we also know that
		 * ddp->dd_linkid is not set. Following this, the link
		 * must still be in the DD_INITIALIZING state because
		 * that flag is removed IFF dd_linkid is set. This is
		 * why we can ASSERT the DD_INITIALIZING flag below if
		 * the call to i_dls_devnet_setzid() fails.
		 */
		if (linkid == DATALINK_INVALID_LINKID ||
		    class != DATALINK_CLASS_PHYS) {
			err = EINVAL;
			goto done;
		}

		ASSERT(ddp->dd_flags & DD_INITIALIZING);

	} else {
		ddp = kmem_cache_alloc(i_dls_devnet_cachep, KM_SLEEP);
		ddp->dd_flags = DD_INITIALIZING;
		ddp->dd_tref = 0;
		ddp->dd_ref++;
		ddp->dd_owner_zid = zoneid;
		/*
		 * If we are creating a new devnet which will be owned by a NGZ
		 * then mark it as transient. This link has never been in the
		 * GZ, the GZ will not have a hold on its reference, and we do
		 * not want to return it to the GZ when the zone halts.
		 */
		if (zoneid != GLOBAL_ZONEID)
			ddp->dd_transient = B_TRUE;
		(void) strlcpy(ddp->dd_mac, macname, sizeof (ddp->dd_mac));
		VERIFY(mod_hash_insert(i_dls_devnet_hash,
		    (mod_hash_key_t)ddp->dd_mac, (mod_hash_val_t)ddp) == 0);
	}

	if (linkid != DATALINK_INVALID_LINKID) {
		ddp->dd_linkid = linkid;
		(void) strlcpy(ddp->dd_linkname, linkname,
		    sizeof (ddp->dd_linkname));
		VERIFY(mod_hash_insert(i_dls_devnet_id_hash,
		    (mod_hash_key_t)(uintptr_t)linkid,
		    (mod_hash_val_t)ddp) == 0);
		devnet_need_rebuild = B_TRUE;
		stat_create = B_TRUE;
	}
	err = 0;
done:
	/*
	 * It is safe to drop the i_dls_devnet_lock at this point. In the case
	 * of physical devices, the softmac framework will fail the device
	 * detach based on the smac_state or smac_hold_cnt. Other cases like
	 * vnic and aggr use their own scheme to serialize creates and deletes
	 * and ensure that *ddp is valid.
	 */
	rw_exit(&i_dls_devnet_lock);

	if (err == 0 && zoneid != GLOBAL_ZONEID) {
		/*
		 * If this link is being created directly within a non-global
		 * zone, then flag it as transient so that it will be cleaned
		 * up when the zone is shut down.
		 */
		err = i_dls_devnet_setzid(ddp, zoneid, B_FALSE, B_TRUE);
		if (err != 0) {
			/*
			 * At this point the link is marked as
			 * DD_INITIALIZING -- there can be no
			 * outstanding temp refs and therefore no need
			 * to wait for them.
			 */
			ASSERT(ddp->dd_flags & DD_INITIALIZING);
			(void) dls_devnet_unset(mh, &linkid, B_FALSE);
			return (err);
		}
	}

	if (err == 0) {
		/*
		 * The kstat subsystem holds its own locks (rather perimeter)
		 * before calling the ks_update (dls_devnet_stat_update) entry
		 * point which in turn grabs the i_dls_devnet_lock. So the
		 * lock hierarchy is kstat locks -> i_dls_devnet_lock.
		 */
		if (stat_create)
			dls_devnet_stat_create(ddp, zoneid);
		if (ddpp != NULL)
			*ddpp = ddp;

		mutex_enter(&ddp->dd_mutex);
		if (linkid != DATALINK_INVALID_LINKID &&
		    !ddp->dd_prop_loaded && ddp->dd_prop_taskid == 0) {
			ddp->dd_prop_taskid = taskq_dispatch(system_taskq,
			    dls_devnet_prop_task, ddp, TQ_SLEEP);
		}
		mutex_exit(&ddp->dd_mutex);

	}
	return (err);
}

/*
 * Disassociate the linkid from the link identified by macname. If
 * wait is B_TRUE, wait until all temporary refs are released and the
 * prop task is finished.
 *
 * If waiting then you SHOULD NOT call this from inside the MAC perim
 * as deadlock will ensue. Otherwise, this function is safe to call
 * from inside or outside the MAC perim.
 */
static int
dls_devnet_unset(mac_handle_t mh, datalink_id_t *id, boolean_t wait)
{
	const char	*macname = mac_name(mh);
	dls_devnet_t	*ddp;
	int		err;
	mod_hash_val_t	val;

	rw_enter(&i_dls_devnet_lock, RW_WRITER);
	if ((err = mod_hash_find(i_dls_devnet_hash,
	    (mod_hash_key_t)macname, (mod_hash_val_t *)&ddp)) != 0) {
		ASSERT(err == MH_ERR_NOTFOUND);
		rw_exit(&i_dls_devnet_lock);
		return (ENOENT);
	}

	mutex_enter(&ddp->dd_mutex);

	/*
	 * Make sure downcalls into softmac_create or softmac_destroy from
	 * devfs don't cv_wait on any devfs related condition for fear of
	 * deadlock. Return EBUSY if the asynchronous thread started for
	 * property loading as part of the post attach hasn't yet completed.
	 */
	VERIFY(ddp->dd_ref != 0);
	if ((ddp->dd_ref != 1) || (!wait &&
	    (ddp->dd_tref != 0 || ddp->dd_prop_taskid != 0))) {
		int zstatus = 0;

		/*
		 * There are a couple of alternatives that might be going on
		 * here; a) the zone is shutting down and it has a transient
		 * link assigned, in which case we want to clean it up instead
		 * of moving it back to the global zone, or b) its possible
		 * that we're trying to clean up an orphaned vnic that was
		 * delegated to a zone and which wasn't cleaned up properly
		 * when the zone went away.  Check for either of these cases
		 * before we simply return EBUSY.
		 *
		 * zstatus indicates which situation we are dealing with:
		 *	 0 - means return EBUSY
		 *	 1 - means case (a), cleanup transient link
		 *	-1 - means case (b), orphaned VNIC
		 */
		if (ddp->dd_ref > 1 && ddp->dd_zid != GLOBAL_ZONEID) {
			zone_t	*zp;

			if ((zp = zone_find_by_id(ddp->dd_zid)) == NULL) {
				zstatus = -1;
			} else {
				if (ddp->dd_transient) {
					zone_status_t s = zone_status_get(zp);

					if (s >= ZONE_IS_SHUTTING_DOWN)
						zstatus = 1;
				}
				zone_rele(zp);
			}
		}

		if (zstatus == 0) {
			mutex_exit(&ddp->dd_mutex);
			rw_exit(&i_dls_devnet_lock);
			return (EBUSY);
		}

		/*
		 * We want to delete the link, reset ref to 1;
		 */
		if (zstatus == -1) {
			/* Log a warning, but continue in this case */
			cmn_err(CE_WARN, "clear orphaned datalink: %s\n",
			    ddp->dd_linkname);
		}
		ddp->dd_ref = 1;
	}

	ddp->dd_flags |= DD_CONDEMNED;
	ddp->dd_ref--;
	*id = ddp->dd_linkid;

	/*
	 * Remove this dls_devnet_t from the hash table.
	 */
	VERIFY(mod_hash_remove(i_dls_devnet_hash,
	    (mod_hash_key_t)ddp->dd_mac, &val) == 0);

	if (ddp->dd_linkid != DATALINK_INVALID_LINKID) {
		VERIFY(mod_hash_remove(i_dls_devnet_id_hash,
		    (mod_hash_key_t)(uintptr_t)ddp->dd_linkid, &val) == 0);

		devnet_need_rebuild = B_TRUE;
	}
	rw_exit(&i_dls_devnet_lock);

	/*
	 * It is important to call i_dls_devnet_setzid() WITHOUT the
	 * i_dls_devnet_lock held. The setzid call grabs the MAC
	 * perim; thus causing DLS -> MAC lock ordering if performed
	 * with the i_dls_devnet_lock held. This forces consumers to
	 * grab the MAC perim before calling dls_devnet_unset() (the
	 * locking rules state MAC -> DLS order). By performing the
	 * setzid outside of the i_dls_devnet_lock consumers can
	 * safely call dls_devnet_unset() outside the MAC perim.
	 */
	if (ddp->dd_zid != GLOBAL_ZONEID) {
		/*
		 * We need to release the dd_mutex before we try and destroy the
		 * stat. When we destroy it, we'll need to grab the lock for the
		 * kstat but if there's a concurrent reader of the kstat, we'll
		 * be blocked on it. This will lead to deadlock because these
		 * kstats employ a ks_update function (dls_devnet_stat_update)
		 * which needs the dd_mutex that we currently hold.
		 *
		 * Because we've already flagged the dls_devnet_t as
		 * DD_CONDEMNED and we still have a write lock on
		 * i_dls_devnet_lock, we should be able to release the dd_mutex.
		 */
		mutex_exit(&ddp->dd_mutex);
		dls_devnet_stat_destroy(ddp, ddp->dd_zid);
		mutex_enter(&ddp->dd_mutex);
		(void) i_dls_devnet_setzid(ddp, GLOBAL_ZONEID, B_FALSE,
		    B_FALSE);
	}

	if (wait) {
		/*
		 * Wait until all temporary references are released.
		 * The holders of the tref need the MAC perim to
		 * perform their work and release the tref. To avoid
		 * deadlock, assert that the perim is never held here.
		 */
		ASSERT0(MAC_PERIM_HELD(mh));
		while ((ddp->dd_tref != 0) || (ddp->dd_prop_taskid != 0))
			cv_wait(&ddp->dd_cv, &ddp->dd_mutex);
	} else {
		VERIFY(ddp->dd_tref == 0);
		VERIFY(ddp->dd_prop_taskid == 0);
	}

	if (ddp->dd_linkid != DATALINK_INVALID_LINKID)
		dls_devnet_stat_destroy(ddp, ddp->dd_owner_zid);

	ddp->dd_prop_loaded = B_FALSE;
	ddp->dd_linkid = DATALINK_INVALID_LINKID;
	ddp->dd_flags = 0;
	mutex_exit(&ddp->dd_mutex);
	kmem_cache_free(i_dls_devnet_cachep, ddp);

	return (0);
}

/*
 * This is a private hold routine used when we already have the dls_link_t, thus
 * we know that it cannot go away.
 */
int
dls_devnet_hold_tmp_by_link(dls_link_t *dlp, dls_dl_handle_t *ddhp)
{
	int err;
	dls_devnet_t *ddp = NULL;

	rw_enter(&i_dls_devnet_lock, RW_WRITER);
	if ((err = mod_hash_find(i_dls_devnet_hash,
	    (mod_hash_key_t)dlp->dl_name, (mod_hash_val_t *)&ddp)) != 0) {
		ASSERT(err == MH_ERR_NOTFOUND);
		rw_exit(&i_dls_devnet_lock);
		return (ENOENT);
	}

	mutex_enter(&ddp->dd_mutex);
	VERIFY(ddp->dd_ref > 0);
	if (DD_NOT_VISIBLE(ddp->dd_flags)) {
		mutex_exit(&ddp->dd_mutex);
		rw_exit(&i_dls_devnet_lock);
		return (ENOENT);
	}
	ddp->dd_tref++;
	mutex_exit(&ddp->dd_mutex);
	rw_exit(&i_dls_devnet_lock);

	*ddhp = ddp;
	return (0);
}

static int
dls_devnet_hold_common(datalink_id_t linkid, dls_devnet_t **ddpp,
    boolean_t tmp_hold)
{
	dls_devnet_t		*ddp;
	int			err;

	rw_enter(&i_dls_devnet_lock, RW_READER);
	if ((err = mod_hash_find(i_dls_devnet_id_hash,
	    (mod_hash_key_t)(uintptr_t)linkid, (mod_hash_val_t *)&ddp)) != 0) {
		ASSERT(err == MH_ERR_NOTFOUND);
		rw_exit(&i_dls_devnet_lock);
		return (ENOENT);
	}

	mutex_enter(&ddp->dd_mutex);
	VERIFY(ddp->dd_ref > 0);
	if (DD_NOT_VISIBLE(ddp->dd_flags)) {
		mutex_exit(&ddp->dd_mutex);
		rw_exit(&i_dls_devnet_lock);
		return (ENOENT);
	}
	if (tmp_hold)
		ddp->dd_tref++;
	else
		ddp->dd_ref++;
	mutex_exit(&ddp->dd_mutex);
	rw_exit(&i_dls_devnet_lock);

	*ddpp = ddp;
	return (0);
}

int
dls_devnet_hold(datalink_id_t linkid, dls_devnet_t **ddpp)
{
	return (dls_devnet_hold_common(linkid, ddpp, B_FALSE));
}

/*
 * Hold the vanity naming structure (dls_devnet_t) temporarily.  The request to
 * delete the dls_devnet_t will wait until the temporary reference is released.
 */
int
dls_devnet_hold_tmp(datalink_id_t linkid, dls_devnet_t **ddpp)
{
	return (dls_devnet_hold_common(linkid, ddpp, B_TRUE));
}

/*
 * This funtion is called when a DLS client tries to open a device node.
 * This dev_t could be a result of a /dev/net node access (returned by
 * devnet_create_rvp->dls_devnet_open()) or a direct /dev node access.
 * In both cases, this function bumps up the reference count of the
 * dls_devnet_t structure. The reference is held as long as the device node
 * is open. In the case of /dev/net while it is true that the initial reference
 * is held when the devnet_create_rvp->dls_devnet_open call happens, this
 * initial reference is released immediately in devnet_inactive_callback ->
 * dls_devnet_close(). (Note that devnet_inactive_callback() is called right
 * after dld_open completes, not when the /dev/net node is being closed).
 * To undo this function, call dls_devnet_rele()
 */
int
dls_devnet_hold_by_dev(dev_t dev, dls_dl_handle_t *ddhp)
{
	char			name[MAXNAMELEN];
	char			*drv;
	dls_devnet_t		*ddp;
	int			err;

	if ((drv = ddi_major_to_name(getmajor(dev))) == NULL)
		return (EINVAL);

	(void) snprintf(name, sizeof (name), "%s%d", drv,
	    DLS_MINOR2INST(getminor(dev)));

	rw_enter(&i_dls_devnet_lock, RW_READER);
	if ((err = mod_hash_find(i_dls_devnet_hash,
	    (mod_hash_key_t)name, (mod_hash_val_t *)&ddp)) != 0) {
		ASSERT(err == MH_ERR_NOTFOUND);
		rw_exit(&i_dls_devnet_lock);
		return (ENOENT);
	}
	mutex_enter(&ddp->dd_mutex);
	VERIFY(ddp->dd_ref > 0);
	if (DD_NOT_VISIBLE(ddp->dd_flags)) {
		mutex_exit(&ddp->dd_mutex);
		rw_exit(&i_dls_devnet_lock);
		return (ENOENT);
	}
	ddp->dd_ref++;
	mutex_exit(&ddp->dd_mutex);
	rw_exit(&i_dls_devnet_lock);

	*ddhp = ddp;
	return (0);
}

void
dls_devnet_rele(dls_devnet_t *ddp)
{
	mutex_enter(&ddp->dd_mutex);
	VERIFY(ddp->dd_ref > 1);
	ddp->dd_ref--;
	if ((ddp->dd_flags & DD_IMPLICIT_IPTUN) && ddp->dd_ref == 1) {
		mutex_exit(&ddp->dd_mutex);
		if (i_dls_devnet_destroy_iptun(ddp->dd_linkid) != 0)
			ddp->dd_flags |= DD_IMPLICIT_IPTUN;
		return;
	}
	mutex_exit(&ddp->dd_mutex);
}

static int
dls_devnet_hold_by_name(const char *link, dls_devnet_t **ddpp)
{
	char			drv[MAXLINKNAMELEN];
	uint_t			ppa;
	major_t			major;
	dev_t			phy_dev, tmp_dev;
	datalink_id_t		linkid;
	dls_dev_handle_t	ddh;
	int			err;

	if ((err = dls_mgmt_get_linkid(link, &linkid)) == 0)
		return (dls_devnet_hold(linkid, ddpp));

	/*
	 * If we failed to get the link's linkid because the dlmgmtd daemon
	 * has not been started, return ENOENT so that the application can
	 * fallback to open the /dev node.
	 */
	if (err == EBADF)
		return (ENOENT);

	if (err != ENOENT)
		return (err);

	/*
	 * If we reach this point it means dlmgmtd is up but has no
	 * mapping for the link name.
	 */
	if (ddi_parse_dlen(link, drv, MAXLINKNAMELEN, &ppa) != DDI_SUCCESS)
		return (ENOENT);

	if (IS_IPTUN_LINK(drv)) {
		if ((err = i_dls_devnet_create_iptun(link, drv, &linkid)) != 0)
			return (err);
		/*
		 * At this point, an IP tunnel MAC has registered, which
		 * resulted in a link being created.
		 */
		err = dls_devnet_hold(linkid, ddpp);
		if (err != 0) {
			VERIFY(i_dls_devnet_destroy_iptun(linkid) == 0);
			return (err);
		}
		/*
		 * dls_devnet_rele() will know to destroy the implicit IP
		 * tunnel on last reference release if DD_IMPLICIT_IPTUN is
		 * set.
		 */
		(*ddpp)->dd_flags |= DD_IMPLICIT_IPTUN;
		return (0);
	}

	/*
	 * If this link:
	 * (a) is a physical device, (b) this is the first boot, (c) the MAC
	 * is not registered yet, and (d) we cannot find its linkid, then the
	 * linkname is the same as the devname.
	 *
	 * First filter out invalid names.
	 */
	if ((major = ddi_name_to_major(drv)) == (major_t)-1)
		return (ENOENT);

	phy_dev = makedevice(major, DLS_PPA2MINOR(ppa));
	if (softmac_hold_device(phy_dev, &ddh) != 0)
		return (ENOENT);

	/*
	 * At this time, the MAC should be registered, check its phy_dev using
	 * the given name.
	 */
	if ((err = dls_mgmt_get_linkid(link, &linkid)) != 0 ||
	    (err = dls_mgmt_get_phydev(linkid, &tmp_dev)) != 0) {
		softmac_rele_device(ddh);
		return (err);
	}
	if (tmp_dev != phy_dev) {
		softmac_rele_device(ddh);
		return (ENOENT);
	}

	err = dls_devnet_hold(linkid, ddpp);
	softmac_rele_device(ddh);
	return (err);
}

int
dls_devnet_macname2linkid(const char *macname, datalink_id_t *linkidp)
{
	dls_devnet_t	*ddp;

	rw_enter(&i_dls_devnet_lock, RW_READER);
	if (mod_hash_find(i_dls_devnet_hash, (mod_hash_key_t)macname,
	    (mod_hash_val_t *)&ddp) != 0) {
		rw_exit(&i_dls_devnet_lock);
		return (ENOENT);
	}

	*linkidp = ddp->dd_linkid;
	rw_exit(&i_dls_devnet_lock);
	return (0);
}

/*
 * Get linkid for the given dev.
 */
int
dls_devnet_dev2linkid(dev_t dev, datalink_id_t *linkidp)
{
	char	macname[MAXNAMELEN];
	char	*drv;

	if ((drv = ddi_major_to_name(getmajor(dev))) == NULL)
		return (EINVAL);

	(void) snprintf(macname, sizeof (macname), "%s%d", drv,
	    DLS_MINOR2INST(getminor(dev)));
	return (dls_devnet_macname2linkid(macname, linkidp));
}

/*
 * Get the link's physical dev_t. It this is a VLAN, get the dev_t of the
 * link this VLAN is created on.
 */
int
dls_devnet_phydev(datalink_id_t vlanid, dev_t *devp)
{
	dls_devnet_t	*ddp;
	int		err;

	if ((err = dls_devnet_hold_tmp(vlanid, &ddp)) != 0)
		return (err);

	err = dls_mgmt_get_phydev(ddp->dd_linkid, devp);
	dls_devnet_rele_tmp(ddp);
	return (err);
}

/*
 * Handle the renaming requests.  There are two rename cases:
 *
 * 1. Request to rename a valid link (id1) to an non-existent link name
 *    (id2). In this case id2 is DATALINK_INVALID_LINKID.  Just check whether
 *    id1 is held by any applications.
 *
 *    In this case, the link's kstats need to be updated using the given name.
 *
 * 2. Request to rename a valid link (id1) to the name of a REMOVED
 *    physical link (id2). In this case, check that id1 and its associated
 *    mac is not held by any application, and update the link's linkid to id2.
 *
 *    This case does not change the <link name, linkid> mapping, so the link's
 *    kstats need to be updated with using name associated the given id2.
 */
int
dls_devnet_rename(datalink_id_t id1, datalink_id_t id2, const char *link)
{
	dls_dev_handle_t	ddh = NULL;
	int			err = 0;
	dev_t			phydev = 0;
	dls_devnet_t		*ddp;
	mac_perim_handle_t	mph = NULL;
	mac_handle_t		mh;
	mod_hash_val_t		val;

	/*
	 * In the second case, id2 must be a REMOVED physical link.
	 */
	if ((id2 != DATALINK_INVALID_LINKID) &&
	    (dls_mgmt_get_phydev(id2, &phydev) == 0) &&
	    softmac_hold_device(phydev, &ddh) == 0) {
		softmac_rele_device(ddh);
		return (EEXIST);
	}

	/*
	 * Hold id1 to prevent it from being detached (if a physical link).
	 */
	if (dls_mgmt_get_phydev(id1, &phydev) == 0)
		(void) softmac_hold_device(phydev, &ddh);

	/*
	 * The framework does not hold hold locks across calls to the
	 * mac perimeter, hence enter the perimeter first. This also waits
	 * for the property loading to finish.
	 */
	if ((err = mac_perim_enter_by_linkid(id1, &mph)) != 0) {
		softmac_rele_device(ddh);
		return (err);
	}

	rw_enter(&i_dls_devnet_lock, RW_WRITER);
	if ((err = mod_hash_find(i_dls_devnet_id_hash,
	    (mod_hash_key_t)(uintptr_t)id1, (mod_hash_val_t *)&ddp)) != 0) {
		ASSERT(err == MH_ERR_NOTFOUND);
		err = ENOENT;
		goto done;
	}

	mutex_enter(&ddp->dd_mutex);
	if (ddp->dd_ref > 1) {
		mutex_exit(&ddp->dd_mutex);
		err = EBUSY;
		goto done;
	}
	mutex_exit(&ddp->dd_mutex);

	if (id2 == DATALINK_INVALID_LINKID) {
		(void) strlcpy(ddp->dd_linkname, link,
		    sizeof (ddp->dd_linkname));

		/* rename mac client name and its flow if exists */
		if ((err = mac_open(ddp->dd_mac, &mh)) != 0)
			goto done;
		(void) mac_rename_primary(mh, link);
		mac_close(mh);
		goto done;
	}

	/*
	 * The second case, check whether the MAC is used by any MAC
	 * user.  This must be a physical link so ddh must not be NULL.
	 */
	if (ddh == NULL) {
		err = EINVAL;
		goto done;
	}

	if ((err = mac_open(ddp->dd_mac, &mh)) != 0)
		goto done;

	/*
	 * We release the reference of the MAC which mac_open() is
	 * holding. Note that this mac will not be unregistered
	 * because the physical device is held.
	 */
	mac_close(mh);

	/*
	 * Check if there is any other MAC clients, if not, hold this mac
	 * exclusively until we are done.
	 */
	if ((err = mac_mark_exclusive(mh)) != 0)
		goto done;

	/*
	 * Update the link's linkid.
	 */
	if ((err = mod_hash_find(i_dls_devnet_id_hash,
	    (mod_hash_key_t)(uintptr_t)id2, &val)) != MH_ERR_NOTFOUND) {
		mac_unmark_exclusive(mh);
		err = EEXIST;
		goto done;
	}

	err = dls_mgmt_get_linkinfo(id2, ddp->dd_linkname, NULL, NULL, NULL);
	if (err != 0) {
		mac_unmark_exclusive(mh);
		goto done;
	}

	(void) mod_hash_remove(i_dls_devnet_id_hash,
	    (mod_hash_key_t)(uintptr_t)id1, &val);

	ddp->dd_linkid = id2;
	(void) mod_hash_insert(i_dls_devnet_id_hash,
	    (mod_hash_key_t)(uintptr_t)ddp->dd_linkid, (mod_hash_val_t)ddp);

	mac_unmark_exclusive(mh);

	/* load properties for new id */
	mutex_enter(&ddp->dd_mutex);
	ddp->dd_prop_loaded = B_FALSE;
	ddp->dd_prop_taskid = taskq_dispatch(system_taskq,
	    dls_devnet_prop_task, ddp, TQ_SLEEP);
	mutex_exit(&ddp->dd_mutex);

done:
	rw_exit(&i_dls_devnet_lock);

	if (err == 0)
		dls_devnet_stat_rename(ddp);

	if (mph != NULL)
		mac_perim_exit(mph);
	softmac_rele_device(ddh);
	return (err);
}

static int
i_dls_devnet_setzid(dls_devnet_t *ddp, zoneid_t new_zoneid, boolean_t setprop,
    boolean_t transient)
{
	int			err;
	mac_perim_handle_t	mph;
	boolean_t		upcall_done = B_FALSE;
	datalink_id_t		linkid = ddp->dd_linkid;
	zoneid_t		old_zoneid = ddp->dd_zid;
	dlmgmt_door_setzoneid_t	setzid;
	dlmgmt_setzoneid_retval_t retval;

	if (old_zoneid == new_zoneid)
		return (0);

	if ((err = mac_perim_enter_by_macname(ddp->dd_mac, &mph)) != 0)
		return (err);

	/*
	 * When changing the zoneid of an existing link, we need to tell
	 * dlmgmtd about it.  dlmgmtd already knows the zoneid associated with
	 * newly created links.
	 */
	if (setprop) {
		setzid.ld_cmd = DLMGMT_CMD_SETZONEID;
		setzid.ld_linkid = linkid;
		setzid.ld_zoneid = new_zoneid;
		err = i_dls_mgmt_upcall(&setzid, sizeof (setzid), &retval,
		    sizeof (retval));
		if (err != 0)
			goto done;

		/*
		 * We set upcall_done only if the upcall is
		 * successful. This way, if dls_link_setzid() fails,
		 * we know another upcall must be done to reset the
		 * dlmgmtd state.
		 */
		upcall_done = B_TRUE;
	}
	if ((err = dls_link_setzid(ddp->dd_mac, new_zoneid)) == 0) {
		ddp->dd_zid = new_zoneid;
		ddp->dd_transient = transient;
		devnet_need_rebuild = B_TRUE;
	}

done:
	if (err != 0 && upcall_done) {
		setzid.ld_zoneid = old_zoneid;
		(void) i_dls_mgmt_upcall(&setzid, sizeof (setzid), &retval,
		    sizeof (retval));
	}
	mac_perim_exit(mph);
	return (err);
}

int
dls_devnet_setzid(dls_dl_handle_t ddh, zoneid_t new_zid)
{
	dls_devnet_t	*ddp;
	int		err;
	zoneid_t	old_zid;
	boolean_t	refheld = B_FALSE;

	old_zid = ddh->dd_zid;

	if (old_zid == new_zid)
		return (0);

	/*
	 * Acquire an additional reference to the link if it is being assigned
	 * to a non-global zone from the global zone.
	 */
	if (old_zid == GLOBAL_ZONEID && new_zid != GLOBAL_ZONEID) {
		if ((err = dls_devnet_hold(ddh->dd_linkid, &ddp)) != 0)
			return (err);
		refheld = B_TRUE;
	}

	if ((err = i_dls_devnet_setzid(ddh, new_zid, B_TRUE, B_FALSE)) != 0) {
		if (refheld)
			dls_devnet_rele(ddp);
		return (err);
	}

	/*
	 * Release the additional reference if the link is returning to the
	 * global zone from a non-global zone.
	 */
	if (old_zid != GLOBAL_ZONEID && new_zid == GLOBAL_ZONEID)
		dls_devnet_rele(ddh);

	/* Re-create kstats in the appropriate zones. */
	if (old_zid != GLOBAL_ZONEID)
		dls_devnet_stat_destroy(ddh, old_zid);
	if (new_zid != GLOBAL_ZONEID)
		dls_devnet_stat_create(ddh, new_zid);

	return (0);
}

zoneid_t
dls_devnet_getzid(dls_dl_handle_t ddh)
{
	return (((dls_devnet_t *)ddh)->dd_zid);
}

zoneid_t
dls_devnet_getownerzid(dls_dl_handle_t ddh)
{
	return (((dls_devnet_t *)ddh)->dd_owner_zid);
}

/*
 * Is linkid visible from zoneid?  A link is visible if it was created in the
 * zone, or if it is currently assigned to the zone.
 */
boolean_t
dls_devnet_islinkvisible(datalink_id_t linkid, zoneid_t zoneid)
{
	dls_devnet_t	*ddp;
	boolean_t	result;

	if (dls_devnet_hold_tmp(linkid, &ddp) != 0)
		return (B_FALSE);
	result = (ddp->dd_owner_zid == zoneid || ddp->dd_zid == zoneid);
	dls_devnet_rele_tmp(ddp);
	return (result);
}

/*
 * Access a vanity naming node.
 */
int
dls_devnet_open(const char *link, dls_dl_handle_t *dhp, dev_t *devp)
{
	dls_devnet_t	*ddp;
	dls_link_t	*dlp;
	zoneid_t	zid = getzoneid();
	int		err;
	mac_perim_handle_t	mph;

	if ((err = dls_devnet_hold_by_name(link, &ddp)) != 0)
		return (err);

	dls_devnet_prop_task_wait(ddp);

	/*
	 * Opening a link that does not belong to the current non-global zone
	 * is not allowed.
	 */
	if (zid != GLOBAL_ZONEID && ddp->dd_zid != zid) {
		dls_devnet_rele(ddp);
		return (ENOENT);
	}

	err = mac_perim_enter_by_macname(ddp->dd_mac, &mph);
	if (err != 0) {
		dls_devnet_rele(ddp);
		return (err);
	}

	err = dls_link_hold_create(ddp->dd_mac, &dlp);
	mac_perim_exit(mph);

	if (err != 0) {
		dls_devnet_rele(ddp);
		return (err);
	}

	*dhp = ddp;
	*devp = dls_link_dev(dlp);
	return (0);
}

/*
 * Close access to a vanity naming node.
 */
void
dls_devnet_close(dls_dl_handle_t dlh)
{
	dls_devnet_t	*ddp = dlh;
	dls_link_t	*dlp;
	mac_perim_handle_t	mph;

	VERIFY(mac_perim_enter_by_macname(ddp->dd_mac, &mph) == 0);
	VERIFY(dls_link_hold(ddp->dd_mac, &dlp) == 0);

	/*
	 * One rele for the hold placed in dls_devnet_open, another for
	 * the hold done just above
	 */
	dls_link_rele(dlp);
	dls_link_rele(dlp);
	mac_perim_exit(mph);

	dls_devnet_rele(ddp);
}

/*
 * This is used by /dev/net to rebuild the nodes for readdir().  It is not
 * critical and no protection is needed.
 */
boolean_t
dls_devnet_rebuild()
{
	boolean_t updated = devnet_need_rebuild;

	devnet_need_rebuild = B_FALSE;
	return (updated);
}

int
dls_devnet_create(mac_handle_t mh, datalink_id_t linkid, zoneid_t zoneid)
{
	dls_link_t	*dlp;
	dls_devnet_t	*ddp;
	int		err;
	mac_perim_handle_t mph;

	/*
	 * Holding the mac perimeter ensures that the downcall from the
	 * dlmgmt daemon which does the property loading does not proceed
	 * until we relinquish the perimeter.
	 */
	mac_perim_enter_by_mh(mh, &mph);
	/*
	 * Make this association before we call dls_link_hold_create as
	 * we need to use the linkid to get the user name for the link
	 * when we create the MAC client.
	 */
	if ((err = dls_devnet_set(mh, linkid, zoneid, &ddp)) == 0) {
		if ((err = dls_link_hold_create(mac_name(mh), &dlp)) != 0) {
			mac_perim_exit(mph);
			(void) dls_devnet_unset(mh, &linkid, B_FALSE);
			return (err);
		}

		/*
		 * If dd_linkid is set then the link was successfully
		 * initialized. In this case we can remove the
		 * initializing flag and make the link visible to the
		 * rest of the system.
		 *
		 * If not set then we were called by softmac and it
		 * was unable to obtain a linkid for the physical link
		 * because dlmgmtd is down. In that case softmac will
		 * eventually obtain a linkid and call
		 * dls_devnet_recreate() to complete initialization.
		 */
		mutex_enter(&ddp->dd_mutex);
		if (ddp->dd_linkid != DATALINK_INVALID_LINKID)
			ddp->dd_flags &= ~DD_INITIALIZING;
		mutex_exit(&ddp->dd_mutex);

	}

	mac_perim_exit(mph);
	return (err);
}

/*
 * Set the linkid of the dls_devnet_t and add it into the i_dls_devnet_id_hash.
 * This is called in the case that the dlmgmtd daemon is started later than
 * the physical devices get attached, and the linkid is only known after the
 * daemon starts.
 */
int
dls_devnet_recreate(mac_handle_t mh, datalink_id_t linkid)
{
	dls_devnet_t	*ddp;
	int		err;

	VERIFY(linkid != DATALINK_INVALID_LINKID);
	if ((err = dls_devnet_set(mh, linkid, GLOBAL_ZONEID, &ddp)) == 0) {
		mutex_enter(&ddp->dd_mutex);
		if (ddp->dd_linkid != DATALINK_INVALID_LINKID)
			ddp->dd_flags &= ~DD_INITIALIZING;
		mutex_exit(&ddp->dd_mutex);
	}

	return (err);

}

int
dls_devnet_destroy(mac_handle_t mh, datalink_id_t *idp, boolean_t wait)
{
	int			err;
	mac_perim_handle_t	mph;

	*idp = DATALINK_INVALID_LINKID;
	err = dls_devnet_unset(mh, idp, wait);

	/*
	 * We continue on in the face of ENOENT because the devnet
	 * unset and DLS link release are not atomic and we may have a
	 * scenario where there is no entry in i_dls_devnet_hash for
	 * the MAC name but there is an entry in i_dls_link_hash. For
	 * example, if the following occurred:
	 *
	 * 1. dls_devnet_unset() returns success, and
	 *
	 * 2. dls_link_rele_by_name() fails with ENOTEMPTY because
	 *    flows still exist, and
	 *
	 * 3. dls_devnet_set() fails to set the zone id and calls
	 *    dls_devnet_unset() -- leaving an entry in
	 *    i_dls_link_hash but no corresponding entry in
	 *    i_dls_devnet_hash.
	 *
	 * Even if #3 wasn't true the dls_devnet_set() may fail for
	 * different reasons in the future; the point is that it _can_
	 * fail as part of its contract. We can't rely on it working
	 * so we must assume that these two pieces of state (devnet
	 * and link hashes), which should always be in sync, can get
	 * out of sync and thus even if we get ENOENT from the devnet
	 * hash we should still try to delete from the link hash just
	 * in case.
	 *
	 * We could prevent the ENOTEMPTY from dls_link_rele_by_name()
	 * by calling mac_disable() before calling
	 * dls_devnet_destroy() but that's not currently possible due
	 * to a long-standing bug. OpenSolaris 6791335: The semantics
	 * of mac_disable() were modified by Crossbow such that
	 * dls_devnet_destroy() needs to be called before
	 * mac_disable() can succeed. This is because of the implicit
	 * reference that dls has on the mac_impl_t.
	 */
	if (err != 0 && err != ENOENT)
		return (err);

	mac_perim_enter_by_mh(mh, &mph);
	err = dls_link_rele_by_name(mac_name(mh));
	mac_perim_exit(mph);

	if (err != 0) {
		dls_devnet_t	*ddp;

		/*
		 * XXX It is a general GLDv3 bug that dls_devnet_set() has to
		 * be called to re-set the link when destroy fails.  The
		 * zoneid below will be incorrect if this function is ever
		 * called from kernel context or from a zone other than that
		 * which initially created the link.
		 */
		(void) dls_devnet_set(mh, *idp, crgetzoneid(CRED()), &ddp);

		/*
		 * You might think dd_linkid should always be set
		 * here, but in the case where dls_devnet_unset()
		 * returns ENOENT it will be DATALINK_INVALID_LINKID.
		 * Stay consistent with the rest of DLS and only
		 * remove the initializing flag if linkid is set.
		 */
		mutex_enter(&ddp->dd_mutex);
		if (ddp->dd_linkid != DATALINK_INVALID_LINKID)
			ddp->dd_flags &= ~DD_INITIALIZING;
		mutex_exit(&ddp->dd_mutex);
	}
	return (err);
}

/*
 * Implicitly create an IP tunnel link.
 */
static int
i_dls_devnet_create_iptun(const char *linkname, const char *drvname,
    datalink_id_t *linkid)
{
	int		err;
	iptun_kparams_t	ik;
	uint32_t	media;
	netstack_t	*ns;
	major_t		iptun_major;
	dev_info_t	*iptun_dip;

	/* First ensure that the iptun device is attached. */
	if ((iptun_major = ddi_name_to_major(IPTUN_DRIVER_NAME)) == (major_t)-1)
		return (EINVAL);
	if ((iptun_dip = ddi_hold_devi_by_instance(iptun_major, 0, 0)) == NULL)
		return (EINVAL);

	if (IS_IPV4_TUN(drvname)) {
		ik.iptun_kparam_type = IPTUN_TYPE_IPV4;
		media = DL_IPV4;
	} else if (IS_6TO4_TUN(drvname)) {
		ik.iptun_kparam_type = IPTUN_TYPE_6TO4;
		media = DL_6TO4;
	} else if (IS_IPV6_TUN(drvname)) {
		ik.iptun_kparam_type = IPTUN_TYPE_IPV6;
		media = DL_IPV6;
	}
	ik.iptun_kparam_flags = (IPTUN_KPARAM_TYPE | IPTUN_KPARAM_IMPLICIT);

	/* Obtain a datalink id for this tunnel. */
	err = dls_mgmt_create((char *)linkname, 0, DATALINK_CLASS_IPTUN, media,
	    B_FALSE, &ik.iptun_kparam_linkid);
	if (err != 0) {
		ddi_release_devi(iptun_dip);
		return (err);
	}

	ns = netstack_get_current();
	err = iptun_create(&ik, CRED());
	netstack_rele(ns);

	if (err != 0)
		VERIFY(dls_mgmt_destroy(ik.iptun_kparam_linkid, B_FALSE) == 0);
	else
		*linkid = ik.iptun_kparam_linkid;

	ddi_release_devi(iptun_dip);
	return (err);
}

static int
i_dls_devnet_destroy_iptun(datalink_id_t linkid)
{
	int err;

	/*
	 * Note the use of zone_kcred() here as opposed to CRED().  This is
	 * because the process that does the last close of this /dev/net node
	 * may not have necessary privileges to delete this IP tunnel, but the
	 * tunnel must always be implicitly deleted on last close.
	 */
	if ((err = iptun_delete(linkid, zone_kcred())) == 0)
		(void) dls_mgmt_destroy(linkid, B_FALSE);
	return (err);
}

const char *
dls_devnet_link(dls_dl_handle_t ddh)
{
	return (ddh->dd_linkname);
}

const char *
dls_devnet_mac(dls_dl_handle_t ddh)
{
	return (ddh->dd_mac);
}

datalink_id_t
dls_devnet_linkid(dls_dl_handle_t ddh)
{
	return (ddh->dd_linkid);
}
