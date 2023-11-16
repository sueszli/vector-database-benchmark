#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/param.h>
#include <sys/systm.h>
#include <sys/kernel.h>
#include <sys/kthread.h>
#include <sys/namei.h>
#include <sys/proc.h>
#include <sys/filedesc.h>
#include <sys/vnode.h>
#include <sys/malloc.h>
#include <sys/unistd.h>
#include <sys/fcntl.h>
#include <sys/buf.h>
#include <sys/mutex.h>
#include <sys/vnode.h>

#include <sys/systm.h>
#include <sys/conf.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/proc.h>
#include <sys/uio.h>
#include <sys/kernel.h>
#include <sys/vnode.h>
#include <sys/namei.h>
#include <sys/malloc.h>
#include <sys/mount.h>
#include <sys/exec.h>
#include <sys/mbuf.h>
#include <sys/poll.h>
#include <sys/select.h>
#include "kernio.h"
struct vnode *
kio_open(const char *file, int flags, int cmode)
{
	struct thread *td = curthread;
	struct nameidata nd;
	int error;

	if (td->td_proc->p_fd->fd_rdir == NULL)
		td->td_proc->p_fd->fd_rdir = rootvnode;
	if (td->td_proc->p_fd->fd_cdir == NULL)
		td->td_proc->p_fd->fd_cdir = rootvnode;

	flags = FFLAGS(flags);
	NDINIT(&nd, LOOKUP, NOFOLLOW, UIO_SYSSPACE, file, td);
	error = vn_open_cred(&nd, &flags, cmode,V_SAVE | V_NORMAL , td->td_ucred, NULL);
	NDFREE(&nd, NDF_ONLY_PNBUF);
	if (error != 0)
		return (NULL);
	/* We just unlock so we hold a reference. */
	VOP_UNLOCK(nd.ni_vp, 0);
	return (nd.ni_vp);
}

void
kio_close(struct vnode *vp)
{
	struct thread *td = curthread;

	vn_close(vp, FWRITE, td->td_ucred, td);
}

int
kio_write(struct vnode *vp, void *buf, size_t size, unsigned long offset)
{
	struct thread *td = curthread;
	struct mount *mp;
	struct uio auio;
	struct iovec aiov;

	bzero(&aiov, sizeof(aiov));
	bzero(&auio, sizeof(auio));

	aiov.iov_base = buf;
	aiov.iov_len = size;

	auio.uio_iov = &aiov;
	auio.uio_offset = offset;
	auio.uio_segflg = UIO_SYSSPACE;
	auio.uio_rw = UIO_WRITE;
	auio.uio_iovcnt = 1;
	auio.uio_resid = size;
	auio.uio_td = td;

	/*
	 * Do all of the junk required to write now.
	 */
	vn_start_write(vp, &mp, V_WAIT);
	vn_lock(vp, LK_EXCLUSIVE | LK_RETRY);
	//VOP_LEASE(vp, td, td->td_ucred, LEASE_WRITE);
	VOP_WRITE(vp, &auio, IO_UNIT | IO_ASYNC, td->td_ucred);
	VOP_UNLOCK(vp, 0);
	vn_finished_write(mp);
	return (0);
}
