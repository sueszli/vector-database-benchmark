/*
 * Copyright (C) 2020 Waldemar Kozaczuk
 *
 * This work is open source software, licensed under the terms of the
 * BSD license as described in the LICENSE file in the top-level directory.
 */

#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tuple>

#include <osv/contiguous_alloc.hh>
#include <osv/debug.h>
#include <osv/device.h>
#include <osv/file.h>
#include <osv/mmio.hh>
#include <osv/mount.h>
#include <osv/prex.h>
#include <osv/sched.hh>
#include <osv/vnode.h>

#include "fuse_kernel.h"
#include "virtiofs.hh"
#include "virtiofs_dax.hh"
#include "virtiofs_i.hh"

static constexpr uint32_t OPEN_FLAGS = O_RDONLY;

int virtiofs_init()
{
    return 0;
}

static int virtiofs_lookup(struct vnode* vnode, char* name, struct vnode** vpp)
{
    auto* inode = static_cast<virtiofs_inode*>(vnode->v_data);

    if (*name == '\0') {
        return ENOENT;
    }

    if (!S_ISDIR(inode->attr.mode)) {
        kprintf("[virtiofs] inode %lld, ABORTED lookup of %s because not a "
                "directory\n", inode->nodeid, name);
        return ENOTDIR;
    }

    auto in_args_len = strlen(name) + 1;
    std::unique_ptr<char[]> in_args {new (std::nothrow) char[in_args_len]};
    std::unique_ptr<fuse_entry_out> out_args {
        new (std::nothrow) fuse_entry_out};
    if (!out_args || !in_args) {
        return ENOMEM;
    }
    strcpy(in_args.get(), name);

    auto* m_data = static_cast<virtiofs_mount_data*>(vnode->v_mount->m_data);
    auto* drv = m_data->drv;
    auto error = fuse_req_send_and_receive_reply(drv, FUSE_LOOKUP,
        inode->nodeid, in_args.get(), in_args_len, out_args.get(),
        sizeof(*out_args)).second;
    if (error) {
        kprintf("[virtiofs] inode %lld, lookup failed to find %s\n",
            inode->nodeid, name);
        // TODO: Implement proper error handling by sending FUSE_FORGET
        return error;
    }

    struct vnode* vp;
    // TODO OPT: Should we even use the cache? (consult spec on metadata)
    if (vget(vnode->v_mount, out_args->nodeid, &vp) == 1) {
        virtiofs_debug("lookup found vp in cache\n");
        *vpp = vp;
        return 0;
    }

    auto* new_inode = new (std::nothrow) virtiofs_inode;
    if (!new_inode) {
        return ENOMEM;
    }
    new_inode->nodeid = out_args->nodeid;
    virtiofs_debug("inode %lld, lookup found inode %lld for %s\n",
        inode->nodeid, new_inode->nodeid, name);
    memcpy(&new_inode->attr, &out_args->attr, sizeof(out_args->attr));

    virtiofs_set_vnode(vp, new_inode);
    *vpp = vp;

    return 0;
}

static int virtiofs_open(struct file* fp)
{
    if ((file_flags(fp) & FWRITE)) {
        // Do not allow opening files to write
        return EROFS;
    }

    auto* vnode = file_dentry(fp)->d_vnode;
    auto* inode = static_cast<virtiofs_inode*>(vnode->v_data);

    std::unique_ptr<fuse_open_in> in_args {new (std::nothrow) fuse_open_in()};
    std::unique_ptr<fuse_open_out> out_args {new (std::nothrow) fuse_open_out};
    if (!out_args || !in_args) {
        return ENOMEM;
    }
    in_args->flags = OPEN_FLAGS;

    auto* m_data = static_cast<virtiofs_mount_data*>(vnode->v_mount->m_data);
    auto* drv = m_data->drv;
    auto operation = S_ISDIR(inode->attr.mode) ? FUSE_OPENDIR : FUSE_OPEN;
    auto error = fuse_req_send_and_receive_reply(drv, operation,
        inode->nodeid, in_args.get(), sizeof(*in_args), out_args.get(),
        sizeof(*out_args)).second;
    if (error) {
        kprintf("[virtiofs] inode %lld, open failed\n", inode->nodeid);
        return error;
    }

    virtiofs_debug("inode %lld, opened\n", inode->nodeid);

    auto* f_data = new (std::nothrow) virtiofs_file_data;
    if (!f_data) {
        return ENOMEM;
    }
    f_data->file_handle = out_args->fh;
    // TODO OPT: Consult and possibly act upon out_args->open_flags
    file_setdata(fp, f_data);

    return 0;
}

static int virtiofs_close(struct vnode* vnode, struct file* fp)
{
    auto* inode = static_cast<virtiofs_inode*>(vnode->v_data);

    std::unique_ptr<fuse_release_in> in_args {
        new (std::nothrow) fuse_release_in()};
    if (!in_args) {
        return ENOMEM;
    }
    auto* f_data = static_cast<virtiofs_file_data*>(file_data(fp));
    in_args->fh = f_data->file_handle;
    in_args->flags = OPEN_FLAGS; // need to be same as in FUSE_OPEN

    auto* m_data = static_cast<virtiofs_mount_data*>(vnode->v_mount->m_data);
    auto* drv = m_data->drv;
    auto operation = S_ISDIR(inode->attr.mode) ? FUSE_RELEASEDIR : FUSE_RELEASE;
    auto error = fuse_req_send_and_receive_reply(drv, operation,
        inode->nodeid, in_args.get(), sizeof(*in_args), nullptr, 0).second;
    if (error) {
        kprintf("[virtiofs] inode %lld, close failed\n", inode->nodeid);
        return error;
    }

    file_setdata(fp, nullptr);
    delete f_data;
    virtiofs_debug("inode %lld, closed\n", inode->nodeid);

    // TODO: Investigate if we should send FUSE_FORGET once all handles to the
    // file closed on our side

    return 0;
}

static int virtiofs_readlink(struct vnode* vnode, struct uio* uio)
{
    auto* inode = static_cast<virtiofs_inode*>(vnode->v_data);

    std::unique_ptr<char[]> link_path {new (std::nothrow) char[PATH_MAX]};
    if (!link_path) {
        return ENOMEM;
    }

    auto* m_data = static_cast<virtiofs_mount_data*>(vnode->v_mount->m_data);
    auto* drv = m_data->drv;
    auto error = fuse_req_send_and_receive_reply(drv, FUSE_READLINK,
        inode->nodeid, nullptr, 0, link_path.get(), PATH_MAX).second;
    if (error) {
        kprintf("[virtiofs] inode %lld, readlink failed\n", inode->nodeid);
        return error;
    }

    virtiofs_debug("inode %lld, read symlink [%s]\n", inode->nodeid,
        link_path.get());
    return uiomove(link_path.get(), strlen(link_path.get()), uio);
}

// Read @read_amt bytes from @inode, using the fallback FUSE_READ mechanism.
static int virtiofs_read_fallback(virtiofs_inode& inode, u64 file_handle,
    u32 read_amt, u32 flags, virtio::fs& drv, struct uio& uio)
{
    std::unique_ptr<fuse_read_in> in_args {new (std::nothrow) fuse_read_in()};
    std::unique_ptr<void, std::function<void(void*)>> buf {
        memory::alloc_phys_contiguous_aligned(read_amt,
        alignof(std::max_align_t)), memory::free_phys_contiguous_aligned };
    if (!in_args || !buf) {
        return ENOMEM;
    }
    in_args->fh = file_handle;
    in_args->offset = uio.uio_offset;
    in_args->size = read_amt;
    in_args->flags = flags;

    virtiofs_debug("inode %lld, reading %lld bytes at offset %lld\n",
        inode.nodeid, read_amt, uio.uio_offset);
    auto error = fuse_req_send_and_receive_reply(&drv, FUSE_READ,
        inode.nodeid, in_args.get(), sizeof(*in_args), buf.get(),
        read_amt).second;
    if (error) {
        kprintf("[virtiofs] inode %lld, read failed\n", inode.nodeid);
        return error;
    }

    return uiomove(buf.get(), read_amt, &uio);
}

static int virtiofs_read(struct vnode* vnode, struct file* fp, struct uio* uio,
    int ioflag)
{
    // Can't read directories
    if (vnode->v_type == VDIR) {
        return EISDIR;
    }
    // Can't read anything but reg
    if (vnode->v_type != VREG) {
        return EINVAL;
    }
    // Can't start reading before the first byte
    if (uio->uio_offset < 0) {
        return EINVAL;
    }
    // Need to read at least 1 byte
    if (uio->uio_resid == 0) {
        return 0;
    }
    // Can't read after the end of the file
    if (uio->uio_offset >= vnode->v_size) {
        return 0;
    }

    auto* inode = static_cast<virtiofs_inode*>(vnode->v_data);
    auto* file_data = static_cast<virtiofs_file_data*>(fp->f_data);
    auto* m_data = static_cast<virtiofs_mount_data*>(vnode->v_mount->m_data);
    auto* drv = m_data->drv;
    auto dax_mgr = m_data->dax_mgr;

    // Total read amount is what they requested, or what is left
    auto read_amt = std::min<uint64_t>(uio->uio_resid,
        inode->attr.size - uio->uio_offset);

    if (dax_mgr) {
        // Try to read from DAX
        if (!dax_mgr->read(*inode, file_data->file_handle, read_amt, *uio)) {
            return 0;
        }
    }
    // DAX unavailable or failed, use fallback
    return virtiofs_read_fallback(*inode, file_data->file_handle, read_amt,
        ioflag, *drv, *uio);
}

// Checks if @buf (with size @len) points to a valid fuse_dirent (with its name
// not exceeding @name_max) and if so returns @buf. Otherwise, returns nullptr.
static fuse_dirent* parse_fuse_dirent(void* buf, size_t len, size_t name_max)
{
    if (len < FUSE_NAME_OFFSET) {
        return nullptr;
    }

    auto* fdir = static_cast<struct fuse_dirent*>(buf);
    if (FUSE_DIRENT_SIZE(fdir) > len || fdir->namelen > name_max) {
        return nullptr;
    }

    return fdir;
}

static int virtiofs_readdir(struct vnode* vnode, struct file* fp,
    struct dirent* dir)
{
    auto* inode = static_cast<virtiofs_inode*>(vnode->v_data);
    auto* file_data = static_cast<virtiofs_file_data*>(fp->f_data);
    auto* m_data = static_cast<virtiofs_mount_data*>(vnode->v_mount->m_data);
    auto* drv = m_data->drv;

    // NOTE: The response consists of a buffer of size <= fuse_read_in.size.
    // This contains multiple (whole) fuse_dirent structs, each padded to 64-bit
    // boundary. We only parse one such struct at a time, for simplicity. The
    // fuse_dirent.name field is _not_ null-terminated, but our dirent.d_name
    // is, requiring caution.
    std::unique_ptr<fuse_read_in> in_args {new (std::nothrow) fuse_read_in()};
    constexpr size_t name_max = sizeof(dir->d_name) - 1;    // account for '\0'
    // The size of a (padded) fuse_dirent with a name_max-long name
    size_t bufsize = FUSE_DIRENT_ALIGN(FUSE_NAME_OFFSET + name_max);
    std::unique_ptr<void, std::function<void(void*)>> buf {
        memory::alloc_phys_contiguous_aligned(bufsize,
        alignof(std::max_align_t)), memory::free_phys_contiguous_aligned };
    if (!in_args || !buf) {
        return ENOMEM;
    }
    in_args->fh = file_data->file_handle;
    in_args->offset = fp->f_offset;
    in_args->size = bufsize;
    in_args->flags = fp->f_flags;

    size_t len;
    int error;
    std::tie(len, error) = fuse_req_send_and_receive_reply(drv, FUSE_READDIR,
        inode->nodeid, in_args.get(), sizeof(*in_args), buf.get(), bufsize);
    if (error) {
        kprintf("[virtiofs] inode %lld, readdir failed\n", inode->nodeid);
        return error;
    }

    if (len == 0) {
        return ENOENT;
    }
    auto* fdir = parse_fuse_dirent(buf.get(), len, name_max);
    if (!fdir) {
        kprintf("[virtiofs] inode %lld, dirent parsing failed\n",
            inode->nodeid);
        return EIO;
    }

    dir->d_ino = fdir->ino;
    dir->d_off = fdir->off;
    dir->d_type = fdir->type;
    // Copy fdir->name (not null-terminated) to dir->d_name (null-terminated)
    memcpy(dir->d_name, fdir->name, fdir->namelen);
    dir->d_name[fdir->namelen] = '\0';

    fp->f_offset = fdir->off;

    virtiofs_debug("inode %lld, read dir entry %s\n", inode->nodeid,
        dir->d_name);
    return 0;
}

static int virtiofs_getattr(struct vnode* vnode, struct vattr* attr)
{
    auto* inode = static_cast<virtiofs_inode*>(vnode->v_data);

    // TODO: Call FUSE_GETATTR? But figure out if fuse_getattr_in.fh is
    // necessary (look at the flags)
    attr->va_mode = 0555; // TODO: Is it really correct?

    if (S_ISDIR(inode->attr.mode)) {
        attr->va_type = VDIR;
    } else if (S_ISREG(inode->attr.mode)) {
        attr->va_type = VREG;
    } else if (S_ISLNK(inode->attr.mode)) {
        attr->va_type = VLNK;
    }

    attr->va_nodeid = vnode->v_ino;
    attr->va_size = inode->attr.size;

    return 0;
}

#define virtiofs_write       ((vnop_write_t)vop_erofs)
#define virtiofs_seek        ((vnop_seek_t)vop_nullop)
#define virtiofs_ioctl       ((vnop_ioctl_t)vop_nullop)
#define virtiofs_create      ((vnop_create_t)vop_erofs)
#define virtiofs_remove      ((vnop_remove_t)vop_erofs)
#define virtiofs_rename      ((vnop_rename_t)vop_erofs)
#define virtiofs_mkdir       ((vnop_mkdir_t)vop_erofs)
#define virtiofs_rmdir       ((vnop_rmdir_t)vop_erofs)
#define virtiofs_setattr     ((vnop_setattr_t)vop_erofs)
#define virtiofs_inactive    ((vnop_inactive_t)vop_nullop)
#define virtiofs_truncate    ((vnop_truncate_t)vop_erofs)
#define virtiofs_link        ((vnop_link_t)vop_erofs)
#define virtiofs_arc         ((vnop_cache_t) nullptr)
#define virtiofs_fallocate   ((vnop_fallocate_t)vop_erofs)
#define virtiofs_fsync       ((vnop_fsync_t)vop_nullop)
#define virtiofs_symlink     ((vnop_symlink_t)vop_erofs)

struct vnops virtiofs_vnops = {
    virtiofs_open,      /* open */
    virtiofs_close,     /* close */
    virtiofs_read,      /* read */
    virtiofs_write,     /* write - returns error when called */
    virtiofs_seek,      /* seek */
    virtiofs_ioctl,     /* ioctl */
    virtiofs_fsync,     /* fsync */
    virtiofs_readdir,   /* readdir */
    virtiofs_lookup,    /* lookup */
    virtiofs_create,    /* create - returns error when called */
    virtiofs_remove,    /* remove - returns error when called */
    virtiofs_rename,    /* rename - returns error when called */
    virtiofs_mkdir,     /* mkdir - returns error when called */
    virtiofs_rmdir,     /* rmdir - returns error when called */
    virtiofs_getattr,   /* getattr */
    virtiofs_setattr,   /* setattr - returns error when called */
    virtiofs_inactive,  /* inactive */
    virtiofs_truncate,  /* truncate - returns error when called */
    virtiofs_link,      /* link - returns error when called */
    virtiofs_arc,       /* arc */ //TODO: Implement to allow memory re-use when
                        // mapping files
    virtiofs_fallocate, /* fallocate - returns error when called */
    virtiofs_readlink,  /* read link */
    virtiofs_symlink    /* symbolic link - returns error when called */
};
