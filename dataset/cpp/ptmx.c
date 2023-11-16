/*
 * Copyright (C) 2020-2022 The opuntiaOS Project Authors.
 *  + Contributed by Nikita Melekhin <nimelehin@gmail.com>
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <fs/devfs/devfs.h>
#include <io/tty/ptmx.h>
#include <io/tty/pty_master.h>
#include <libkern/libkern.h>
#include <libkern/log.h>

// #define PTY_DEBUG

bool ptmx_can_read(file_t* file, size_t start)
{
    return true;
}

bool ptmx_can_write(file_t* file, size_t start)
{
    return true;
}

int ptmx_read(file_t* file, void __user* buf, size_t start, size_t len)
{
    return 0;
}

int ptmx_write(file_t* file, void __user* buf, size_t start, size_t len)
{
    return 0;
}

int ptmx_open(const path_t* path, struct file_descriptor* fd, uint32_t flags)
{
#ifdef PTY_DEBUG
    log("Opening ptmx");
#endif
    return pty_master_alloc(fd);
}

int ptmx_install()
{
    path_t vfspth;
    if (vfs_resolve_path("/dev", &vfspth) < 0) {
        return -1;
    }

    file_ops_t fops = { 0 };
    fops.open = ptmx_open;
    fops.can_read = ptmx_can_read;
    fops.can_write = ptmx_can_write;
    fops.read = ptmx_read;
    fops.write = ptmx_write;
    devfs_inode_t* res = devfs_register(&vfspth, MKDEV(5, 2), "ptmx", 4, S_IFCHR | 0777, &fops);
    path_put(&vfspth);
    return 0;
}
