/*
 * Copyright (C) 2020-2022 The opuntiaOS Project Authors.
 *  + Contributed by Nikita Melekhin <nimelehin@gmail.com>
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <libkern/bits/errno.h>
#include <libkern/libkern.h>
#include <libkern/log.h>
#include <platform/generic/syscalls/params.h>
#include <syscalls/handlers.h>
#include <tasking/sched.h>
#include <tasking/tasking.h>

void sys_getpid(trapframe_t* tf)
{
    return_with_val(RUNNING_THREAD->tid);
}

void sys_getuid(trapframe_t* tf)
{
    return_with_val(RUNNING_THREAD->process->uid);
}

void sys_setuid(trapframe_t* tf)
{
    uid_t new_uid = SYSCALL_VAR1(tf);
    proc_t* proc = RUNNING_THREAD->process;

    spinlock_acquire(&proc->lock);

    if (proc->uid != new_uid && proc->euid != new_uid && !proc_is_su(proc)) {
        spinlock_release(&proc->lock);
        return_with_val(-EPERM);
    }

    proc->uid = new_uid;
    proc->euid = new_uid;
    proc->suid = new_uid;
    spinlock_release(&proc->lock);
    return_with_val(0);
}

void sys_setgid(trapframe_t* tf)
{
    gid_t new_gid = SYSCALL_VAR1(tf);
    proc_t* proc = RUNNING_THREAD->process;

    spinlock_acquire(&proc->lock);

    if (proc->gid != new_gid && proc->egid != new_gid && !proc_is_su(proc)) {
        spinlock_release(&proc->lock);
        return_with_val(-EPERM);
    }

    proc->gid = new_gid;
    proc->egid = new_gid;
    proc->sgid = new_gid;
    spinlock_release(&proc->lock);
    return_with_val(0);
}

void sys_setreuid(trapframe_t* tf)
{
    proc_t* proc = RUNNING_THREAD->process;
    uid_t new_ruid = SYSCALL_VAR1(tf);
    uid_t new_euid = SYSCALL_VAR2(tf);

    spinlock_acquire(&proc->lock);

    if (new_ruid == (uid_t)-1) {
        new_ruid = proc->uid;
    }

    if (new_euid == (uid_t)-1) {
        new_euid = proc->euid;
    }

    if (proc->uid != new_euid && proc->euid != new_euid && proc->suid != new_euid) {
        spinlock_release(&proc->lock);
        return_with_val(-EPERM);
    }

    if (proc->uid != new_ruid && proc->euid != new_ruid && proc->suid != new_ruid) {
        spinlock_release(&proc->lock);
        return_with_val(-EPERM);
    }

    proc->uid = new_ruid;
    proc->euid = new_euid;
    spinlock_release(&proc->lock);
    return_with_val(0);
}

void sys_setregid(trapframe_t* tf)
{
    proc_t* proc = RUNNING_THREAD->process;
    gid_t new_rgid = SYSCALL_VAR1(tf);
    gid_t new_egid = SYSCALL_VAR2(tf);

    spinlock_acquire(&proc->lock);

    if (new_rgid == (gid_t)-1) {
        new_rgid = proc->gid;
    }

    if (new_egid == (gid_t)-1) {
        new_egid = proc->egid;
    }

    if (proc->gid != new_egid && proc->egid != new_egid && proc->sgid != new_egid) {
        spinlock_release(&proc->lock);
        return_with_val(-EPERM);
    }

    if (proc->gid != new_rgid && proc->egid != new_rgid && proc->sgid != new_rgid) {
        spinlock_release(&proc->lock);
        return_with_val(-EPERM);
    }

    proc->gid = new_rgid;
    proc->egid = new_egid;

    spinlock_release(&proc->lock);
    return_with_val(0);
}