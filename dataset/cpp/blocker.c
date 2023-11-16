/*
 * Copyright (C) 2020-2022 The opuntiaOS Project Authors.
 *  + Contributed by Nikita Melekhin <nimelehin@gmail.com>
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <libkern/atomic.h>
#include <libkern/libkern.h>
#include <libkern/log.h>
#include <libkern/syscall_structs.h>
#include <libkern/time.h>
#include <tasking/sched.h>
#include <tasking/thread.h>
#include <time/time_manager.h>

bool should_unblock_join_block(thread_t* thread)
{
    if (thread_is_freed(thread->blocker_data.join.joinee) || thread->blocker_data.join.join_pid != thread->blocker_data.join.joinee->tid) {
        return true;
    }

    const int status = thread->blocker_data.join.joinee->status;
    if (status == THREAD_STATUS_DYING) {
        return true;
    }
    return false;
}

extern thread_t* tasking_get_thread(pid_t tid);
int init_join_blocker(thread_t* thread, int wait_for_pid)
{
    thread_t* joinee_thread = tasking_get_thread(wait_for_pid);

    thread->blocker_data.join.joinee = joinee_thread;
    thread->blocker_data.join.join_pid = wait_for_pid;

    if (should_unblock_join_block(thread)) {
        return 0;
    }

    thread->status = THREAD_STATUS_BLOCKED;
    thread->blocker.reason = BLOCKER_JOIN;
    thread->blocker.should_unblock = should_unblock_join_block;
    thread->blocker.should_unblock_for_signal = true;
    sched_dequeue(thread);
    resched();

    return 0;
}

bool should_unblock_read_block(thread_t* thread)
{
    if (!thread->blocker_data.rw.fd->file->ops->can_read) {
        return true;
    }
    return thread->blocker_data.rw.fd->file->ops->can_read(thread->blocker_data.rw.fd->file, thread->blocker_data.rw.fd->offset);
}

int init_read_blocker(thread_t* thread, file_descriptor_t* bfd)
{
    thread->blocker_data.rw.fd = bfd;

    if (should_unblock_read_block(thread)) {
        return 0;
    }

    thread->status = THREAD_STATUS_BLOCKED;
    thread->blocker.reason = BLOCKER_READ;
    thread->blocker.should_unblock = should_unblock_read_block;
    thread->blocker.should_unblock_for_signal = true;
    sched_dequeue(thread);
    resched();
    return 0;
}

bool should_unblock_write_block(thread_t* thread)
{
    if (!thread->blocker_data.rw.fd->file->ops->can_write) {
        return true;
    }
    return thread->blocker_data.rw.fd->file->ops->can_write(thread->blocker_data.rw.fd->file, thread->blocker_data.rw.fd->offset);
}

int init_write_blocker(thread_t* thread, file_descriptor_t* bfd)
{
    thread->blocker_data.rw.fd = bfd;

    if (should_unblock_write_block(thread)) {
        return 0;
    }

    thread->status = THREAD_STATUS_BLOCKED;
    thread->blocker.reason = BLOCKER_WRITE;
    thread->blocker.should_unblock = should_unblock_write_block;
    thread->blocker.should_unblock_for_signal = true;
    sched_dequeue(thread);
    resched();
    return 0;
}

bool should_unblock_sleep_block(thread_t* thread)
{
    timespec_t ts = timeman_timespec_since_epoch();
    return timespec_cmp(&thread->blocker_data.sleep.until, &ts) <= 0;
}

int init_sleep_blocker(thread_t* thread, timespec_t ts)
{
    thread->blocker_data.sleep.until = ts;

    if (should_unblock_sleep_block(thread)) {
        return 0;
    }

    thread->status = THREAD_STATUS_BLOCKED;
    thread->blocker.reason = BLOCKER_SLEEP;
    thread->blocker.should_unblock = should_unblock_sleep_block;
    thread->blocker.should_unblock_for_signal = true;
    sched_dequeue(thread);
    resched();
    return 0;
}

bool should_unblock_select_block(thread_t* thread)
{
    timespec_t ts = timeman_timespec_since_epoch();
    if (thread->blocker_data.select.is_until_time_set && timespec_cmp(&thread->blocker_data.sleep.until, &ts) <= 0) {
        return true;
    }

    file_descriptor_t* fd;
    for (int i = 0; i < thread->blocker_data.select.nfds; i++) {
        if (FD_ISSET(i, &thread->blocker_data.select.readfds)) {
            fd = proc_get_fd(thread->process, i);
            if (fd && fd->file->ops->can_read(fd->file, fd->offset)) {
                return true;
            }
        }
    }

    for (int i = 0; i < thread->blocker_data.select.nfds; i++) {
        if (FD_ISSET(i, &thread->blocker_data.select.writefds)) {
            fd = proc_get_fd(thread->process, i);
            if (fd && fd->file->ops->can_write(fd->file, fd->offset)) {
                return true;
            }
        }
    }
    return false;
}

int init_select_blocker(thread_t* thread, int nfds, fd_set_t* readfds, fd_set_t* writefds, fd_set_t* exceptfds, timeval_t* timeout)
{
    FD_ZERO(&(thread->blocker_data.select.readfds));
    FD_ZERO(&(thread->blocker_data.select.writefds));
    FD_ZERO(&(thread->blocker_data.select.exceptfds));
    thread->blocker_data.select.is_until_time_set = false;

    if (readfds) {
        thread->blocker_data.select.readfds = *readfds;
    }
    if (writefds) {
        thread->blocker_data.select.writefds = *writefds;
    }
    if (exceptfds) {
        thread->blocker_data.select.exceptfds = *exceptfds;
    }
    if (timeout) {
        thread->blocker_data.select.until = timeman_timespec_since_epoch();
        timespec_add_timeval(&thread->blocker_data.select.until, timeout);
        thread->blocker_data.select.is_until_time_set = true;
    }
    thread->blocker_data.select.nfds = nfds;

    if (should_unblock_select_block(thread)) {
        return 0;
    }

    thread->status = THREAD_STATUS_BLOCKED;
    thread->blocker.reason = BLOCKER_SELECT;
    thread->blocker.should_unblock = should_unblock_select_block;
    thread->blocker.should_unblock_for_signal = true;
    sched_dequeue(thread);
    resched();
    return 0;
}
