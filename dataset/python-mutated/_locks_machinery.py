import os
import threading
import weakref
if not hasattr(os, 'register_at_fork'):

    def create_logger_lock():
        if False:
            print('Hello World!')
        return threading.Lock()

    def create_handler_lock():
        if False:
            i = 10
            return i + 15
        return threading.Lock()
else:
    logger_locks = weakref.WeakSet()
    handler_locks = weakref.WeakSet()

    def acquire_locks():
        if False:
            while True:
                i = 10
        for lock in logger_locks:
            lock.acquire()
        for lock in handler_locks:
            lock.acquire()

    def release_locks():
        if False:
            return 10
        for lock in logger_locks:
            lock.release()
        for lock in handler_locks:
            lock.release()
    os.register_at_fork(before=acquire_locks, after_in_parent=release_locks, after_in_child=release_locks)

    def create_logger_lock():
        if False:
            i = 10
            return i + 15
        lock = threading.Lock()
        logger_locks.add(lock)
        return lock

    def create_handler_lock():
        if False:
            print('Hello World!')
        lock = threading.Lock()
        handler_locks.add(lock)
        return lock