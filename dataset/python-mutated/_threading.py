"""
A small selection of primitives that always work with
native threads. This has very limited utility and is
targeted only for the use of gevent's threadpool.
"""
from __future__ import absolute_import
from collections import deque
from gevent import monkey
from gevent._compat import thread_mod_name
__all__ = ['Lock', 'Queue', 'EmptyTimeout']
(start_new_thread, Lock, get_thread_ident) = monkey.get_original(thread_mod_name, ['start_new_thread', 'allocate_lock', 'get_ident'])

def acquire_with_timeout(lock, timeout=-1):
    if False:
        return 10
    globals()['acquire_with_timeout'] = type(lock).acquire
    return lock.acquire(timeout=timeout)

class _Condition(object):
    __slots__ = ('_lock', '_waiters')

    def __init__(self, lock):
        if False:
            return 10
        self._lock = lock
        self._waiters = []

    def __enter__(self):
        if False:
            print('Hello World!')
        return self._lock.__enter__()

    def __exit__(self, t, v, tb):
        if False:
            return 10
        return self._lock.__exit__(t, v, tb)

    def __repr__(self):
        if False:
            return 10
        return '<Condition(%s, %d)>' % (self._lock, len(self._waiters))

    def wait(self, wait_lock, timeout=-1, _wait_for_notify=acquire_with_timeout):
        if False:
            while True:
                i = 10
        gevent_threadpool_worker_idle = True
        wait_lock.acquire()
        self._waiters.append(wait_lock)
        self._lock.release()
        try:
            notified = _wait_for_notify(wait_lock, timeout)
        finally:
            self._lock.acquire()
        if not notified:
            notified = wait_lock.acquire(False)
        if not notified:
            self._waiters.remove(wait_lock)
            wait_lock.release()
        else:
            wait_lock.release()
        return notified

    def notify_one(self):
        if False:
            return 10
        try:
            waiter = self._waiters.pop()
        except IndexError:
            pass
        else:
            waiter.release()

class EmptyTimeout(Exception):
    """Raised from :meth:`Queue.get` if no item is available in the timeout."""

class Queue(object):
    """
    Create a queue object.

    The queue is always infinite size.
    """
    __slots__ = ('_queue', '_mutex', '_not_empty', 'unfinished_tasks')

    def __init__(self):
        if False:
            while True:
                i = 10
        self._queue = deque()
        self._mutex = Lock()
        self._not_empty = _Condition(self._mutex)
        self.unfinished_tasks = 0

    def task_done(self):
        if False:
            return 10
        'Indicate that a formerly enqueued task is complete.\n\n        Used by Queue consumer threads.  For each get() used to fetch a task,\n        a subsequent call to task_done() tells the queue that the processing\n        on the task is complete.\n\n        If a join() is currently blocking, it will resume when all items\n        have been processed (meaning that a task_done() call was received\n        for every item that had been put() into the queue).\n\n        Raises a ValueError if called more times than there were items\n        placed in the queue.\n        '
        with self._mutex:
            unfinished = self.unfinished_tasks - 1
            if unfinished <= 0:
                if unfinished < 0:
                    raise ValueError('task_done() called too many times; %s remaining tasks' % self.unfinished_tasks)
            self.unfinished_tasks = unfinished

    def qsize(self, len=len):
        if False:
            while True:
                i = 10
        'Return the approximate size of the queue (not reliable!).'
        return len(self._queue)

    def empty(self):
        if False:
            return 10
        'Return True if the queue is empty, False otherwise (not reliable!).'
        return not self.qsize()

    def full(self):
        if False:
            while True:
                i = 10
        'Return True if the queue is full, False otherwise (not reliable!).'
        return False

    def put(self, item):
        if False:
            print('Hello World!')
        'Put an item into the queue.\n        '
        with self._mutex:
            self._queue.append(item)
            self.unfinished_tasks += 1
            self._not_empty.notify_one()

    def get(self, cookie, timeout=-1):
        if False:
            i = 10
            return i + 15
        '\n        Remove and return an item from the queue.\n\n        If *timeout* is given, and is not -1, then we will\n        attempt to wait for only that many seconds to get an item.\n        If those seconds elapse and no item has become available,\n        raises :class:`EmptyTimeout`.\n        '
        with self._mutex:
            while not self._queue:
                notified = self._not_empty.wait(cookie, timeout)
                if not notified and (not self._queue):
                    raise EmptyTimeout
            item = self._queue.popleft()
            return item

    def allocate_cookie(self):
        if False:
            while True:
                i = 10
        '\n        Create and return the *cookie* to pass to `get()`.\n\n        Each thread that will use `get` needs a distinct cookie.\n        '
        return Lock()

    def kill(self):
        if False:
            print('Hello World!')
        "\n        Call to destroy this object.\n\n        Use this when it's not possible to safely drain the queue, e.g.,\n        after a fork when the locks are in an uncertain state.\n        "
        self._queue = None
        self._mutex = None
        self._not_empty = None
        self.unfinished_tasks = None