"""
Implementation of an L{IWorker} based on native threads and queues.
"""
from typing import Callable
from zope.interface import implementer
from ._convenience import Quit
from ._ithreads import IExclusiveWorker
_stop = object()

@implementer(IExclusiveWorker)
class ThreadWorker:
    """
    An L{IExclusiveWorker} implemented based on a single thread and a queue.

    This worker ensures exclusivity (i.e. it is an L{IExclusiveWorker} and not
    an L{IWorker}) by performing all of the work passed to C{do} on the I{same}
    thread.
    """

    def __init__(self, startThread, queue):
        if False:
            return 10
        '\n        Create a L{ThreadWorker} with a function to start a thread and a queue\n        to use to communicate with that thread.\n\n        @param startThread: a callable that takes a callable to run in another\n            thread.\n        @type startThread: callable taking a 0-argument callable and returning\n            nothing.\n\n        @param queue: A L{Queue} to use to give tasks to the thread created by\n            C{startThread}.\n        @type queue: L{Queue}\n        '
        self._q = queue
        self._hasQuit = Quit()

        def work():
            if False:
                i = 10
                return i + 15
            for task in iter(queue.get, _stop):
                task()
        startThread(work)

    def do(self, task: Callable[[], None]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform the given task on the thread owned by this L{ThreadWorker}.\n\n        @param task: the function to call on a thread.\n        '
        self._hasQuit.check()
        self._q.put(task)

    def quit(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reject all future work and stop the thread started by C{__init__}.\n        '
        self._hasQuit.set()
        self._q.put(_stop)

@implementer(IExclusiveWorker)
class LockWorker:
    """
    An L{IWorker} implemented based on a mutual-exclusion lock.
    """

    def __init__(self, lock, local):
        if False:
            print('Hello World!')
        '\n        @param lock: A mutual-exclusion lock, with C{acquire} and C{release}\n            methods.\n        @type lock: L{threading.Lock}\n\n        @param local: Local storage.\n        @type local: L{threading.local}\n        '
        self._quit = Quit()
        self._lock = lock
        self._local = local

    def do(self, work: Callable[[], None]) -> None:
        if False:
            return 10
        '\n        Do the given work on this thread, with the mutex acquired.  If this is\n        called re-entrantly, return and wait for the outer invocation to do the\n        work.\n\n        @param work: the work to do with the lock held.\n        '
        lock = self._lock
        local = self._local
        self._quit.check()
        working = getattr(local, 'working', None)
        if working is None:
            working = local.working = []
            working.append(work)
            lock.acquire()
            try:
                while working:
                    working.pop(0)()
            finally:
                lock.release()
                local.working = None
        else:
            working.append(work)

    def quit(self):
        if False:
            i = 10
            return i + 15
        '\n        Quit this L{LockWorker}.\n        '
        self._quit.set()
        self._lock = None