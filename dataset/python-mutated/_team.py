"""
Implementation of a L{Team} of workers; a thread-pool that can allocate work to
workers.
"""
from __future__ import annotations
from collections import deque
from typing import Callable, Optional, Set
from zope.interface import implementer
from . import IWorker
from ._convenience import Quit
from ._ithreads import IExclusiveWorker

class Statistics:
    """
    Statistics about a L{Team}'s current activity.

    @ivar idleWorkerCount: The number of idle workers.
    @type idleWorkerCount: L{int}

    @ivar busyWorkerCount: The number of busy workers.
    @type busyWorkerCount: L{int}

    @ivar backloggedWorkCount: The number of work items passed to L{Team.do}
        which have not yet been sent to a worker to be performed because not
        enough workers are available.
    @type backloggedWorkCount: L{int}
    """

    def __init__(self, idleWorkerCount: int, busyWorkerCount: int, backloggedWorkCount: int) -> None:
        if False:
            print('Hello World!')
        self.idleWorkerCount = idleWorkerCount
        self.busyWorkerCount = busyWorkerCount
        self.backloggedWorkCount = backloggedWorkCount

@implementer(IWorker)
class Team:
    """
    A composite L{IWorker} implementation.

    @ivar _quit: A L{Quit} flag indicating whether this L{Team} has been quit
        yet.  This may be set by an arbitrary thread since L{Team.quit} may be
        called from anywhere.

    @ivar _coordinator: the L{IExclusiveWorker} coordinating access to this
        L{Team}'s internal resources.

    @ivar _createWorker: a callable that will create new workers.

    @ivar _logException: a 0-argument callable called in an exception context
        when there is an unhandled error from a task passed to L{Team.do}

    @ivar _idle: a L{set} of idle workers.

    @ivar _busyCount: the number of workers currently busy.

    @ivar _pending: a C{deque} of tasks - that is, 0-argument callables passed
        to L{Team.do} - that are outstanding.

    @ivar _shouldQuitCoordinator: A flag indicating that the coordinator should
        be quit at the next available opportunity.  Unlike L{Team._quit}, this
        flag is only set by the coordinator.

    @ivar _toShrink: the number of workers to shrink this L{Team} by at the
        next available opportunity; set in the coordinator.
    """

    def __init__(self, coordinator: IExclusiveWorker, createWorker: Callable[[], Optional[IWorker]], logException: Callable[[], None]):
        if False:
            i = 10
            return i + 15
        '\n        @param coordinator: an L{IExclusiveWorker} which will coordinate access\n            to resources on this L{Team}; that is to say, an\n            L{IExclusiveWorker} whose C{do} method ensures that its given work\n            will be executed in a mutually exclusive context, not in parallel\n            with other work enqueued by C{do} (although possibly in parallel\n            with the caller).\n\n        @param createWorker: A 0-argument callable that will create an\n            L{IWorker} to perform work.\n\n        @param logException: A 0-argument callable called in an exception\n            context when the work passed to C{do} raises an exception.\n        '
        self._quit = Quit()
        self._coordinator = coordinator
        self._createWorker = createWorker
        self._logException = logException
        self._idle: Set[IWorker] = set()
        self._busyCount = 0
        self._pending: 'deque[Callable[..., object]]' = deque()
        self._shouldQuitCoordinator = False
        self._toShrink = 0

    def statistics(self) -> Statistics:
        if False:
            print('Hello World!')
        '\n        Gather information on the current status of this L{Team}.\n\n        @return: a L{Statistics} describing the current state of this L{Team}.\n        '
        return Statistics(len(self._idle), self._busyCount, len(self._pending))

    def grow(self, n: int) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Increase the the number of idle workers by C{n}.\n\n        @param n: The number of new idle workers to create.\n        @type n: L{int}\n        '
        self._quit.check()

        @self._coordinator.do
        def createOneWorker() -> None:
            if False:
                for i in range(10):
                    print('nop')
            for x in range(n):
                worker = self._createWorker()
                if worker is None:
                    return
                self._recycleWorker(worker)

    def shrink(self, n: Optional[int]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Decrease the number of idle workers by C{n}.\n\n        @param n: The number of idle workers to shut down, or L{None} (or\n            unspecified) to shut down all workers.\n        @type n: L{int} or L{None}\n        '
        self._quit.check()
        self._coordinator.do(lambda : self._quitIdlers(n))

    def _quitIdlers(self, n: Optional[int]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        The implmentation of C{shrink}, performed by the coordinator worker.\n\n        @param n: see L{Team.shrink}\n        '
        if n is None:
            n = len(self._idle) + self._busyCount
        for x in range(n):
            if self._idle:
                self._idle.pop().quit()
            else:
                self._toShrink += 1
        if self._shouldQuitCoordinator and self._busyCount == 0:
            self._coordinator.quit()

    def do(self, task: Callable[[], None]) -> None:
        if False:
            print('Hello World!')
        '\n        Perform some work in a worker created by C{createWorker}.\n\n        @param task: the callable to run\n        '
        self._quit.check()
        self._coordinator.do(lambda : self._coordinateThisTask(task))

    def _coordinateThisTask(self, task: Callable[..., object]) -> None:
        if False:
            while True:
                i = 10
        '\n        Select a worker to dispatch to, either an idle one or a new one, and\n        perform it.\n\n        This method should run on the coordinator worker.\n\n        @param task: the task to dispatch\n        @type task: 0-argument callable\n        '
        worker = self._idle.pop() if self._idle else self._createWorker()
        if worker is None:
            self._pending.append(task)
            return
        not_none_worker = worker
        self._busyCount += 1

        @worker.do
        def doWork() -> None:
            if False:
                while True:
                    i = 10
            try:
                task()
            except BaseException:
                self._logException()

            @self._coordinator.do
            def idleAndPending() -> None:
                if False:
                    while True:
                        i = 10
                self._busyCount -= 1
                self._recycleWorker(not_none_worker)

    def _recycleWorker(self, worker: IWorker) -> None:
        if False:
            return 10
        '\n        Called only from coordinator.\n\n        Recycle the given worker into the idle pool.\n\n        @param worker: a worker created by C{createWorker} and now idle.\n        @type worker: L{IWorker}\n        '
        self._idle.add(worker)
        if self._pending:
            self._coordinateThisTask(self._pending.popleft())
        elif self._shouldQuitCoordinator:
            self._quitIdlers()
        elif self._toShrink > 0:
            self._toShrink -= 1
            self._idle.remove(worker)
            worker.quit()

    def quit(self) -> None:
        if False:
            return 10
        '\n        Stop doing work and shut down all idle workers.\n        '
        self._quit.set()

        @self._coordinator.do
        def startFinishing() -> None:
            if False:
                while True:
                    i = 10
            self._shouldQuitCoordinator = True
            self._quitIdlers()