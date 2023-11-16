"""
Top level thread pool interface, used to implement
L{twisted.python.threadpool}.
"""
from queue import Queue
from threading import Lock, Thread, local as LocalStorage
from typing import Callable, Optional
from typing_extensions import Protocol
from twisted.python.log import err
from ._ithreads import IWorker
from ._team import Team
from ._threadworker import LockWorker, ThreadWorker

class _ThreadFactory(Protocol):

    def __call__(self, *, target: Callable[..., object]) -> Thread:
        if False:
            i = 10
            return i + 15
        ...

def pool(currentLimit: Callable[[], int], threadFactory: _ThreadFactory=Thread) -> Team:
    if False:
        return 10
    "\n    Construct a L{Team} that spawns threads as a thread pool, with the given\n    limiting function.\n\n    @note: Future maintainers: while the public API for the eventual move to\n        twisted.threads should look I{something} like this, and while this\n        function is necessary to implement the API described by\n        L{twisted.python.threadpool}, I am starting to think the idea of a hard\n        upper limit on threadpool size is just bad (turning memory performance\n        issues into correctness issues well before we run into memory\n        pressure), and instead we should build something with reactor\n        integration for slowly releasing idle threads when they're not needed\n        and I{rate} limiting the creation of new threads rather than just\n        hard-capping it.\n\n    @param currentLimit: a callable that returns the current limit on the\n        number of workers that the returned L{Team} should create; if it\n        already has more workers than that value, no new workers will be\n        created.\n    @type currentLimit: 0-argument callable returning L{int}\n\n    @param threadFactory: Factory that, when given a C{target} keyword argument,\n        returns a L{threading.Thread} that will run that target.\n    @type threadFactory: callable returning a L{threading.Thread}\n\n    @return: a new L{Team}.\n    "

    def startThread(target: Callable[..., object]) -> None:
        if False:
            while True:
                i = 10
        return threadFactory(target=target).start()

    def limitedWorkerCreator() -> Optional[IWorker]:
        if False:
            print('Hello World!')
        stats = team.statistics()
        if stats.busyWorkerCount + stats.idleWorkerCount >= currentLimit():
            return None
        return ThreadWorker(startThread, Queue())
    team = Team(coordinator=LockWorker(Lock(), LocalStorage()), createWorker=limitedWorkerCreator, logException=err)
    return team