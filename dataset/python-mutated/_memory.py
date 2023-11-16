"""
Implementation of an in-memory worker that defers execution.
"""
from zope.interface import implementer
from . import IWorker
from ._convenience import Quit
NoMoreWork = object()

@implementer(IWorker)
class MemoryWorker:
    """
    An L{IWorker} that queues work for later performance.

    @ivar _quit: a flag indicating
    @type _quit: L{Quit}
    """

    def __init__(self, pending=list):
        if False:
            print('Hello World!')
        '\n        Create a L{MemoryWorker}.\n        '
        self._quit = Quit()
        self._pending = pending()

    def do(self, work):
        if False:
            i = 10
            return i + 15
        '\n        Queue some work for to perform later; see L{createMemoryWorker}.\n\n        @param work: The work to perform.\n        '
        self._quit.check()
        self._pending.append(work)

    def quit(self):
        if False:
            i = 10
            return i + 15
        '\n        Quit this worker.\n        '
        self._quit.set()
        self._pending.append(NoMoreWork)

def createMemoryWorker():
    if False:
        print('Hello World!')
    '\n    Create an L{IWorker} that does nothing but defer work, to be performed\n    later.\n\n    @return: a worker that will enqueue work to perform later, and a callable\n        that will perform one element of that work.\n    @rtype: 2-L{tuple} of (L{IWorker}, L{callable})\n    '

    def perform():
        if False:
            return 10
        if not worker._pending:
            return False
        if worker._pending[0] is NoMoreWork:
            return False
        worker._pending.pop(0)()
        return True
    worker = MemoryWorker()
    return (worker, perform)