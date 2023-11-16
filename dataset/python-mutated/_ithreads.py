"""
Interfaces related to threads.
"""
from typing import Callable
from zope.interface import Interface

class AlreadyQuit(Exception):
    """
    This worker worker is dead and cannot execute more instructions.
    """

class IWorker(Interface):
    """
    A worker that can perform some work concurrently.

    All methods on this interface must be thread-safe.
    """

    def do(task: Callable[[], None]) -> None:
        if False:
            while True:
                i = 10
        "\n        Perform the given task.\n\n        As an interface, this method makes no specific claims about concurrent\n        execution.  An L{IWorker}'s C{do} implementation may defer execution\n        for later on the same thread, immediately on a different thread, or\n        some combination of the two.  It is valid for a C{do} method to\n        schedule C{task} in such a way that it may never be executed.\n\n        It is important for some implementations to provide specific properties\n        with respect to where C{task} is executed, of course, and client code\n        may rely on a more specific implementation of C{do} than L{IWorker}.\n\n        @param task: a task to call in a thread or other concurrent context.\n        @type task: 0-argument callable\n\n        @raise AlreadyQuit: if C{quit} has been called.\n        "

    def quit():
        if False:
            print('Hello World!')
        '\n        Free any resources associated with this L{IWorker} and cause it to\n        reject all future work.\n\n        @raise AlreadyQuit: if this method has already been called.\n        '

class IExclusiveWorker(IWorker):
    """
    Like L{IWorker}, but with the additional guarantee that the callables
    passed to C{do} will not be called exclusively with each other.
    """