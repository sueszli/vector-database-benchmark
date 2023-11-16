"""
Replacement for ``multiprocessing`` library in coala's debug mode.
"""
import logging
import sys
import queue
from functools import partial
from coalib.processes.communication.LogMessage import LogMessage
__all__ = ['Manager', 'Process', 'Queue']

class Manager:
    """
    A debug replacement for ``multiprocessing.Manager``, just offering
    ``builtins.dict`` as ``.dict`` member.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Just add ``dict`` as instance member.\n        '
        self.dict = dict

class Process(partial):
    """
    A debug replacement for ``multiprocessing.Process``, running the callable
    target without any process parallelization or threading.
    """

    def __new__(cls, target, kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Just pass the arguments to underlying ``functools.partial``.\n        '
        return partial.__new__(cls, target, **kwargs)

    def start(self):
        if False:
            while True:
                i = 10
        '\n        Just call the underlying ``functools.partial`` instaed of any thread\n        or parallel process creation.\n        '
        return self()

class Queue(queue.Queue):
    """
    A debug replacement for ``multiprocessing.Queue``, directly processing
    any incoming :class:`coalib.processes.communication.LogMessage.LogMessage`
    instances (if the queue was instantiated from a function with a local
    ``log_printer``).
    """

    def __init__(self):
        if False:
            return 10
        '\n        Gets local ``log_printer`` from function that created this instance.\n        '
        super().__init__()
        self.log_printer = sys._getframe(1).f_locals.get('log_printer')

    def put(self, item):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add `item` to queue.\n\n        Except `item` is an instance of\n        :class:`coalib.processes.communication.LogMessage.LogMessage` and\n        there is a ``self.log_printer``. Then `item` is just sent to logger\n        instead.\n        '
        if isinstance(item, LogMessage):
            logging.log(item.log_level, item.message)
        else:
            super().put(item)