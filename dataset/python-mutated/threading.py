"""
Threading utilities.
"""
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import itertools
import os
from queue import Queue

def concurrent_chain(generators, jobs=None):
    if False:
        return 10
    "\n    Similar to itertools.chain(), but runs the individual generators in a\n    thread pool. The resulting items may be out of order accordingly.\n\n    When one generator raises an exception, all other currently-running\n    generators are stopped (they may run until their next 'yield' statement).\n    The exception is then raised.\n    "
    if jobs is None:
        jobs = os.cpu_count()
    if jobs == 1:
        for generator in generators:
            yield from generator
        return
    queue = ClosableQueue()
    running_generator_count = 0
    with ThreadPoolExecutor(jobs) as pool:
        for generator in generators:
            pool.submit(generator_to_queue, generator, queue)
            running_generator_count += 1
        while running_generator_count > 0:
            (event_type, value) = queue.get()
            if event_type == GeneratorEvent.VALUE:
                yield value
            elif event_type == GeneratorEvent.EXCEPTION:
                queue.close('Exception in different generator')
                raise value
            elif event_type == GeneratorEvent.STOP_ITERATION:
                running_generator_count -= 1

class ClosableQueue(Queue):
    """
    For use in concurrent_chain.

    Behaves like Queue until close() has been called.
    After that, any call to put() raises RuntimeError.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.closed = False
        self.close_reason = None

    def put(self, item, block=True, timeout=None):
        if False:
            i = 10
            return i + 15
        self.raise_if_closed()
        super().put(item, block, timeout)

    def close(self, reason=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Any subsequent calls to put() or raise_if_closed()\n        will raise RuntimeError(reason).\n        '
        with self.mutex:
            self.closed = True
            self.close_reason = reason

    def raise_if_closed(self):
        if False:
            while True:
                i = 10
        '\n        Raises RuntimeError(reason) if the queue has been closed.\n        Returns None elsewise.\n        '
        with self.mutex:
            if self.closed:
                raise RuntimeError(self.close_reason)

class GeneratorEvent(Enum):
    """
    For use by concurrent_chain.
    Represents any event that a generator may cause.
    """
    VALUE = 0
    EXCEPTION = 1
    STOP_ITERATION = 2

def generator_to_queue(generator, queue: ClosableQueue) -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    For use by concurrent_chain.\n    Appends all of the generator's events to the queue,\n    as tuples of (event type, value).\n    "
    try:
        queue.raise_if_closed()
        for item in generator:
            queue.put((GeneratorEvent.VALUE, item))
        queue.put((GeneratorEvent.STOP_ITERATION, None))
    except BaseException as exc:
        queue.put((GeneratorEvent.EXCEPTION, exc))

def test_concurrent_chain() -> None:
    if False:
        while True:
            i = 10
    ' Tests concurrent_chain '
    from ..testing.testing import assert_value, assert_raises, result

    def errorgen():
        if False:
            print('Hello World!')
        ' Test generator that raises an exception '
        yield 'errorgen'
        raise ValueError()
    assert_value(list(concurrent_chain([], 2)), [])
    assert_value(list(concurrent_chain([range(10)], 2)), list(range(10)))
    assert_value(sorted(list(concurrent_chain([range(10), range(20)], 2))), sorted(list(itertools.chain(range(10), range(20)))))
    chain = concurrent_chain([range(10), range(20), errorgen(), range(30)], 2)
    with assert_raises(ValueError):
        result(list(chain))