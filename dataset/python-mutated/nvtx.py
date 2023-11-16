"""This package adds support for NVIDIA Tools Extension (NVTX) used in profiling."""
from contextlib import contextmanager
try:
    from torch._C import _nvtx
except ImportError:

    class _NVTXStub:

        @staticmethod
        def _fail(*args, **kwargs):
            if False:
                return 10
            raise RuntimeError('NVTX functions not installed. Are you sure you have a CUDA build?')
        rangePushA = _fail
        rangePop = _fail
        markA = _fail
    _nvtx = _NVTXStub()
__all__ = ['range_push', 'range_pop', 'range_start', 'range_end', 'mark', 'range']

def range_push(msg):
    if False:
        for i in range(10):
            print('nop')
    '\n    Push a range onto a stack of nested range span.  Returns zero-based depth of the range that is started.\n\n    Args:\n        msg (str): ASCII message to associate with range\n    '
    return _nvtx.rangePushA(msg)

def range_pop():
    if False:
        return 10
    'Pop a range off of a stack of nested range spans.  Returns the  zero-based depth of the range that is ended.'
    return _nvtx.rangePop()

def range_start(msg) -> int:
    if False:
        print('Hello World!')
    '\n    Mark the start of a range with string message. It returns an unique handle\n    for this range to pass to the corresponding call to rangeEnd().\n\n    A key difference between this and range_push/range_pop is that the\n    range_start/range_end version supports range across threads (start on one\n    thread and end on another thread).\n\n    Returns: A range handle (uint64_t) that can be passed to range_end().\n\n    Args:\n        msg (str): ASCII message to associate with the range.\n    '
    return _nvtx.rangeStartA(msg)

def range_end(range_id) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Mark the end of a range for a given range_id.\n\n    Args:\n        range_id (int): an unique handle for the start range.\n    '
    _nvtx.rangeEnd(range_id)

def mark(msg):
    if False:
        for i in range(10):
            print('nop')
    '\n    Describe an instantaneous event that occurred at some point.\n\n    Args:\n        msg (str): ASCII message to associate with the event.\n    '
    return _nvtx.markA(msg)

@contextmanager
def range(msg, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Context manager / decorator that pushes an NVTX range at the beginning\n    of its scope, and pops it at the end. If extra arguments are given,\n    they are passed as arguments to msg.format().\n\n    Args:\n        msg (str): message to associate with the range\n    '
    range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        range_pop()