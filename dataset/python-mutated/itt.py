from contextlib import contextmanager
try:
    from torch._C import _itt
except ImportError:

    class _ITTStub:

        @staticmethod
        def _fail(*args, **kwargs):
            if False:
                while True:
                    i = 10
            raise RuntimeError('ITT functions not installed. Are you sure you have a ITT build?')

        @staticmethod
        def is_available():
            if False:
                print('Hello World!')
            return False
        rangePush = _fail
        rangePop = _fail
        mark = _fail
    _itt = _ITTStub()
__all__ = ['is_available', 'range_push', 'range_pop', 'mark', 'range']

def is_available():
    if False:
        while True:
            i = 10
    '\n    Check if ITT feature is available or not\n    '
    return _itt.is_available()

def range_push(msg):
    if False:
        i = 10
        return i + 15
    '\n    Pushes a range onto a stack of nested range span.  Returns zero-based\n    depth of the range that is started.\n\n    Arguments:\n        msg (str): ASCII message to associate with range\n    '
    return _itt.rangePush(msg)

def range_pop():
    if False:
        print('Hello World!')
    '\n    Pops a range off of a stack of nested range spans. Returns the\n    zero-based depth of the range that is ended.\n    '
    return _itt.rangePop()

def mark(msg):
    if False:
        i = 10
        return i + 15
    '\n    Describe an instantaneous event that occurred at some point.\n\n    Arguments:\n        msg (str): ASCII message to associate with the event.\n    '
    return _itt.mark(msg)

@contextmanager
def range(msg, *args, **kwargs):
    if False:
        return 10
    '\n    Context manager / decorator that pushes an ITT range at the beginning\n    of its scope, and pops it at the end. If extra arguments are given,\n    they are passed as arguments to msg.format().\n\n    Args:\n        msg (str): message to associate with the range\n    '
    range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        range_pop()