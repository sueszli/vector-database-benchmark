import threading
import scipy._lib.decorator
__all__ = ['ReentrancyError', 'ReentrancyLock', 'non_reentrant']

class ReentrancyError(RuntimeError):
    pass

class ReentrancyLock:
    """
    Threading lock that raises an exception for reentrant calls.

    Calls from different threads are serialized, and nested calls from the
    same thread result to an error.

    The object can be used as a context manager or to decorate functions
    via the decorate() method.

    """

    def __init__(self, err_msg):
        if False:
            return 10
        self._rlock = threading.RLock()
        self._entered = False
        self._err_msg = err_msg

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self._rlock.acquire()
        if self._entered:
            self._rlock.release()
            raise ReentrancyError(self._err_msg)
        self._entered = True

    def __exit__(self, type, value, traceback):
        if False:
            return 10
        self._entered = False
        self._rlock.release()

    def decorate(self, func):
        if False:
            for i in range(10):
                print('nop')

        def caller(func, *a, **kw):
            if False:
                print('Hello World!')
            with self:
                return func(*a, **kw)
        return scipy._lib.decorator.decorate(func, caller)

def non_reentrant(err_msg=None):
    if False:
        return 10
    '\n    Decorate a function with a threading lock and prevent reentrant calls.\n    '

    def decorator(func):
        if False:
            i = 10
            return i + 15
        msg = err_msg
        if msg is None:
            msg = '%s is not re-entrant' % func.__name__
        lock = ReentrancyLock(msg)
        return lock.decorate(func)
    return decorator