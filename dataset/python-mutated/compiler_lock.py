import threading
import functools
import numba.core.event as ev

class _CompilerLock(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._lock = threading.RLock()

    def acquire(self):
        if False:
            print('Hello World!')
        ev.start_event('numba:compiler_lock')
        self._lock.acquire()

    def release(self):
        if False:
            for i in range(10):
                print('nop')
        self._lock.release()
        ev.end_event('numba:compiler_lock')

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.acquire()

    def __exit__(self, exc_val, exc_type, traceback):
        if False:
            return 10
        self.release()

    def is_locked(self):
        if False:
            print('Hello World!')
        is_owned = getattr(self._lock, '_is_owned')
        if not callable(is_owned):
            is_owned = self._is_owned
        return is_owned()

    def __call__(self, func):
        if False:
            return 10

        @functools.wraps(func)
        def _acquire_compile_lock(*args, **kwargs):
            if False:
                print('Hello World!')
            with self:
                return func(*args, **kwargs)
        return _acquire_compile_lock

    def _is_owned(self):
        if False:
            while True:
                i = 10
        if self._lock.acquire(0):
            self._lock.release()
            return False
        else:
            return True
global_compiler_lock = _CompilerLock()

def require_global_compiler_lock():
    if False:
        return 10
    'Sentry that checks the global_compiler_lock is acquired.\n    '
    assert global_compiler_lock.is_locked()