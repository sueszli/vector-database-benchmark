import os
import sys
import tempfile
import threading
import _multiprocessing
from time import time as _time
from multiprocessing import process, util
from multiprocessing.context import assert_spawning
from . import resource_tracker
__all__ = ['Lock', 'RLock', 'Semaphore', 'BoundedSemaphore', 'Condition', 'Event']
try:
    from _multiprocessing import SemLock as _SemLock
    from _multiprocessing import sem_unlink
except ImportError:
    raise ImportError('This platform lacks a functioning sem_open implementation, therefore, the required synchronization primitives needed will not function, see issue 3770.')
(RECURSIVE_MUTEX, SEMAPHORE) = range(2)
SEM_VALUE_MAX = _multiprocessing.SemLock.SEM_VALUE_MAX

class SemLock:
    _rand = tempfile._RandomNameSequence()

    def __init__(self, kind, value, maxvalue, name=None):
        if False:
            for i in range(10):
                print('nop')
        unlink_now = False
        if name is None:
            for _ in range(100):
                try:
                    self._semlock = _SemLock(kind, value, maxvalue, SemLock._make_name(), unlink_now)
                except FileExistsError:
                    pass
                else:
                    break
            else:
                raise FileExistsError('cannot find name for semaphore')
        else:
            self._semlock = _SemLock(kind, value, maxvalue, name, unlink_now)
        self.name = name
        util.debug(f'created semlock with handle {self._semlock.handle} and name "{self.name}"')
        self._make_methods()

        def _after_fork(obj):
            if False:
                i = 10
                return i + 15
            obj._semlock._after_fork()
        util.register_after_fork(self, _after_fork)
        resource_tracker.register(self._semlock.name, 'semlock')
        util.Finalize(self, SemLock._cleanup, (self._semlock.name,), exitpriority=0)

    @staticmethod
    def _cleanup(name):
        if False:
            i = 10
            return i + 15
        try:
            sem_unlink(name)
        except FileNotFoundError:
            pass
        finally:
            resource_tracker.unregister(name, 'semlock')

    def _make_methods(self):
        if False:
            for i in range(10):
                print('nop')
        self.acquire = self._semlock.acquire
        self.release = self._semlock.release

    def __enter__(self):
        if False:
            print('Hello World!')
        return self._semlock.acquire()

    def __exit__(self, *args):
        if False:
            return 10
        return self._semlock.release()

    def __getstate__(self):
        if False:
            print('Hello World!')
        assert_spawning(self)
        sl = self._semlock
        h = sl.handle
        return (h, sl.kind, sl.maxvalue, sl.name)

    def __setstate__(self, state):
        if False:
            return 10
        self._semlock = _SemLock._rebuild(*state)
        util.debug(f'recreated blocker with handle {state[0]!r} and name "{state[3]}"')
        self._make_methods()

    @staticmethod
    def _make_name():
        if False:
            return 10
        return f'/loky-{os.getpid()}-{next(SemLock._rand)}'

class Semaphore(SemLock):

    def __init__(self, value=1):
        if False:
            for i in range(10):
                print('nop')
        SemLock.__init__(self, SEMAPHORE, value, SEM_VALUE_MAX)

    def get_value(self):
        if False:
            for i in range(10):
                print('nop')
        if sys.platform == 'darwin':
            raise NotImplementedError('OSX does not implement sem_getvalue')
        return self._semlock._get_value()

    def __repr__(self):
        if False:
            while True:
                i = 10
        try:
            value = self._semlock._get_value()
        except Exception:
            value = 'unknown'
        return f'<{self.__class__.__name__}(value={value})>'

class BoundedSemaphore(Semaphore):

    def __init__(self, value=1):
        if False:
            while True:
                i = 10
        SemLock.__init__(self, SEMAPHORE, value, value)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        try:
            value = self._semlock._get_value()
        except Exception:
            value = 'unknown'
        return f'<{self.__class__.__name__}(value={value}, maxvalue={self._semlock.maxvalue})>'

class Lock(SemLock):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__(SEMAPHORE, 1, 1)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        try:
            if self._semlock._is_mine():
                name = process.current_process().name
                if threading.current_thread().name != 'MainThread':
                    name = f'{name}|{threading.current_thread().name}'
            elif self._semlock._get_value() == 1:
                name = 'None'
            elif self._semlock._count() > 0:
                name = 'SomeOtherThread'
            else:
                name = 'SomeOtherProcess'
        except Exception:
            name = 'unknown'
        return f'<{self.__class__.__name__}(owner={name})>'

class RLock(SemLock):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__(RECURSIVE_MUTEX, 1, 1)

    def __repr__(self):
        if False:
            print('Hello World!')
        try:
            if self._semlock._is_mine():
                name = process.current_process().name
                if threading.current_thread().name != 'MainThread':
                    name = f'{name}|{threading.current_thread().name}'
                count = self._semlock._count()
            elif self._semlock._get_value() == 1:
                (name, count) = ('None', 0)
            elif self._semlock._count() > 0:
                (name, count) = ('SomeOtherThread', 'nonzero')
            else:
                (name, count) = ('SomeOtherProcess', 'nonzero')
        except Exception:
            (name, count) = ('unknown', 'unknown')
        return f'<{self.__class__.__name__}({name}, {count})>'

class Condition:

    def __init__(self, lock=None):
        if False:
            print('Hello World!')
        self._lock = lock or RLock()
        self._sleeping_count = Semaphore(0)
        self._woken_count = Semaphore(0)
        self._wait_semaphore = Semaphore(0)
        self._make_methods()

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        assert_spawning(self)
        return (self._lock, self._sleeping_count, self._woken_count, self._wait_semaphore)

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        (self._lock, self._sleeping_count, self._woken_count, self._wait_semaphore) = state
        self._make_methods()

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._lock.__enter__()

    def __exit__(self, *args):
        if False:
            print('Hello World!')
        return self._lock.__exit__(*args)

    def _make_methods(self):
        if False:
            for i in range(10):
                print('nop')
        self.acquire = self._lock.acquire
        self.release = self._lock.release

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        try:
            num_waiters = self._sleeping_count._semlock._get_value() - self._woken_count._semlock._get_value()
        except Exception:
            num_waiters = 'unknown'
        return f'<{self.__class__.__name__}({self._lock}, {num_waiters})>'

    def wait(self, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        assert self._lock._semlock._is_mine(), 'must acquire() condition before using wait()'
        self._sleeping_count.release()
        count = self._lock._semlock._count()
        for _ in range(count):
            self._lock.release()
        try:
            return self._wait_semaphore.acquire(True, timeout)
        finally:
            self._woken_count.release()
            for _ in range(count):
                self._lock.acquire()

    def notify(self):
        if False:
            return 10
        assert self._lock._semlock._is_mine(), 'lock is not owned'
        assert not self._wait_semaphore.acquire(False)
        while self._woken_count.acquire(False):
            res = self._sleeping_count.acquire(False)
            assert res
        if self._sleeping_count.acquire(False):
            self._wait_semaphore.release()
            self._woken_count.acquire()
            self._wait_semaphore.acquire(False)

    def notify_all(self):
        if False:
            for i in range(10):
                print('nop')
        assert self._lock._semlock._is_mine(), 'lock is not owned'
        assert not self._wait_semaphore.acquire(False)
        while self._woken_count.acquire(False):
            res = self._sleeping_count.acquire(False)
            assert res
        sleepers = 0
        while self._sleeping_count.acquire(False):
            self._wait_semaphore.release()
            sleepers += 1
        if sleepers:
            for _ in range(sleepers):
                self._woken_count.acquire()
            while self._wait_semaphore.acquire(False):
                pass

    def wait_for(self, predicate, timeout=None):
        if False:
            i = 10
            return i + 15
        result = predicate()
        if result:
            return result
        if timeout is not None:
            endtime = _time() + timeout
        else:
            endtime = None
            waittime = None
        while not result:
            if endtime is not None:
                waittime = endtime - _time()
                if waittime <= 0:
                    break
            self.wait(waittime)
            result = predicate()
        return result

class Event:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._cond = Condition(Lock())
        self._flag = Semaphore(0)

    def is_set(self):
        if False:
            print('Hello World!')
        with self._cond:
            if self._flag.acquire(False):
                self._flag.release()
                return True
            return False

    def set(self):
        if False:
            return 10
        with self._cond:
            self._flag.acquire(False)
            self._flag.release()
            self._cond.notify_all()

    def clear(self):
        if False:
            print('Hello World!')
        with self._cond:
            self._flag.acquire(False)

    def wait(self, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        with self._cond:
            if self._flag.acquire(False):
                self._flag.release()
            else:
                self._cond.wait(timeout)
            if self._flag.acquire(False):
                self._flag.release()
                return True
            return False