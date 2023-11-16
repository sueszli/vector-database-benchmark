"""
Utilities for cancellation in synchronous and asynchronous contexts.
"""
import abc
import asyncio
import contextlib
import ctypes
import math
import os
import signal
import sys
import threading
import time
from typing import Callable, Dict, Optional, Type
import anyio
from prefect._internal.concurrency import logger
from prefect._internal.concurrency.event_loop import get_running_loop
_THREAD_SHIELDS: Dict[threading.Thread, 'ThreadShield'] = {}
_THREAD_SHIELDS_LOCK = threading.Lock()

class ThreadShield:
    """
    A wrapper around a reentrant lock for shielding a thread from remote exceptions.
    This can be used in two ways:

    1. As a context manager from _another_ thread to wait until the shield is released
      by a target before sending an exception.

    2. From the current thread, using `set_exception` to throw the exception when the
      shield is released.

    A reentrant lock means that shields can be nested and the exception will only be
    raised when the last context is exited.
    """

    def __init__(self, owner: threading.Thread):
        if False:
            while True:
                i = 10
        self._lock = threading._RLock()
        self._exception = None
        self._owner = owner

    def __enter__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._lock.__enter__()

    def __exit__(self, *exc_info):
        if False:
            for i in range(10):
                print('nop')
        retval = self._lock.__exit__(*exc_info)
        if not self.active() and self._exception and (self._owner.ident == threading.current_thread().ident):
            exc = self._exception
            self._exception = None
            raise exc from None
        return retval

    def set_exception(self, exc: Exception):
        if False:
            while True:
                i = 10
        self._exception = exc

    def active(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns true if the shield is active.\n        '
        return self._lock._count > 0

class CancelledError(asyncio.CancelledError):
    pass

def _get_thread_shield(thread) -> ThreadShield:
    if False:
        i = 10
        return i + 15
    with _THREAD_SHIELDS_LOCK:
        if thread not in _THREAD_SHIELDS:
            _THREAD_SHIELDS[thread] = ThreadShield(thread)
        for thread_ in tuple(_THREAD_SHIELDS.keys()):
            if not thread_.is_alive():
                _THREAD_SHIELDS.pop(thread_)
        return _THREAD_SHIELDS[thread]

@contextlib.contextmanager
def shield():
    if False:
        for i in range(10):
            print('nop')
    '\n    Prevent code from within the scope from being cancelled.\n\n    This guards against cancellation from alarm signals and injected exceptions as used\n    in this module.\n\n    If an event loop is running in the thread where this is called, it will be shielded\n    from asynchronous cancellation as well.\n    '
    with anyio.CancelScope(shield=True) if get_running_loop() else contextlib.nullcontext():
        with _get_thread_shield(threading.current_thread()):
            yield

class CancelScope(abc.ABC):
    """
    Defines a context where cancellation can be requested.

    If cancelled, any code within the context should be interrupted. The cancellation
    implementation varies depending on the environment and may not interrupt some system
    calls.

    A timeout can be defined to automatically cancel the scope after a given duration if
    it has not exited.
    """

    def __init__(self, name: Optional[str]=None, timeout: Optional[float]=None) -> None:
        if False:
            i = 10
            return i + 15
        self.name = name
        self._deadline = None
        self._cancelled = False
        self._completed = False
        self._started = False
        self._start_time = None
        self._end_time = None
        self._timeout = timeout
        self._lock = threading.Lock()
        self._callbacks = []
        super().__init__()

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            self._deadline = get_deadline(self._timeout)
            self._started = True
            self._start_time = time.monotonic()
        logger.debug('%r entered', self)
        return self

    def __exit__(self, *_):
        if False:
            return 10
        with self._lock:
            if not self._cancelled:
                self._completed = True
            self._end_time = time.monotonic()
        logger.debug('%r exited', self)

    @property
    def timeout(self):
        if False:
            for i in range(10):
                print('nop')
        return self._timeout

    def started(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            return self._started

    def cancelled(self) -> bool:
        if False:
            while True:
                i = 10
        with self._lock:
            return self._cancelled

    def timedout(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            if not self._end_time or not self._deadline:
                return False
            return self._cancelled and self._end_time > self._deadline

    def set_timeout(self, timeout: float):
        if False:
            i = 10
            return i + 15
        with self._lock:
            if self._started:
                raise RuntimeError('Cannot set timeout after scope has started.')
            self._timeout = timeout

    def completed(self):
        if False:
            i = 10
            return i + 15
        with self._lock:
            return self._completed

    def cancel(self, throw: bool=True) -> bool:
        if False:
            return 10
        '\n        Cancel this scope.\n\n        If `throw` is not set, this will only mark the scope as cancelled and will not\n        throw the cancelled error.\n        '
        with self._lock:
            if not self.started:
                raise RuntimeError('Scope has not been entered.')
            if self._completed:
                return False
            if self._cancelled:
                return True
            self._cancelled = True
        logger.info('%r cancelling', self)
        for callback in self._callbacks:
            callback()
        return True

    def add_cancel_callback(self, callback: Callable[[], None]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a callback to execute on cancellation.\n        '
        self._callbacks.append(callback)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        with self._lock:
            state = ('completed' if self._completed else 'cancelled' if self._cancelled else 'running' if self._started else 'pending').upper()
            timeout = f', timeout={self._timeout:.2f}' if self._timeout else ''
            runtime = f', runtime={(self._end_time or time.monotonic()) - self._start_time:.2f}' if self._start_time else ''
            name = f', name={self.name!r}' if self.name else f'at {hex(id(self))}'
        return f'<{type(self).__name__}{name} {state}{timeout}{runtime}>'

class AsyncCancelScope(CancelScope):

    def __init__(self, name: Optional[str]=None, timeout: Optional[float]=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(name=name, timeout=timeout)
        self.loop = None

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.loop = asyncio.get_running_loop()
        super().__enter__()
        self._anyio_scope = anyio.CancelScope(deadline=self._deadline if self._deadline is not None else math.inf).__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        if self._anyio_scope.cancel_called:
            self.cancel(throw=False)
        super().__exit__(exc_type, exc_val, exc_tb)
        if self.cancelled() and exc_type is not CancelledError:
            raise CancelledError() from exc_val
        return False

    def cancel(self, throw: bool=True):
        if False:
            while True:
                i = 10
        if not super().cancel():
            return False
        if throw:
            if self.loop is get_running_loop():
                self._anyio_scope.cancel()
            else:
                self.loop.call_soon_threadsafe(self._anyio_scope.cancel)
        return True

class NullCancelScope(CancelScope):
    """
    A cancel scope that does nothing.

    This is used for environments where cancellation is not supported.
    """

    def __init__(self, name: Optional[str]=None, timeout: Optional[float]=None, reason: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(name, timeout)
        self.reason = reason or 'null cancel scope'

    def cancel(self):
        if False:
            print('Hello World!')
        logger.warning('%r cannot cancel %s.', self, self.reason)
        return False

class AlarmCancelScope(CancelScope):
    """
    A cancel scope that uses an alarm signal which can interrupt long-running system
    calls.

    Only the main thread can be cancelled with an alarm signal, so this scope is only
    available in the main thread.
    """

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        super().__enter__()
        current_thread = threading.current_thread()
        self._previous_timer = None
        if current_thread is not threading.main_thread():
            raise ValueError('Alarm based timeouts can only be used in the main thread.')
        self._previous_alarm_handler = signal.getsignal(signal.SIGALRM)
        if self._previous_alarm_handler != signal.SIG_DFL:
            logger.warning('%r overriding existing alarm handler %s', self, self._previous_alarm_handler)
        signal.signal(signal.SIGALRM, self._sigalarm_to_error)
        if self.timeout is not None:
            logger.debug('%r set alarm timer for %f seconds', self, self.timeout)
            self._previous_timer = signal.setitimer(signal.ITIMER_REAL, self.timeout)
        return self

    def _sigalarm_to_error(self, *args):
        if False:
            i = 10
            return i + 15
        logger.debug('%r captured alarm raising as cancelled error', self)
        if self.cancel(throw=False):
            shield = _get_thread_shield(threading.main_thread())
            if shield.active():
                logger.debug('%r thread shield active; delaying exception', self)
                shield.set_exception(CancelledError())
            else:
                raise CancelledError()

    def __exit__(self, *_):
        if False:
            print('Hello World!')
        retval = super().__exit__(*_)
        if self.timeout is not None:
            signal.setitimer(signal.ITIMER_REAL, *self._previous_timer)
        signal.signal(signal.SIGALRM, self._previous_alarm_handler)
        return retval

    def cancel(self, throw: bool=True):
        if False:
            print('Hello World!')
        if not super().cancel():
            return False
        if throw:
            logger.debug('%r sending alarm signal to main thread', self)
            os.kill(os.getpid(), signal.SIGALRM)
        return True

class WatcherThreadCancelScope(CancelScope):
    """
    A cancel scope that uses a watcher thread and an injected exception to enforce
    cancellation.

    The injected exception cannot interrupt calls and will be raised on the ~next
    instruction. This can raise exceptions in unexpected places. See `shield` for
    guarding against interruption.

    If a timeout is specified, a watcher thread is spawned that will run for `timeout`
    seconds then send the exception to the supervised thread.
    """

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__enter__()
        self._event = threading.Event()
        self._enforcer_thread = None
        self._supervised_thread = threading.current_thread()
        if self.timeout is not None:
            name = self.name or f'for scope {hex(id(self))}'
            self._enforcer_thread = threading.Thread(target=self._timeout_enforcer, name=f'timeout-watcher {name} {self.timeout:.2f}')
            self._enforcer_thread.start()
        return self

    def __exit__(self, *_):
        if False:
            i = 10
            return i + 15
        retval = super().__exit__(*_)
        self._event.set()
        if self._enforcer_thread:
            logger.debug('%r joining enforcer thread %r', self, self._enforcer_thread)
            self._enforcer_thread.join()
        return retval

    def _send_cancelled_error(self):
        if False:
            print('Hello World!')
        '\n        Send a cancelled error to the supervised thread.\n        '
        if self._supervised_thread.is_alive():
            logger.debug('%r sending exception to supervised thread %r', self, self._supervised_thread)
            with _get_thread_shield(self._supervised_thread):
                try:
                    _send_exception_to_thread(self._supervised_thread, CancelledError)
                except ValueError:
                    logger.debug('Thread missing!')

    def _timeout_enforcer(self):
        if False:
            print('Hello World!')
        '\n        Target for a thread that enforces a timeout.\n        '
        if not self._event.wait(self.timeout):
            logger.debug('%r enforcer detected timeout!', self)
            if self.cancel(throw=False):
                with _get_thread_shield(self._supervised_thread):
                    self._send_cancelled_error()
        logger.debug('%r waiting for supervised thread to exit', self)
        self._event.wait()

    def cancel(self, throw: bool=True):
        if False:
            return 10
        if not super().cancel():
            return False
        if throw:
            self._send_cancelled_error()
        return True

def get_deadline(timeout: Optional[float]):
    if False:
        return 10
    '\n    Compute an deadline given a timeout.\n\n    Uses a monotonic clock.\n    '
    if timeout is None:
        return None
    return time.monotonic() + timeout

def get_timeout(deadline: Optional[float]):
    if False:
        return 10
    '\n    Compute an timeout given a deadline.\n\n    Uses a monotonic clock.\n    '
    if deadline is None:
        return None
    return max(0, deadline - time.monotonic())

@contextlib.contextmanager
def cancel_async_at(deadline: Optional[float], name: Optional[str]=None):
    if False:
        while True:
            i = 10
    '\n    Cancel any async calls within the context if it does not exit by the given deadline.\n\n    Deadlines must be computed with the monotonic clock. See `get_deadline`.\n\n    A timeout error will be raised on the next `await` when the timeout expires.\n\n    Yields a `CancelContext`.\n    '
    with cancel_async_after(get_timeout(deadline), name=name) as ctx:
        yield ctx

@contextlib.contextmanager
def cancel_async_after(timeout: Optional[float], name: Optional[str]=None):
    if False:
        i = 10
        return i + 15
    '\n    Cancel any async calls within the context if it does not exit after the given\n    timeout.\n\n    A timeout error will be raised on the next `await` when the timeout expires.\n\n    Yields a `CancelContext`.\n    '
    with AsyncCancelScope(timeout=timeout, name=name) as ctx:
        yield ctx

@contextlib.contextmanager
def cancel_sync_at(deadline: Optional[float], name: Optional[str]=None):
    if False:
        while True:
            i = 10
    '\n    Cancel any sync calls within the context if it does not exit by the given deadline.\n\n    Deadlines must be computed with the monotonic clock. See `get_deadline`.\n\n    The cancel method varies depending on if this is called in the main thread or not.\n    See `cancel_sync_after` for details\n\n    Yields a `CancelContext`.\n    '
    timeout = max(0, deadline - time.monotonic()) if deadline is not None else None
    with cancel_sync_after(timeout, name=name) as ctx:
        yield ctx

@contextlib.contextmanager
def cancel_sync_after(timeout: Optional[float], name: Optional[str]=None):
    if False:
        i = 10
        return i + 15
    '\n    Cancel any sync calls within the context if it does not exit after the given\n    timeout.\n\n    The timeout method varies depending on if this is called in the main thread or not.\n    See `AlarmCancelScope` and `WatcherThreadCancelScope` for details.\n\n    Yields a `CancelContext`.\n    '
    if sys.platform.startswith('win'):
        yield NullCancelScope(reason='cancellation is not supported on Windows')
        return
    thread = threading.current_thread()
    existing_alarm_handler = signal.getsignal(signal.SIGALRM) != signal.SIG_DFL
    if thread is threading.main_thread() and (not existing_alarm_handler) and (timeout is not None):
        scope = AlarmCancelScope(name=name, timeout=timeout)
    else:
        scope = WatcherThreadCancelScope(name=name, timeout=timeout)
    with scope:
        yield scope

def _send_exception_to_thread(thread: threading.Thread, exc_type: Type[BaseException]):
    if False:
        for i in range(10):
            print('nop')
    '\n    Raise an exception in a thread.\n\n    This will not interrupt long-running system calls like `sleep` or `wait`.\n    '
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), ctypes.py_object(exc_type))
    if ret == 0:
        raise ValueError('Thread not found.')