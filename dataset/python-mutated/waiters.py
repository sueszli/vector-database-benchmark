"""
Implementations of `Waiter`s, which allow work to be sent back to a thread while it
waits for the result of the call.
"""
import abc
import asyncio
import contextlib
import inspect
import queue
import threading
import weakref
from collections import deque
from typing import Awaitable, Generic, List, Optional, TypeVar, Union
import anyio
from prefect._internal.concurrency import logger
from prefect._internal.concurrency.calls import Call, Portal
from prefect._internal.concurrency.event_loop import call_soon_in_loop
from prefect._internal.concurrency.primitives import Event
T = TypeVar('T')
_WAITERS_BY_THREAD: 'weakref.WeakKeyDictionary[threading.Thread, deque[Waiter]]' = weakref.WeakKeyDictionary()

def get_waiter_for_thread(thread: threading.Thread) -> Optional['Waiter']:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the current waiter for a thread.\n\n    Returns `None` if one does not exist.\n    '
    waiters = _WAITERS_BY_THREAD.get(thread)
    if waiters:
        idx = -1
        while abs(idx) <= len(waiters):
            try:
                waiter = waiters[idx]
                if not waiter.call_is_done():
                    return waiter
                idx = idx - 1
            except IndexError:
                break
    return None

def add_waiter_for_thread(waiter: 'Waiter', thread: threading.Thread):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add a waiter for a thread.\n    '
    if thread not in _WAITERS_BY_THREAD:
        _WAITERS_BY_THREAD[thread] = deque()
    _WAITERS_BY_THREAD[thread].append(waiter)

class Waiter(Portal, abc.ABC, Generic[T]):
    """
    A waiter allows a waiting for a call while routing callbacks to the
    the current thread.

    Calls sent back to the waiter will be executed when waiting for the result.
    """

    def __init__(self, call: Call[T]) -> None:
        if False:
            i = 10
            return i + 15
        if not isinstance(call, Call):
            raise TypeError(f'Expected call of type `Call`; got {call!r}.')
        self._call = call
        self._owner_thread = threading.current_thread()
        add_waiter_for_thread(self, self._owner_thread)
        super().__init__()

    def call_is_done(self) -> bool:
        if False:
            print('Hello World!')
        return self._call.future.done()

    @abc.abstractmethod
    def wait(self) -> Union[Awaitable[None], None]:
        if False:
            while True:
                i = 10
        '\n        Wait for the call to finish.\n\n        Watch for and execute any waiting callbacks.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def add_done_callback(self, callback: Call) -> Call:
        if False:
            i = 10
            return i + 15
        '\n        Schedule a call to run when the waiter is done waiting.\n\n        If the waiter is already done, a `RuntimeError` error will be thrown.\n        '
        raise NotImplementedError()

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'<{self.__class__.__name__} call={self._call}, owner={self._owner_thread.name!r}>'

class SyncWaiter(Waiter[T]):

    def __init__(self, call: Call[T]) -> None:
        if False:
            while True:
                i = 10
        super().__init__(call=call)
        self._queue: queue.Queue = queue.Queue()
        self._done_callbacks = []
        self._done_event = threading.Event()

    def submit(self, call: Call):
        if False:
            while True:
                i = 10
        '\n        Submit a callback to execute while waiting.\n        '
        if self.call_is_done():
            raise RuntimeError(f'The call {self._call} is already done.')
        self._queue.put_nowait(call)
        call.set_runner(self)
        return call

    def _handle_waiting_callbacks(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('Waiter %r watching for callbacks', self)
        while True:
            callback: Call = self._queue.get()
            if callback is None:
                break
            self._call.future.add_cancel_callback(callback.future.cancel)
            callback.run()
            del callback

    @contextlib.contextmanager
    def _handle_done_callbacks(self):
        if False:
            i = 10
            return i + 15
        try:
            yield
        finally:
            while self._done_callbacks:
                callback = self._done_callbacks.pop()
                if callback:
                    callback.run()

    def add_done_callback(self, callback: Call):
        if False:
            return 10
        if self._done_event.is_set():
            raise RuntimeError('Cannot add done callbacks to done waiters.')
        else:
            self._done_callbacks.append(callback)

    def wait(self) -> T:
        if False:
            for i in range(10):
                print('nop')
        self._call.future.add_done_callback(lambda _: self._queue.put_nowait(None))
        self._call.future.add_done_callback(lambda _: self._done_event.set())
        with self._handle_done_callbacks():
            self._handle_waiting_callbacks()
            self._done_event.wait()
        _WAITERS_BY_THREAD[self._owner_thread].remove(self)
        return self._call

class AsyncWaiter(Waiter[T]):

    def __init__(self, call: Call[T]) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(call=call)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._early_submissions: List[Call] = []
        self._done_callbacks = []
        self._done_event = Event()
        self._done_waiting = False

    def submit(self, call: Call):
        if False:
            return 10
        '\n        Submit a callback to execute while waiting.\n        '
        if self.call_is_done():
            raise RuntimeError(f'The call {self._call} is already done.')
        call.set_runner(self)
        if not self._queue:
            self._early_submissions.append(call)
            return call
        call_soon_in_loop(self._loop, self._queue.put_nowait, call)
        return call

    def _resubmit_early_submissions(self):
        if False:
            print('Hello World!')
        assert self._queue
        for call in self._early_submissions:
            call_soon_in_loop(self._loop, self._queue.put_nowait, call)
        self._early_submissions = []

    async def _handle_waiting_callbacks(self):
        logger.debug('Waiter %r watching for callbacks', self)
        tasks = []
        try:
            while True:
                callback: Call = await self._queue.get()
                if callback is None:
                    break
                self._call.future.add_cancel_callback(callback.future.cancel)
                retval = callback.run()
                if inspect.isawaitable(retval):
                    tasks.append(retval)
                del callback
            await asyncio.gather(*tasks)
        finally:
            self._done_waiting = True

    @contextlib.asynccontextmanager
    async def _handle_done_callbacks(self):
        try:
            yield
        finally:
            while self._done_callbacks:
                callback = self._done_callbacks.pop()
                if callback:
                    with anyio.CancelScope(shield=True):
                        await self._run_done_callback(callback)

    async def _run_done_callback(self, callback: Call):
        coro = callback.run()
        if coro:
            await coro

    def add_done_callback(self, callback: Call):
        if False:
            for i in range(10):
                print('nop')
        if self._done_event.is_set():
            raise RuntimeError('Cannot add done callbacks to done waiters.')
        else:
            self._done_callbacks.append(callback)

    def _signal_stop_waiting(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._done_waiting:
            call_soon_in_loop(self._loop, self._queue.put_nowait, None)

    async def wait(self) -> Call[T]:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._resubmit_early_submissions()
        self._call.future.add_done_callback(lambda _: self._signal_stop_waiting())
        self._call.future.add_done_callback(lambda _: self._done_event.set())
        async with self._handle_done_callbacks():
            await self._handle_waiting_callbacks()
            await self._done_event.wait()
        _WAITERS_BY_THREAD[self._owner_thread].remove(self)
        return self._call