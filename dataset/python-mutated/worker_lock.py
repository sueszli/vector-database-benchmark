import random
from types import TracebackType
from typing import TYPE_CHECKING, AsyncContextManager, Collection, Dict, Optional, Tuple, Type, Union
from weakref import WeakSet
import attr
from twisted.internet import defer
from twisted.internet.interfaces import IReactorTime
from synapse.logging.context import PreserveLoggingContext
from synapse.logging.opentracing import start_active_span
from synapse.metrics.background_process_metrics import wrap_as_background_process
from synapse.storage.databases.main.lock import Lock, LockStore
from synapse.util.async_helpers import timeout_deferred
if TYPE_CHECKING:
    from synapse.logging.opentracing import opentracing
    from synapse.server import HomeServer
NEW_EVENT_DURING_PURGE_LOCK_NAME = 'new_event_during_purge_lock'

class WorkerLocksHandler:
    """A class for waiting on taking out locks, rather than using the storage
    functions directly (which don't support awaiting).
    """

    def __init__(self, hs: 'HomeServer') -> None:
        if False:
            i = 10
            return i + 15
        self._reactor = hs.get_reactor()
        self._store = hs.get_datastores().main
        self._clock = hs.get_clock()
        self._notifier = hs.get_notifier()
        self._instance_name = hs.get_instance_name()
        self._locks: Dict[Tuple[str, str], WeakSet[Union[WaitingLock, WaitingMultiLock]]] = {}
        self._clock.looping_call(self._cleanup_locks, 30000)
        self._notifier.add_lock_released_callback(self._on_lock_released)

    def acquire_lock(self, lock_name: str, lock_key: str) -> 'WaitingLock':
        if False:
            while True:
                i = 10
        'Acquire a standard lock, returns a context manager that will block\n        until the lock is acquired.\n\n        Note: Care must be taken to avoid deadlocks. In particular, this\n        function does *not* timeout.\n\n        Usage:\n            async with handler.acquire_lock(name, key):\n                # Do work while holding the lock...\n        '
        lock = WaitingLock(reactor=self._reactor, store=self._store, handler=self, lock_name=lock_name, lock_key=lock_key, write=None)
        self._locks.setdefault((lock_name, lock_key), WeakSet()).add(lock)
        return lock

    def acquire_read_write_lock(self, lock_name: str, lock_key: str, *, write: bool) -> 'WaitingLock':
        if False:
            i = 10
            return i + 15
        'Acquire a read/write lock, returns a context manager that will block\n        until the lock is acquired.\n\n        Note: Care must be taken to avoid deadlocks. In particular, this\n        function does *not* timeout.\n\n        Usage:\n            async with handler.acquire_read_write_lock(name, key, write=True):\n                # Do work while holding the lock...\n        '
        lock = WaitingLock(reactor=self._reactor, store=self._store, handler=self, lock_name=lock_name, lock_key=lock_key, write=write)
        self._locks.setdefault((lock_name, lock_key), WeakSet()).add(lock)
        return lock

    def acquire_multi_read_write_lock(self, lock_names: Collection[Tuple[str, str]], *, write: bool) -> 'WaitingMultiLock':
        if False:
            while True:
                i = 10
        'Acquires multi read/write locks at once, returns a context manager\n        that will block until all the locks are acquired.\n\n        This will try and acquire all locks at once, and will never hold on to a\n        subset of the locks. (This avoids accidentally creating deadlocks).\n\n        Note: Care must be taken to avoid deadlocks. In particular, this\n        function does *not* timeout.\n        '
        lock = WaitingMultiLock(lock_names=lock_names, write=write, reactor=self._reactor, store=self._store, handler=self)
        for (lock_name, lock_key) in lock_names:
            self._locks.setdefault((lock_name, lock_key), WeakSet()).add(lock)
        return lock

    def notify_lock_released(self, lock_name: str, lock_key: str) -> None:
        if False:
            return 10
        'Notify that a lock has been released.\n\n        Pokes both the notifier and replication.\n        '
        self._notifier.notify_lock_released(self._instance_name, lock_name, lock_key)

    def _on_lock_released(self, instance_name: str, lock_name: str, lock_key: str) -> None:
        if False:
            i = 10
            return i + 15
        'Called when a lock has been released.\n\n        Wakes up any locks that might be waiting on this.\n        '
        locks = self._locks.get((lock_name, lock_key))
        if not locks:
            return

        def _wake_deferred(deferred: defer.Deferred) -> None:
            if False:
                print('Hello World!')
            if not deferred.called:
                deferred.callback(None)
        for lock in locks:
            self._clock.call_later(0, _wake_deferred, lock.deferred)

    @wrap_as_background_process('_cleanup_locks')
    async def _cleanup_locks(self) -> None:
        """Periodically cleans out stale entries in the locks map"""
        self._locks = {key: value for (key, value) in self._locks.items() if value}

@attr.s(auto_attribs=True, eq=False)
class WaitingLock:
    reactor: IReactorTime
    store: LockStore
    handler: WorkerLocksHandler
    lock_name: str
    lock_key: str
    write: Optional[bool]
    deferred: 'defer.Deferred[None]' = attr.Factory(defer.Deferred)
    _inner_lock: Optional[Lock] = None
    _retry_interval: float = 0.1
    _lock_span: 'opentracing.Scope' = attr.Factory(lambda : start_active_span('WaitingLock.lock'))

    async def __aenter__(self) -> None:
        self._lock_span.__enter__()
        with start_active_span('WaitingLock.waiting_for_lock'):
            while self._inner_lock is None:
                self.deferred = defer.Deferred()
                if self.write is not None:
                    lock = await self.store.try_acquire_read_write_lock(self.lock_name, self.lock_key, write=self.write)
                else:
                    lock = await self.store.try_acquire_lock(self.lock_name, self.lock_key)
                if lock:
                    self._inner_lock = lock
                    break
                try:
                    with PreserveLoggingContext():
                        await timeout_deferred(deferred=self.deferred, timeout=self._get_next_retry_interval(), reactor=self.reactor)
                except Exception:
                    pass
        return await self._inner_lock.__aenter__()

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> Optional[bool]:
        assert self._inner_lock
        self.handler.notify_lock_released(self.lock_name, self.lock_key)
        try:
            r = await self._inner_lock.__aexit__(exc_type, exc, tb)
        finally:
            self._lock_span.__exit__(exc_type, exc, tb)
        return r

    def _get_next_retry_interval(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        next = self._retry_interval
        self._retry_interval = max(5, next * 2)
        return next * random.uniform(0.9, 1.1)

@attr.s(auto_attribs=True, eq=False)
class WaitingMultiLock:
    lock_names: Collection[Tuple[str, str]]
    write: bool
    reactor: IReactorTime
    store: LockStore
    handler: WorkerLocksHandler
    deferred: 'defer.Deferred[None]' = attr.Factory(defer.Deferred)
    _inner_lock_cm: Optional[AsyncContextManager] = None
    _retry_interval: float = 0.1
    _lock_span: 'opentracing.Scope' = attr.Factory(lambda : start_active_span('WaitingLock.lock'))

    async def __aenter__(self) -> None:
        self._lock_span.__enter__()
        with start_active_span('WaitingLock.waiting_for_lock'):
            while self._inner_lock_cm is None:
                self.deferred = defer.Deferred()
                lock_cm = await self.store.try_acquire_multi_read_write_lock(self.lock_names, write=self.write)
                if lock_cm:
                    self._inner_lock_cm = lock_cm
                    break
                try:
                    with PreserveLoggingContext():
                        await timeout_deferred(deferred=self.deferred, timeout=self._get_next_retry_interval(), reactor=self.reactor)
                except Exception:
                    pass
        assert self._inner_lock_cm
        await self._inner_lock_cm.__aenter__()
        return

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> Optional[bool]:
        assert self._inner_lock_cm
        for (lock_name, lock_key) in self.lock_names:
            self.handler.notify_lock_released(lock_name, lock_key)
        try:
            r = await self._inner_lock_cm.__aexit__(exc_type, exc, tb)
        finally:
            self._lock_span.__exit__(exc_type, exc, tb)
        return r

    def _get_next_retry_interval(self) -> float:
        if False:
            while True:
                i = 10
        next = self._retry_interval
        self._retry_interval = max(5, next * 2)
        return next * random.uniform(0.9, 1.1)