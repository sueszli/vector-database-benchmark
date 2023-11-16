import abc
import enum
import threading
from typing import Callable, Collection, Dict, Generic, MutableMapping, Optional, Set, Sized, Tuple, TypeVar, Union, cast
from prometheus_client import Gauge
from twisted.internet import defer
from twisted.python.failure import Failure
from synapse.util.async_helpers import ObservableDeferred
from synapse.util.caches.lrucache import LruCache
from synapse.util.caches.treecache import TreeCache, iterate_tree_cache_entry
cache_pending_metric = Gauge('synapse_util_caches_cache_pending', 'Number of lookups currently pending for this cache', ['name'])
T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')

class _Sentinel(enum.Enum):
    sentinel = object()

class DeferredCache(Generic[KT, VT]):
    """Wraps an LruCache, adding support for Deferred results.

    It expects that each entry added with set() will be a Deferred; likewise get()
    will return a Deferred.
    """
    __slots__ = ('cache', 'thread', '_pending_deferred_cache')

    def __init__(self, name: str, max_entries: int=1000, tree: bool=False, iterable: bool=False, apply_cache_factor_from_config: bool=True, prune_unread_entries: bool=True):
        if False:
            print('Hello World!')
        "\n        Args:\n            name: The name of the cache\n            max_entries: Maximum amount of entries that the cache will hold\n            tree: Use a TreeCache instead of a dict as the underlying cache type\n            iterable: If True, count each item in the cached object as an entry,\n                rather than each cached object\n            apply_cache_factor_from_config: Whether cache factors specified in the\n                config file affect `max_entries`\n            prune_unread_entries: If True, cache entries that haven't been read recently\n                will be evicted from the cache in the background. Set to False to\n                opt-out of this behaviour.\n        "
        cache_type = TreeCache if tree else dict
        self._pending_deferred_cache: Union[TreeCache, 'MutableMapping[KT, CacheEntry[KT, VT]]'] = cache_type()

        def metrics_cb() -> None:
            if False:
                return 10
            cache_pending_metric.labels(name).set(len(self._pending_deferred_cache))
        self.cache: LruCache[KT, VT] = LruCache(max_size=max_entries, cache_name=name, cache_type=cache_type, size_callback=(lambda d: len(cast(Sized, d)) or 1) if iterable else None, metrics_collection_callback=metrics_cb, apply_cache_factor_from_config=apply_cache_factor_from_config, prune_unread_entries=prune_unread_entries)
        self.thread: Optional[threading.Thread] = None

    @property
    def max_entries(self) -> int:
        if False:
            print('Hello World!')
        return self.cache.max_size

    def check_thread(self) -> None:
        if False:
            while True:
                i = 10
        expected_thread = self.thread
        if expected_thread is None:
            self.thread = threading.current_thread()
        elif expected_thread is not threading.current_thread():
            raise ValueError('Cache objects can only be accessed from the main thread')

    def get(self, key: KT, callback: Optional[Callable[[], None]]=None, update_metrics: bool=True) -> defer.Deferred:
        if False:
            print('Hello World!')
        'Looks the key up in the caches.\n\n        For symmetry with set(), this method does *not* follow the synapse logcontext\n        rules: the logcontext will not be cleared on return, and the Deferred will run\n        its callbacks in the sentinel context. In other words: wrap the result with\n        make_deferred_yieldable() before `await`ing it.\n\n        Args:\n            key:\n            callback: Gets called when the entry in the cache is invalidated\n            update_metrics: whether to update the cache hit rate metrics\n\n        Returns:\n            A Deferred which completes with the result. Note that this may later fail\n            if there is an ongoing set() operation which later completes with a failure.\n\n        Raises:\n            KeyError if the key is not found in the cache\n        '
        val = self._pending_deferred_cache.get(key, _Sentinel.sentinel)
        if val is not _Sentinel.sentinel:
            val.add_invalidation_callback(key, callback)
            if update_metrics:
                m = self.cache.metrics
                assert m
                m.inc_hits()
            return val.deferred(key)
        callbacks = (callback,) if callback else ()
        val2 = self.cache.get(key, _Sentinel.sentinel, callbacks=callbacks, update_metrics=update_metrics)
        if val2 is _Sentinel.sentinel:
            raise KeyError()
        else:
            return defer.succeed(val2)

    def get_bulk(self, keys: Collection[KT], callback: Optional[Callable[[], None]]=None) -> Tuple[Dict[KT, VT], Optional['defer.Deferred[Dict[KT, VT]]'], Collection[KT]]:
        if False:
            i = 10
            return i + 15
        "Bulk lookup of items in the cache.\n\n        Returns:\n            A 3-tuple of:\n                1. a dict of key/value of items already cached;\n                2. a deferred that resolves to a dict of key/value of items\n                   we're already fetching; and\n                3. a collection of keys that don't appear in the previous two.\n        "
        cached = {}
        pending = []
        pending_results = {}
        missing = []
        callbacks = (callback,) if callback else ()
        for key in keys:
            immediate_value = self.cache.get(key, _Sentinel.sentinel, callbacks=callbacks)
            if immediate_value is not _Sentinel.sentinel:
                cached[key] = immediate_value
                continue
            pending_value = self._pending_deferred_cache.get(key, _Sentinel.sentinel)
            if pending_value is not _Sentinel.sentinel:
                pending_value.add_invalidation_callback(key, callback)

                def completed_cb(value: VT, key: KT) -> VT:
                    if False:
                        return 10
                    pending_results[key] = value
                    return value
                d = pending_value.deferred(key).addCallback(completed_cb, key)
                pending.append(d)
                continue
            missing.append(key)
        pending_deferred = None
        if pending:
            pending_deferred = defer.gatherResults(pending, consumeErrors=True).addCallback(lambda _: pending_results)
        return (cached, pending_deferred, missing)

    def get_immediate(self, key: KT, default: T, update_metrics: bool=True) -> Union[VT, T]:
        if False:
            print('Hello World!')
        'If we have a *completed* cached value, return it.'
        return self.cache.get(key, default, update_metrics=update_metrics)

    def set(self, key: KT, value: 'defer.Deferred[VT]', callback: Optional[Callable[[], None]]=None) -> defer.Deferred:
        if False:
            while True:
                i = 10
        'Adds a new entry to the cache (or updates an existing one).\n\n        The given `value` *must* be a Deferred.\n\n        First any existing entry for the same key is invalidated. Then a new entry\n        is added to the cache for the given key.\n\n        Until the `value` completes, calls to `get()` for the key will also result in an\n        incomplete Deferred, which will ultimately complete with the same result as\n        `value`.\n\n        If `value` completes successfully, subsequent calls to `get()` will then return\n        a completed deferred with the same result. If it *fails*, the cache is\n        invalidated and subequent calls to `get()` will raise a KeyError.\n\n        If another call to `set()` happens before `value` completes, then (a) any\n        invalidation callbacks registered in the interim will be called, (b) any\n        `get()`s in the interim will continue to complete with the result from the\n        *original* `value`, (c) any future calls to `get()` will complete with the\n        result from the *new* `value`.\n\n        It is expected that `value` does *not* follow the synapse logcontext rules - ie,\n        if it is incomplete, it runs its callbacks in the sentinel context.\n\n        Args:\n            key: Key to be set\n            value: a deferred which will complete with a result to add to the cache\n            callback: An optional callback to be called when the entry is invalidated\n        '
        self.check_thread()
        self._pending_deferred_cache.pop(key, None)
        entry = CacheEntrySingle[KT, VT](value)
        entry.add_invalidation_callback(key, callback)
        self._pending_deferred_cache[key] = entry
        deferred = entry.deferred(key).addCallbacks(self._completed_callback, self._error_callback, callbackArgs=(entry, key), errbackArgs=(entry, key))
        return deferred

    def start_bulk_input(self, keys: Collection[KT], callback: Optional[Callable[[], None]]=None) -> 'CacheMultipleEntries[KT, VT]':
        if False:
            i = 10
            return i + 15
        'Bulk set API for use when fetching multiple keys at once from the DB.\n\n        Called *before* starting the fetch from the DB, and the caller *must*\n        call either `complete_bulk(..)` or `error_bulk(..)` on the return value.\n        '
        entry = CacheMultipleEntries[KT, VT]()
        entry.add_global_invalidation_callback(callback)
        for key in keys:
            self._pending_deferred_cache[key] = entry
        return entry

    def _completed_callback(self, value: VT, entry: 'CacheEntry[KT, VT]', key: KT) -> VT:
        if False:
            while True:
                i = 10
        'Called when a deferred is completed.'
        current_entry = self._pending_deferred_cache.pop(key, None)
        if current_entry is not entry:
            if current_entry:
                self._pending_deferred_cache[key] = current_entry
            return value
        self.cache.set(key, value, entry.get_invalidation_callbacks(key))
        return value

    def _error_callback(self, failure: Failure, entry: 'CacheEntry[KT, VT]', key: KT) -> Failure:
        if False:
            return 10
        'Called when a deferred errors.'
        current_entry = self._pending_deferred_cache.pop(key, None)
        if current_entry is not entry:
            if current_entry:
                self._pending_deferred_cache[key] = current_entry
            return failure
        for cb in entry.get_invalidation_callbacks(key):
            cb()
        return failure

    def prefill(self, key: KT, value: VT, callback: Optional[Callable[[], None]]=None) -> None:
        if False:
            print('Hello World!')
        callbacks = (callback,) if callback else ()
        self.cache.set(key, value, callbacks=callbacks)
        self._pending_deferred_cache.pop(key, None)

    def invalidate(self, key: KT) -> None:
        if False:
            return 10
        'Delete a key, or tree of entries\n\n        If the cache is backed by a regular dict, then "key" must be of\n        the right type for this cache\n\n        If the cache is backed by a TreeCache, then "key" must be a tuple, but\n        may be of lower cardinality than the TreeCache - in which case the whole\n        subtree is deleted.\n        '
        self.check_thread()
        self.cache.del_multi(key)
        entry = self._pending_deferred_cache.pop(key, None)
        if entry:
            for iter_entry in iterate_tree_cache_entry(entry):
                for cb in iter_entry.get_invalidation_callbacks(key):
                    cb()

    def invalidate_all(self) -> None:
        if False:
            return 10
        self.check_thread()
        self.cache.clear()
        for (key, entry) in self._pending_deferred_cache.items():
            for cb in entry.get_invalidation_callbacks(key):
                cb()
        self._pending_deferred_cache.clear()

class CacheEntry(Generic[KT, VT], metaclass=abc.ABCMeta):
    """Abstract class for entries in `DeferredCache[KT, VT]`"""

    @abc.abstractmethod
    def deferred(self, key: KT) -> 'defer.Deferred[VT]':
        if False:
            while True:
                i = 10
        'Get a deferred that a caller can wait on to get the value at the\n        given key'
        ...

    @abc.abstractmethod
    def add_invalidation_callback(self, key: KT, callback: Optional[Callable[[], None]]) -> None:
        if False:
            while True:
                i = 10
        'Add an invalidation callback'
        ...

    @abc.abstractmethod
    def get_invalidation_callbacks(self, key: KT) -> Collection[Callable[[], None]]:
        if False:
            for i in range(10):
                print('nop')
        'Get all invalidation callbacks'
        ...

class CacheEntrySingle(CacheEntry[KT, VT]):
    """An implementation of `CacheEntry` wrapping a deferred that results in a
    single cache entry.
    """
    __slots__ = ['_deferred', '_callbacks']

    def __init__(self, deferred: 'defer.Deferred[VT]') -> None:
        if False:
            while True:
                i = 10
        self._deferred = ObservableDeferred(deferred, consumeErrors=True)
        self._callbacks: Set[Callable[[], None]] = set()

    def deferred(self, key: KT) -> 'defer.Deferred[VT]':
        if False:
            i = 10
            return i + 15
        return self._deferred.observe()

    def add_invalidation_callback(self, key: KT, callback: Optional[Callable[[], None]]) -> None:
        if False:
            return 10
        if callback is None:
            return
        self._callbacks.add(callback)

    def get_invalidation_callbacks(self, key: KT) -> Collection[Callable[[], None]]:
        if False:
            for i in range(10):
                print('nop')
        return self._callbacks

class CacheMultipleEntries(CacheEntry[KT, VT]):
    """Cache entry that is used for bulk lookups and insertions."""
    __slots__ = ['_deferred', '_callbacks', '_global_callbacks']

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self._deferred: Optional[ObservableDeferred[Dict[KT, VT]]] = None
        self._callbacks: Dict[KT, Set[Callable[[], None]]] = {}
        self._global_callbacks: Set[Callable[[], None]] = set()

    def deferred(self, key: KT) -> 'defer.Deferred[VT]':
        if False:
            while True:
                i = 10
        if not self._deferred:
            self._deferred = ObservableDeferred(defer.Deferred(), consumeErrors=True)
        return self._deferred.observe().addCallback(lambda res: res[key])

    def add_invalidation_callback(self, key: KT, callback: Optional[Callable[[], None]]) -> None:
        if False:
            i = 10
            return i + 15
        if callback is None:
            return
        self._callbacks.setdefault(key, set()).add(callback)

    def get_invalidation_callbacks(self, key: KT) -> Collection[Callable[[], None]]:
        if False:
            i = 10
            return i + 15
        return self._callbacks.get(key, set()) | self._global_callbacks

    def add_global_invalidation_callback(self, callback: Optional[Callable[[], None]]) -> None:
        if False:
            i = 10
            return i + 15
        'Add a callback for when any keys get invalidated.'
        if callback is None:
            return
        self._global_callbacks.add(callback)

    def complete_bulk(self, cache: DeferredCache[KT, VT], result: Dict[KT, VT]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Called when there is a result'
        for (key, value) in result.items():
            cache._completed_callback(value, self, key)
        if self._deferred:
            self._deferred.callback(result)

    def error_bulk(self, cache: DeferredCache[KT, VT], keys: Collection[KT], failure: Failure) -> None:
        if False:
            i = 10
            return i + 15
        'Called when bulk lookup failed.'
        for key in keys:
            cache._error_callback(failure, self, key)
        if self._deferred:
            self._deferred.errback(failure)