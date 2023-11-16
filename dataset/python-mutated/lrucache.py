import logging
import math
import threading
import weakref
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Collection, Dict, Generic, Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast, overload
from typing_extensions import Literal
from twisted.internet import reactor
from twisted.internet.interfaces import IReactorTime
from synapse.config import cache as cache_config
from synapse.metrics.background_process_metrics import wrap_as_background_process
from synapse.metrics.jemalloc import get_jemalloc_stats
from synapse.util import Clock, caches
from synapse.util.caches import CacheMetric, EvictionReason, register_cache
from synapse.util.caches.treecache import TreeCache, iterate_tree_cache_entry, iterate_tree_cache_items
from synapse.util.linked_list import ListNode
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)
try:
    from pympler.asizeof import Asizer

    def _get_size_of(val: Any, *, recurse: bool=True) -> int:
        if False:
            i = 10
            return i + 15
        'Get an estimate of the size in bytes of the object.\n\n        Args:\n            val: The object to size.\n            recurse: If true will include referenced values in the size,\n                otherwise only sizes the given object.\n        '
        if val in ((), None, ''):
            return 0
        sizer = Asizer()
        sizer.exclude_refs((), None, '')
        return sizer.asizeof(val, limit=100 if recurse else 0)
except ImportError:

    def _get_size_of(val: Any, *, recurse: bool=True) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 0
FT = TypeVar('FT', bound=Callable[..., Any])
KT = TypeVar('KT')
VT = TypeVar('VT')
T = TypeVar('T')

class _TimedListNode(ListNode[T]):
    """A `ListNode` that tracks last access time."""
    __slots__ = ['last_access_ts_secs']

    def update_last_access(self, clock: Clock) -> None:
        if False:
            return 10
        self.last_access_ts_secs = int(clock.time())
USE_GLOBAL_LIST = False
GLOBAL_ROOT = ListNode['_Node'].create_root_node()

@wrap_as_background_process('LruCache._expire_old_entries')
async def _expire_old_entries(clock: Clock, expiry_seconds: float, autotune_config: Optional[dict]) -> None:
    """Walks the global cache list to find cache entries that haven't been
    accessed in the given number of seconds, or if a given memory threshold has been breached.
    """
    if autotune_config:
        max_cache_memory_usage = autotune_config['max_cache_memory_usage']
        target_cache_memory_usage = autotune_config['target_cache_memory_usage']
        min_cache_ttl = autotune_config['min_cache_ttl'] / 1000
    now = int(clock.time())
    node = GLOBAL_ROOT.prev_node
    assert node is not None
    i = 0
    logger.debug('Searching for stale caches')
    evicting_due_to_memory = False
    jemalloc_interface = get_jemalloc_stats()
    if jemalloc_interface and autotune_config:
        try:
            jemalloc_interface.refresh_stats()
            mem_usage = jemalloc_interface.get_stat('allocated')
            if mem_usage > max_cache_memory_usage:
                logger.info('Begin memory-based cache eviction.')
                evicting_due_to_memory = True
        except Exception:
            logger.warning('Unable to read allocated memory, skipping memory-based cache eviction.')
    while node is not GLOBAL_ROOT:
        assert isinstance(node, _TimedListNode)
        if node.last_access_ts_secs > now - expiry_seconds and (not evicting_due_to_memory):
            break
        if evicting_due_to_memory and now - node.last_access_ts_secs < min_cache_ttl:
            break
        cache_entry = node.get_cache_entry()
        next_node = node.prev_node
        assert next_node is not None
        assert cache_entry is not None
        cache_entry.drop_from_cache()
        if jemalloc_interface and evicting_due_to_memory and ((i + 1) % 100 == 0):
            try:
                jemalloc_interface.refresh_stats()
                mem_usage = jemalloc_interface.get_stat('allocated')
                if mem_usage < target_cache_memory_usage:
                    evicting_due_to_memory = False
                    logger.info('Stop memory-based cache eviction.')
            except Exception:
                logger.warning('Unable to read allocated memory, this may affect memory-based cache eviction.')
                evicting_due_to_memory = False
        if (i + 1) % 10000 == 0:
            logger.debug('Waiting during drop')
            if node.last_access_ts_secs > now - expiry_seconds:
                await clock.sleep(0.5)
            else:
                await clock.sleep(0)
            logger.debug('Waking during drop')
        node = next_node
        if node.prev_node is None:
            break
        i += 1
    logger.info('Dropped %d items from caches', i)

def setup_expire_lru_cache_entries(hs: 'HomeServer') -> None:
    if False:
        return 10
    'Start a background job that expires all cache entries if they have not\n    been accessed for the given number of seconds, or if a given memory usage threshold has been\n    breached.\n    '
    if not hs.config.caches.expiry_time_msec and (not hs.config.caches.cache_autotuning):
        return
    if hs.config.caches.expiry_time_msec:
        expiry_time = hs.config.caches.expiry_time_msec / 1000
        logger.info('Expiring LRU caches after %d seconds', expiry_time)
    else:
        expiry_time = math.inf
    global USE_GLOBAL_LIST
    USE_GLOBAL_LIST = True
    clock = hs.get_clock()
    clock.looping_call(_expire_old_entries, 30 * 1000, clock, expiry_time, hs.config.caches.cache_autotuning)

class _Node(Generic[KT, VT]):
    __slots__ = ['_list_node', '_global_list_node', '_cache', 'key', 'value', 'callbacks', 'memory']

    def __init__(self, root: 'ListNode[_Node]', key: KT, value: VT, cache: 'weakref.ReferenceType[LruCache[KT, VT]]', clock: Clock, callbacks: Collection[Callable[[], None]]=(), prune_unread_entries: bool=True):
        if False:
            for i in range(10):
                print('nop')
        self._list_node = ListNode.insert_after(self, root)
        self._global_list_node: Optional[_TimedListNode] = None
        if USE_GLOBAL_LIST and prune_unread_entries:
            self._global_list_node = _TimedListNode.insert_after(self, GLOBAL_ROOT)
            self._global_list_node.update_last_access(clock)
        self._cache = cache
        self.key = key
        self.value = value
        self.callbacks: Optional[List[Callable[[], None]]] = None
        self.add_callbacks(callbacks)
        self.memory = 0
        if caches.TRACK_MEMORY_USAGE:
            self.memory = _get_size_of(key) + _get_size_of(value) + _get_size_of(self._list_node, recurse=False) + _get_size_of(self.callbacks, recurse=False) + _get_size_of(self, recurse=False)
            self.memory += _get_size_of(self.memory, recurse=False)
            if self._global_list_node:
                self.memory += _get_size_of(self._global_list_node, recurse=False)
                self.memory += _get_size_of(self._global_list_node.last_access_ts_secs)

    def add_callbacks(self, callbacks: Collection[Callable[[], None]]) -> None:
        if False:
            print('Hello World!')
        'Add to stored list of callbacks, removing duplicates.'
        if not callbacks:
            return
        if not self.callbacks:
            self.callbacks = []
        for callback in callbacks:
            if callback not in self.callbacks:
                self.callbacks.append(callback)

    def run_and_clear_callbacks(self) -> None:
        if False:
            while True:
                i = 10
        'Run all callbacks and clear the stored list of callbacks. Used when\n        the node is being deleted.\n        '
        if not self.callbacks:
            return
        for callback in self.callbacks:
            callback()
        self.callbacks = None

    def drop_from_cache(self) -> None:
        if False:
            i = 10
            return i + 15
        'Drop this node from the cache.\n\n        Ensures that the entry gets removed from the cache and that we get\n        removed from all lists.\n        '
        cache = self._cache()
        if cache is None or cache.pop(self.key, _Sentinel.sentinel) is _Sentinel.sentinel:
            self.drop_from_lists()

    def drop_from_lists(self) -> None:
        if False:
            return 10
        'Remove this node from the cache lists.'
        self._list_node.remove_from_list()
        if self._global_list_node:
            self._global_list_node.remove_from_list()

    def move_to_front(self, clock: Clock, cache_list_root: ListNode) -> None:
        if False:
            i = 10
            return i + 15
        'Moves this node to the front of all the lists its in.'
        self._list_node.move_after(cache_list_root)
        if self._global_list_node:
            self._global_list_node.move_after(GLOBAL_ROOT)
            self._global_list_node.update_last_access(clock)

class _Sentinel(Enum):
    sentinel = object()

class LruCache(Generic[KT, VT]):
    """
    Least-recently-used cache, supporting prometheus metrics and invalidation callbacks.

    If cache_type=TreeCache, all keys must be tuples.
    """

    def __init__(self, max_size: int, cache_name: Optional[str]=None, cache_type: Type[Union[dict, TreeCache]]=dict, size_callback: Optional[Callable[[VT], int]]=None, metrics_collection_callback: Optional[Callable[[], None]]=None, apply_cache_factor_from_config: bool=True, clock: Optional[Clock]=None, prune_unread_entries: bool=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Args:\n            max_size: The maximum amount of entries the cache can hold\n\n            cache_name: The name of this cache, for the prometheus metrics. If unset,\n                no metrics will be reported on this cache.\n\n            cache_type:\n                type of underlying cache to be used. Typically one of dict\n                or TreeCache.\n\n            size_callback:\n\n            metrics_collection_callback:\n                metrics collection callback. This is called early in the metrics\n                collection process, before any of the metrics registered with the\n                prometheus Registry are collected, so can be used to update any dynamic\n                metrics.\n\n                Ignored if cache_name is None.\n\n            apply_cache_factor_from_config: If true, `max_size` will be\n                multiplied by a cache factor derived from the homeserver config\n\n            clock:\n\n            prune_unread_entries: If True, cache entries that haven't been read recently\n                will be evicted from the cache in the background. Set to False to\n                opt-out of this behaviour.\n        "
        if clock is None:
            real_clock = Clock(cast(IReactorTime, reactor))
        else:
            real_clock = clock
        cache: Union[Dict[KT, _Node[KT, VT]], TreeCache] = cache_type()
        self.cache = cache
        self.apply_cache_factor_from_config = apply_cache_factor_from_config
        self._original_max_size = max_size
        if apply_cache_factor_from_config:
            self.max_size = int(max_size * cache_config.properties.default_factor_size)
        else:
            self.max_size = int(max_size)
        self._on_resize: Optional[Callable[[], None]] = None
        if cache_name is not None:
            metrics: Optional[CacheMetric] = register_cache('lru_cache', cache_name, self, collect_callback=metrics_collection_callback)
        else:
            metrics = None
        self.metrics = metrics
        weak_ref_to_self = weakref.ref(self)
        list_root = ListNode[_Node[KT, VT]].create_root_node()
        lock = threading.Lock()

        def evict() -> None:
            if False:
                print('Hello World!')
            while cache_len() > self.max_size:
                todelete = list_root.prev_node
                assert todelete is not None
                node = todelete.get_cache_entry()
                assert node is not None
                evicted_len = delete_node(node)
                cache.pop(node.key, None)
                if metrics:
                    metrics.inc_evictions(EvictionReason.size, evicted_len)

        def synchronized(f: FT) -> FT:
            if False:
                i = 10
                return i + 15

            @wraps(f)
            def inner(*args: Any, **kwargs: Any) -> Any:
                if False:
                    return 10
                with lock:
                    return f(*args, **kwargs)
            return cast(FT, inner)
        cached_cache_len = [0]
        if size_callback is not None:

            def cache_len() -> int:
                if False:
                    while True:
                        i = 10
                return cached_cache_len[0]
        else:

            def cache_len() -> int:
                if False:
                    return 10
                return len(cache)
        self.len = synchronized(cache_len)

        def add_node(key: KT, value: VT, callbacks: Collection[Callable[[], None]]=()) -> None:
            if False:
                return 10
            node: _Node[KT, VT] = _Node(list_root, key, value, weak_ref_to_self, real_clock, callbacks, prune_unread_entries)
            cache[key] = node
            if size_callback:
                cached_cache_len[0] += size_callback(node.value)
            if caches.TRACK_MEMORY_USAGE and metrics:
                metrics.inc_memory_usage(node.memory)

        def move_node_to_front(node: _Node[KT, VT]) -> None:
            if False:
                while True:
                    i = 10
            node.move_to_front(real_clock, list_root)

        def delete_node(node: _Node[KT, VT]) -> int:
            if False:
                i = 10
                return i + 15
            node.drop_from_lists()
            deleted_len = 1
            if size_callback:
                deleted_len = size_callback(node.value)
                cached_cache_len[0] -= deleted_len
            node.run_and_clear_callbacks()
            if caches.TRACK_MEMORY_USAGE and metrics:
                metrics.dec_memory_usage(node.memory)
            return deleted_len

        @overload
        def cache_get(key: KT, default: Literal[None]=None, callbacks: Collection[Callable[[], None]]=..., update_metrics: bool=..., update_last_access: bool=...) -> Optional[VT]:
            if False:
                print('Hello World!')
            ...

        @overload
        def cache_get(key: KT, default: T, callbacks: Collection[Callable[[], None]]=..., update_metrics: bool=..., update_last_access: bool=...) -> Union[T, VT]:
            if False:
                for i in range(10):
                    print('nop')
            ...

        @synchronized
        def cache_get(key: KT, default: Optional[T]=None, callbacks: Collection[Callable[[], None]]=(), update_metrics: bool=True, update_last_access: bool=True) -> Union[None, T, VT]:
            if False:
                while True:
                    i = 10
            'Look up a key in the cache\n\n            Args:\n                key\n                default\n                callbacks: A collection of callbacks that will fire when the\n                    node is removed from the cache (either due to invalidation\n                    or expiry).\n                update_metrics: Whether to update the hit rate metrics\n                update_last_access: Whether to update the last access metrics\n                    on a node if successfully fetched. These metrics are used\n                    to determine when to remove the node from the cache. Set\n                    to False if this fetch should *not* prevent a node from\n                    being expired.\n            '
            node = cache.get(key, None)
            if node is not None:
                if update_last_access:
                    move_node_to_front(node)
                node.add_callbacks(callbacks)
                if update_metrics and metrics:
                    metrics.inc_hits()
                return node.value
            else:
                if update_metrics and metrics:
                    metrics.inc_misses()
                return default

        @overload
        def cache_get_multi(key: tuple, default: Literal[None]=None, update_metrics: bool=True) -> Union[None, Iterable[Tuple[KT, VT]]]:
            if False:
                for i in range(10):
                    print('nop')
            ...

        @overload
        def cache_get_multi(key: tuple, default: T, update_metrics: bool=True) -> Union[T, Iterable[Tuple[KT, VT]]]:
            if False:
                while True:
                    i = 10
            ...

        @synchronized
        def cache_get_multi(key: tuple, default: Optional[T]=None, update_metrics: bool=True) -> Union[None, T, Iterable[Tuple[KT, VT]]]:
            if False:
                print('Hello World!')
            'Returns a generator yielding all entries under the given key.\n\n            Can only be used if backed by a tree cache.\n\n            Example:\n\n                cache = LruCache(10, cache_type=TreeCache)\n                cache[(1, 1)] = "a"\n                cache[(1, 2)] = "b"\n                cache[(2, 1)] = "c"\n\n                items = cache.get_multi((1,))\n                assert list(items) == [((1, 1), "a"), ((1, 2), "b")]\n\n            Returns:\n                Either default if the key doesn\'t exist, or a generator of the\n                key/value pairs.\n            '
            assert isinstance(cache, TreeCache)
            node = cache.get(key, None)
            if node is not None:
                if update_metrics and metrics:
                    metrics.inc_hits()
                return ((full_key, lru_node.value) for (full_key, lru_node) in iterate_tree_cache_items(key, node))
            else:
                if update_metrics and metrics:
                    metrics.inc_misses()
                return default

        @synchronized
        def cache_set(key: KT, value: VT, callbacks: Collection[Callable[[], None]]=()) -> None:
            if False:
                i = 10
                return i + 15
            node = cache.get(key, None)
            if node is not None:
                if value != node.value:
                    node.run_and_clear_callbacks()
                if size_callback:
                    cached_cache_len[0] -= size_callback(node.value)
                    cached_cache_len[0] += size_callback(value)
                node.add_callbacks(callbacks)
                move_node_to_front(node)
                node.value = value
            else:
                add_node(key, value, set(callbacks))
            evict()

        @synchronized
        def cache_set_default(key: KT, value: VT) -> VT:
            if False:
                for i in range(10):
                    print('nop')
            node = cache.get(key, None)
            if node is not None:
                return node.value
            else:
                add_node(key, value)
                evict()
                return value

        @overload
        def cache_pop(key: KT, default: Literal[None]=None) -> Optional[VT]:
            if False:
                for i in range(10):
                    print('nop')
            ...

        @overload
        def cache_pop(key: KT, default: T) -> Union[T, VT]:
            if False:
                i = 10
                return i + 15
            ...

        @synchronized
        def cache_pop(key: KT, default: Optional[T]=None) -> Union[None, T, VT]:
            if False:
                while True:
                    i = 10
            node = cache.get(key, None)
            if node:
                evicted_len = delete_node(node)
                cache.pop(node.key, None)
                if metrics:
                    metrics.inc_evictions(EvictionReason.invalidation, evicted_len)
                return node.value
            else:
                return default

        @synchronized
        def cache_del_multi(key: KT) -> None:
            if False:
                i = 10
                return i + 15
            'Delete an entry, or tree of entries\n\n            If the LruCache is backed by a regular dict, then "key" must be of\n            the right type for this cache\n\n            If the LruCache is backed by a TreeCache, then "key" must be a tuple, but\n            may be of lower cardinality than the TreeCache - in which case the whole\n            subtree is deleted.\n            '
            popped = cache.pop(key, None)
            if popped is None:
                return
            for leaf in iterate_tree_cache_entry(popped):
                delete_node(leaf)

        @synchronized
        def cache_clear() -> None:
            if False:
                for i in range(10):
                    print('nop')
            for node in cache.values():
                node.run_and_clear_callbacks()
                node.drop_from_lists()
            assert list_root.next_node == list_root
            assert list_root.prev_node == list_root
            cache.clear()
            if size_callback:
                cached_cache_len[0] = 0
            if caches.TRACK_MEMORY_USAGE and metrics:
                metrics.clear_memory_usage()

        @synchronized
        def cache_contains(key: KT) -> bool:
            if False:
                print('Hello World!')
            return key in cache
        self._on_resize = evict
        self.get = cache_get
        self.set = cache_set
        self.setdefault = cache_set_default
        self.pop = cache_pop
        self.del_multi = cache_del_multi
        if cache_type is TreeCache:
            self.get_multi = cache_get_multi
        self.invalidate = cache_del_multi
        self.len = synchronized(cache_len)
        self.contains = cache_contains
        self.clear = cache_clear

    def __getitem__(self, key: KT) -> VT:
        if False:
            print('Hello World!')
        result = self.get(key, _Sentinel.sentinel)
        if result is _Sentinel.sentinel:
            raise KeyError()
        else:
            return result

    def __setitem__(self, key: KT, value: VT) -> None:
        if False:
            return 10
        self.set(key, value)

    def __delitem__(self, key: KT, value: VT) -> None:
        if False:
            for i in range(10):
                print('nop')
        result = self.pop(key, _Sentinel.sentinel)
        if result is _Sentinel.sentinel:
            raise KeyError()

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.len()

    def __contains__(self, key: KT) -> bool:
        if False:
            i = 10
            return i + 15
        return self.contains(key)

    def set_cache_factor(self, factor: float) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set the cache factor for this individual cache.\n\n        This will trigger a resize if it changes, which may require evicting\n        items from the cache.\n        '
        if not self.apply_cache_factor_from_config:
            return
        new_size = int(self._original_max_size * factor)
        if new_size != self.max_size:
            self.max_size = new_size
            if self._on_resize:
                self._on_resize()

    def __del__(self) -> None:
        if False:
            while True:
                i = 10
        self.clear()

class AsyncLruCache(Generic[KT, VT]):
    """
    An asynchronous wrapper around a subset of the LruCache API.

    On its own this doesn't change the behaviour but allows subclasses that
    utilize external cache systems that require await behaviour to be created.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        if False:
            while True:
                i = 10
        self._lru_cache: LruCache[KT, VT] = LruCache(*args, **kwargs)

    async def get(self, key: KT, default: Optional[T]=None, update_metrics: bool=True) -> Optional[VT]:
        return self._lru_cache.get(key, update_metrics=update_metrics)

    async def get_external(self, key: KT, default: Optional[T]=None, update_metrics: bool=True) -> Optional[VT]:
        return None

    def get_local(self, key: KT, default: Optional[T]=None, update_metrics: bool=True) -> Optional[VT]:
        if False:
            i = 10
            return i + 15
        return self._lru_cache.get(key, update_metrics=update_metrics)

    async def set(self, key: KT, value: VT) -> None:
        self.set_local(key, value)
        await self.set_external(key, value)

    async def set_external(self, key: KT, value: VT) -> None:
        pass

    def set_local(self, key: KT, value: VT) -> None:
        if False:
            print('Hello World!')
        self._lru_cache.set(key, value)

    async def invalidate(self, key: KT) -> None:
        return self._lru_cache.invalidate(key)

    def invalidate_local(self, key: KT) -> None:
        if False:
            i = 10
            return i + 15
        'Remove an entry from the local cache\n\n        This variant of `invalidate` is useful if we know that the external\n        cache has already been invalidated.\n        '
        return self._lru_cache.invalidate(key)

    async def contains(self, key: KT) -> bool:
        return self._lru_cache.contains(key)

    def clear(self) -> None:
        if False:
            i = 10
            return i + 15
        self._lru_cache.clear()