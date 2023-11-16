import enum
import logging
import threading
from typing import Dict, Generic, Iterable, Optional, Set, Tuple, TypeVar, Union
import attr
from typing_extensions import Literal
from synapse.util.caches.lrucache import LruCache
from synapse.util.caches.treecache import TreeCache
logger = logging.getLogger(__name__)
KT = TypeVar('KT')
DKT = TypeVar('DKT')
DV = TypeVar('DV')

@attr.s(slots=True, frozen=True, auto_attribs=True)
class DictionaryEntry(Generic[DKT, DV]):
    """Returned when getting an entry from the cache

    If `full` is true then `known_absent` will be the empty set.

    Attributes:
        full: Whether the cache has the full or dict or just some keys.
            If not full then not all requested keys will necessarily be present
            in `value`
        known_absent: Keys that were looked up in the dict and were not there.
        value: The full or partial dict value
    """
    full: bool
    known_absent: Set[DKT]
    value: Dict[DKT, DV]

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self.value)

class _FullCacheKey(enum.Enum):
    """The key we use to cache the full dict."""
    KEY = object()

class _Sentinel(enum.Enum):
    sentinel = object()

class _PerKeyValue(Generic[DV]):
    """The cached value of a dictionary key. If `value` is the sentinel,
    indicates that the requested key is known to *not* be in the full dict.
    """
    __slots__ = ['value']

    def __init__(self, value: Union[DV, Literal[_Sentinel.sentinel]]) -> None:
        if False:
            return 10
        self.value = value

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return 1

class DictionaryCache(Generic[KT, DKT, DV]):
    """Caches key -> dictionary lookups, supporting caching partial dicts, i.e.
    fetching a subset of dictionary keys for a particular key.

    This cache has two levels of key. First there is the "cache key" (of type
    `KT`), which maps to a dict. The keys to that dict are the "dict key" (of
    type `DKT`). The overall structure is therefore `KT->DKT->DV`. For
    example, it might look like:

       {
           1: { 1: "a", 2: "b" },
           2: { 1: "c" },
       }

    It is possible to look up either individual dict keys, or the *complete*
    dict for a given cache key.

    Each dict item, and the complete dict is treated as a separate LRU
    entry for the purpose of cache expiry. For example, given:
        dict_cache.get(1, None)  -> DictionaryEntry({1: "a", 2: "b"})
        dict_cache.get(1, [1])  -> DictionaryEntry({1: "a"})
        dict_cache.get(1, [2])  -> DictionaryEntry({2: "b"})

    ... then the cache entry for the complete dict will expire first,
    followed by the cache entry for the '1' dict key, and finally that
    for the '2' dict key.
    """

    def __init__(self, name: str, max_entries: int=1000):
        if False:
            return 10
        self.cache: LruCache[Tuple[KT, Union[DKT, Literal[_FullCacheKey.KEY]]], Union[_PerKeyValue, Dict[DKT, DV]]] = LruCache(max_size=max_entries, cache_name=name, cache_type=TreeCache, size_callback=len)
        self.name = name
        self.sequence = 0
        self.thread: Optional[threading.Thread] = None

    def check_thread(self) -> None:
        if False:
            i = 10
            return i + 15
        expected_thread = self.thread
        if expected_thread is None:
            self.thread = threading.current_thread()
        elif expected_thread is not threading.current_thread():
            raise ValueError('Cache objects can only be accessed from the main thread')

    def get(self, key: KT, dict_keys: Optional[Iterable[DKT]]=None) -> DictionaryEntry:
        if False:
            return 10
        "Fetch an entry out of the cache\n\n        Args:\n            key\n            dict_keys: If given a set of keys then return only those keys\n                that exist in the cache. If None then returns the full dict\n                if it is in the cache.\n\n        Returns:\n            If `dict_keys` is not None then `DictionaryEntry` will contain include\n            the keys that are in the cache.\n\n            If None then will either return the full dict if in the cache, or the\n            empty dict (with `full` set to False) if it isn't.\n        "
        if dict_keys is None:
            return self._get_full_dict(key)
        values = {}
        known_absent = set()
        missing = []
        for dict_key in dict_keys:
            entry = self.cache.get((key, dict_key), _Sentinel.sentinel)
            if entry is _Sentinel.sentinel:
                missing.append(dict_key)
                continue
            assert isinstance(entry, _PerKeyValue)
            if entry.value is _Sentinel.sentinel:
                known_absent.add(dict_key)
            else:
                values[dict_key] = entry.value
        if not missing:
            return DictionaryEntry(False, known_absent, values)
        entry = self.cache.get((key, _FullCacheKey.KEY), _Sentinel.sentinel, update_last_access=False)
        if entry is _Sentinel.sentinel:
            return DictionaryEntry(False, known_absent, values)
        assert isinstance(entry, dict)
        for dict_key in missing:
            value = entry.get(dict_key, _Sentinel.sentinel)
            self.cache[key, dict_key] = _PerKeyValue(value)
            if value is not _Sentinel.sentinel:
                values[dict_key] = value
        return DictionaryEntry(True, set(), values)

    def _get_full_dict(self, key: KT) -> DictionaryEntry:
        if False:
            for i in range(10):
                print('nop')
        'Fetch the full dict for the given key.'
        entry = self.cache.get((key, _FullCacheKey.KEY), _Sentinel.sentinel)
        if entry is not _Sentinel.sentinel:
            assert isinstance(entry, dict)
            return DictionaryEntry(True, set(), entry)
        return DictionaryEntry(False, set(), {})

    def invalidate(self, key: KT) -> None:
        if False:
            print('Hello World!')
        self.check_thread()
        self.sequence += 1
        self.cache.del_multi((key,))

    def invalidate_all(self) -> None:
        if False:
            return 10
        self.check_thread()
        self.sequence += 1
        self.cache.clear()

    def update(self, sequence: int, key: KT, value: Dict[DKT, DV], fetched_keys: Optional[Iterable[DKT]]=None) -> None:
        if False:
            while True:
                i = 10
        'Updates the entry in the cache.\n\n        Note: This does *not* invalidate any existing entries for the `key`.\n        In particular, if we add an entry for the cached "full dict" with\n        `fetched_keys=None`, existing entries for individual dict keys are\n        not invalidated. Likewise, adding entries for individual keys does\n        not invalidate any cached value for the full dict.\n\n        In other words: if the underlying data is *changed*, the cache must\n        be explicitly invalidated via `.invalidate()`.\n\n        Args:\n            sequence\n            key\n            value: The value to update the cache with.\n            fetched_keys: All of the dictionary keys which were\n                fetched from the database.\n\n                If None, this is the complete value for key K. Otherwise, it\n                is used to infer a list of keys which we know don\'t exist in\n                the full dict.\n        '
        self.check_thread()
        if self.sequence == sequence:
            if fetched_keys is None:
                self.cache[key, _FullCacheKey.KEY] = value
            else:
                self._update_subset(key, value, fetched_keys)

    def _update_subset(self, key: KT, value: Dict[DKT, DV], fetched_keys: Iterable[DKT]) -> None:
        if False:
            while True:
                i = 10
        'Add the given dictionary values as explicit keys in the cache.\n\n        Args:\n            key: top-level cache key\n            value: The dictionary with all the values that we should cache\n            fetched_keys: The full set of dict keys that were looked up. Any keys\n                here not in `value` should be marked as "known absent".\n        '
        for (dict_key, dict_value) in value.items():
            self.cache[key, dict_key] = _PerKeyValue(dict_value)
        for dict_key in fetched_keys:
            if dict_key in value:
                continue
            self.cache[key, dict_key] = _PerKeyValue(_Sentinel.sentinel)