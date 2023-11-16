import logging
import time
from typing import Any, Callable, Dict, Generic, Tuple, TypeVar, Union
import attr
from sortedcontainers import SortedList
from synapse.util.caches import register_cache
logger = logging.getLogger(__name__)
SENTINEL: Any = object()
T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')

class TTLCache(Generic[KT, VT]):
    """A key/value cache implementation where each entry has its own TTL"""

    def __init__(self, cache_name: str, timer: Callable[[], float]=time.time):
        if False:
            while True:
                i = 10
        self._data: Dict[KT, _CacheEntry[KT, VT]] = {}
        self._expiry_list: SortedList[_CacheEntry[KT, VT]] = SortedList()
        self._timer = timer
        self._metrics = register_cache('ttl', cache_name, self, resizable=False)

    def set(self, key: KT, value: VT, ttl: float) -> None:
        if False:
            i = 10
            return i + 15
        'Add/update an entry in the cache\n\n        Args:\n            key: key for this entry\n            value: value for this entry\n            ttl: TTL for this entry, in seconds\n        '
        expiry = self._timer() + ttl
        self.expire()
        e = self._data.pop(key, SENTINEL)
        if e is not SENTINEL:
            assert isinstance(e, _CacheEntry)
            self._expiry_list.remove(e)
        entry = _CacheEntry(expiry_time=expiry, ttl=ttl, key=key, value=value)
        self._data[key] = entry
        self._expiry_list.add(entry)

    def get(self, key: KT, default: T=SENTINEL) -> Union[VT, T]:
        if False:
            print('Hello World!')
        'Get a value from the cache\n\n        Args:\n            key: key to look up\n            default: default value to return, if key is not found. If not set, and the\n                key is not found, a KeyError will be raised\n\n        Returns:\n            value from the cache, or the default\n        '
        self.expire()
        e = self._data.get(key, SENTINEL)
        if e is SENTINEL:
            self._metrics.inc_misses()
            if default is SENTINEL:
                raise KeyError(key)
            return default
        assert isinstance(e, _CacheEntry)
        self._metrics.inc_hits()
        return e.value

    def get_with_expiry(self, key: KT) -> Tuple[VT, float, float]:
        if False:
            print('Hello World!')
        'Get a value, and its expiry time, from the cache\n\n        Args:\n            key: key to look up\n\n        Returns:\n            A tuple of  the value from the cache, the expiry time and the TTL\n\n        Raises:\n            KeyError if the entry is not found\n        '
        self.expire()
        try:
            e = self._data[key]
        except KeyError:
            self._metrics.inc_misses()
            raise
        self._metrics.inc_hits()
        return (e.value, e.expiry_time, e.ttl)

    def pop(self, key: KT, default: T=SENTINEL) -> Union[VT, T]:
        if False:
            for i in range(10):
                print('nop')
        'Remove a value from the cache\n\n        If key is in the cache, remove it and return its value, else return default.\n        If default is not given and key is not in the cache, a KeyError is raised.\n\n        Args:\n            key: key to look up\n            default: default value to return, if key is not found. If not set, and the\n                key is not found, a KeyError will be raised\n\n        Returns:\n            value from the cache, or the default\n        '
        self.expire()
        e = self._data.pop(key, SENTINEL)
        if e is SENTINEL:
            self._metrics.inc_misses()
            if default is SENTINEL:
                raise KeyError(key)
            return default
        assert isinstance(e, _CacheEntry)
        self._expiry_list.remove(e)
        self._metrics.inc_hits()
        return e.value

    def __getitem__(self, key: KT) -> VT:
        if False:
            for i in range(10):
                print('nop')
        return self.get(key)

    def __delitem__(self, key: KT) -> None:
        if False:
            i = 10
            return i + 15
        self.pop(key)

    def __contains__(self, key: KT) -> bool:
        if False:
            i = 10
            return i + 15
        return key in self._data

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        self.expire()
        return len(self._data)

    def expire(self) -> None:
        if False:
            print('Hello World!')
        'Run the expiry on the cache. Any entries whose expiry times are due will\n        be removed\n        '
        now = self._timer()
        while self._expiry_list:
            first_entry = self._expiry_list[0]
            if first_entry.expiry_time - now > 0.0:
                break
            del self._data[first_entry.key]
            del self._expiry_list[0]

@attr.s(frozen=True, slots=True, auto_attribs=True)
class _CacheEntry(Generic[KT, VT]):
    """TTLCache entry"""
    expiry_time: float
    ttl: float
    key: KT
    value: VT