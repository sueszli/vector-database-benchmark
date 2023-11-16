"""Utilities for memoization

Note: a memoized function should always return an _immutable_
result to avoid later modifications polluting cached results.
"""
from collections import OrderedDict
from functools import wraps

class DoNotCache:
    """Wrapper to return a result without caching it.

    In a function decorated with `@lru_cache_key`:

        return DoNotCache(result)

    is equivalent to:

        return result  # but don't cache it!
    """

    def __init__(self, result):
        if False:
            i = 10
            return i + 15
        self.result = result

class LRUCache:
    """A simple Least-Recently-Used (LRU) cache with a max size"""

    def __init__(self, maxsize=1024):
        if False:
            for i in range(10):
                print('nop')
        self._cache = OrderedDict()
        self.maxsize = maxsize

    def __contains__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return key in self._cache

    def get(self, key, default=None):
        if False:
            return 10
        'Get an item from the cache'
        if key in self._cache:
            result = self._cache[key]
            self._cache.move_to_end(key)
            return result
        return default

    def set(self, key, value):
        if False:
            return 10
        'Store an entry in the cache\n\n        Purges oldest entry if cache is full\n        '
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
    __getitem__ = get
    __setitem__ = set

def lru_cache_key(key_func, maxsize=1024):
    if False:
        while True:
            i = 10
    'Like functools.lru_cache, but takes a custom key function,\n    as seen in sorted(key=func).\n\n    Useful for non-hashable arguments which have a known hashable equivalent (e.g. sets, lists),\n    or mutable objects where only immutable fields might be used\n    (e.g. User, where only username affects output).\n\n    For safety: Cached results should always be immutable,\n    such as using `frozenset` instead of mutable `set`.\n\n    Example:\n\n        @lru_cache_key(lambda user: user.name)\n        def func_user(user):\n            # output only varies by name\n\n    Args:\n        key (callable):\n            Should have the same signature as the decorated function.\n            Returns a hashable key to use in the cache\n        maxsize (int):\n            The maximum size of the cache.\n    '

    def cache_func(func):
        if False:
            i = 10
            return i + 15
        cache = LRUCache(maxsize=maxsize)

        @wraps(func)
        def cached(*args, **kwargs):
            if False:
                while True:
                    i = 10
            cache_key = key_func(*args, **kwargs)
            if cache_key in cache:
                return cache[cache_key]
            else:
                result = func(*args, **kwargs)
                if isinstance(result, DoNotCache):
                    result = result.result
                else:
                    cache[cache_key] = result
            return result
        return cached
    return cache_func

class FrozenDict(dict):
    """A frozen dictionary subclass

    Immutable and hashable, so it can be used as a cache key

    Values will be frozen with `.freeze(value)`
    and must be hashable after freezing.

    Not rigorous, but enough for our purposes.
    """
    _hash = None

    def __init__(self, d):
        if False:
            while True:
                i = 10
        dict_set = dict.__setitem__
        for (key, value) in d.items():
            dict.__setitem__(self, key, self._freeze(value))

    def _freeze(self, item):
        if False:
            i = 10
            return i + 15
        'Make values of a dict hashable\n        - list, set -> frozenset\n        - dict -> recursive _FrozenDict\n        - anything else: assumed hashable\n        '
        if isinstance(item, FrozenDict):
            return item
        elif isinstance(item, list):
            return tuple((self._freeze(e) for e in item))
        elif isinstance(item, set):
            return frozenset(item)
        elif isinstance(item, dict):
            return FrozenDict(item)
        else:
            return item

    def __setitem__(self, key):
        if False:
            print('Hello World!')
        raise RuntimeError('Cannot modify frozen {type(self).__name__}')

    def update(self, other):
        if False:
            print('Hello World!')
        raise RuntimeError('Cannot modify frozen {type(self).__name__}')

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        'Cache hash because we are immutable'
        if self._hash is None:
            self._hash = hash(tuple(((key, value) for (key, value) in self.items())))
        return self._hash