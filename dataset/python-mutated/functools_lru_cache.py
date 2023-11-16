from __future__ import absolute_import
import functools
from collections import namedtuple
from threading import RLock
_CacheInfo = namedtuple('_CacheInfo', ['hits', 'misses', 'maxsize', 'currsize'])

@functools.wraps(functools.update_wrapper)
def update_wrapper(wrapper, wrapped, assigned=functools.WRAPPER_ASSIGNMENTS, updated=functools.WRAPPER_UPDATES):
    if False:
        for i in range(10):
            print('nop')
    '\n    Patch two bugs in functools.update_wrapper.\n    '
    assigned = tuple((attr for attr in assigned if hasattr(wrapped, attr)))
    wrapper = functools.update_wrapper(wrapper, wrapped, assigned, updated)
    wrapper.__wrapped__ = wrapped
    return wrapper

class _HashedSeq(list):
    __slots__ = 'hashvalue'

    def __init__(self, tup, hash=hash):
        if False:
            return 10
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        if False:
            print('Hello World!')
        return self.hashvalue

def _make_key(args, kwds, typed, kwd_mark=(object(),), fasttypes=set([int, str, frozenset, type(None)]), sorted=sorted, tuple=tuple, type=type, len=len):
    if False:
        return 10
    'Make a cache key from optionally typed positional and keyword arguments'
    key = args
    if kwds:
        sorted_items = sorted(kwds.items())
        key += kwd_mark
        for item in sorted_items:
            key += item
    if typed:
        key += tuple((type(v) for v in args))
        if kwds:
            key += tuple((type(v) for (k, v) in sorted_items))
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)

def lru_cache(maxsize=100, typed=False):
    if False:
        return 10
    'Least-recently-used cache decorator.\n\n    If *maxsize* is set to None, the LRU features are disabled and the cache\n    can grow without bound.\n\n    If *typed* is True, arguments of different types will be cached separately.\n    For example, f(3.0) and f(3) will be treated as distinct calls with\n    distinct results.\n\n    Arguments to the cached function must be hashable.\n\n    View the cache statistics named tuple (hits, misses, maxsize, currsize) with\n    f.cache_info().  Clear the cache and statistics with f.cache_clear().\n    Access the underlying function with f.__wrapped__.\n\n    See:  http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used\n\n    '

    def decorating_function(user_function):
        if False:
            for i in range(10):
                print('nop')
        cache = dict()
        stats = [0, 0]
        (HITS, MISSES) = (0, 1)
        make_key = _make_key
        cache_get = cache.get
        _len = len
        lock = RLock()
        root = []
        root[:] = [root, root, None, None]
        nonlocal_root = [root]
        (PREV, NEXT, KEY, RESULT) = (0, 1, 2, 3)
        if maxsize == 0:

            def wrapper(*args, **kwds):
                if False:
                    return 10
                result = user_function(*args, **kwds)
                stats[MISSES] += 1
                return result
        elif maxsize is None:

            def wrapper(*args, **kwds):
                if False:
                    i = 10
                    return i + 15
                key = make_key(args, kwds, typed)
                result = cache_get(key, root)
                if result is not root:
                    stats[HITS] += 1
                    return result
                result = user_function(*args, **kwds)
                cache[key] = result
                stats[MISSES] += 1
                return result
        else:

            def wrapper(*args, **kwds):
                if False:
                    while True:
                        i = 10
                key = make_key(args, kwds, typed) if kwds or typed else args
                with lock:
                    link = cache_get(key)
                    if link is not None:
                        (root,) = nonlocal_root
                        (link_prev, link_next, key, result) = link
                        link_prev[NEXT] = link_next
                        link_next[PREV] = link_prev
                        last = root[PREV]
                        last[NEXT] = root[PREV] = link
                        link[PREV] = last
                        link[NEXT] = root
                        stats[HITS] += 1
                        return result
                result = user_function(*args, **kwds)
                with lock:
                    (root,) = nonlocal_root
                    if key in cache:
                        pass
                    elif _len(cache) >= maxsize:
                        oldroot = root
                        oldroot[KEY] = key
                        oldroot[RESULT] = result
                        root = nonlocal_root[0] = oldroot[NEXT]
                        oldkey = root[KEY]
                        root[KEY] = root[RESULT] = None
                        del cache[oldkey]
                        cache[key] = oldroot
                    else:
                        last = root[PREV]
                        link = [last, root, key, result]
                        last[NEXT] = root[PREV] = cache[key] = link
                    stats[MISSES] += 1
                return result

        def cache_info():
            if False:
                return 10
            'Report cache statistics'
            with lock:
                return _CacheInfo(stats[HITS], stats[MISSES], maxsize, len(cache))

        def cache_clear():
            if False:
                return 10
            'Clear the cache and cache statistics'
            with lock:
                cache.clear()
                root = nonlocal_root[0]
                root[:] = [root, root, None, None]
                stats[:] = [0, 0]
        wrapper.__wrapped__ = user_function
        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return update_wrapper(wrapper, user_function)
    return decorating_function