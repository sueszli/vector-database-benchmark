"""
Tools for memoization of function results.
"""
from collections import OrderedDict, Sequence
from itertools import compress
from weakref import WeakKeyDictionary, ref
from six.moves._thread import allocate_lock as Lock
from toolz.sandbox import unzip
from trading_calendars.utils.memoize import lazyval
from zipline.utils.compat import wraps

class classlazyval(lazyval):
    """ Decorator that marks that an attribute of a class should not be
    computed until needed, and that the value should be memoized.

    Example
    -------

    >>> from zipline.utils.memoize import classlazyval
    >>> class C(object):
    ...     count = 0
    ...     @classlazyval
    ...     def val(cls):
    ...         cls.count += 1
    ...         return "val"
    ...
    >>> C.count
    0
    >>> C.val, C.count
    ('val', 1)
    >>> C.val, C.count
    ('val', 1)
    """

    def __get__(self, instance, owner):
        if False:
            while True:
                i = 10
        return super(classlazyval, self).__get__(owner, owner)

def _weak_lru_cache(maxsize=100):
    if False:
        for i in range(10):
            print('nop')
    '\n    Users should only access the lru_cache through its public API:\n    cache_info, cache_clear\n    The internals of the lru_cache are encapsulated for thread safety and\n    to allow the implementation to change.\n    '

    def decorating_function(user_function, tuple=tuple, sorted=sorted, len=len, KeyError=KeyError):
        if False:
            while True:
                i = 10
        (hits, misses) = ([0], [0])
        kwd_mark = (object(),)
        lock = Lock()
        if maxsize is None:
            cache = _WeakArgsDict()

            @wraps(user_function)
            def wrapper(*args, **kwds):
                if False:
                    while True:
                        i = 10
                key = args
                if kwds:
                    key += kwd_mark + tuple(sorted(kwds.items()))
                try:
                    result = cache[key]
                    hits[0] += 1
                    return result
                except KeyError:
                    pass
                result = user_function(*args, **kwds)
                cache[key] = result
                misses[0] += 1
                return result
        else:
            cache = _WeakArgsOrderedDict()
            cache_popitem = cache.popitem
            cache_renew = cache.move_to_end

            @wraps(user_function)
            def wrapper(*args, **kwds):
                if False:
                    for i in range(10):
                        print('nop')
                key = args
                if kwds:
                    key += kwd_mark + tuple(sorted(kwds.items()))
                with lock:
                    try:
                        result = cache[key]
                        cache_renew(key)
                        hits[0] += 1
                        return result
                    except KeyError:
                        pass
                result = user_function(*args, **kwds)
                with lock:
                    cache[key] = result
                    misses[0] += 1
                    if len(cache) > maxsize:
                        cache_popitem(False)
                return result

        def cache_info():
            if False:
                return 10
            'Report cache statistics'
            with lock:
                return (hits[0], misses[0], maxsize, len(cache))

        def cache_clear():
            if False:
                while True:
                    i = 10
            'Clear the cache and cache statistics'
            with lock:
                cache.clear()
                hits[0] = misses[0] = 0
        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return wrapper
    return decorating_function

class _WeakArgs(Sequence):
    """
    Works with _WeakArgsDict to provide a weak cache for function args.
    When any of those args are gc'd, the pair is removed from the cache.
    """

    def __init__(self, items, dict_remove=None):
        if False:
            while True:
                i = 10

        def remove(k, selfref=ref(self), dict_remove=dict_remove):
            if False:
                return 10
            self = selfref()
            if self is not None and dict_remove is not None:
                dict_remove(self)
        (self._items, self._selectors) = unzip((self._try_ref(item, remove) for item in items))
        self._items = tuple(self._items)
        self._selectors = tuple(self._selectors)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        return self._items[index]

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._items)

    @staticmethod
    def _try_ref(item, callback):
        if False:
            for i in range(10):
                print('nop')
        try:
            return (ref(item, callback), True)
        except TypeError:
            return (item, False)

    @property
    def alive(self):
        if False:
            return 10
        return all((item() is not None for item in compress(self._items, self._selectors)))

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self._items == other._items

    def __hash__(self):
        if False:
            print('Hello World!')
        try:
            return self.__hash
        except AttributeError:
            h = self.__hash = hash(self._items)
            return h

class _WeakArgsDict(WeakKeyDictionary, object):

    def __delitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        del self.data[_WeakArgs(key)]

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self.data[_WeakArgs(key)]

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s(%r)' % (type(self).__name__, self.data)

    def __setitem__(self, key, value):
        if False:
            return 10
        self.data[_WeakArgs(key, self._remove)] = value

    def __contains__(self, key):
        if False:
            while True:
                i = 10
        try:
            wr = _WeakArgs(key)
        except TypeError:
            return False
        return wr in self.data

    def pop(self, key, *args):
        if False:
            i = 10
            return i + 15
        return self.data.pop(_WeakArgs(key), *args)

class _WeakArgsOrderedDict(_WeakArgsDict, object):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(_WeakArgsOrderedDict, self).__init__()
        self.data = OrderedDict()

    def popitem(self, last=True):
        if False:
            return 10
        while True:
            (key, value) = self.data.popitem(last)
            if key.alive:
                return (tuple(key), value)

    def move_to_end(self, key):
        if False:
            while True:
                i = 10
        'Move an existing element to the end.\n\n        Raises KeyError if the element does not exist.\n        '
        self[key] = self.pop(key)

def weak_lru_cache(maxsize=100):
    if False:
        while True:
            i = 10
    'Weak least-recently-used cache decorator.\n\n    If *maxsize* is set to None, the LRU features are disabled and the cache\n    can grow without bound.\n\n    Arguments to the cached function must be hashable. Any that are weak-\n    referenceable will be stored by weak reference.  Once any of the args have\n    been garbage collected, the entry will be removed from the cache.\n\n    View the cache statistics named tuple (hits, misses, maxsize, currsize)\n    with f.cache_info().  Clear the cache and statistics with f.cache_clear().\n\n    See:  http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used\n\n    '

    class desc(lazyval):

        def __get__(self, instance, owner):
            if False:
                print('Hello World!')
            if instance is None:
                return self
            try:
                return self._cache[instance]
            except KeyError:
                inst = ref(instance)

                @_weak_lru_cache(maxsize)
                @wraps(self._get)
                def wrapper(*args, **kwargs):
                    if False:
                        i = 10
                        return i + 15
                    return self._get(inst(), *args, **kwargs)
                self._cache[instance] = wrapper
                return wrapper

        @_weak_lru_cache(maxsize)
        def __call__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            return self._get(*args, **kwargs)
    return desc
remember_last = weak_lru_cache(1)