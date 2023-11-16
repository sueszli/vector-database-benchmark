"""``cacheutils`` contains consistent implementations of fundamental
cache types. Currently there are two to choose from:

  * :class:`LRI` - Least-recently inserted
  * :class:`LRU` - Least-recently used

Both caches are :class:`dict` subtypes, designed to be as
interchangeable as possible, to facilitate experimentation. A key
practice with performance enhancement with caching is ensuring that
the caching strategy is working. If the cache is constantly missing,
it is just adding more overhead and code complexity. The standard
statistics are:

  * ``hit_count`` - the number of times the queried key has been in
    the cache
  * ``miss_count`` - the number of times a key has been absent and/or
    fetched by the cache
  * ``soft_miss_count`` - the number of times a key has been absent,
    but a default has been provided by the caller, as with
    :meth:`dict.get` and :meth:`dict.setdefault`. Soft misses are a
    subset of misses, so this number is always less than or equal to
    ``miss_count``.

Additionally, ``cacheutils`` provides :class:`ThresholdCounter`, a
cache-like bounded counter useful for online statistics collection.

Learn more about `caching algorithms on Wikipedia
<https://en.wikipedia.org/wiki/Cache_algorithms#Examples>`_.

"""
import heapq
import weakref
import itertools
from operator import attrgetter
try:
    from threading import RLock
except Exception:

    class RLock(object):
        """Dummy reentrant lock for builds without threads"""

        def __enter__(self):
            if False:
                i = 10
                return i + 15
            pass

        def __exit__(self, exctype, excinst, exctb):
            if False:
                print('Hello World!')
            pass
try:
    from .typeutils import make_sentinel
    _MISSING = make_sentinel(var_name='_MISSING')
    _KWARG_MARK = make_sentinel(var_name='_KWARG_MARK')
except ImportError:
    _MISSING = object()
    _KWARG_MARK = object()
try:
    xrange
except NameError:
    xrange = range
    (unicode, str, bytes, basestring) = (str, bytes, bytes, (str, bytes))
(PREV, NEXT, KEY, VALUE) = range(4)
DEFAULT_MAX_SIZE = 128

class LRI(dict):
    """The ``LRI`` implements the basic *Least Recently Inserted* strategy to
    caching. One could also think of this as a ``SizeLimitedDefaultDict``.

    *on_miss* is a callable that accepts the missing key (as opposed
    to :class:`collections.defaultdict`'s "default_factory", which
    accepts no arguments.) Also note that, like the :class:`LRI`,
    the ``LRI`` is instrumented with statistics tracking.

    >>> cap_cache = LRI(max_size=2)
    >>> cap_cache['a'], cap_cache['b'] = 'A', 'B'
    >>> from pprint import pprint as pp
    >>> pp(dict(cap_cache))
    {'a': 'A', 'b': 'B'}
    >>> [cap_cache['b'] for i in range(3)][0]
    'B'
    >>> cap_cache['c'] = 'C'
    >>> print(cap_cache.get('a'))
    None
    >>> cap_cache.hit_count, cap_cache.miss_count, cap_cache.soft_miss_count
    (3, 1, 1)
    """

    def __init__(self, max_size=DEFAULT_MAX_SIZE, values=None, on_miss=None):
        if False:
            while True:
                i = 10
        if max_size <= 0:
            raise ValueError('expected max_size > 0, not %r' % max_size)
        self.hit_count = self.miss_count = self.soft_miss_count = 0
        self.max_size = max_size
        self._lock = RLock()
        self._init_ll()
        if on_miss is not None and (not callable(on_miss)):
            raise TypeError('expected on_miss to be a callable (or None), not %r' % on_miss)
        self.on_miss = on_miss
        if values:
            self.update(values)

    def _init_ll(self):
        if False:
            for i in range(10):
                print('nop')
        anchor = []
        anchor[:] = [anchor, anchor, _MISSING, _MISSING]
        self._link_lookup = {}
        self._anchor = anchor

    def _print_ll(self):
        if False:
            while True:
                i = 10
        print('***')
        for (key, val) in self._get_flattened_ll():
            print(key, val)
        print('***')
        return

    def _get_flattened_ll(self):
        if False:
            for i in range(10):
                print('nop')
        flattened_list = []
        link = self._anchor
        while True:
            flattened_list.append((link[KEY], link[VALUE]))
            link = link[NEXT]
            if link is self._anchor:
                break
        return flattened_list

    def _get_link_and_move_to_front_of_ll(self, key):
        if False:
            while True:
                i = 10
        newest = self._link_lookup[key]
        newest[PREV][NEXT] = newest[NEXT]
        newest[NEXT][PREV] = newest[PREV]
        anchor = self._anchor
        second_newest = anchor[PREV]
        second_newest[NEXT] = anchor[PREV] = newest
        newest[PREV] = second_newest
        newest[NEXT] = anchor
        return newest

    def _set_key_and_add_to_front_of_ll(self, key, value):
        if False:
            print('Hello World!')
        anchor = self._anchor
        second_newest = anchor[PREV]
        newest = [second_newest, anchor, key, value]
        second_newest[NEXT] = anchor[PREV] = newest
        self._link_lookup[key] = newest

    def _set_key_and_evict_last_in_ll(self, key, value):
        if False:
            while True:
                i = 10
        oldanchor = self._anchor
        oldanchor[KEY] = key
        oldanchor[VALUE] = value
        self._anchor = anchor = oldanchor[NEXT]
        evicted = anchor[KEY]
        anchor[KEY] = anchor[VALUE] = _MISSING
        del self._link_lookup[evicted]
        self._link_lookup[key] = oldanchor
        return evicted

    def _remove_from_ll(self, key):
        if False:
            print('Hello World!')
        link = self._link_lookup.pop(key)
        link[PREV][NEXT] = link[NEXT]
        link[NEXT][PREV] = link[PREV]

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        with self._lock:
            try:
                link = self._get_link_and_move_to_front_of_ll(key)
            except KeyError:
                if len(self) < self.max_size:
                    self._set_key_and_add_to_front_of_ll(key, value)
                else:
                    evicted = self._set_key_and_evict_last_in_ll(key, value)
                    super(LRI, self).__delitem__(evicted)
            else:
                link[VALUE] = value
            super(LRI, self).__setitem__(key, value)
        return

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            try:
                link = self._link_lookup[key]
            except KeyError:
                self.miss_count += 1
                if not self.on_miss:
                    raise
                ret = self[key] = self.on_miss(key)
                return ret
            self.hit_count += 1
            return link[VALUE]

    def get(self, key, default=None):
        if False:
            while True:
                i = 10
        try:
            return self[key]
        except KeyError:
            self.soft_miss_count += 1
            return default

    def __delitem__(self, key):
        if False:
            print('Hello World!')
        with self._lock:
            super(LRI, self).__delitem__(key)
            self._remove_from_ll(key)

    def pop(self, key, default=_MISSING):
        if False:
            while True:
                i = 10
        with self._lock:
            try:
                ret = super(LRI, self).pop(key)
            except KeyError:
                if default is _MISSING:
                    raise
                ret = default
            else:
                self._remove_from_ll(key)
            return ret

    def popitem(self):
        if False:
            i = 10
            return i + 15
        with self._lock:
            item = super(LRI, self).popitem()
            self._remove_from_ll(item[0])
            return item

    def clear(self):
        if False:
            return 10
        with self._lock:
            super(LRI, self).clear()
            self._init_ll()

    def copy(self):
        if False:
            i = 10
            return i + 15
        return self.__class__(max_size=self.max_size, values=self)

    def setdefault(self, key, default=None):
        if False:
            i = 10
            return i + 15
        with self._lock:
            try:
                return self[key]
            except KeyError:
                self.soft_miss_count += 1
                self[key] = default
                return default

    def update(self, E, **F):
        if False:
            return 10
        with self._lock:
            if E is self:
                return
            setitem = self.__setitem__
            if callable(getattr(E, 'keys', None)):
                for k in E.keys():
                    setitem(k, E[k])
            else:
                for (k, v) in E:
                    setitem(k, v)
            for k in F:
                setitem(k, F[k])
            return

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        with self._lock:
            if self is other:
                return True
            if len(other) != len(self):
                return False
            if not isinstance(other, LRI):
                return other == self
            return super(LRI, self).__eq__(other)

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self == other

    def __repr__(self):
        if False:
            while True:
                i = 10
        cn = self.__class__.__name__
        val_map = super(LRI, self).__repr__()
        return '%s(max_size=%r, on_miss=%r, values=%s)' % (cn, self.max_size, self.on_miss, val_map)

class LRU(LRI):
    """The ``LRU`` is :class:`dict` subtype implementation of the
    *Least-Recently Used* caching strategy.

    Args:
        max_size (int): Max number of items to cache. Defaults to ``128``.
        values (iterable): Initial values for the cache. Defaults to ``None``.
        on_miss (callable): a callable which accepts a single argument, the
            key not present in the cache, and returns the value to be cached.

    >>> cap_cache = LRU(max_size=2)
    >>> cap_cache['a'], cap_cache['b'] = 'A', 'B'
    >>> from pprint import pprint as pp
    >>> pp(dict(cap_cache))
    {'a': 'A', 'b': 'B'}
    >>> [cap_cache['b'] for i in range(3)][0]
    'B'
    >>> cap_cache['c'] = 'C'
    >>> print(cap_cache.get('a'))
    None

    This cache is also instrumented with statistics
    collection. ``hit_count``, ``miss_count``, and ``soft_miss_count``
    are all integer members that can be used to introspect the
    performance of the cache. ("Soft" misses are misses that did not
    raise :exc:`KeyError`, e.g., ``LRU.get()`` or ``on_miss`` was used to
    cache a default.

    >>> cap_cache.hit_count, cap_cache.miss_count, cap_cache.soft_miss_count
    (3, 1, 1)

    Other than the size-limiting caching behavior and statistics,
    ``LRU`` acts like its parent class, the built-in Python :class:`dict`.
    """

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        with self._lock:
            try:
                link = self._get_link_and_move_to_front_of_ll(key)
            except KeyError:
                self.miss_count += 1
                if not self.on_miss:
                    raise
                ret = self[key] = self.on_miss(key)
                return ret
            self.hit_count += 1
            return link[VALUE]

class _HashedKey(list):
    """The _HashedKey guarantees that hash() will be called no more than once
    per cached function invocation.
    """
    __slots__ = 'hash_value'

    def __init__(self, key):
        if False:
            print('Hello World!')
        self[:] = key
        self.hash_value = hash(tuple(key))

    def __hash__(self):
        if False:
            while True:
                i = 10
        return self.hash_value

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s(%s)' % (self.__class__.__name__, list.__repr__(self))

def make_cache_key(args, kwargs, typed=False, kwarg_mark=_KWARG_MARK, fasttypes=frozenset([int, str, frozenset, type(None)])):
    if False:
        for i in range(10):
            print('nop')
    "Make a generic key from a function's positional and keyword\n    arguments, suitable for use in caches. Arguments within *args* and\n    *kwargs* must be `hashable`_. If *typed* is ``True``, ``3`` and\n    ``3.0`` will be treated as separate keys.\n\n    The key is constructed in a way that is flat as possible rather than\n    as a nested structure that would take more memory.\n\n    If there is only a single argument and its data type is known to cache\n    its hash value, then that argument is returned without a wrapper.  This\n    saves space and improves lookup speed.\n\n    >>> tuple(make_cache_key(('a', 'b'), {'c': ('d')}))\n    ('a', 'b', _KWARG_MARK, ('c', 'd'))\n\n    .. _hashable: https://docs.python.org/2/glossary.html#term-hashable\n    "
    key = list(args)
    if kwargs:
        sorted_items = sorted(kwargs.items())
        key.append(kwarg_mark)
        key.extend(sorted_items)
    if typed:
        key.extend([type(v) for v in args])
        if kwargs:
            key.extend([type(v) for (k, v) in sorted_items])
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedKey(key)
_make_cache_key = make_cache_key

class CachedFunction(object):
    """This type is used by :func:`cached`, below. Instances of this
    class are used to wrap functions in caching logic.
    """

    def __init__(self, func, cache, scoped=True, typed=False, key=None):
        if False:
            for i in range(10):
                print('nop')
        self.func = func
        if callable(cache):
            self.get_cache = cache
        elif not (callable(getattr(cache, '__getitem__', None)) and callable(getattr(cache, '__setitem__', None))):
            raise TypeError('expected cache to be a dict-like object, or callable returning a dict-like object, not %r' % cache)
        else:

            def _get_cache():
                if False:
                    for i in range(10):
                        print('nop')
                return cache
            self.get_cache = _get_cache
        self.scoped = scoped
        self.typed = typed
        self.key_func = key or make_cache_key

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        cache = self.get_cache()
        key = self.key_func(args, kwargs, typed=self.typed)
        try:
            ret = cache[key]
        except KeyError:
            ret = cache[key] = self.func(*args, **kwargs)
        return ret

    def __repr__(self):
        if False:
            while True:
                i = 10
        cn = self.__class__.__name__
        if self.typed or not self.scoped:
            return '%s(func=%r, scoped=%r, typed=%r)' % (cn, self.func, self.scoped, self.typed)
        return '%s(func=%r)' % (cn, self.func)

class CachedMethod(object):
    """Similar to :class:`CachedFunction`, this type is used by
    :func:`cachedmethod` to wrap methods in caching logic.
    """

    def __init__(self, func, cache, scoped=True, typed=False, key=None):
        if False:
            while True:
                i = 10
        self.func = func
        self.__isabstractmethod__ = getattr(func, '__isabstractmethod__', False)
        if isinstance(cache, basestring):
            self.get_cache = attrgetter(cache)
        elif callable(cache):
            self.get_cache = cache
        elif not (callable(getattr(cache, '__getitem__', None)) and callable(getattr(cache, '__setitem__', None))):
            raise TypeError('expected cache to be an attribute name, dict-like object, or callable returning a dict-like object, not %r' % cache)
        else:

            def _get_cache(obj):
                if False:
                    i = 10
                    return i + 15
                return cache
            self.get_cache = _get_cache
        self.scoped = scoped
        self.typed = typed
        self.key_func = key or make_cache_key
        self.bound_to = None

    def __get__(self, obj, objtype=None):
        if False:
            for i in range(10):
                print('nop')
        if obj is None:
            return self
        cls = self.__class__
        ret = cls(self.func, self.get_cache, typed=self.typed, scoped=self.scoped, key=self.key_func)
        ret.bound_to = obj
        return ret

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        obj = args[0] if self.bound_to is None else self.bound_to
        cache = self.get_cache(obj)
        key_args = (self.bound_to, self.func) + args if self.scoped else args
        key = self.key_func(key_args, kwargs, typed=self.typed)
        try:
            ret = cache[key]
        except KeyError:
            if self.bound_to is not None:
                args = (self.bound_to,) + args
            ret = cache[key] = self.func(*args, **kwargs)
        return ret

    def __repr__(self):
        if False:
            print('Hello World!')
        cn = self.__class__.__name__
        args = (cn, self.func, self.scoped, self.typed)
        if self.bound_to is not None:
            args += (self.bound_to,)
            return '<%s func=%r scoped=%r typed=%r bound_to=%r>' % args
        return '%s(func=%r, scoped=%r, typed=%r)' % args

def cached(cache, scoped=True, typed=False, key=None):
    if False:
        while True:
            i = 10
    'Cache any function with the cache object of your choosing. Note\n    that the function wrapped should take only `hashable`_ arguments.\n\n    Args:\n        cache (Mapping): Any :class:`dict`-like object suitable for\n            use as a cache. Instances of the :class:`LRU` and\n            :class:`LRI` are good choices, but a plain :class:`dict`\n            can work in some cases, as well. This argument can also be\n            a callable which accepts no arguments and returns a mapping.\n        scoped (bool): Whether the function itself is part of the\n            cache key.  ``True`` by default, different functions will\n            not read one another\'s cache entries, but can evict one\n            another\'s results. ``False`` can be useful for certain\n            shared cache use cases. More advanced behavior can be\n            produced through the *key* argument.\n        typed (bool): Whether to factor argument types into the cache\n            check. Default ``False``, setting to ``True`` causes the\n            cache keys for ``3`` and ``3.0`` to be considered unequal.\n\n    >>> my_cache = LRU()\n    >>> @cached(my_cache)\n    ... def cached_lower(x):\n    ...     return x.lower()\n    ...\n    >>> cached_lower("CaChInG\'s FuN AgAiN!")\n    "caching\'s fun again!"\n    >>> len(my_cache)\n    1\n\n    .. _hashable: https://docs.python.org/2/glossary.html#term-hashable\n\n    '

    def cached_func_decorator(func):
        if False:
            while True:
                i = 10
        return CachedFunction(func, cache, scoped=scoped, typed=typed, key=key)
    return cached_func_decorator

def cachedmethod(cache, scoped=True, typed=False, key=None):
    if False:
        i = 10
        return i + 15
    "Similar to :func:`cached`, ``cachedmethod`` is used to cache\n    methods based on their arguments, using any :class:`dict`-like\n    *cache* object.\n\n    Args:\n        cache (str/Mapping/callable): Can be the name of an attribute\n            on the instance, any Mapping/:class:`dict`-like object, or\n            a callable which returns a Mapping.\n        scoped (bool): Whether the method itself and the object it is\n            bound to are part of the cache keys. ``True`` by default,\n            different methods will not read one another's cache\n            results. ``False`` can be useful for certain shared cache\n            use cases. More advanced behavior can be produced through\n            the *key* arguments.\n        typed (bool): Whether to factor argument types into the cache\n            check. Default ``False``, setting to ``True`` causes the\n            cache keys for ``3`` and ``3.0`` to be considered unequal.\n        key (callable): A callable with a signature that matches\n            :func:`make_cache_key` that returns a tuple of hashable\n            values to be used as the key in the cache.\n\n    >>> class Lowerer(object):\n    ...     def __init__(self):\n    ...         self.cache = LRI()\n    ...\n    ...     @cachedmethod('cache')\n    ...     def lower(self, text):\n    ...         return text.lower()\n    ...\n    >>> lowerer = Lowerer()\n    >>> lowerer.lower('WOW WHO COULD GUESS CACHING COULD BE SO NEAT')\n    'wow who could guess caching could be so neat'\n    >>> len(lowerer.cache)\n    1\n\n    "

    def cached_method_decorator(func):
        if False:
            while True:
                i = 10
        return CachedMethod(func, cache, scoped=scoped, typed=typed, key=key)
    return cached_method_decorator

class cachedproperty(object):
    """The ``cachedproperty`` is used similar to :class:`property`, except
    that the wrapped method is only called once. This is commonly used
    to implement lazy attributes.

    After the property has been accessed, the value is stored on the
    instance itself, using the same name as the cachedproperty. This
    allows the cache to be cleared with :func:`delattr`, or through
    manipulating the object's ``__dict__``.
    """

    def __init__(self, func):
        if False:
            i = 10
            return i + 15
        self.__doc__ = getattr(func, '__doc__')
        self.__isabstractmethod__ = getattr(func, '__isabstractmethod__', False)
        self.func = func

    def __get__(self, obj, objtype=None):
        if False:
            print('Hello World!')
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

    def __repr__(self):
        if False:
            while True:
                i = 10
        cn = self.__class__.__name__
        return '<%s func=%s>' % (cn, self.func)

class ThresholdCounter(object):
    """A **bounded** dict-like Mapping from keys to counts. The
    ThresholdCounter automatically compacts after every (1 /
    *threshold*) additions, maintaining exact counts for any keys
    whose count represents at least a *threshold* ratio of the total
    data. In other words, if a particular key is not present in the
    ThresholdCounter, its count represents less than *threshold* of
    the total data.

    >>> tc = ThresholdCounter(threshold=0.1)
    >>> tc.add(1)
    >>> tc.items()
    [(1, 1)]
    >>> tc.update([2] * 10)
    >>> tc.get(1)
    0
    >>> tc.add(5)
    >>> 5 in tc
    True
    >>> len(list(tc.elements()))
    11

    As you can see above, the API is kept similar to
    :class:`collections.Counter`. The most notable feature omissions
    being that counted items cannot be set directly, uncounted, or
    removed, as this would disrupt the math.

    Use the ThresholdCounter when you need best-effort long-lived
    counts for dynamically-keyed data. Without a bounded datastructure
    such as this one, the dynamic keys often represent a memory leak
    and can impact application reliability. The ThresholdCounter's
    item replacement strategy is fully deterministic and can be
    thought of as *Amortized Least Relevant*. The absolute upper bound
    of keys it will store is *(2/threshold)*, but realistically
    *(1/threshold)* is expected for uniformly random datastreams, and
    one or two orders of magnitude better for real-world data.

    This algorithm is an implementation of the Lossy Counting
    algorithm described in "Approximate Frequency Counts over Data
    Streams" by Manku & Motwani. Hat tip to Kurt Rose for discovery
    and initial implementation.

    """

    def __init__(self, threshold=0.001):
        if False:
            return 10
        if not 0 < threshold < 1:
            raise ValueError('expected threshold between 0 and 1, not: %r' % threshold)
        self.total = 0
        self._count_map = {}
        self._threshold = threshold
        self._thresh_count = int(1 / threshold)
        self._cur_bucket = 1

    @property
    def threshold(self):
        if False:
            for i in range(10):
                print('nop')
        return self._threshold

    def add(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Increment the count of *key* by 1, automatically adding it if it\n        does not exist.\n\n        Cache compaction is triggered every *1/threshold* additions.\n        '
        self.total += 1
        try:
            self._count_map[key][0] += 1
        except KeyError:
            self._count_map[key] = [1, self._cur_bucket - 1]
        if self.total % self._thresh_count == 0:
            self._count_map = dict([(k, v) for (k, v) in self._count_map.items() if sum(v) > self._cur_bucket])
            self._cur_bucket += 1
        return

    def elements(self):
        if False:
            for i in range(10):
                print('nop')
        'Return an iterator of all the common elements tracked by the\n        counter. Yields each key as many times as it has been seen.\n        '
        repeaters = itertools.starmap(itertools.repeat, self.iteritems())
        return itertools.chain.from_iterable(repeaters)

    def most_common(self, n=None):
        if False:
            while True:
                i = 10
        'Get the top *n* keys and counts as tuples. If *n* is omitted,\n        returns all the pairs.\n        '
        if n <= 0:
            return []
        ret = sorted(self.iteritems(), key=lambda x: x[1], reverse=True)
        if n is None or n >= len(ret):
            return ret
        return ret[:n]

    def get_common_count(self):
        if False:
            print('Hello World!')
        'Get the sum of counts for keys exceeding the configured data\n        threshold.\n        '
        return sum([count for (count, _) in self._count_map.values()])

    def get_uncommon_count(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the sum of counts for keys that were culled because the\n        associated counts represented less than the configured\n        threshold. The long-tail counts.\n        '
        return self.total - self.get_common_count()

    def get_commonality(self):
        if False:
            while True:
                i = 10
        'Get a float representation of the effective count accuracy. The\n        higher the number, the less uniform the keys being added, and\n        the higher accuracy and efficiency of the ThresholdCounter.\n\n        If a stronger measure of data cardinality is required,\n        consider using hyperloglog.\n        '
        return float(self.get_common_count()) / self.total

    def __getitem__(self, key):
        if False:
            return 10
        return self._count_map[key][0]

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._count_map)

    def __contains__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return key in self._count_map

    def iterkeys(self):
        if False:
            return 10
        return iter(self._count_map)

    def keys(self):
        if False:
            return 10
        return list(self.iterkeys())

    def itervalues(self):
        if False:
            return 10
        count_map = self._count_map
        for k in count_map:
            yield count_map[k][0]

    def values(self):
        if False:
            while True:
                i = 10
        return list(self.itervalues())

    def iteritems(self):
        if False:
            for i in range(10):
                print('nop')
        count_map = self._count_map
        for k in count_map:
            yield (k, count_map[k][0])

    def items(self):
        if False:
            while True:
                i = 10
        return list(self.iteritems())

    def get(self, key, default=0):
        if False:
            i = 10
            return i + 15
        'Get count for *key*, defaulting to 0.'
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, iterable, **kwargs):
        if False:
            while True:
                i = 10
        'Like dict.update() but add counts instead of replacing them, used\n        to add multiple items in one call.\n\n        Source can be an iterable of keys to add, or a mapping of keys\n        to integer counts.\n        '
        if iterable is not None:
            if callable(getattr(iterable, 'iteritems', None)):
                for (key, count) in iterable.iteritems():
                    for i in xrange(count):
                        self.add(key)
            else:
                for key in iterable:
                    self.add(key)
        if kwargs:
            self.update(kwargs)

class MinIDMap(object):
    """
    Assigns arbitrary weakref-able objects the smallest possible unique
    integer IDs, such that no two objects have the same ID at the same
    time.

    Maps arbitrary hashable objects to IDs.

    Based on https://gist.github.com/kurtbrose/25b48114de216a5e55df
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.mapping = weakref.WeakKeyDictionary()
        self.ref_map = {}
        self.free = []

    def get(self, a):
        if False:
            print('Hello World!')
        try:
            return self.mapping[a][0]
        except KeyError:
            pass
        if self.free:
            nxt = heapq.heappop(self.free)
        else:
            nxt = len(self.mapping)
        ref = weakref.ref(a, self._clean)
        self.mapping[a] = (nxt, ref)
        self.ref_map[ref] = nxt
        return nxt

    def drop(self, a):
        if False:
            while True:
                i = 10
        (freed, ref) = self.mapping[a]
        del self.mapping[a]
        del self.ref_map[ref]
        heapq.heappush(self.free, freed)

    def _clean(self, ref):
        if False:
            print('Hello World!')
        print(self.ref_map[ref])
        heapq.heappush(self.free, self.ref_map[ref])
        del self.ref_map[ref]

    def __contains__(self, a):
        if False:
            print('Hello World!')
        return a in self.mapping

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self.mapping)

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.mapping.__len__()

    def iteritems(self):
        if False:
            i = 10
            return i + 15
        return iter(((k, self.mapping[k][0]) for k in iter(self.mapping)))