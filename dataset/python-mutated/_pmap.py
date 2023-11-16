from collections.abc import Mapping, Hashable
from itertools import chain
from pyrsistent._pvector import pvector
from pyrsistent._transformations import transform

class PMap(object):
    """
    Persistent map/dict. Tries to follow the same naming conventions as the built in dict where feasible.

    Do not instantiate directly, instead use the factory functions :py:func:`m` or :py:func:`pmap` to
    create an instance.

    Was originally written as a very close copy of the Clojure equivalent but was later rewritten to closer
    re-assemble the python dict. This means that a sparse vector (a PVector) of buckets is used. The keys are
    hashed and the elements inserted at position hash % len(bucket_vector). Whenever the map size exceeds 2/3 of
    the containing vectors size the map is reallocated to a vector of double the size. This is done to avoid
    excessive hash collisions.

    This structure corresponds most closely to the built in dict type and is intended as a replacement. Where the
    semantics are the same (more or less) the same function names have been used but for some cases it is not possible,
    for example assignments and deletion of values.

    PMap implements the Mapping protocol and is Hashable. It also supports dot-notation for
    element access.

    Random access and insert is log32(n) where n is the size of the map.

    The following are examples of some common operations on persistent maps

    >>> m1 = m(a=1, b=3)
    >>> m2 = m1.set('c', 3)
    >>> m3 = m2.remove('a')
    >>> m1
    pmap({'b': 3, 'a': 1})
    >>> m2
    pmap({'c': 3, 'b': 3, 'a': 1})
    >>> m3
    pmap({'c': 3, 'b': 3})
    >>> m3['c']
    3
    >>> m3.c
    3
    """
    __slots__ = ('_size', '_buckets', '__weakref__', '_cached_hash')

    def __new__(cls, size, buckets):
        if False:
            print('Hello World!')
        self = super(PMap, cls).__new__(cls)
        self._size = size
        self._buckets = buckets
        return self

    @staticmethod
    def _get_bucket(buckets, key):
        if False:
            for i in range(10):
                print('nop')
        index = hash(key) % len(buckets)
        bucket = buckets[index]
        return (index, bucket)

    @staticmethod
    def _getitem(buckets, key):
        if False:
            print('Hello World!')
        (_, bucket) = PMap._get_bucket(buckets, key)
        if bucket:
            for (k, v) in bucket:
                if k == key:
                    return v
        raise KeyError(key)

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return PMap._getitem(self._buckets, key)

    @staticmethod
    def _contains(buckets, key):
        if False:
            i = 10
            return i + 15
        (_, bucket) = PMap._get_bucket(buckets, key)
        if bucket:
            for (k, _) in bucket:
                if k == key:
                    return True
            return False
        return False

    def __contains__(self, key):
        if False:
            print('Hello World!')
        return self._contains(self._buckets, key)
    get = Mapping.get

    def __iter__(self):
        if False:
            print('Hello World!')
        return self.iterkeys()

    def __getattr__(self, key):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError("{0} has no attribute '{1}'".format(type(self).__name__, key)) from e

    def iterkeys(self):
        if False:
            return 10
        for (k, _) in self.iteritems():
            yield k

    def itervalues(self):
        if False:
            for i in range(10):
                print('nop')
        for (_, v) in self.iteritems():
            yield v

    def iteritems(self):
        if False:
            for i in range(10):
                print('nop')
        for bucket in self._buckets:
            if bucket:
                for (k, v) in bucket:
                    yield (k, v)

    def values(self):
        if False:
            print('Hello World!')
        return pvector(self.itervalues())

    def keys(self):
        if False:
            print('Hello World!')
        return pvector(self.iterkeys())

    def items(self):
        if False:
            while True:
                i = 10
        return pvector(self.iteritems())

    def __len__(self):
        if False:
            return 10
        return self._size

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'pmap({0})'.format(str(dict(self)))

    def __eq__(self, other):
        if False:
            return 10
        if self is other:
            return True
        if not isinstance(other, Mapping):
            return NotImplemented
        if len(self) != len(other):
            return False
        if isinstance(other, PMap):
            if hasattr(self, '_cached_hash') and hasattr(other, '_cached_hash') and (self._cached_hash != other._cached_hash):
                return False
            if self._buckets == other._buckets:
                return True
            return dict(self.iteritems()) == dict(other.iteritems())
        elif isinstance(other, dict):
            return dict(self.iteritems()) == other
        return dict(self.iteritems()) == dict(other.items())
    __ne__ = Mapping.__ne__

    def __lt__(self, other):
        if False:
            print('Hello World!')
        raise TypeError('PMaps are not orderable')
    __le__ = __lt__
    __gt__ = __lt__
    __ge__ = __lt__

    def __str__(self):
        if False:
            return 10
        return self.__repr__()

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(self, '_cached_hash'):
            self._cached_hash = hash(frozenset(self.iteritems()))
        return self._cached_hash

    def set(self, key, val):
        if False:
            while True:
                i = 10
        "\n        Return a new PMap with key and val inserted.\n\n        >>> m1 = m(a=1, b=2)\n        >>> m2 = m1.set('a', 3)\n        >>> m3 = m1.set('c' ,4)\n        >>> m1\n        pmap({'b': 2, 'a': 1})\n        >>> m2\n        pmap({'b': 2, 'a': 3})\n        >>> m3\n        pmap({'c': 4, 'b': 2, 'a': 1})\n        "
        return self.evolver().set(key, val).persistent()

    def remove(self, key):
        if False:
            return 10
        "\n        Return a new PMap without the element specified by key. Raises KeyError if the element\n        is not present.\n\n        >>> m1 = m(a=1, b=2)\n        >>> m1.remove('a')\n        pmap({'b': 2})\n        "
        return self.evolver().remove(key).persistent()

    def discard(self, key):
        if False:
            while True:
                i = 10
        "\n        Return a new PMap without the element specified by key. Returns reference to itself\n        if element is not present.\n\n        >>> m1 = m(a=1, b=2)\n        >>> m1.discard('a')\n        pmap({'b': 2})\n        >>> m1 is m1.discard('c')\n        True\n        "
        try:
            return self.remove(key)
        except KeyError:
            return self

    def update(self, *maps):
        if False:
            return 10
        "\n        Return a new PMap with the items in Mappings inserted. If the same key is present in multiple\n        maps the rightmost (last) value is inserted.\n\n        >>> m1 = m(a=1, b=2)\n        >>> m1.update(m(a=2, c=3), {'a': 17, 'd': 35})\n        pmap({'c': 3, 'b': 2, 'a': 17, 'd': 35})\n        "
        return self.update_with(lambda l, r: r, *maps)

    def update_with(self, update_fn, *maps):
        if False:
            i = 10
            return i + 15
        "\n        Return a new PMap with the items in Mappings maps inserted. If the same key is present in multiple\n        maps the values will be merged using merge_fn going from left to right.\n\n        >>> from operator import add\n        >>> m1 = m(a=1, b=2)\n        >>> m1.update_with(add, m(a=2))\n        pmap({'b': 2, 'a': 3})\n\n        The reverse behaviour of the regular merge. Keep the leftmost element instead of the rightmost.\n\n        >>> m1 = m(a=1)\n        >>> m1.update_with(lambda l, r: l, m(a=2), {'a':3})\n        pmap({'a': 1})\n        "
        evolver = self.evolver()
        for map in maps:
            for (key, value) in map.items():
                evolver.set(key, update_fn(evolver[key], value) if key in evolver else value)
        return evolver.persistent()

    def __add__(self, other):
        if False:
            while True:
                i = 10
        return self.update(other)
    __or__ = __add__

    def __reduce__(self):
        if False:
            return 10
        return (pmap, (dict(self),))

    def transform(self, *transformations):
        if False:
            return 10
        "\n        Transform arbitrarily complex combinations of PVectors and PMaps. A transformation\n        consists of two parts. One match expression that specifies which elements to transform\n        and one transformation function that performs the actual transformation.\n\n        >>> from pyrsistent import freeze, ny\n        >>> news_paper = freeze({'articles': [{'author': 'Sara', 'content': 'A short article'},\n        ...                                   {'author': 'Steve', 'content': 'A slightly longer article'}],\n        ...                      'weather': {'temperature': '11C', 'wind': '5m/s'}})\n        >>> short_news = news_paper.transform(['articles', ny, 'content'], lambda c: c[:25] + '...' if len(c) > 25 else c)\n        >>> very_short_news = news_paper.transform(['articles', ny, 'content'], lambda c: c[:15] + '...' if len(c) > 15 else c)\n        >>> very_short_news.articles[0].content\n        'A short article'\n        >>> very_short_news.articles[1].content\n        'A slightly long...'\n\n        When nothing has been transformed the original data structure is kept\n\n        >>> short_news is news_paper\n        True\n        >>> very_short_news is news_paper\n        False\n        >>> very_short_news.articles[0] is news_paper.articles[0]\n        True\n        "
        return transform(self, transformations)

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    class _Evolver(object):
        __slots__ = ('_buckets_evolver', '_size', '_original_pmap')

        def __init__(self, original_pmap):
            if False:
                return 10
            self._original_pmap = original_pmap
            self._buckets_evolver = original_pmap._buckets.evolver()
            self._size = original_pmap._size

        def __getitem__(self, key):
            if False:
                while True:
                    i = 10
            return PMap._getitem(self._buckets_evolver, key)

        def __setitem__(self, key, val):
            if False:
                i = 10
                return i + 15
            self.set(key, val)

        def set(self, key, val):
            if False:
                for i in range(10):
                    print('nop')
            if len(self._buckets_evolver) < 0.67 * self._size:
                self._reallocate(2 * len(self._buckets_evolver))
            kv = (key, val)
            (index, bucket) = PMap._get_bucket(self._buckets_evolver, key)
            if bucket:
                for (k, v) in bucket:
                    if k == key:
                        if v is not val:
                            new_bucket = [(k2, v2) if k2 != k else (k2, val) for (k2, v2) in bucket]
                            self._buckets_evolver[index] = new_bucket
                        return self
                new_bucket = [kv]
                new_bucket.extend(bucket)
                self._buckets_evolver[index] = new_bucket
                self._size += 1
            else:
                self._buckets_evolver[index] = [kv]
                self._size += 1
            return self

        def _reallocate(self, new_size):
            if False:
                i = 10
                return i + 15
            new_list = new_size * [None]
            buckets = self._buckets_evolver.persistent()
            for (k, v) in chain.from_iterable((x for x in buckets if x)):
                index = hash(k) % new_size
                if new_list[index]:
                    new_list[index].append((k, v))
                else:
                    new_list[index] = [(k, v)]
            self._buckets_evolver = pvector().evolver()
            self._buckets_evolver.extend(new_list)

        def is_dirty(self):
            if False:
                return 10
            return self._buckets_evolver.is_dirty()

        def persistent(self):
            if False:
                return 10
            if self.is_dirty():
                self._original_pmap = PMap(self._size, self._buckets_evolver.persistent())
            return self._original_pmap

        def __len__(self):
            if False:
                return 10
            return self._size

        def __contains__(self, key):
            if False:
                while True:
                    i = 10
            return PMap._contains(self._buckets_evolver, key)

        def __delitem__(self, key):
            if False:
                while True:
                    i = 10
            self.remove(key)

        def remove(self, key):
            if False:
                for i in range(10):
                    print('nop')
            (index, bucket) = PMap._get_bucket(self._buckets_evolver, key)
            if bucket:
                new_bucket = [(k, v) for (k, v) in bucket if k != key]
                if len(bucket) > len(new_bucket):
                    self._buckets_evolver[index] = new_bucket if new_bucket else None
                    self._size -= 1
                    return self
            raise KeyError('{0}'.format(key))

    def evolver(self):
        if False:
            return 10
        "\n        Create a new evolver for this pmap. For a discussion on evolvers in general see the\n        documentation for the pvector evolver.\n\n        Create the evolver and perform various mutating updates to it:\n\n        >>> m1 = m(a=1, b=2)\n        >>> e = m1.evolver()\n        >>> e['c'] = 3\n        >>> len(e)\n        3\n        >>> del e['a']\n\n        The underlying pmap remains the same:\n\n        >>> m1\n        pmap({'b': 2, 'a': 1})\n\n        The changes are kept in the evolver. An updated pmap can be created using the\n        persistent() function on the evolver.\n\n        >>> m2 = e.persistent()\n        >>> m2\n        pmap({'c': 3, 'b': 2})\n\n        The new pmap will share data with the original pmap in the same way that would have\n        been done if only using operations on the pmap.\n        "
        return self._Evolver(self)
Mapping.register(PMap)
Hashable.register(PMap)

def _turbo_mapping(initial, pre_size):
    if False:
        for i in range(10):
            print('nop')
    if pre_size:
        size = pre_size
    else:
        try:
            size = 2 * len(initial) or 8
        except Exception:
            size = 8
    buckets = size * [None]
    if not isinstance(initial, Mapping):
        initial = dict(initial)
    for (k, v) in initial.items():
        h = hash(k)
        index = h % size
        bucket = buckets[index]
        if bucket:
            bucket.append((k, v))
        else:
            buckets[index] = [(k, v)]
    return PMap(len(initial), pvector().extend(buckets))
_EMPTY_PMAP = _turbo_mapping({}, 0)

def pmap(initial={}, pre_size=0):
    if False:
        print('Hello World!')
    "\n    Create new persistent map, inserts all elements in initial into the newly created map.\n    The optional argument pre_size may be used to specify an initial size of the underlying bucket vector. This\n    may have a positive performance impact in the cases where you know beforehand that a large number of elements\n    will be inserted into the map eventually since it will reduce the number of reallocations required.\n\n    >>> pmap({'a': 13, 'b': 14})\n    pmap({'b': 14, 'a': 13})\n    "
    if not initial and pre_size == 0:
        return _EMPTY_PMAP
    return _turbo_mapping(initial, pre_size)

def m(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Creates a new persistent map. Inserts all key value arguments into the newly created map.\n\n    >>> m(a=13, b=14)\n    pmap({'b': 14, 'a': 13})\n    "
    return pmap(kwargs)