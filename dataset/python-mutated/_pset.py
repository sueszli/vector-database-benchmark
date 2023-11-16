from collections.abc import Set, Hashable
import sys
from pyrsistent._pmap import pmap

class PSet(object):
    """
    Persistent set implementation. Built on top of the persistent map. The set supports all operations
    in the Set protocol and is Hashable.

    Do not instantiate directly, instead use the factory functions :py:func:`s` or :py:func:`pset`
    to create an instance.

    Random access and insert is log32(n) where n is the size of the set.

    Some examples:

    >>> s = pset([1, 2, 3, 1])
    >>> s2 = s.add(4)
    >>> s3 = s2.remove(2)
    >>> s
    pset([1, 2, 3])
    >>> s2
    pset([1, 2, 3, 4])
    >>> s3
    pset([1, 3, 4])
    """
    __slots__ = ('_map', '__weakref__')

    def __new__(cls, m):
        if False:
            i = 10
            return i + 15
        self = super(PSet, cls).__new__(cls)
        self._map = m
        return self

    def __contains__(self, element):
        if False:
            while True:
                i = 10
        return element in self._map

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self._map)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self._map)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if not self:
            return 'p' + str(set(self))
        return 'pset([{0}])'.format(str(set(self))[1:-1])

    def __str__(self):
        if False:
            print('Hello World!')
        return self.__repr__()

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self._map)

    def __reduce__(self):
        if False:
            while True:
                i = 10
        return (pset, (list(self),))

    @classmethod
    def _from_iterable(cls, it, pre_size=8):
        if False:
            i = 10
            return i + 15
        return PSet(pmap(dict(((k, True) for k in it)), pre_size=pre_size))

    def add(self, element):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a new PSet with element added\n\n        >>> s1 = s(1, 2)\n        >>> s1.add(3)\n        pset([1, 2, 3])\n        '
        return self.evolver().add(element).persistent()

    def update(self, iterable):
        if False:
            while True:
                i = 10
        '\n        Return a new PSet with elements in iterable added\n\n        >>> s1 = s(1, 2)\n        >>> s1.update([3, 4, 4])\n        pset([1, 2, 3, 4])\n        '
        e = self.evolver()
        for element in iterable:
            e.add(element)
        return e.persistent()

    def remove(self, element):
        if False:
            print('Hello World!')
        '\n        Return a new PSet with element removed. Raises KeyError if element is not present.\n\n        >>> s1 = s(1, 2)\n        >>> s1.remove(2)\n        pset([1])\n        '
        if element in self._map:
            return self.evolver().remove(element).persistent()
        raise KeyError("Element '%s' not present in PSet" % repr(element))

    def discard(self, element):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a new PSet with element removed. Returns itself if element is not present.\n        '
        if element in self._map:
            return self.evolver().remove(element).persistent()
        return self

    class _Evolver(object):
        __slots__ = ('_original_pset', '_pmap_evolver')

        def __init__(self, original_pset):
            if False:
                return 10
            self._original_pset = original_pset
            self._pmap_evolver = original_pset._map.evolver()

        def add(self, element):
            if False:
                for i in range(10):
                    print('nop')
            self._pmap_evolver[element] = True
            return self

        def remove(self, element):
            if False:
                i = 10
                return i + 15
            del self._pmap_evolver[element]
            return self

        def is_dirty(self):
            if False:
                i = 10
                return i + 15
            return self._pmap_evolver.is_dirty()

        def persistent(self):
            if False:
                while True:
                    i = 10
            if not self.is_dirty():
                return self._original_pset
            return PSet(self._pmap_evolver.persistent())

        def __len__(self):
            if False:
                for i in range(10):
                    print('nop')
            return len(self._pmap_evolver)

    def copy(self):
        if False:
            while True:
                i = 10
        return self

    def evolver(self):
        if False:
            i = 10
            return i + 15
        '\n        Create a new evolver for this pset. For a discussion on evolvers in general see the\n        documentation for the pvector evolver.\n\n        Create the evolver and perform various mutating updates to it:\n\n        >>> s1 = s(1, 2, 3)\n        >>> e = s1.evolver()\n        >>> _ = e.add(4)\n        >>> len(e)\n        4\n        >>> _ = e.remove(1)\n\n        The underlying pset remains the same:\n\n        >>> s1\n        pset([1, 2, 3])\n\n        The changes are kept in the evolver. An updated pmap can be created using the\n        persistent() function on the evolver.\n\n        >>> s2 = e.persistent()\n        >>> s2\n        pset([2, 3, 4])\n\n        The new pset will share data with the original pset in the same way that would have\n        been done if only using operations on the pset.\n        '
        return PSet._Evolver(self)
    __le__ = Set.__le__
    __lt__ = Set.__lt__
    __gt__ = Set.__gt__
    __ge__ = Set.__ge__
    __eq__ = Set.__eq__
    __ne__ = Set.__ne__
    __and__ = Set.__and__
    __or__ = Set.__or__
    __sub__ = Set.__sub__
    __xor__ = Set.__xor__
    issubset = __le__
    issuperset = __ge__
    union = __or__
    intersection = __and__
    difference = __sub__
    symmetric_difference = __xor__
    isdisjoint = Set.isdisjoint
Set.register(PSet)
Hashable.register(PSet)
_EMPTY_PSET = PSet(pmap())

def pset(iterable=(), pre_size=8):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a persistent set from iterable. Optionally takes a sizing parameter equivalent to that\n    used for :py:func:`pmap`.\n\n    >>> s1 = pset([1, 2, 3, 2])\n    >>> s1\n    pset([1, 2, 3])\n    '
    if not iterable:
        return _EMPTY_PSET
    return PSet._from_iterable(iterable, pre_size=pre_size)

def s(*elements):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a persistent set.\n\n    Takes an arbitrary number of arguments to insert into the new set.\n\n    >>> s1 = s(1, 2, 3, 2)\n    >>> s1\n    pset([1, 2, 3])\n    '
    return pset(elements)