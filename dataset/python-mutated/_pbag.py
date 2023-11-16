from collections.abc import Container, Iterable, Sized, Hashable
from functools import reduce
from pyrsistent._pmap import pmap

def _add_to_counters(counters, element):
    if False:
        i = 10
        return i + 15
    return counters.set(element, counters.get(element, 0) + 1)

class PBag(object):
    """
    A persistent bag/multiset type.

    Requires elements to be hashable, and allows duplicates, but has no
    ordering. Bags are hashable.

    Do not instantiate directly, instead use the factory functions :py:func:`b`
    or :py:func:`pbag` to create an instance.

    Some examples:

    >>> s = pbag([1, 2, 3, 1])
    >>> s2 = s.add(4)
    >>> s3 = s2.remove(1)
    >>> s
    pbag([1, 1, 2, 3])
    >>> s2
    pbag([1, 1, 2, 3, 4])
    >>> s3
    pbag([1, 2, 3, 4])
    """
    __slots__ = ('_counts', '__weakref__')

    def __init__(self, counts):
        if False:
            for i in range(10):
                print('nop')
        self._counts = counts

    def add(self, element):
        if False:
            i = 10
            return i + 15
        '\n        Add an element to the bag.\n\n        >>> s = pbag([1])\n        >>> s2 = s.add(1)\n        >>> s3 = s.add(2)\n        >>> s2\n        pbag([1, 1])\n        >>> s3\n        pbag([1, 2])\n        '
        return PBag(_add_to_counters(self._counts, element))

    def update(self, iterable):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update bag with all elements in iterable.\n\n        >>> s = pbag([1])\n        >>> s.update([1, 2])\n        pbag([1, 1, 2])\n        '
        if iterable:
            return PBag(reduce(_add_to_counters, iterable, self._counts))
        return self

    def remove(self, element):
        if False:
            while True:
                i = 10
        '\n        Remove an element from the bag.\n\n        >>> s = pbag([1, 1, 2])\n        >>> s2 = s.remove(1)\n        >>> s3 = s.remove(2)\n        >>> s2\n        pbag([1, 2])\n        >>> s3\n        pbag([1, 1])\n        '
        if element not in self._counts:
            raise KeyError(element)
        elif self._counts[element] == 1:
            newc = self._counts.remove(element)
        else:
            newc = self._counts.set(element, self._counts[element] - 1)
        return PBag(newc)

    def count(self, element):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return the number of times an element appears.\n\n\n        >>> pbag([]).count('non-existent')\n        0\n        >>> pbag([1, 1, 2]).count(1)\n        2\n        "
        return self._counts.get(element, 0)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the length including duplicates.\n\n        >>> len(pbag([1, 1, 2]))\n        3\n        '
        return sum(self._counts.itervalues())

    def __iter__(self):
        if False:
            while True:
                i = 10
        '\n        Return an iterator of all elements, including duplicates.\n\n        >>> list(pbag([1, 1, 2]))\n        [1, 1, 2]\n        >>> list(pbag([1, 2]))\n        [1, 2]\n        '
        for (elt, count) in self._counts.iteritems():
            for i in range(count):
                yield elt

    def __contains__(self, elt):
        if False:
            print('Hello World!')
        '\n        Check if an element is in the bag.\n\n        >>> 1 in pbag([1, 1, 2])\n        True\n        >>> 0 in pbag([1, 2])\n        False\n        '
        return elt in self._counts

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'pbag({0})'.format(list(self))

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if two bags are equivalent, honoring the number of duplicates,\n        and ignoring insertion order.\n\n        >>> pbag([1, 1, 2]) == pbag([1, 2])\n        False\n        >>> pbag([2, 1, 0]) == pbag([0, 1, 2])\n        True\n        '
        if type(other) is not PBag:
            raise TypeError('Can only compare PBag with PBags')
        return self._counts == other._counts

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        raise TypeError('PBags are not orderable')
    __le__ = __lt__
    __gt__ = __lt__
    __ge__ = __lt__

    def __add__(self, other):
        if False:
            return 10
        '\n        Combine elements from two PBags.\n\n        >>> pbag([1, 2, 2]) + pbag([2, 3, 3])\n        pbag([1, 2, 2, 2, 3, 3])\n        '
        if not isinstance(other, PBag):
            return NotImplemented
        result = self._counts.evolver()
        for (elem, other_count) in other._counts.iteritems():
            result[elem] = self.count(elem) + other_count
        return PBag(result.persistent())

    def __sub__(self, other):
        if False:
            print('Hello World!')
        '\n        Remove elements from one PBag that are present in another.\n\n        >>> pbag([1, 2, 2, 2, 3]) - pbag([2, 3, 3, 4])\n        pbag([1, 2, 2])\n        '
        if not isinstance(other, PBag):
            return NotImplemented
        result = self._counts.evolver()
        for (elem, other_count) in other._counts.iteritems():
            newcount = self.count(elem) - other_count
            if newcount > 0:
                result[elem] = newcount
            elif elem in self:
                result.remove(elem)
        return PBag(result.persistent())

    def __or__(self, other):
        if False:
            print('Hello World!')
        '\n        Union: Keep elements that are present in either of two PBags.\n\n        >>> pbag([1, 2, 2, 2]) | pbag([2, 3, 3])\n        pbag([1, 2, 2, 2, 3, 3])\n        '
        if not isinstance(other, PBag):
            return NotImplemented
        result = self._counts.evolver()
        for (elem, other_count) in other._counts.iteritems():
            count = self.count(elem)
            newcount = max(count, other_count)
            result[elem] = newcount
        return PBag(result.persistent())

    def __and__(self, other):
        if False:
            while True:
                i = 10
        '\n        Intersection: Only keep elements that are present in both PBags.\n\n        >>> pbag([1, 2, 2, 2]) & pbag([2, 3, 3])\n        pbag([2])\n        '
        if not isinstance(other, PBag):
            return NotImplemented
        result = pmap().evolver()
        for (elem, count) in self._counts.iteritems():
            newcount = min(count, other.count(elem))
            if newcount > 0:
                result[elem] = newcount
        return PBag(result.persistent())

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Hash based on value of elements.\n\n        >>> m = pmap({pbag([1, 2]): "it\'s here!"})\n        >>> m[pbag([2, 1])]\n        "it\'s here!"\n        >>> pbag([1, 1, 2]) in m\n        False\n        '
        return hash(self._counts)
Container.register(PBag)
Iterable.register(PBag)
Sized.register(PBag)
Hashable.register(PBag)

def b(*elements):
    if False:
        return 10
    '\n    Construct a persistent bag.\n\n    Takes an arbitrary number of arguments to insert into the new persistent\n    bag.\n\n    >>> b(1, 2, 3, 2)\n    pbag([1, 2, 2, 3])\n    '
    return pbag(elements)

def pbag(elements):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert an iterable to a persistent bag.\n\n    Takes an iterable with elements to insert.\n\n    >>> pbag([1, 2, 3, 2])\n    pbag([1, 2, 2, 3])\n    '
    if not elements:
        return _EMPTY_PBAG
    return PBag(reduce(_add_to_counters, elements, pmap()))
_EMPTY_PBAG = PBag(pmap())