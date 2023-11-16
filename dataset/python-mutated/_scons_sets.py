"""Classes to represent arbitrary sets (including sets of sets).

This module implements sets using dictionaries whose values are
ignored.  The usual operations (union, intersection, deletion, etc.)
are provided as both methods and operators.

Important: sets are not sequences!  While they support 'x in s',
'len(s)', and 'for x in s', none of those operations are unique for
sequences; for example, mappings support all three as well.  The
characteristic operation for sequences is subscripting with small
integers: s[i], for i in range(len(s)).  Sets don't support
subscripting at all.  Also, sequences allow multiple occurrences and
their elements have a definite order; sets on the other hand don't
record multiple occurrences and don't remember the order of element
insertion (which is why they don't support s[i]).

The following classes are provided:

BaseSet -- All the operations common to both mutable and immutable
    sets. This is an abstract class, not meant to be directly
    instantiated.

Set -- Mutable sets, subclass of BaseSet; not hashable.

ImmutableSet -- Immutable sets, subclass of BaseSet; hashable.
    An iterable argument is mandatory to create an ImmutableSet.

_TemporarilyImmutableSet -- A wrapper around a Set, hashable,
    giving the same hash value as the immutable set equivalent
    would have.  Do not use this class directly.

Only hashable objects can be added to a Set. In particular, you cannot
really add a Set as an element to another Set; if you try, what is
actually added is an ImmutableSet built from it (it compares equal to
the one you tried adding).

When you ask if `x in y' where x is a Set and y is a Set or
ImmutableSet, x is wrapped into a _TemporarilyImmutableSet z, and
what's tested is actually `z in y'.

"""
exec('from itertools import ifilterfalse as filterfalse')
__all__ = ['BaseSet', 'Set', 'ImmutableSet']

class BaseSet(object):
    """Common base class for mutable and immutable sets."""
    __slots__ = ['_data']

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'This is an abstract class.'
        if self.__class__ is BaseSet:
            raise TypeError('BaseSet is an abstract class.  Use Set or ImmutableSet.')

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the number of elements of a set.'
        return len(self._data)

    def __repr__(self):
        if False:
            return 10
        "Return string representation of a set.\n\n        This looks like 'Set([<list of elements>])'.\n        "
        return self._repr()
    __str__ = __repr__

    def _repr(self, sort_them=False):
        if False:
            print('Hello World!')
        elements = list(self._data.keys())
        if sort_them:
            elements.sort()
        return '%s(%r)' % (self.__class__.__name__, elements)

    def __iter__(self):
        if False:
            print('Hello World!')
        'Return an iterator over the elements or a set.\n\n        This is the keys iterator for the underlying dict.\n        '
        return self._data.iterkeys()

    def __cmp__(self, other):
        if False:
            print('Hello World!')
        raise TypeError("can't compare sets using cmp()")

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, BaseSet):
            return self._data == other._data
        else:
            return False

    def __ne__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, BaseSet):
            return self._data != other._data
        else:
            return True

    def copy(self):
        if False:
            print('Hello World!')
        'Return a shallow copy of a set.'
        result = self.__class__()
        result._data.update(self._data)
        return result
    __copy__ = copy

    def __deepcopy__(self, memo):
        if False:
            i = 10
            return i + 15
        'Return a deep copy of a set; used by copy module.'
        from copy import deepcopy
        result = self.__class__()
        memo[id(self)] = result
        data = result._data
        value = True
        for elt in self:
            data[deepcopy(elt, memo)] = value
        return result

    def __or__(self, other):
        if False:
            print('Hello World!')
        'Return the union of two sets as a new set.\n\n        (I.e. all elements that are in either set.)\n        '
        if not isinstance(other, BaseSet):
            return NotImplemented
        return self.union(other)

    def union(self, other):
        if False:
            print('Hello World!')
        'Return the union of two sets as a new set.\n\n        (I.e. all elements that are in either set.)\n        '
        result = self.__class__(self)
        result._update(other)
        return result

    def __and__(self, other):
        if False:
            print('Hello World!')
        'Return the intersection of two sets as a new set.\n\n        (I.e. all elements that are in both sets.)\n        '
        if not isinstance(other, BaseSet):
            return NotImplemented
        return self.intersection(other)

    def intersection(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Return the intersection of two sets as a new set.\n\n        (I.e. all elements that are in both sets.)\n        '
        if not isinstance(other, BaseSet):
            other = Set(other)
        if len(self) <= len(other):
            (little, big) = (self, other)
        else:
            (little, big) = (other, self)
        common = iter(filter(big._data.has_key, little))
        return self.__class__(common)

    def __xor__(self, other):
        if False:
            i = 10
            return i + 15
        'Return the symmetric difference of two sets as a new set.\n\n        (I.e. all elements that are in exactly one of the sets.)\n        '
        if not isinstance(other, BaseSet):
            return NotImplemented
        return self.symmetric_difference(other)

    def symmetric_difference(self, other):
        if False:
            i = 10
            return i + 15
        'Return the symmetric difference of two sets as a new set.\n\n        (I.e. all elements that are in exactly one of the sets.)\n        '
        result = self.__class__()
        data = result._data
        value = True
        selfdata = self._data
        try:
            otherdata = other._data
        except AttributeError:
            otherdata = Set(other)._data
        for elt in filterfalse(otherdata.has_key, selfdata):
            data[elt] = value
        for elt in filterfalse(selfdata.has_key, otherdata):
            data[elt] = value
        return result

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        'Return the difference of two sets as a new Set.\n\n        (I.e. all elements that are in this set and not in the other.)\n        '
        if not isinstance(other, BaseSet):
            return NotImplemented
        return self.difference(other)

    def difference(self, other):
        if False:
            print('Hello World!')
        'Return the difference of two sets as a new Set.\n\n        (I.e. all elements that are in this set and not in the other.)\n        '
        result = self.__class__()
        data = result._data
        try:
            otherdata = other._data
        except AttributeError:
            otherdata = Set(other)._data
        value = True
        for elt in filterfalse(otherdata.has_key, self):
            data[elt] = value
        return result

    def __contains__(self, element):
        if False:
            i = 10
            return i + 15
        "Report whether an element is a member of a set.\n\n        (Called in response to the expression `element in self'.)\n        "
        try:
            return element in self._data
        except TypeError:
            transform = getattr(element, '__as_temporarily_immutable__', None)
            if transform is None:
                raise
            return transform() in self._data

    def issubset(self, other):
        if False:
            i = 10
            return i + 15
        'Report whether another set contains this set.'
        self._binary_sanity_check(other)
        if len(self) > len(other):
            return False
        for elt in filterfalse(other._data.has_key, self):
            return False
        return True

    def issuperset(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Report whether this set contains another set.'
        self._binary_sanity_check(other)
        if len(self) < len(other):
            return False
        for elt in filterfalse(self._data.has_key, other):
            return False
        return True
    __le__ = issubset
    __ge__ = issuperset

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        self._binary_sanity_check(other)
        return len(self) < len(other) and self.issubset(other)

    def __gt__(self, other):
        if False:
            while True:
                i = 10
        self._binary_sanity_check(other)
        return len(self) > len(other) and self.issuperset(other)

    def _binary_sanity_check(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, BaseSet):
            raise TypeError('Binary operation only permitted between sets')

    def _compute_hash(self):
        if False:
            while True:
                i = 10
        result = 0
        for elt in self:
            result ^= hash(elt)
        return result

    def _update(self, iterable):
        if False:
            print('Hello World!')
        data = self._data
        if isinstance(iterable, BaseSet):
            data.update(iterable._data)
            return
        value = True
        if type(iterable) in (list, tuple, xrange):
            it = iter(iterable)
            while True:
                try:
                    for element in it:
                        data[element] = value
                    return
                except TypeError:
                    transform = getattr(element, '__as_immutable__', None)
                    if transform is None:
                        raise
                    data[transform()] = value
        else:
            for element in iterable:
                try:
                    data[element] = value
                except TypeError:
                    transform = getattr(element, '__as_immutable__', None)
                    if transform is None:
                        raise
                    data[transform()] = value

class ImmutableSet(BaseSet):
    """Immutable set class."""
    __slots__ = ['_hashcode']

    def __init__(self, iterable=None):
        if False:
            while True:
                i = 10
        'Construct an immutable set from an optional iterable.'
        self._hashcode = None
        self._data = {}
        if iterable is not None:
            self._update(iterable)

    def __hash__(self):
        if False:
            return 10
        if self._hashcode is None:
            self._hashcode = self._compute_hash()
        return self._hashcode

    def __getstate__(self):
        if False:
            while True:
                i = 10
        return (self._data, self._hashcode)

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        (self._data, self._hashcode) = state

class Set(BaseSet):
    """ Mutable set class."""
    __slots__ = []

    def __init__(self, iterable=None):
        if False:
            return 10
        'Construct a set from an optional iterable.'
        self._data = {}
        if iterable is not None:
            self._update(iterable)

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        return (self._data,)

    def __setstate__(self, data):
        if False:
            i = 10
            return i + 15
        (self._data,) = data

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        'A Set cannot be hashed.'
        raise TypeError("Can't hash a Set, only an ImmutableSet.")

    def __ior__(self, other):
        if False:
            return 10
        'Update a set with the union of itself and another.'
        self._binary_sanity_check(other)
        self._data.update(other._data)
        return self

    def union_update(self, other):
        if False:
            print('Hello World!')
        'Update a set with the union of itself and another.'
        self._update(other)

    def __iand__(self, other):
        if False:
            i = 10
            return i + 15
        'Update a set with the intersection of itself and another.'
        self._binary_sanity_check(other)
        self._data = (self & other)._data
        return self

    def intersection_update(self, other):
        if False:
            i = 10
            return i + 15
        'Update a set with the intersection of itself and another.'
        if isinstance(other, BaseSet):
            self &= other
        else:
            self._data = self.intersection(other)._data

    def __ixor__(self, other):
        if False:
            print('Hello World!')
        'Update a set with the symmetric difference of itself and another.'
        self._binary_sanity_check(other)
        self.symmetric_difference_update(other)
        return self

    def symmetric_difference_update(self, other):
        if False:
            while True:
                i = 10
        'Update a set with the symmetric difference of itself and another.'
        data = self._data
        value = True
        if not isinstance(other, BaseSet):
            other = Set(other)
        if self is other:
            self.clear()
        for elt in other:
            if elt in data:
                del data[elt]
            else:
                data[elt] = value

    def __isub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Remove all elements of another set from this set.'
        self._binary_sanity_check(other)
        self.difference_update(other)
        return self

    def difference_update(self, other):
        if False:
            i = 10
            return i + 15
        'Remove all elements of another set from this set.'
        data = self._data
        if not isinstance(other, BaseSet):
            other = Set(other)
        if self is other:
            self.clear()
        for elt in filter(data.has_key, other):
            del data[elt]

    def update(self, iterable):
        if False:
            for i in range(10):
                print('nop')
        'Add all values from an iterable (such as a list or file).'
        self._update(iterable)

    def clear(self):
        if False:
            return 10
        'Remove all elements from this set.'
        self._data.clear()

    def add(self, element):
        if False:
            for i in range(10):
                print('nop')
        'Add an element to a set.\n\n        This has no effect if the element is already present.\n        '
        try:
            self._data[element] = True
        except TypeError:
            transform = getattr(element, '__as_immutable__', None)
            if transform is None:
                raise
            self._data[transform()] = True

    def remove(self, element):
        if False:
            print('Hello World!')
        'Remove an element from a set; it must be a member.\n\n        If the element is not a member, raise a KeyError.\n        '
        try:
            del self._data[element]
        except TypeError:
            transform = getattr(element, '__as_temporarily_immutable__', None)
            if transform is None:
                raise
            del self._data[transform()]

    def discard(self, element):
        if False:
            print('Hello World!')
        'Remove an element from a set if it is a member.\n\n        If the element is not a member, do nothing.\n        '
        try:
            self.remove(element)
        except KeyError:
            pass

    def pop(self):
        if False:
            print('Hello World!')
        'Remove and return an arbitrary set element.'
        return self._data.popitem()[0]

    def __as_immutable__(self):
        if False:
            for i in range(10):
                print('nop')
        return ImmutableSet(self)

    def __as_temporarily_immutable__(self):
        if False:
            i = 10
            return i + 15
        return _TemporarilyImmutableSet(self)

class _TemporarilyImmutableSet(BaseSet):

    def __init__(self, set):
        if False:
            i = 10
            return i + 15
        self._set = set
        self._data = set._data

    def __hash__(self):
        if False:
            print('Hello World!')
        return self._set._compute_hash()