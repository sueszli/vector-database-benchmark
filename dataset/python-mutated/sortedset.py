"""Sorted Set
=============

:doc:`Sorted Containers<index>` is an Apache2 licensed Python sorted
collections library, written in pure-Python, and fast as C-extensions. The
:doc:`introduction<introduction>` is the best way to get started.

Sorted set implementations:

.. currentmodule:: sortedcontainers

* :class:`SortedSet`

"""
from itertools import chain
from operator import eq, ne, gt, ge, lt, le
from textwrap import dedent
from .sortedlist import SortedList, recursive_repr
try:
    from collections.abc import MutableSet, Sequence, Set
except ImportError:
    from collections import MutableSet, Sequence, Set

class SortedSet(MutableSet, Sequence):
    """Sorted set is a sorted mutable set.

    Sorted set values are maintained in sorted order. The design of sorted set
    is simple: sorted set uses a set for set-operations and maintains a sorted
    list of values.

    Sorted set values must be hashable and comparable. The hash and total
    ordering of values must not change while they are stored in the sorted set.

    Mutable set methods:

    * :func:`SortedSet.__contains__`
    * :func:`SortedSet.__iter__`
    * :func:`SortedSet.__len__`
    * :func:`SortedSet.add`
    * :func:`SortedSet.discard`

    Sequence methods:

    * :func:`SortedSet.__getitem__`
    * :func:`SortedSet.__delitem__`
    * :func:`SortedSet.__reversed__`

    Methods for removing values:

    * :func:`SortedSet.clear`
    * :func:`SortedSet.pop`
    * :func:`SortedSet.remove`

    Set-operation methods:

    * :func:`SortedSet.difference`
    * :func:`SortedSet.difference_update`
    * :func:`SortedSet.intersection`
    * :func:`SortedSet.intersection_update`
    * :func:`SortedSet.symmetric_difference`
    * :func:`SortedSet.symmetric_difference_update`
    * :func:`SortedSet.union`
    * :func:`SortedSet.update`

    Methods for miscellany:

    * :func:`SortedSet.copy`
    * :func:`SortedSet.count`
    * :func:`SortedSet.__repr__`
    * :func:`SortedSet._check`

    Sorted list methods available:

    * :func:`SortedList.bisect_left`
    * :func:`SortedList.bisect_right`
    * :func:`SortedList.index`
    * :func:`SortedList.irange`
    * :func:`SortedList.islice`
    * :func:`SortedList._reset`

    Additional sorted list methods available, if key-function used:

    * :func:`SortedKeyList.bisect_key_left`
    * :func:`SortedKeyList.bisect_key_right`
    * :func:`SortedKeyList.irange_key`

    Sorted set comparisons use subset and superset relations. Two sorted sets
    are equal if and only if every element of each sorted set is contained in
    the other (each is a subset of the other). A sorted set is less than
    another sorted set if and only if the first sorted set is a proper subset
    of the second sorted set (is a subset, but is not equal). A sorted set is
    greater than another sorted set if and only if the first sorted set is a
    proper superset of the second sorted set (is a superset, but is not equal).

    """

    def __init__(self, iterable=None, key=None):
        if False:
            return 10
        "Initialize sorted set instance.\n\n        Optional `iterable` argument provides an initial iterable of values to\n        initialize the sorted set.\n\n        Optional `key` argument defines a callable that, like the `key`\n        argument to Python's `sorted` function, extracts a comparison key from\n        each value. The default, none, compares values directly.\n\n        Runtime complexity: `O(n*log(n))`\n\n        >>> ss = SortedSet([3, 1, 2, 5, 4])\n        >>> ss\n        SortedSet([1, 2, 3, 4, 5])\n        >>> from operator import neg\n        >>> ss = SortedSet([3, 1, 2, 5, 4], neg)\n        >>> ss\n        SortedSet([5, 4, 3, 2, 1], key=<built-in function neg>)\n\n        :param iterable: initial values (optional)\n        :param key: function used to extract comparison key (optional)\n\n        "
        self._key = key
        if not hasattr(self, '_set'):
            self._set = set()
        self._list = SortedList(self._set, key=key)
        _set = self._set
        self.isdisjoint = _set.isdisjoint
        self.issubset = _set.issubset
        self.issuperset = _set.issuperset
        _list = self._list
        self.bisect_left = _list.bisect_left
        self.bisect = _list.bisect
        self.bisect_right = _list.bisect_right
        self.index = _list.index
        self.irange = _list.irange
        self.islice = _list.islice
        self._reset = _list._reset
        if key is not None:
            self.bisect_key_left = _list.bisect_key_left
            self.bisect_key_right = _list.bisect_key_right
            self.bisect_key = _list.bisect_key
            self.irange_key = _list.irange_key
        if iterable is not None:
            self._update(iterable)

    @classmethod
    def _fromset(cls, values, key=None):
        if False:
            while True:
                i = 10
        'Initialize sorted set from existing set.\n\n        Used internally by set operations that return a new set.\n\n        '
        sorted_set = object.__new__(cls)
        sorted_set._set = values
        sorted_set.__init__(key=key)
        return sorted_set

    @property
    def key(self):
        if False:
            while True:
                i = 10
        'Function used to extract comparison key from values.\n\n        Sorted set compares values directly when the key function is none.\n\n        '
        return self._key

    def __contains__(self, value):
        if False:
            while True:
                i = 10
        'Return true if `value` is an element of the sorted set.\n\n        ``ss.__contains__(value)`` <==> ``value in ss``\n\n        Runtime complexity: `O(1)`\n\n        >>> ss = SortedSet([1, 2, 3, 4, 5])\n        >>> 3 in ss\n        True\n\n        :param value: search for value in sorted set\n        :return: true if `value` in sorted set\n\n        '
        return value in self._set

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        "Lookup value at `index` in sorted set.\n\n        ``ss.__getitem__(index)`` <==> ``ss[index]``\n\n        Supports slicing.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> ss = SortedSet('abcde')\n        >>> ss[2]\n        'c'\n        >>> ss[-1]\n        'e'\n        >>> ss[2:5]\n        ['c', 'd', 'e']\n\n        :param index: integer or slice for indexing\n        :return: value or list of values\n        :raises IndexError: if index out of range\n\n        "
        return self._list[index]

    def __delitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        "Remove value at `index` from sorted set.\n\n        ``ss.__delitem__(index)`` <==> ``del ss[index]``\n\n        Supports slicing.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> ss = SortedSet('abcde')\n        >>> del ss[2]\n        >>> ss\n        SortedSet(['a', 'b', 'd', 'e'])\n        >>> del ss[:2]\n        >>> ss\n        SortedSet(['d', 'e'])\n\n        :param index: integer or slice for indexing\n        :raises IndexError: if index out of range\n\n        "
        _set = self._set
        _list = self._list
        if isinstance(index, slice):
            values = _list[index]
            _set.difference_update(values)
        else:
            value = _list[index]
            _set.remove(value)
        del _list[index]

    def __make_cmp(set_op, symbol, doc):
        if False:
            while True:
                i = 10
        'Make comparator method.'

        def comparer(self, other):
            if False:
                i = 10
                return i + 15
            'Compare method for sorted set and set.'
            if isinstance(other, SortedSet):
                return set_op(self._set, other._set)
            elif isinstance(other, Set):
                return set_op(self._set, other)
            return NotImplemented
        set_op_name = set_op.__name__
        comparer.__name__ = '__{0}__'.format(set_op_name)
        doc_str = 'Return true if and only if sorted set is {0} `other`.\n\n        ``ss.__{1}__(other)`` <==> ``ss {2} other``\n\n        Comparisons use subset and superset semantics as with sets.\n\n        Runtime complexity: `O(n)`\n\n        :param other: `other` set\n        :return: true if sorted set is {0} `other`\n\n        '
        comparer.__doc__ = dedent(doc_str.format(doc, set_op_name, symbol))
        return comparer
    __eq__ = __make_cmp(eq, '==', 'equal to')
    __ne__ = __make_cmp(ne, '!=', 'not equal to')
    __lt__ = __make_cmp(lt, '<', 'a proper subset of')
    __gt__ = __make_cmp(gt, '>', 'a proper superset of')
    __le__ = __make_cmp(le, '<=', 'a subset of')
    __ge__ = __make_cmp(ge, '>=', 'a superset of')
    __make_cmp = staticmethod(__make_cmp)

    def __len__(self):
        if False:
            print('Hello World!')
        'Return the size of the sorted set.\n\n        ``ss.__len__()`` <==> ``len(ss)``\n\n        :return: size of sorted set\n\n        '
        return len(self._set)

    def __iter__(self):
        if False:
            print('Hello World!')
        'Return an iterator over the sorted set.\n\n        ``ss.__iter__()`` <==> ``iter(ss)``\n\n        Iterating the sorted set while adding or deleting values may raise a\n        :exc:`RuntimeError` or fail to iterate over all values.\n\n        '
        return iter(self._list)

    def __reversed__(self):
        if False:
            return 10
        'Return a reverse iterator over the sorted set.\n\n        ``ss.__reversed__()`` <==> ``reversed(ss)``\n\n        Iterating the sorted set while adding or deleting values may raise a\n        :exc:`RuntimeError` or fail to iterate over all values.\n\n        '
        return reversed(self._list)

    def add(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Add `value` to sorted set.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> ss = SortedSet()\n        >>> ss.add(3)\n        >>> ss.add(1)\n        >>> ss.add(2)\n        >>> ss\n        SortedSet([1, 2, 3])\n\n        :param value: value to add to sorted set\n\n        '
        _set = self._set
        if value not in _set:
            _set.add(value)
            self._list.add(value)
    _add = add

    def clear(self):
        if False:
            while True:
                i = 10
        'Remove all values from sorted set.\n\n        Runtime complexity: `O(n)`\n\n        '
        self._set.clear()
        self._list.clear()

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a shallow copy of the sorted set.\n\n        Runtime complexity: `O(n)`\n\n        :return: new sorted set\n\n        '
        return self._fromset(set(self._set), key=self._key)
    __copy__ = copy

    def count(self, value):
        if False:
            print('Hello World!')
        'Return number of occurrences of `value` in the sorted set.\n\n        Runtime complexity: `O(1)`\n\n        >>> ss = SortedSet([1, 2, 3, 4, 5])\n        >>> ss.count(3)\n        1\n\n        :param value: value to count in sorted set\n        :return: count\n\n        '
        return 1 if value in self._set else 0

    def discard(self, value):
        if False:
            while True:
                i = 10
        'Remove `value` from sorted set if it is a member.\n\n        If `value` is not a member, do nothing.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> ss = SortedSet([1, 2, 3, 4, 5])\n        >>> ss.discard(5)\n        >>> ss.discard(0)\n        >>> ss == set([1, 2, 3, 4])\n        True\n\n        :param value: `value` to discard from sorted set\n\n        '
        _set = self._set
        if value in _set:
            _set.remove(value)
            self._list.remove(value)
    _discard = discard

    def pop(self, index=-1):
        if False:
            print('Hello World!')
        "Remove and return value at `index` in sorted set.\n\n        Raise :exc:`IndexError` if the sorted set is empty or index is out of\n        range.\n\n        Negative indices are supported.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> ss = SortedSet('abcde')\n        >>> ss.pop()\n        'e'\n        >>> ss.pop(2)\n        'c'\n        >>> ss\n        SortedSet(['a', 'b', 'd'])\n\n        :param int index: index of value (default -1)\n        :return: value\n        :raises IndexError: if index is out of range\n\n        "
        value = self._list.pop(index)
        self._set.remove(value)
        return value

    def remove(self, value):
        if False:
            i = 10
            return i + 15
        'Remove `value` from sorted set; `value` must be a member.\n\n        If `value` is not a member, raise :exc:`KeyError`.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> ss = SortedSet([1, 2, 3, 4, 5])\n        >>> ss.remove(5)\n        >>> ss == set([1, 2, 3, 4])\n        True\n        >>> ss.remove(0)\n        Traceback (most recent call last):\n          ...\n        KeyError: 0\n\n        :param value: `value` to remove from sorted set\n        :raises KeyError: if `value` is not in sorted set\n\n        '
        self._set.remove(value)
        self._list.remove(value)

    def difference(self, *iterables):
        if False:
            return 10
        'Return the difference of two or more sets as a new sorted set.\n\n        The `difference` method also corresponds to operator ``-``.\n\n        ``ss.__sub__(iterable)`` <==> ``ss - iterable``\n\n        The difference is all values that are in this sorted set but not the\n        other `iterables`.\n\n        >>> ss = SortedSet([1, 2, 3, 4, 5])\n        >>> ss.difference([4, 5, 6, 7])\n        SortedSet([1, 2, 3])\n\n        :param iterables: iterable arguments\n        :return: new sorted set\n\n        '
        diff = self._set.difference(*iterables)
        return self._fromset(diff, key=self._key)
    __sub__ = difference

    def difference_update(self, *iterables):
        if False:
            print('Hello World!')
        'Remove all values of `iterables` from this sorted set.\n\n        The `difference_update` method also corresponds to operator ``-=``.\n\n        ``ss.__isub__(iterable)`` <==> ``ss -= iterable``\n\n        >>> ss = SortedSet([1, 2, 3, 4, 5])\n        >>> _ = ss.difference_update([4, 5, 6, 7])\n        >>> ss\n        SortedSet([1, 2, 3])\n\n        :param iterables: iterable arguments\n        :return: itself\n\n        '
        _set = self._set
        _list = self._list
        values = set(chain(*iterables))
        if 4 * len(values) > len(_set):
            _set.difference_update(values)
            _list.clear()
            _list.update(_set)
        else:
            _discard = self._discard
            for value in values:
                _discard(value)
        return self
    __isub__ = difference_update

    def intersection(self, *iterables):
        if False:
            for i in range(10):
                print('nop')
        'Return the intersection of two or more sets as a new sorted set.\n\n        The `intersection` method also corresponds to operator ``&``.\n\n        ``ss.__and__(iterable)`` <==> ``ss & iterable``\n\n        The intersection is all values that are in this sorted set and each of\n        the other `iterables`.\n\n        >>> ss = SortedSet([1, 2, 3, 4, 5])\n        >>> ss.intersection([4, 5, 6, 7])\n        SortedSet([4, 5])\n\n        :param iterables: iterable arguments\n        :return: new sorted set\n\n        '
        intersect = self._set.intersection(*iterables)
        return self._fromset(intersect, key=self._key)
    __and__ = intersection
    __rand__ = __and__

    def intersection_update(self, *iterables):
        if False:
            i = 10
            return i + 15
        'Update the sorted set with the intersection of `iterables`.\n\n        The `intersection_update` method also corresponds to operator ``&=``.\n\n        ``ss.__iand__(iterable)`` <==> ``ss &= iterable``\n\n        Keep only values found in itself and all `iterables`.\n\n        >>> ss = SortedSet([1, 2, 3, 4, 5])\n        >>> _ = ss.intersection_update([4, 5, 6, 7])\n        >>> ss\n        SortedSet([4, 5])\n\n        :param iterables: iterable arguments\n        :return: itself\n\n        '
        _set = self._set
        _list = self._list
        _set.intersection_update(*iterables)
        _list.clear()
        _list.update(_set)
        return self
    __iand__ = intersection_update

    def symmetric_difference(self, other):
        if False:
            return 10
        'Return the symmetric difference with `other` as a new sorted set.\n\n        The `symmetric_difference` method also corresponds to operator ``^``.\n\n        ``ss.__xor__(other)`` <==> ``ss ^ other``\n\n        The symmetric difference is all values tha are in exactly one of the\n        sets.\n\n        >>> ss = SortedSet([1, 2, 3, 4, 5])\n        >>> ss.symmetric_difference([4, 5, 6, 7])\n        SortedSet([1, 2, 3, 6, 7])\n\n        :param other: `other` iterable\n        :return: new sorted set\n\n        '
        diff = self._set.symmetric_difference(other)
        return self._fromset(diff, key=self._key)
    __xor__ = symmetric_difference
    __rxor__ = __xor__

    def symmetric_difference_update(self, other):
        if False:
            while True:
                i = 10
        'Update the sorted set with the symmetric difference with `other`.\n\n        The `symmetric_difference_update` method also corresponds to operator\n        ``^=``.\n\n        ``ss.__ixor__(other)`` <==> ``ss ^= other``\n\n        Keep only values found in exactly one of itself and `other`.\n\n        >>> ss = SortedSet([1, 2, 3, 4, 5])\n        >>> _ = ss.symmetric_difference_update([4, 5, 6, 7])\n        >>> ss\n        SortedSet([1, 2, 3, 6, 7])\n\n        :param other: `other` iterable\n        :return: itself\n\n        '
        _set = self._set
        _list = self._list
        _set.symmetric_difference_update(other)
        _list.clear()
        _list.update(_set)
        return self
    __ixor__ = symmetric_difference_update

    def union(self, *iterables):
        if False:
            i = 10
            return i + 15
        'Return new sorted set with values from itself and all `iterables`.\n\n        The `union` method also corresponds to operator ``|``.\n\n        ``ss.__or__(iterable)`` <==> ``ss | iterable``\n\n        >>> ss = SortedSet([1, 2, 3, 4, 5])\n        >>> ss.union([4, 5, 6, 7])\n        SortedSet([1, 2, 3, 4, 5, 6, 7])\n\n        :param iterables: iterable arguments\n        :return: new sorted set\n\n        '
        return self.__class__(chain(iter(self), *iterables), key=self._key)
    __or__ = union
    __ror__ = __or__

    def update(self, *iterables):
        if False:
            i = 10
            return i + 15
        'Update the sorted set adding values from all `iterables`.\n\n        The `update` method also corresponds to operator ``|=``.\n\n        ``ss.__ior__(iterable)`` <==> ``ss |= iterable``\n\n        >>> ss = SortedSet([1, 2, 3, 4, 5])\n        >>> _ = ss.update([4, 5, 6, 7])\n        >>> ss\n        SortedSet([1, 2, 3, 4, 5, 6, 7])\n\n        :param iterables: iterable arguments\n        :return: itself\n\n        '
        _set = self._set
        _list = self._list
        values = set(chain(*iterables))
        if 4 * len(values) > len(_set):
            _list = self._list
            _set.update(values)
            _list.clear()
            _list.update(_set)
        else:
            _add = self._add
            for value in values:
                _add(value)
        return self
    __ior__ = update
    _update = update

    def __reduce__(self):
        if False:
            for i in range(10):
                print('nop')
        'Support for pickle.\n\n        The tricks played with exposing methods in :func:`SortedSet.__init__`\n        confuse pickle so customize the reducer.\n\n        '
        return (type(self), (self._set, self._key))

    @recursive_repr()
    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Return string representation of sorted set.\n\n        ``ss.__repr__()`` <==> ``repr(ss)``\n\n        :return: string representation\n\n        '
        _key = self._key
        key = '' if _key is None else ', key={0!r}'.format(_key)
        type_name = type(self).__name__
        return '{0}({1!r}{2})'.format(type_name, list(self), key)

    def _check(self):
        if False:
            return 10
        'Check invariants of sorted set.\n\n        Runtime complexity: `O(n)`\n\n        '
        _set = self._set
        _list = self._list
        _list._check()
        assert len(_set) == len(_list)
        assert all((value in _set for value in _list))