"""Sorted List
==============

:doc:`Sorted Containers<index>` is an Apache2 licensed Python sorted
collections library, written in pure-Python, and fast as C-extensions. The
:doc:`introduction<introduction>` is the best way to get started.

Sorted list implementations:

.. currentmodule:: sortedcontainers

* :class:`SortedList`
* :class:`SortedKeyList`

"""
from __future__ import print_function
import sys
import traceback
from bisect import bisect_left, bisect_right, insort
from itertools import chain, repeat, starmap
from math import log
from operator import add, eq, ne, gt, ge, lt, le, iadd
from textwrap import dedent
try:
    from collections.abc import Sequence, MutableSequence
except ImportError:
    from collections import Sequence, MutableSequence
from functools import wraps
from sys import hexversion
if hexversion < 50331648:
    from itertools import imap as map
    from itertools import izip as zip
    try:
        from thread import get_ident
    except ImportError:
        from dummy_thread import get_ident
else:
    from functools import reduce
    try:
        from _thread import get_ident
    except ImportError:
        from _dummy_thread import get_ident

def recursive_repr(fillvalue='...'):
    if False:
        return 10
    'Decorator to make a repr function return fillvalue for a recursive call.'

    def decorating_function(user_function):
        if False:
            return 10
        repr_running = set()

        @wraps(user_function)
        def wrapper(self):
            if False:
                for i in range(10):
                    print('nop')
            key = (id(self), get_ident())
            if key in repr_running:
                return fillvalue
            repr_running.add(key)
            try:
                result = user_function(self)
            finally:
                repr_running.discard(key)
            return result
        return wrapper
    return decorating_function

class SortedList(MutableSequence):
    """Sorted list is a sorted mutable sequence.

    Sorted list values are maintained in sorted order.

    Sorted list values must be comparable. The total ordering of values must
    not change while they are stored in the sorted list.

    Methods for adding values:

    * :func:`SortedList.add`
    * :func:`SortedList.update`
    * :func:`SortedList.__add__`
    * :func:`SortedList.__iadd__`
    * :func:`SortedList.__mul__`
    * :func:`SortedList.__imul__`

    Methods for removing values:

    * :func:`SortedList.clear`
    * :func:`SortedList.discard`
    * :func:`SortedList.remove`
    * :func:`SortedList.pop`
    * :func:`SortedList.__delitem__`

    Methods for looking up values:

    * :func:`SortedList.bisect_left`
    * :func:`SortedList.bisect_right`
    * :func:`SortedList.count`
    * :func:`SortedList.index`
    * :func:`SortedList.__contains__`
    * :func:`SortedList.__getitem__`

    Methods for iterating values:

    * :func:`SortedList.irange`
    * :func:`SortedList.islice`
    * :func:`SortedList.__iter__`
    * :func:`SortedList.__reversed__`

    Methods for miscellany:

    * :func:`SortedList.copy`
    * :func:`SortedList.__len__`
    * :func:`SortedList.__repr__`
    * :func:`SortedList._check`
    * :func:`SortedList._reset`

    Sorted lists use lexicographical ordering semantics when compared to other
    sequences.

    Some methods of mutable sequences are not supported and will raise
    not-implemented error.

    """
    DEFAULT_LOAD_FACTOR = 1000

    def __init__(self, iterable=None, key=None):
        if False:
            i = 10
            return i + 15
        'Initialize sorted list instance.\n\n        Optional `iterable` argument provides an initial iterable of values to\n        initialize the sorted list.\n\n        Runtime complexity: `O(n*log(n))`\n\n        >>> sl = SortedList()\n        >>> sl\n        SortedList([])\n        >>> sl = SortedList([3, 1, 2, 5, 4])\n        >>> sl\n        SortedList([1, 2, 3, 4, 5])\n\n        :param iterable: initial values (optional)\n\n        '
        assert key is None
        self._len = 0
        self._load = self.DEFAULT_LOAD_FACTOR
        self._lists = []
        self._maxes = []
        self._index = []
        self._offset = 0
        if iterable is not None:
            self._update(iterable)

    def __new__(cls, iterable=None, key=None):
        if False:
            i = 10
            return i + 15
        'Create new sorted list or sorted-key list instance.\n\n        Optional `key`-function argument will return an instance of subtype\n        :class:`SortedKeyList`.\n\n        >>> sl = SortedList()\n        >>> isinstance(sl, SortedList)\n        True\n        >>> sl = SortedList(key=lambda x: -x)\n        >>> isinstance(sl, SortedList)\n        True\n        >>> isinstance(sl, SortedKeyList)\n        True\n\n        :param iterable: initial values (optional)\n        :param key: function used to extract comparison key (optional)\n        :return: sorted list or sorted-key list instance\n\n        '
        if key is None:
            return object.__new__(cls)
        elif cls is SortedList:
            return object.__new__(SortedKeyList)
        else:
            raise TypeError('inherit SortedKeyList for key argument')

    @property
    def key(self):
        if False:
            while True:
                i = 10
        'Function used to extract comparison key from values.\n\n        Sorted list compares values directly so the key function is none.\n\n        '
        return None

    def _reset(self, load):
        if False:
            for i in range(10):
                print('nop')
        "Reset sorted list load factor.\n\n        The `load` specifies the load-factor of the list. The default load\n        factor of 1000 works well for lists from tens to tens-of-millions of\n        values. Good practice is to use a value that is the cube root of the\n        list size. With billions of elements, the best load factor depends on\n        your usage. It's best to leave the load factor at the default until you\n        start benchmarking.\n\n        See :doc:`implementation` and :doc:`performance-scale` for more\n        information.\n\n        Runtime complexity: `O(n)`\n\n        :param int load: load-factor for sorted list sublists\n\n        "
        values = reduce(iadd, self._lists, [])
        self._clear()
        self._load = load
        self._update(values)

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        'Remove all values from sorted list.\n\n        Runtime complexity: `O(n)`\n\n        '
        self._len = 0
        del self._lists[:]
        del self._maxes[:]
        del self._index[:]
        self._offset = 0
    _clear = clear

    def add(self, value):
        if False:
            print('Hello World!')
        'Add `value` to sorted list.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sl = SortedList()\n        >>> sl.add(3)\n        >>> sl.add(1)\n        >>> sl.add(2)\n        >>> sl\n        SortedList([1, 2, 3])\n\n        :param value: value to add to sorted list\n\n        '
        _lists = self._lists
        _maxes = self._maxes
        if _maxes:
            pos = bisect_right(_maxes, value)
            if pos == len(_maxes):
                pos -= 1
                _lists[pos].append(value)
                _maxes[pos] = value
            else:
                insort(_lists[pos], value)
            self._expand(pos)
        else:
            _lists.append([value])
            _maxes.append(value)
        self._len += 1

    def _expand(self, pos):
        if False:
            print('Hello World!')
        'Split sublists with length greater than double the load-factor.\n\n        Updates the index when the sublist length is less than double the load\n        level. This requires incrementing the nodes in a traversal from the\n        leaf node to the root. For an example traversal see\n        ``SortedList._loc``.\n\n        '
        _load = self._load
        _lists = self._lists
        _index = self._index
        if len(_lists[pos]) > _load << 1:
            _maxes = self._maxes
            _lists_pos = _lists[pos]
            half = _lists_pos[_load:]
            del _lists_pos[_load:]
            _maxes[pos] = _lists_pos[-1]
            _lists.insert(pos + 1, half)
            _maxes.insert(pos + 1, half[-1])
            del _index[:]
        elif _index:
            child = self._offset + pos
            while child:
                _index[child] += 1
                child = child - 1 >> 1
            _index[0] += 1

    def update(self, iterable):
        if False:
            i = 10
            return i + 15
        'Update sorted list by adding all values from `iterable`.\n\n        Runtime complexity: `O(k*log(n))` -- approximate.\n\n        >>> sl = SortedList()\n        >>> sl.update([3, 1, 2])\n        >>> sl\n        SortedList([1, 2, 3])\n\n        :param iterable: iterable of values to add\n\n        '
        _lists = self._lists
        _maxes = self._maxes
        values = sorted(iterable)
        if _maxes:
            if len(values) * 4 >= self._len:
                _lists.append(values)
                values = reduce(iadd, _lists, [])
                values.sort()
                self._clear()
            else:
                _add = self.add
                for val in values:
                    _add(val)
                return
        _load = self._load
        _lists.extend((values[pos:pos + _load] for pos in range(0, len(values), _load)))
        _maxes.extend((sublist[-1] for sublist in _lists))
        self._len = len(values)
        del self._index[:]
    _update = update

    def __contains__(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Return true if `value` is an element of the sorted list.\n\n        ``sl.__contains__(value)`` <==> ``value in sl``\n\n        Runtime complexity: `O(log(n))`\n\n        >>> sl = SortedList([1, 2, 3, 4, 5])\n        >>> 3 in sl\n        True\n\n        :param value: search for value in sorted list\n        :return: true if `value` in sorted list\n\n        '
        _maxes = self._maxes
        if not _maxes:
            return False
        pos = bisect_left(_maxes, value)
        if pos == len(_maxes):
            return False
        _lists = self._lists
        idx = bisect_left(_lists[pos], value)
        return _lists[pos][idx] == value

    def discard(self, value):
        if False:
            print('Hello World!')
        'Remove `value` from sorted list if it is a member.\n\n        If `value` is not a member, do nothing.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sl = SortedList([1, 2, 3, 4, 5])\n        >>> sl.discard(5)\n        >>> sl.discard(0)\n        >>> sl == [1, 2, 3, 4]\n        True\n\n        :param value: `value` to discard from sorted list\n\n        '
        _maxes = self._maxes
        if not _maxes:
            return
        pos = bisect_left(_maxes, value)
        if pos == len(_maxes):
            return
        _lists = self._lists
        idx = bisect_left(_lists[pos], value)
        if _lists[pos][idx] == value:
            self._delete(pos, idx)

    def remove(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Remove `value` from sorted list; `value` must be a member.\n\n        If `value` is not a member, raise ValueError.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sl = SortedList([1, 2, 3, 4, 5])\n        >>> sl.remove(5)\n        >>> sl == [1, 2, 3, 4]\n        True\n        >>> sl.remove(0)\n        Traceback (most recent call last):\n          ...\n        ValueError: 0 not in list\n\n        :param value: `value` to remove from sorted list\n        :raises ValueError: if `value` is not in sorted list\n\n        '
        _maxes = self._maxes
        if not _maxes:
            raise ValueError('{0!r} not in list'.format(value))
        pos = bisect_left(_maxes, value)
        if pos == len(_maxes):
            raise ValueError('{0!r} not in list'.format(value))
        _lists = self._lists
        idx = bisect_left(_lists[pos], value)
        if _lists[pos][idx] == value:
            self._delete(pos, idx)
        else:
            raise ValueError('{0!r} not in list'.format(value))

    def _delete(self, pos, idx):
        if False:
            for i in range(10):
                print('nop')
        'Delete value at the given `(pos, idx)`.\n\n        Combines lists that are less than half the load level.\n\n        Updates the index when the sublist length is more than half the load\n        level. This requires decrementing the nodes in a traversal from the\n        leaf node to the root. For an example traversal see\n        ``SortedList._loc``.\n\n        :param int pos: lists index\n        :param int idx: sublist index\n\n        '
        _lists = self._lists
        _maxes = self._maxes
        _index = self._index
        _lists_pos = _lists[pos]
        del _lists_pos[idx]
        self._len -= 1
        len_lists_pos = len(_lists_pos)
        if len_lists_pos > self._load >> 1:
            _maxes[pos] = _lists_pos[-1]
            if _index:
                child = self._offset + pos
                while child > 0:
                    _index[child] -= 1
                    child = child - 1 >> 1
                _index[0] -= 1
        elif len(_lists) > 1:
            if not pos:
                pos += 1
            prev = pos - 1
            _lists[prev].extend(_lists[pos])
            _maxes[prev] = _lists[prev][-1]
            del _lists[pos]
            del _maxes[pos]
            del _index[:]
            self._expand(prev)
        elif len_lists_pos:
            _maxes[pos] = _lists_pos[-1]
        else:
            del _lists[pos]
            del _maxes[pos]
            del _index[:]

    def _loc(self, pos, idx):
        if False:
            return 10
        'Convert an index pair (lists index, sublist index) into a single\n        index number that corresponds to the position of the value in the\n        sorted list.\n\n        Many queries require the index be built. Details of the index are\n        described in ``SortedList._build_index``.\n\n        Indexing requires traversing the tree from a leaf node to the root. The\n        parent of each node is easily computable at ``(pos - 1) // 2``.\n\n        Left-child nodes are always at odd indices and right-child nodes are\n        always at even indices.\n\n        When traversing up from a right-child node, increment the total by the\n        left-child node.\n\n        The final index is the sum from traversal and the index in the sublist.\n\n        For example, using the index from ``SortedList._build_index``::\n\n            _index = 14 5 9 3 2 4 5\n            _offset = 3\n\n        Tree::\n\n                 14\n              5      9\n            3   2  4   5\n\n        Converting an index pair (2, 3) into a single index involves iterating\n        like so:\n\n        1. Starting at the leaf node: offset + alpha = 3 + 2 = 5. We identify\n           the node as a left-child node. At such nodes, we simply traverse to\n           the parent.\n\n        2. At node 9, position 2, we recognize the node as a right-child node\n           and accumulate the left-child in our total. Total is now 5 and we\n           traverse to the parent at position 0.\n\n        3. Iteration ends at the root.\n\n        The index is then the sum of the total and sublist index: 5 + 3 = 8.\n\n        :param int pos: lists index\n        :param int idx: sublist index\n        :return: index in sorted list\n\n        '
        if not pos:
            return idx
        _index = self._index
        if not _index:
            self._build_index()
        total = 0
        pos += self._offset
        while pos:
            if not pos & 1:
                total += _index[pos - 1]
            pos = pos - 1 >> 1
        return total + idx

    def _pos(self, idx):
        if False:
            for i in range(10):
                print('nop')
        'Convert an index into an index pair (lists index, sublist index)\n        that can be used to access the corresponding lists position.\n\n        Many queries require the index be built. Details of the index are\n        described in ``SortedList._build_index``.\n\n        Indexing requires traversing the tree to a leaf node. Each node has two\n        children which are easily computable. Given an index, pos, the\n        left-child is at ``pos * 2 + 1`` and the right-child is at ``pos * 2 +\n        2``.\n\n        When the index is less than the left-child, traversal moves to the\n        left sub-tree. Otherwise, the index is decremented by the left-child\n        and traversal moves to the right sub-tree.\n\n        At a child node, the indexing pair is computed from the relative\n        position of the child node as compared with the offset and the remaining\n        index.\n\n        For example, using the index from ``SortedList._build_index``::\n\n            _index = 14 5 9 3 2 4 5\n            _offset = 3\n\n        Tree::\n\n                 14\n              5      9\n            3   2  4   5\n\n        Indexing position 8 involves iterating like so:\n\n        1. Starting at the root, position 0, 8 is compared with the left-child\n           node (5) which it is greater than. When greater the index is\n           decremented and the position is updated to the right child node.\n\n        2. At node 9 with index 3, we again compare the index to the left-child\n           node with value 4. Because the index is the less than the left-child\n           node, we simply traverse to the left.\n\n        3. At node 4 with index 3, we recognize that we are at a leaf node and\n           stop iterating.\n\n        4. To compute the sublist index, we subtract the offset from the index\n           of the leaf node: 5 - 3 = 2. To compute the index in the sublist, we\n           simply use the index remaining from iteration. In this case, 3.\n\n        The final index pair from our example is (2, 3) which corresponds to\n        index 8 in the sorted list.\n\n        :param int idx: index in sorted list\n        :return: (lists index, sublist index) pair\n\n        '
        if idx < 0:
            last_len = len(self._lists[-1])
            if -idx <= last_len:
                return (len(self._lists) - 1, last_len + idx)
            idx += self._len
            if idx < 0:
                raise IndexError('list index out of range')
        elif idx >= self._len:
            raise IndexError('list index out of range')
        if idx < len(self._lists[0]):
            return (0, idx)
        _index = self._index
        if not _index:
            self._build_index()
        pos = 0
        child = 1
        len_index = len(_index)
        while child < len_index:
            index_child = _index[child]
            if idx < index_child:
                pos = child
            else:
                idx -= index_child
                pos = child + 1
            child = (pos << 1) + 1
        return (pos - self._offset, idx)

    def _build_index(self):
        if False:
            for i in range(10):
                print('nop')
        'Build a positional index for indexing the sorted list.\n\n        Indexes are represented as binary trees in a dense array notation\n        similar to a binary heap.\n\n        For example, given a lists representation storing integers::\n\n            0: [1, 2, 3]\n            1: [4, 5]\n            2: [6, 7, 8, 9]\n            3: [10, 11, 12, 13, 14]\n\n        The first transformation maps the sub-lists by their length. The\n        first row of the index is the length of the sub-lists::\n\n            0: [3, 2, 4, 5]\n\n        Each row after that is the sum of consecutive pairs of the previous\n        row::\n\n            1: [5, 9]\n            2: [14]\n\n        Finally, the index is built by concatenating these lists together::\n\n            _index = [14, 5, 9, 3, 2, 4, 5]\n\n        An offset storing the start of the first row is also stored::\n\n            _offset = 3\n\n        When built, the index can be used for efficient indexing into the list.\n        See the comment and notes on ``SortedList._pos`` for details.\n\n        '
        row0 = list(map(len, self._lists))
        if len(row0) == 1:
            self._index[:] = row0
            self._offset = 0
            return
        head = iter(row0)
        tail = iter(head)
        row1 = list(starmap(add, zip(head, tail)))
        if len(row0) & 1:
            row1.append(row0[-1])
        if len(row1) == 1:
            self._index[:] = row1 + row0
            self._offset = 1
            return
        size = 2 ** (int(log(len(row1) - 1, 2)) + 1)
        row1.extend(repeat(0, size - len(row1)))
        tree = [row0, row1]
        while len(tree[-1]) > 1:
            head = iter(tree[-1])
            tail = iter(head)
            row = list(starmap(add, zip(head, tail)))
            tree.append(row)
        reduce(iadd, reversed(tree), self._index)
        self._offset = size * 2 - 1

    def __delitem__(self, index):
        if False:
            return 10
        "Remove value at `index` from sorted list.\n\n        ``sl.__delitem__(index)`` <==> ``del sl[index]``\n\n        Supports slicing.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sl = SortedList('abcde')\n        >>> del sl[2]\n        >>> sl\n        SortedList(['a', 'b', 'd', 'e'])\n        >>> del sl[:2]\n        >>> sl\n        SortedList(['d', 'e'])\n\n        :param index: integer or slice for indexing\n        :raises IndexError: if index out of range\n\n        "
        if isinstance(index, slice):
            (start, stop, step) = index.indices(self._len)
            if step == 1 and start < stop:
                if start == 0 and stop == self._len:
                    return self._clear()
                elif self._len <= 8 * (stop - start):
                    values = self._getitem(slice(None, start))
                    if stop < self._len:
                        values += self._getitem(slice(stop, None))
                    self._clear()
                    return self._update(values)
            indices = range(start, stop, step)
            if step > 0:
                indices = reversed(indices)
            (_pos, _delete) = (self._pos, self._delete)
            for index in indices:
                (pos, idx) = _pos(index)
                _delete(pos, idx)
        else:
            (pos, idx) = self._pos(index)
            self._delete(pos, idx)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        "Lookup value at `index` in sorted list.\n\n        ``sl.__getitem__(index)`` <==> ``sl[index]``\n\n        Supports slicing.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sl = SortedList('abcde')\n        >>> sl[1]\n        'b'\n        >>> sl[-1]\n        'e'\n        >>> sl[2:5]\n        ['c', 'd', 'e']\n\n        :param index: integer or slice for indexing\n        :return: value or list of values\n        :raises IndexError: if index out of range\n\n        "
        _lists = self._lists
        if isinstance(index, slice):
            (start, stop, step) = index.indices(self._len)
            if step == 1 and start < stop:
                if start == 0 and stop == self._len:
                    return reduce(iadd, self._lists, [])
                (start_pos, start_idx) = self._pos(start)
                start_list = _lists[start_pos]
                stop_idx = start_idx + stop - start
                if len(start_list) >= stop_idx:
                    return start_list[start_idx:stop_idx]
                if stop == self._len:
                    stop_pos = len(_lists) - 1
                    stop_idx = len(_lists[stop_pos])
                else:
                    (stop_pos, stop_idx) = self._pos(stop)
                prefix = _lists[start_pos][start_idx:]
                middle = _lists[start_pos + 1:stop_pos]
                result = reduce(iadd, middle, prefix)
                result += _lists[stop_pos][:stop_idx]
                return result
            if step == -1 and start > stop:
                result = self._getitem(slice(stop + 1, start + 1))
                result.reverse()
                return result
            indices = range(start, stop, step)
            return list((self._getitem(index) for index in indices))
        else:
            if self._len:
                if index == 0:
                    return _lists[0][0]
                elif index == -1:
                    return _lists[-1][-1]
            else:
                raise IndexError('list index out of range')
            if 0 <= index < len(_lists[0]):
                return _lists[0][index]
            len_last = len(_lists[-1])
            if -len_last < index < 0:
                return _lists[-1][len_last + index]
            (pos, idx) = self._pos(index)
            return _lists[pos][idx]
    _getitem = __getitem__

    def __setitem__(self, index, value):
        if False:
            i = 10
            return i + 15
        'Raise not-implemented error.\n\n        ``sl.__setitem__(index, value)`` <==> ``sl[index] = value``\n\n        :raises NotImplementedError: use ``del sl[index]`` and\n            ``sl.add(value)`` instead\n\n        '
        message = 'use ``del sl[index]`` and ``sl.add(value)`` instead'
        raise NotImplementedError(message)

    def __iter__(self):
        if False:
            print('Hello World!')
        'Return an iterator over the sorted list.\n\n        ``sl.__iter__()`` <==> ``iter(sl)``\n\n        Iterating the sorted list while adding or deleting values may raise a\n        :exc:`RuntimeError` or fail to iterate over all values.\n\n        '
        return chain.from_iterable(self._lists)

    def __reversed__(self):
        if False:
            print('Hello World!')
        'Return a reverse iterator over the sorted list.\n\n        ``sl.__reversed__()`` <==> ``reversed(sl)``\n\n        Iterating the sorted list while adding or deleting values may raise a\n        :exc:`RuntimeError` or fail to iterate over all values.\n\n        '
        return chain.from_iterable(map(reversed, reversed(self._lists)))

    def reverse(self):
        if False:
            print('Hello World!')
        'Raise not-implemented error.\n\n        Sorted list maintains values in ascending sort order. Values may not be\n        reversed in-place.\n\n        Use ``reversed(sl)`` for an iterator over values in descending sort\n        order.\n\n        Implemented to override `MutableSequence.reverse` which provides an\n        erroneous default implementation.\n\n        :raises NotImplementedError: use ``reversed(sl)`` instead\n\n        '
        raise NotImplementedError('use ``reversed(sl)`` instead')

    def islice(self, start=None, stop=None, reverse=False):
        if False:
            for i in range(10):
                print('nop')
        "Return an iterator that slices sorted list from `start` to `stop`.\n\n        The `start` and `stop` index are treated inclusive and exclusive,\n        respectively.\n\n        Both `start` and `stop` default to `None` which is automatically\n        inclusive of the beginning and end of the sorted list.\n\n        When `reverse` is `True` the values are yielded from the iterator in\n        reverse order; `reverse` defaults to `False`.\n\n        >>> sl = SortedList('abcdefghij')\n        >>> it = sl.islice(2, 6)\n        >>> list(it)\n        ['c', 'd', 'e', 'f']\n\n        :param int start: start index (inclusive)\n        :param int stop: stop index (exclusive)\n        :param bool reverse: yield values in reverse order\n        :return: iterator\n\n        "
        _len = self._len
        if not _len:
            return iter(())
        (start, stop, _) = slice(start, stop).indices(self._len)
        if start >= stop:
            return iter(())
        _pos = self._pos
        (min_pos, min_idx) = _pos(start)
        if stop == _len:
            max_pos = len(self._lists) - 1
            max_idx = len(self._lists[-1])
        else:
            (max_pos, max_idx) = _pos(stop)
        return self._islice(min_pos, min_idx, max_pos, max_idx, reverse)

    def _islice(self, min_pos, min_idx, max_pos, max_idx, reverse):
        if False:
            return 10
        'Return an iterator that slices sorted list using two index pairs.\n\n        The index pairs are (min_pos, min_idx) and (max_pos, max_idx), the\n        first inclusive and the latter exclusive. See `_pos` for details on how\n        an index is converted to an index pair.\n\n        When `reverse` is `True`, values are yielded from the iterator in\n        reverse order.\n\n        '
        _lists = self._lists
        if min_pos > max_pos:
            return iter(())
        if min_pos == max_pos:
            if reverse:
                indices = reversed(range(min_idx, max_idx))
                return map(_lists[min_pos].__getitem__, indices)
            indices = range(min_idx, max_idx)
            return map(_lists[min_pos].__getitem__, indices)
        next_pos = min_pos + 1
        if next_pos == max_pos:
            if reverse:
                min_indices = range(min_idx, len(_lists[min_pos]))
                max_indices = range(max_idx)
                return chain(map(_lists[max_pos].__getitem__, reversed(max_indices)), map(_lists[min_pos].__getitem__, reversed(min_indices)))
            min_indices = range(min_idx, len(_lists[min_pos]))
            max_indices = range(max_idx)
            return chain(map(_lists[min_pos].__getitem__, min_indices), map(_lists[max_pos].__getitem__, max_indices))
        if reverse:
            min_indices = range(min_idx, len(_lists[min_pos]))
            sublist_indices = range(next_pos, max_pos)
            sublists = map(_lists.__getitem__, reversed(sublist_indices))
            max_indices = range(max_idx)
            return chain(map(_lists[max_pos].__getitem__, reversed(max_indices)), chain.from_iterable(map(reversed, sublists)), map(_lists[min_pos].__getitem__, reversed(min_indices)))
        min_indices = range(min_idx, len(_lists[min_pos]))
        sublist_indices = range(next_pos, max_pos)
        sublists = map(_lists.__getitem__, sublist_indices)
        max_indices = range(max_idx)
        return chain(map(_lists[min_pos].__getitem__, min_indices), chain.from_iterable(sublists), map(_lists[max_pos].__getitem__, max_indices))

    def irange(self, minimum=None, maximum=None, inclusive=(True, True), reverse=False):
        if False:
            while True:
                i = 10
        "Create an iterator of values between `minimum` and `maximum`.\n\n        Both `minimum` and `maximum` default to `None` which is automatically\n        inclusive of the beginning and end of the sorted list.\n\n        The argument `inclusive` is a pair of booleans that indicates whether\n        the minimum and maximum ought to be included in the range,\n        respectively. The default is ``(True, True)`` such that the range is\n        inclusive of both minimum and maximum.\n\n        When `reverse` is `True` the values are yielded from the iterator in\n        reverse order; `reverse` defaults to `False`.\n\n        >>> sl = SortedList('abcdefghij')\n        >>> it = sl.irange('c', 'f')\n        >>> list(it)\n        ['c', 'd', 'e', 'f']\n\n        :param minimum: minimum value to start iterating\n        :param maximum: maximum value to stop iterating\n        :param inclusive: pair of booleans\n        :param bool reverse: yield values in reverse order\n        :return: iterator\n\n        "
        _maxes = self._maxes
        if not _maxes:
            return iter(())
        _lists = self._lists
        if minimum is None:
            min_pos = 0
            min_idx = 0
        elif inclusive[0]:
            min_pos = bisect_left(_maxes, minimum)
            if min_pos == len(_maxes):
                return iter(())
            min_idx = bisect_left(_lists[min_pos], minimum)
        else:
            min_pos = bisect_right(_maxes, minimum)
            if min_pos == len(_maxes):
                return iter(())
            min_idx = bisect_right(_lists[min_pos], minimum)
        if maximum is None:
            max_pos = len(_maxes) - 1
            max_idx = len(_lists[max_pos])
        elif inclusive[1]:
            max_pos = bisect_right(_maxes, maximum)
            if max_pos == len(_maxes):
                max_pos -= 1
                max_idx = len(_lists[max_pos])
            else:
                max_idx = bisect_right(_lists[max_pos], maximum)
        else:
            max_pos = bisect_left(_maxes, maximum)
            if max_pos == len(_maxes):
                max_pos -= 1
                max_idx = len(_lists[max_pos])
            else:
                max_idx = bisect_left(_lists[max_pos], maximum)
        return self._islice(min_pos, min_idx, max_pos, max_idx, reverse)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        'Return the size of the sorted list.\n\n        ``sl.__len__()`` <==> ``len(sl)``\n\n        :return: size of sorted list\n\n        '
        return self._len

    def bisect_left(self, value):
        if False:
            return 10
        'Return an index to insert `value` in the sorted list.\n\n        If the `value` is already present, the insertion point will be before\n        (to the left of) any existing values.\n\n        Similar to the `bisect` module in the standard library.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sl = SortedList([10, 11, 12, 13, 14])\n        >>> sl.bisect_left(12)\n        2\n\n        :param value: insertion index of value in sorted list\n        :return: index\n\n        '
        _maxes = self._maxes
        if not _maxes:
            return 0
        pos = bisect_left(_maxes, value)
        if pos == len(_maxes):
            return self._len
        idx = bisect_left(self._lists[pos], value)
        return self._loc(pos, idx)

    def bisect_right(self, value):
        if False:
            return 10
        'Return an index to insert `value` in the sorted list.\n\n        Similar to `bisect_left`, but if `value` is already present, the\n        insertion point will be after (to the right of) any existing values.\n\n        Similar to the `bisect` module in the standard library.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sl = SortedList([10, 11, 12, 13, 14])\n        >>> sl.bisect_right(12)\n        3\n\n        :param value: insertion index of value in sorted list\n        :return: index\n\n        '
        _maxes = self._maxes
        if not _maxes:
            return 0
        pos = bisect_right(_maxes, value)
        if pos == len(_maxes):
            return self._len
        idx = bisect_right(self._lists[pos], value)
        return self._loc(pos, idx)
    bisect = bisect_right
    _bisect_right = bisect_right

    def count(self, value):
        if False:
            while True:
                i = 10
        'Return number of occurrences of `value` in the sorted list.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sl = SortedList([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])\n        >>> sl.count(3)\n        3\n\n        :param value: value to count in sorted list\n        :return: count\n\n        '
        _maxes = self._maxes
        if not _maxes:
            return 0
        pos_left = bisect_left(_maxes, value)
        if pos_left == len(_maxes):
            return 0
        _lists = self._lists
        idx_left = bisect_left(_lists[pos_left], value)
        pos_right = bisect_right(_maxes, value)
        if pos_right == len(_maxes):
            return self._len - self._loc(pos_left, idx_left)
        idx_right = bisect_right(_lists[pos_right], value)
        if pos_left == pos_right:
            return idx_right - idx_left
        right = self._loc(pos_right, idx_right)
        left = self._loc(pos_left, idx_left)
        return right - left

    def copy(self):
        if False:
            while True:
                i = 10
        'Return a shallow copy of the sorted list.\n\n        Runtime complexity: `O(n)`\n\n        :return: new sorted list\n\n        '
        return self.__class__(self)
    __copy__ = copy

    def append(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Raise not-implemented error.\n\n        Implemented to override `MutableSequence.append` which provides an\n        erroneous default implementation.\n\n        :raises NotImplementedError: use ``sl.add(value)`` instead\n\n        '
        raise NotImplementedError('use ``sl.add(value)`` instead')

    def extend(self, values):
        if False:
            while True:
                i = 10
        'Raise not-implemented error.\n\n        Implemented to override `MutableSequence.extend` which provides an\n        erroneous default implementation.\n\n        :raises NotImplementedError: use ``sl.update(values)`` instead\n\n        '
        raise NotImplementedError('use ``sl.update(values)`` instead')

    def insert(self, index, value):
        if False:
            return 10
        'Raise not-implemented error.\n\n        :raises NotImplementedError: use ``sl.add(value)`` instead\n\n        '
        raise NotImplementedError('use ``sl.add(value)`` instead')

    def pop(self, index=-1):
        if False:
            while True:
                i = 10
        "Remove and return value at `index` in sorted list.\n\n        Raise :exc:`IndexError` if the sorted list is empty or index is out of\n        range.\n\n        Negative indices are supported.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sl = SortedList('abcde')\n        >>> sl.pop()\n        'e'\n        >>> sl.pop(2)\n        'c'\n        >>> sl\n        SortedList(['a', 'b', 'd'])\n\n        :param int index: index of value (default -1)\n        :return: value\n        :raises IndexError: if index is out of range\n\n        "
        if not self._len:
            raise IndexError('pop index out of range')
        _lists = self._lists
        if index == 0:
            val = _lists[0][0]
            self._delete(0, 0)
            return val
        if index == -1:
            pos = len(_lists) - 1
            loc = len(_lists[pos]) - 1
            val = _lists[pos][loc]
            self._delete(pos, loc)
            return val
        if 0 <= index < len(_lists[0]):
            val = _lists[0][index]
            self._delete(0, index)
            return val
        len_last = len(_lists[-1])
        if -len_last < index < 0:
            pos = len(_lists) - 1
            loc = len_last + index
            val = _lists[pos][loc]
            self._delete(pos, loc)
            return val
        (pos, idx) = self._pos(index)
        val = _lists[pos][idx]
        self._delete(pos, idx)
        return val

    def index(self, value, start=None, stop=None):
        if False:
            return 10
        "Return first index of value in sorted list.\n\n        Raise ValueError if `value` is not present.\n\n        Index must be between `start` and `stop` for the `value` to be\n        considered present. The default value, None, for `start` and `stop`\n        indicate the beginning and end of the sorted list.\n\n        Negative indices are supported.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sl = SortedList('abcde')\n        >>> sl.index('d')\n        3\n        >>> sl.index('z')\n        Traceback (most recent call last):\n          ...\n        ValueError: 'z' is not in list\n\n        :param value: value in sorted list\n        :param int start: start index (default None, start of sorted list)\n        :param int stop: stop index (default None, end of sorted list)\n        :return: index of value\n        :raises ValueError: if value is not present\n\n        "
        _len = self._len
        if not _len:
            raise ValueError('{0!r} is not in list'.format(value))
        if start is None:
            start = 0
        if start < 0:
            start += _len
        if start < 0:
            start = 0
        if stop is None:
            stop = _len
        if stop < 0:
            stop += _len
        if stop > _len:
            stop = _len
        if stop <= start:
            raise ValueError('{0!r} is not in list'.format(value))
        _maxes = self._maxes
        pos_left = bisect_left(_maxes, value)
        if pos_left == len(_maxes):
            raise ValueError('{0!r} is not in list'.format(value))
        _lists = self._lists
        idx_left = bisect_left(_lists[pos_left], value)
        if _lists[pos_left][idx_left] != value:
            raise ValueError('{0!r} is not in list'.format(value))
        stop -= 1
        left = self._loc(pos_left, idx_left)
        if start <= left:
            if left <= stop:
                return left
        else:
            right = self._bisect_right(value) - 1
            if start <= right:
                return start
        raise ValueError('{0!r} is not in list'.format(value))

    def __add__(self, other):
        if False:
            while True:
                i = 10
        "Return new sorted list containing all values in both sequences.\n\n        ``sl.__add__(other)`` <==> ``sl + other``\n\n        Values in `other` do not need to be in sorted order.\n\n        Runtime complexity: `O(n*log(n))`\n\n        >>> sl1 = SortedList('bat')\n        >>> sl2 = SortedList('cat')\n        >>> sl1 + sl2\n        SortedList(['a', 'a', 'b', 'c', 't', 't'])\n\n        :param other: other iterable\n        :return: new sorted list\n\n        "
        values = reduce(iadd, self._lists, [])
        values.extend(other)
        return self.__class__(values)
    __radd__ = __add__

    def __iadd__(self, other):
        if False:
            i = 10
            return i + 15
        "Update sorted list with values from `other`.\n\n        ``sl.__iadd__(other)`` <==> ``sl += other``\n\n        Values in `other` do not need to be in sorted order.\n\n        Runtime complexity: `O(k*log(n))` -- approximate.\n\n        >>> sl = SortedList('bat')\n        >>> sl += 'cat'\n        >>> sl\n        SortedList(['a', 'a', 'b', 'c', 't', 't'])\n\n        :param other: other iterable\n        :return: existing sorted list\n\n        "
        self._update(other)
        return self

    def __mul__(self, num):
        if False:
            while True:
                i = 10
        "Return new sorted list with `num` shallow copies of values.\n\n        ``sl.__mul__(num)`` <==> ``sl * num``\n\n        Runtime complexity: `O(n*log(n))`\n\n        >>> sl = SortedList('abc')\n        >>> sl * 3\n        SortedList(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'])\n\n        :param int num: count of shallow copies\n        :return: new sorted list\n\n        "
        values = reduce(iadd, self._lists, []) * num
        return self.__class__(values)
    __rmul__ = __mul__

    def __imul__(self, num):
        if False:
            i = 10
            return i + 15
        "Update the sorted list with `num` shallow copies of values.\n\n        ``sl.__imul__(num)`` <==> ``sl *= num``\n\n        Runtime complexity: `O(n*log(n))`\n\n        >>> sl = SortedList('abc')\n        >>> sl *= 3\n        >>> sl\n        SortedList(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'])\n\n        :param int num: count of shallow copies\n        :return: existing sorted list\n\n        "
        values = reduce(iadd, self._lists, []) * num
        self._clear()
        self._update(values)
        return self

    def __make_cmp(seq_op, symbol, doc):
        if False:
            print('Hello World!')
        'Make comparator method.'

        def comparer(self, other):
            if False:
                for i in range(10):
                    print('nop')
            'Compare method for sorted list and sequence.'
            if not isinstance(other, Sequence):
                return NotImplemented
            self_len = self._len
            len_other = len(other)
            if self_len != len_other:
                if seq_op is eq:
                    return False
                if seq_op is ne:
                    return True
            for (alpha, beta) in zip(self, other):
                if alpha != beta:
                    return seq_op(alpha, beta)
            return seq_op(self_len, len_other)
        seq_op_name = seq_op.__name__
        comparer.__name__ = '__{0}__'.format(seq_op_name)
        doc_str = 'Return true if and only if sorted list is {0} `other`.\n\n        ``sl.__{1}__(other)`` <==> ``sl {2} other``\n\n        Comparisons use lexicographical order as with sequences.\n\n        Runtime complexity: `O(n)`\n\n        :param other: `other` sequence\n        :return: true if sorted list is {0} `other`\n\n        '
        comparer.__doc__ = dedent(doc_str.format(doc, seq_op_name, symbol))
        return comparer
    __eq__ = __make_cmp(eq, '==', 'equal to')
    __ne__ = __make_cmp(ne, '!=', 'not equal to')
    __lt__ = __make_cmp(lt, '<', 'less than')
    __gt__ = __make_cmp(gt, '>', 'greater than')
    __le__ = __make_cmp(le, '<=', 'less than or equal to')
    __ge__ = __make_cmp(ge, '>=', 'greater than or equal to')
    __make_cmp = staticmethod(__make_cmp)

    def __reduce__(self):
        if False:
            for i in range(10):
                print('nop')
        values = reduce(iadd, self._lists, [])
        return (type(self), (values,))

    @recursive_repr()
    def __repr__(self):
        if False:
            while True:
                i = 10
        'Return string representation of sorted list.\n\n        ``sl.__repr__()`` <==> ``repr(sl)``\n\n        :return: string representation\n\n        '
        return '{0}({1!r})'.format(type(self).__name__, list(self))

    def _check(self):
        if False:
            while True:
                i = 10
        'Check invariants of sorted list.\n\n        Runtime complexity: `O(n)`\n\n        '
        try:
            assert self._load >= 4
            assert len(self._maxes) == len(self._lists)
            assert self._len == sum((len(sublist) for sublist in self._lists))
            for sublist in self._lists:
                for pos in range(1, len(sublist)):
                    assert sublist[pos - 1] <= sublist[pos]
            for pos in range(1, len(self._lists)):
                assert self._lists[pos - 1][-1] <= self._lists[pos][0]
            for pos in range(len(self._maxes)):
                assert self._maxes[pos] == self._lists[pos][-1]
            double = self._load << 1
            assert all((len(sublist) <= double for sublist in self._lists))
            half = self._load >> 1
            for pos in range(0, len(self._lists) - 1):
                assert len(self._lists[pos]) >= half
            if self._index:
                assert self._len == self._index[0]
                assert len(self._index) == self._offset + len(self._lists)
                for pos in range(len(self._lists)):
                    leaf = self._index[self._offset + pos]
                    assert leaf == len(self._lists[pos])
                for pos in range(self._offset):
                    child = (pos << 1) + 1
                    if child >= len(self._index):
                        assert self._index[pos] == 0
                    elif child + 1 == len(self._index):
                        assert self._index[pos] == self._index[child]
                    else:
                        child_sum = self._index[child] + self._index[child + 1]
                        assert child_sum == self._index[pos]
        except:
            traceback.print_exc(file=sys.stdout)
            print('len', self._len)
            print('load', self._load)
            print('offset', self._offset)
            print('len_index', len(self._index))
            print('index', self._index)
            print('len_maxes', len(self._maxes))
            print('maxes', self._maxes)
            print('len_lists', len(self._lists))
            print('lists', self._lists)
            raise

def identity(value):
    if False:
        i = 10
        return i + 15
    'Identity function.'
    return value

class SortedKeyList(SortedList):
    """Sorted-key list is a subtype of sorted list.

    The sorted-key list maintains values in comparison order based on the
    result of a key function applied to every value.

    All the same methods that are available in :class:`SortedList` are also
    available in :class:`SortedKeyList`.

    Additional methods provided:

    * :attr:`SortedKeyList.key`
    * :func:`SortedKeyList.bisect_key_left`
    * :func:`SortedKeyList.bisect_key_right`
    * :func:`SortedKeyList.irange_key`

    Some examples below use:

    >>> from operator import neg
    >>> neg
    <built-in function neg>
    >>> neg(1)
    -1

    """

    def __init__(self, iterable=None, key=identity):
        if False:
            while True:
                i = 10
        "Initialize sorted-key list instance.\n\n        Optional `iterable` argument provides an initial iterable of values to\n        initialize the sorted-key list.\n\n        Optional `key` argument defines a callable that, like the `key`\n        argument to Python's `sorted` function, extracts a comparison key from\n        each value. The default is the identity function.\n\n        Runtime complexity: `O(n*log(n))`\n\n        >>> from operator import neg\n        >>> skl = SortedKeyList(key=neg)\n        >>> skl\n        SortedKeyList([], key=<built-in function neg>)\n        >>> skl = SortedKeyList([3, 1, 2], key=neg)\n        >>> skl\n        SortedKeyList([3, 2, 1], key=<built-in function neg>)\n\n        :param iterable: initial values (optional)\n        :param key: function used to extract comparison key (optional)\n\n        "
        self._key = key
        self._len = 0
        self._load = self.DEFAULT_LOAD_FACTOR
        self._lists = []
        self._keys = []
        self._maxes = []
        self._index = []
        self._offset = 0
        if iterable is not None:
            self._update(iterable)

    def __new__(cls, iterable=None, key=identity):
        if False:
            i = 10
            return i + 15
        return object.__new__(cls)

    @property
    def key(self):
        if False:
            print('Hello World!')
        'Function used to extract comparison key from values.'
        return self._key

    def clear(self):
        if False:
            i = 10
            return i + 15
        'Remove all values from sorted-key list.\n\n        Runtime complexity: `O(n)`\n\n        '
        self._len = 0
        del self._lists[:]
        del self._keys[:]
        del self._maxes[:]
        del self._index[:]
    _clear = clear

    def add(self, value):
        if False:
            print('Hello World!')
        'Add `value` to sorted-key list.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> from operator import neg\n        >>> skl = SortedKeyList(key=neg)\n        >>> skl.add(3)\n        >>> skl.add(1)\n        >>> skl.add(2)\n        >>> skl\n        SortedKeyList([3, 2, 1], key=<built-in function neg>)\n\n        :param value: value to add to sorted-key list\n\n        '
        _lists = self._lists
        _keys = self._keys
        _maxes = self._maxes
        key = self._key(value)
        if _maxes:
            pos = bisect_right(_maxes, key)
            if pos == len(_maxes):
                pos -= 1
                _lists[pos].append(value)
                _keys[pos].append(key)
                _maxes[pos] = key
            else:
                idx = bisect_right(_keys[pos], key)
                _lists[pos].insert(idx, value)
                _keys[pos].insert(idx, key)
            self._expand(pos)
        else:
            _lists.append([value])
            _keys.append([key])
            _maxes.append(key)
        self._len += 1

    def _expand(self, pos):
        if False:
            while True:
                i = 10
        'Split sublists with length greater than double the load-factor.\n\n        Updates the index when the sublist length is less than double the load\n        level. This requires incrementing the nodes in a traversal from the\n        leaf node to the root. For an example traversal see\n        ``SortedList._loc``.\n\n        '
        _lists = self._lists
        _keys = self._keys
        _index = self._index
        if len(_keys[pos]) > self._load << 1:
            _maxes = self._maxes
            _load = self._load
            _lists_pos = _lists[pos]
            _keys_pos = _keys[pos]
            half = _lists_pos[_load:]
            half_keys = _keys_pos[_load:]
            del _lists_pos[_load:]
            del _keys_pos[_load:]
            _maxes[pos] = _keys_pos[-1]
            _lists.insert(pos + 1, half)
            _keys.insert(pos + 1, half_keys)
            _maxes.insert(pos + 1, half_keys[-1])
            del _index[:]
        elif _index:
            child = self._offset + pos
            while child:
                _index[child] += 1
                child = child - 1 >> 1
            _index[0] += 1

    def update(self, iterable):
        if False:
            return 10
        'Update sorted-key list by adding all values from `iterable`.\n\n        Runtime complexity: `O(k*log(n))` -- approximate.\n\n        >>> from operator import neg\n        >>> skl = SortedKeyList(key=neg)\n        >>> skl.update([3, 1, 2])\n        >>> skl\n        SortedKeyList([3, 2, 1], key=<built-in function neg>)\n\n        :param iterable: iterable of values to add\n\n        '
        _lists = self._lists
        _keys = self._keys
        _maxes = self._maxes
        values = sorted(iterable, key=self._key)
        if _maxes:
            if len(values) * 4 >= self._len:
                _lists.append(values)
                values = reduce(iadd, _lists, [])
                values.sort(key=self._key)
                self._clear()
            else:
                _add = self.add
                for val in values:
                    _add(val)
                return
        _load = self._load
        _lists.extend((values[pos:pos + _load] for pos in range(0, len(values), _load)))
        _keys.extend((list(map(self._key, _list)) for _list in _lists))
        _maxes.extend((sublist[-1] for sublist in _keys))
        self._len = len(values)
        del self._index[:]
    _update = update

    def __contains__(self, value):
        if False:
            while True:
                i = 10
        'Return true if `value` is an element of the sorted-key list.\n\n        ``skl.__contains__(value)`` <==> ``value in skl``\n\n        Runtime complexity: `O(log(n))`\n\n        >>> from operator import neg\n        >>> skl = SortedKeyList([1, 2, 3, 4, 5], key=neg)\n        >>> 3 in skl\n        True\n\n        :param value: search for value in sorted-key list\n        :return: true if `value` in sorted-key list\n\n        '
        _maxes = self._maxes
        if not _maxes:
            return False
        key = self._key(value)
        pos = bisect_left(_maxes, key)
        if pos == len(_maxes):
            return False
        _lists = self._lists
        _keys = self._keys
        idx = bisect_left(_keys[pos], key)
        len_keys = len(_keys)
        len_sublist = len(_keys[pos])
        while True:
            if _keys[pos][idx] != key:
                return False
            if _lists[pos][idx] == value:
                return True
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    return False
                len_sublist = len(_keys[pos])
                idx = 0

    def discard(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Remove `value` from sorted-key list if it is a member.\n\n        If `value` is not a member, do nothing.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> from operator import neg\n        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)\n        >>> skl.discard(1)\n        >>> skl.discard(0)\n        >>> skl == [5, 4, 3, 2]\n        True\n\n        :param value: `value` to discard from sorted-key list\n\n        '
        _maxes = self._maxes
        if not _maxes:
            return
        key = self._key(value)
        pos = bisect_left(_maxes, key)
        if pos == len(_maxes):
            return
        _lists = self._lists
        _keys = self._keys
        idx = bisect_left(_keys[pos], key)
        len_keys = len(_keys)
        len_sublist = len(_keys[pos])
        while True:
            if _keys[pos][idx] != key:
                return
            if _lists[pos][idx] == value:
                self._delete(pos, idx)
                return
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    return
                len_sublist = len(_keys[pos])
                idx = 0

    def remove(self, value):
        if False:
            i = 10
            return i + 15
        'Remove `value` from sorted-key list; `value` must be a member.\n\n        If `value` is not a member, raise ValueError.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> from operator import neg\n        >>> skl = SortedKeyList([1, 2, 3, 4, 5], key=neg)\n        >>> skl.remove(5)\n        >>> skl == [4, 3, 2, 1]\n        True\n        >>> skl.remove(0)\n        Traceback (most recent call last):\n          ...\n        ValueError: 0 not in list\n\n        :param value: `value` to remove from sorted-key list\n        :raises ValueError: if `value` is not in sorted-key list\n\n        '
        _maxes = self._maxes
        if not _maxes:
            raise ValueError('{0!r} not in list'.format(value))
        key = self._key(value)
        pos = bisect_left(_maxes, key)
        if pos == len(_maxes):
            raise ValueError('{0!r} not in list'.format(value))
        _lists = self._lists
        _keys = self._keys
        idx = bisect_left(_keys[pos], key)
        len_keys = len(_keys)
        len_sublist = len(_keys[pos])
        while True:
            if _keys[pos][idx] != key:
                raise ValueError('{0!r} not in list'.format(value))
            if _lists[pos][idx] == value:
                self._delete(pos, idx)
                return
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    raise ValueError('{0!r} not in list'.format(value))
                len_sublist = len(_keys[pos])
                idx = 0

    def _delete(self, pos, idx):
        if False:
            print('Hello World!')
        'Delete value at the given `(pos, idx)`.\n\n        Combines lists that are less than half the load level.\n\n        Updates the index when the sublist length is more than half the load\n        level. This requires decrementing the nodes in a traversal from the\n        leaf node to the root. For an example traversal see\n        ``SortedList._loc``.\n\n        :param int pos: lists index\n        :param int idx: sublist index\n\n        '
        _lists = self._lists
        _keys = self._keys
        _maxes = self._maxes
        _index = self._index
        keys_pos = _keys[pos]
        lists_pos = _lists[pos]
        del keys_pos[idx]
        del lists_pos[idx]
        self._len -= 1
        len_keys_pos = len(keys_pos)
        if len_keys_pos > self._load >> 1:
            _maxes[pos] = keys_pos[-1]
            if _index:
                child = self._offset + pos
                while child > 0:
                    _index[child] -= 1
                    child = child - 1 >> 1
                _index[0] -= 1
        elif len(_keys) > 1:
            if not pos:
                pos += 1
            prev = pos - 1
            _keys[prev].extend(_keys[pos])
            _lists[prev].extend(_lists[pos])
            _maxes[prev] = _keys[prev][-1]
            del _lists[pos]
            del _keys[pos]
            del _maxes[pos]
            del _index[:]
            self._expand(prev)
        elif len_keys_pos:
            _maxes[pos] = keys_pos[-1]
        else:
            del _lists[pos]
            del _keys[pos]
            del _maxes[pos]
            del _index[:]

    def irange(self, minimum=None, maximum=None, inclusive=(True, True), reverse=False):
        if False:
            i = 10
            return i + 15
        'Create an iterator of values between `minimum` and `maximum`.\n\n        Both `minimum` and `maximum` default to `None` which is automatically\n        inclusive of the beginning and end of the sorted-key list.\n\n        The argument `inclusive` is a pair of booleans that indicates whether\n        the minimum and maximum ought to be included in the range,\n        respectively. The default is ``(True, True)`` such that the range is\n        inclusive of both minimum and maximum.\n\n        When `reverse` is `True` the values are yielded from the iterator in\n        reverse order; `reverse` defaults to `False`.\n\n        >>> from operator import neg\n        >>> skl = SortedKeyList([11, 12, 13, 14, 15], key=neg)\n        >>> it = skl.irange(14.5, 11.5)\n        >>> list(it)\n        [14, 13, 12]\n\n        :param minimum: minimum value to start iterating\n        :param maximum: maximum value to stop iterating\n        :param inclusive: pair of booleans\n        :param bool reverse: yield values in reverse order\n        :return: iterator\n\n        '
        min_key = self._key(minimum) if minimum is not None else None
        max_key = self._key(maximum) if maximum is not None else None
        return self._irange_key(min_key=min_key, max_key=max_key, inclusive=inclusive, reverse=reverse)

    def irange_key(self, min_key=None, max_key=None, inclusive=(True, True), reverse=False):
        if False:
            print('Hello World!')
        'Create an iterator of values between `min_key` and `max_key`.\n\n        Both `min_key` and `max_key` default to `None` which is automatically\n        inclusive of the beginning and end of the sorted-key list.\n\n        The argument `inclusive` is a pair of booleans that indicates whether\n        the minimum and maximum ought to be included in the range,\n        respectively. The default is ``(True, True)`` such that the range is\n        inclusive of both minimum and maximum.\n\n        When `reverse` is `True` the values are yielded from the iterator in\n        reverse order; `reverse` defaults to `False`.\n\n        >>> from operator import neg\n        >>> skl = SortedKeyList([11, 12, 13, 14, 15], key=neg)\n        >>> it = skl.irange_key(-14, -12)\n        >>> list(it)\n        [14, 13, 12]\n\n        :param min_key: minimum key to start iterating\n        :param max_key: maximum key to stop iterating\n        :param inclusive: pair of booleans\n        :param bool reverse: yield values in reverse order\n        :return: iterator\n\n        '
        _maxes = self._maxes
        if not _maxes:
            return iter(())
        _keys = self._keys
        if min_key is None:
            min_pos = 0
            min_idx = 0
        elif inclusive[0]:
            min_pos = bisect_left(_maxes, min_key)
            if min_pos == len(_maxes):
                return iter(())
            min_idx = bisect_left(_keys[min_pos], min_key)
        else:
            min_pos = bisect_right(_maxes, min_key)
            if min_pos == len(_maxes):
                return iter(())
            min_idx = bisect_right(_keys[min_pos], min_key)
        if max_key is None:
            max_pos = len(_maxes) - 1
            max_idx = len(_keys[max_pos])
        elif inclusive[1]:
            max_pos = bisect_right(_maxes, max_key)
            if max_pos == len(_maxes):
                max_pos -= 1
                max_idx = len(_keys[max_pos])
            else:
                max_idx = bisect_right(_keys[max_pos], max_key)
        else:
            max_pos = bisect_left(_maxes, max_key)
            if max_pos == len(_maxes):
                max_pos -= 1
                max_idx = len(_keys[max_pos])
            else:
                max_idx = bisect_left(_keys[max_pos], max_key)
        return self._islice(min_pos, min_idx, max_pos, max_idx, reverse)
    _irange_key = irange_key

    def bisect_left(self, value):
        if False:
            print('Hello World!')
        'Return an index to insert `value` in the sorted-key list.\n\n        If the `value` is already present, the insertion point will be before\n        (to the left of) any existing values.\n\n        Similar to the `bisect` module in the standard library.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> from operator import neg\n        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)\n        >>> skl.bisect_left(1)\n        4\n\n        :param value: insertion index of value in sorted-key list\n        :return: index\n\n        '
        return self._bisect_key_left(self._key(value))

    def bisect_right(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Return an index to insert `value` in the sorted-key list.\n\n        Similar to `bisect_left`, but if `value` is already present, the\n        insertion point will be after (to the right of) any existing values.\n\n        Similar to the `bisect` module in the standard library.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> from operator import neg\n        >>> skl = SortedList([5, 4, 3, 2, 1], key=neg)\n        >>> skl.bisect_right(1)\n        5\n\n        :param value: insertion index of value in sorted-key list\n        :return: index\n\n        '
        return self._bisect_key_right(self._key(value))
    bisect = bisect_right

    def bisect_key_left(self, key):
        if False:
            while True:
                i = 10
        'Return an index to insert `key` in the sorted-key list.\n\n        If the `key` is already present, the insertion point will be before (to\n        the left of) any existing keys.\n\n        Similar to the `bisect` module in the standard library.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> from operator import neg\n        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)\n        >>> skl.bisect_key_left(-1)\n        4\n\n        :param key: insertion index of key in sorted-key list\n        :return: index\n\n        '
        _maxes = self._maxes
        if not _maxes:
            return 0
        pos = bisect_left(_maxes, key)
        if pos == len(_maxes):
            return self._len
        idx = bisect_left(self._keys[pos], key)
        return self._loc(pos, idx)
    _bisect_key_left = bisect_key_left

    def bisect_key_right(self, key):
        if False:
            print('Hello World!')
        'Return an index to insert `key` in the sorted-key list.\n\n        Similar to `bisect_key_left`, but if `key` is already present, the\n        insertion point will be after (to the right of) any existing keys.\n\n        Similar to the `bisect` module in the standard library.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> from operator import neg\n        >>> skl = SortedList([5, 4, 3, 2, 1], key=neg)\n        >>> skl.bisect_key_right(-1)\n        5\n\n        :param key: insertion index of key in sorted-key list\n        :return: index\n\n        '
        _maxes = self._maxes
        if not _maxes:
            return 0
        pos = bisect_right(_maxes, key)
        if pos == len(_maxes):
            return self._len
        idx = bisect_right(self._keys[pos], key)
        return self._loc(pos, idx)
    bisect_key = bisect_key_right
    _bisect_key_right = bisect_key_right

    def count(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Return number of occurrences of `value` in the sorted-key list.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> from operator import neg\n        >>> skl = SortedKeyList([4, 4, 4, 4, 3, 3, 3, 2, 2, 1], key=neg)\n        >>> skl.count(2)\n        2\n\n        :param value: value to count in sorted-key list\n        :return: count\n\n        '
        _maxes = self._maxes
        if not _maxes:
            return 0
        key = self._key(value)
        pos = bisect_left(_maxes, key)
        if pos == len(_maxes):
            return 0
        _lists = self._lists
        _keys = self._keys
        idx = bisect_left(_keys[pos], key)
        total = 0
        len_keys = len(_keys)
        len_sublist = len(_keys[pos])
        while True:
            if _keys[pos][idx] != key:
                return total
            if _lists[pos][idx] == value:
                total += 1
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    return total
                len_sublist = len(_keys[pos])
                idx = 0

    def copy(self):
        if False:
            return 10
        'Return a shallow copy of the sorted-key list.\n\n        Runtime complexity: `O(n)`\n\n        :return: new sorted-key list\n\n        '
        return self.__class__(self, key=self._key)
    __copy__ = copy

    def index(self, value, start=None, stop=None):
        if False:
            i = 10
            return i + 15
        'Return first index of value in sorted-key list.\n\n        Raise ValueError if `value` is not present.\n\n        Index must be between `start` and `stop` for the `value` to be\n        considered present. The default value, None, for `start` and `stop`\n        indicate the beginning and end of the sorted-key list.\n\n        Negative indices are supported.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> from operator import neg\n        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)\n        >>> skl.index(2)\n        3\n        >>> skl.index(0)\n        Traceback (most recent call last):\n          ...\n        ValueError: 0 is not in list\n\n        :param value: value in sorted-key list\n        :param int start: start index (default None, start of sorted-key list)\n        :param int stop: stop index (default None, end of sorted-key list)\n        :return: index of value\n        :raises ValueError: if value is not present\n\n        '
        _len = self._len
        if not _len:
            raise ValueError('{0!r} is not in list'.format(value))
        if start is None:
            start = 0
        if start < 0:
            start += _len
        if start < 0:
            start = 0
        if stop is None:
            stop = _len
        if stop < 0:
            stop += _len
        if stop > _len:
            stop = _len
        if stop <= start:
            raise ValueError('{0!r} is not in list'.format(value))
        _maxes = self._maxes
        key = self._key(value)
        pos = bisect_left(_maxes, key)
        if pos == len(_maxes):
            raise ValueError('{0!r} is not in list'.format(value))
        stop -= 1
        _lists = self._lists
        _keys = self._keys
        idx = bisect_left(_keys[pos], key)
        len_keys = len(_keys)
        len_sublist = len(_keys[pos])
        while True:
            if _keys[pos][idx] != key:
                raise ValueError('{0!r} is not in list'.format(value))
            if _lists[pos][idx] == value:
                loc = self._loc(pos, idx)
                if start <= loc <= stop:
                    return loc
                elif loc > stop:
                    break
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    raise ValueError('{0!r} is not in list'.format(value))
                len_sublist = len(_keys[pos])
                idx = 0
        raise ValueError('{0!r} is not in list'.format(value))

    def __add__(self, other):
        if False:
            while True:
                i = 10
        'Return new sorted-key list containing all values in both sequences.\n\n        ``skl.__add__(other)`` <==> ``skl + other``\n\n        Values in `other` do not need to be in sorted-key order.\n\n        Runtime complexity: `O(n*log(n))`\n\n        >>> from operator import neg\n        >>> skl1 = SortedKeyList([5, 4, 3], key=neg)\n        >>> skl2 = SortedKeyList([2, 1, 0], key=neg)\n        >>> skl1 + skl2\n        SortedKeyList([5, 4, 3, 2, 1, 0], key=<built-in function neg>)\n\n        :param other: other iterable\n        :return: new sorted-key list\n\n        '
        values = reduce(iadd, self._lists, [])
        values.extend(other)
        return self.__class__(values, key=self._key)
    __radd__ = __add__

    def __mul__(self, num):
        if False:
            i = 10
            return i + 15
        'Return new sorted-key list with `num` shallow copies of values.\n\n        ``skl.__mul__(num)`` <==> ``skl * num``\n\n        Runtime complexity: `O(n*log(n))`\n\n        >>> from operator import neg\n        >>> skl = SortedKeyList([3, 2, 1], key=neg)\n        >>> skl * 2\n        SortedKeyList([3, 3, 2, 2, 1, 1], key=<built-in function neg>)\n\n        :param int num: count of shallow copies\n        :return: new sorted-key list\n\n        '
        values = reduce(iadd, self._lists, []) * num
        return self.__class__(values, key=self._key)

    def __reduce__(self):
        if False:
            for i in range(10):
                print('nop')
        values = reduce(iadd, self._lists, [])
        return (type(self), (values, self.key))

    @recursive_repr()
    def __repr__(self):
        if False:
            print('Hello World!')
        'Return string representation of sorted-key list.\n\n        ``skl.__repr__()`` <==> ``repr(skl)``\n\n        :return: string representation\n\n        '
        type_name = type(self).__name__
        return '{0}({1!r}, key={2!r})'.format(type_name, list(self), self._key)

    def _check(self):
        if False:
            i = 10
            return i + 15
        'Check invariants of sorted-key list.\n\n        Runtime complexity: `O(n)`\n\n        '
        try:
            assert self._load >= 4
            assert len(self._maxes) == len(self._lists) == len(self._keys)
            assert self._len == sum((len(sublist) for sublist in self._lists))
            for sublist in self._keys:
                for pos in range(1, len(sublist)):
                    assert sublist[pos - 1] <= sublist[pos]
            for pos in range(1, len(self._keys)):
                assert self._keys[pos - 1][-1] <= self._keys[pos][0]
            for (val_sublist, key_sublist) in zip(self._lists, self._keys):
                assert len(val_sublist) == len(key_sublist)
                for (val, key) in zip(val_sublist, key_sublist):
                    assert self._key(val) == key
            for pos in range(len(self._maxes)):
                assert self._maxes[pos] == self._keys[pos][-1]
            double = self._load << 1
            assert all((len(sublist) <= double for sublist in self._lists))
            half = self._load >> 1
            for pos in range(0, len(self._lists) - 1):
                assert len(self._lists[pos]) >= half
            if self._index:
                assert self._len == self._index[0]
                assert len(self._index) == self._offset + len(self._lists)
                for pos in range(len(self._lists)):
                    leaf = self._index[self._offset + pos]
                    assert leaf == len(self._lists[pos])
                for pos in range(self._offset):
                    child = (pos << 1) + 1
                    if child >= len(self._index):
                        assert self._index[pos] == 0
                    elif child + 1 == len(self._index):
                        assert self._index[pos] == self._index[child]
                    else:
                        child_sum = self._index[child] + self._index[child + 1]
                        assert child_sum == self._index[pos]
        except:
            traceback.print_exc(file=sys.stdout)
            print('len', self._len)
            print('load', self._load)
            print('offset', self._offset)
            print('len_index', len(self._index))
            print('index', self._index)
            print('len_maxes', len(self._maxes))
            print('maxes', self._maxes)
            print('len_keys', len(self._keys))
            print('keys', self._keys)
            print('len_lists', len(self._lists))
            print('lists', self._lists)
            raise
SortedListWithKey = SortedKeyList