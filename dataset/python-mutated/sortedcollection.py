"""Python Sorted Collection Module by Raymond Hettinger

Copied from http://code.activestate.com/recipes/577197-sortedcollection/
Retrieved on April 22, 2018

"""
from bisect import bisect_left, bisect_right

class SortedCollection(object):
    """Sequence sorted by a key function.

    SortedCollection() is much easier to work with than using bisect() directly.
    It supports key functions like those use in sorted(), min(), and max().
    The result of the key function call is saved so that keys can be searched
    efficiently.

    Instead of returning an insertion-point which can be hard to interpret, the
    five find-methods return a specific item in the sequence. They can scan for
    exact matches, the last item less-than-or-equal to a key, or the first item
    greater-than-or-equal to a key.

    Once found, an item's ordinal position can be located with the index() method.
    New items can be added with the insert() and insert_right() methods.
    Old items can be deleted with the remove() method.

    The usual sequence methods are provided to support indexing, slicing,
    length lookup, clearing, copying, forward and reverse iteration, contains
    checking, item counts, item removal, and a nice looking repr.

    Finding and indexing are O(log n) operations while iteration and insertion
    are O(n).  The initial sort is O(n log n).

    The key function is stored in the 'key' attibute for easy introspection or
    so that you can assign a new key function (triggering an automatic re-sort).

    In short, the class was designed to handle all of the common use cases for
    bisect but with a simpler API and support for key functions.

    >>> from pprint import pprint
    >>> from operator import itemgetter

    >>> s = SortedCollection(key=itemgetter(2))
    >>> for record in [
    ...         ('roger', 'young', 30),
    ...         ('angela', 'jones', 28),
    ...         ('bill', 'smith', 22),
    ...         ('david', 'thomas', 32)]:
    ...     s.insert(record)

    >>> pprint(list(s))         # show records sorted by age
    [('bill', 'smith', 22),
     ('angela', 'jones', 28),
     ('roger', 'young', 30),
     ('david', 'thomas', 32)]

    >>> s.find_le(29)           # find oldest person aged 29 or younger
    ('angela', 'jones', 28)
    >>> s.find_lt(28)           # find oldest person under 28
    ('bill', 'smith', 22)
    >>> s.find_gt(28)           # find youngest person over 28
    ('roger', 'young', 30)

    >>> r = s.find_ge(32)       # find youngest person aged 32 or older
    >>> s.index(r)              # get the index of their record
    3
    >>> s[3]                    # fetch the record at that index
    ('david', 'thomas', 32)

    >>> s.key = itemgetter(0)   # now sort by first name
    >>> pprint(list(s))
    [('angela', 'jones', 28),
     ('bill', 'smith', 22),
     ('david', 'thomas', 32),
     ('roger', 'young', 30)]

    """

    def __init__(self, iterable=(), key=None):
        if False:
            while True:
                i = 10
        self._given_key = key
        key = (lambda x: x) if key is None else key
        decorated = sorted(((key(item), item) for item in iterable))
        self._keys = [k for (k, item) in decorated]
        self._items = [item for (k, item) in decorated]
        self._key = key

    def _getkey(self):
        if False:
            while True:
                i = 10
        return self._key

    def _setkey(self, key):
        if False:
            i = 10
            return i + 15
        if key is not self._key:
            self.__init__(self._items, key=key)

    def _delkey(self):
        if False:
            i = 10
            return i + 15
        self._setkey(None)
    key = property(_getkey, _setkey, _delkey, 'key function')

    def clear(self):
        if False:
            i = 10
            return i + 15
        self.__init__([], self._key)

    def copy(self):
        if False:
            return 10
        return self.__class__(self, self._key)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._items)

    def __getitem__(self, i):
        if False:
            return 10
        return self._items[i]

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self._items)

    def __reversed__(self):
        if False:
            print('Hello World!')
        return reversed(self._items)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s(%r, key=%s)' % (self.__class__.__name__, self._items, getattr(self._given_key, '__name__', repr(self._given_key)))

    def __reduce__(self):
        if False:
            return 10
        return (self.__class__, (self._items, self._given_key))

    def __contains__(self, item):
        if False:
            for i in range(10):
                print('nop')
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return item in self._items[i:j]

    def index(self, item):
        if False:
            return 10
        'Find the position of an item.  Raise ValueError if not found.'
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return self._items[i:j].index(item) + i

    def count(self, item):
        if False:
            for i in range(10):
                print('nop')
        'Return number of occurrences of item'
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return self._items[i:j].count(item)

    def insert(self, item):
        if False:
            while True:
                i = 10
        'Insert a new item.  If equal keys are found, add to the left'
        k = self._key(item)
        i = bisect_left(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)

    def insert_right(self, item):
        if False:
            print('Hello World!')
        'Insert a new item.  If equal keys are found, add to the right'
        k = self._key(item)
        i = bisect_right(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)

    def remove(self, item):
        if False:
            print('Hello World!')
        'Remove first occurrence of item.  Raise ValueError if not found'
        i = self.index(item)
        del self._keys[i]
        del self._items[i]

    def find(self, k):
        if False:
            return 10
        'Return first item with a key == k.  Raise ValueError if not found.'
        i = bisect_left(self._keys, k)
        if i != len(self) and self._keys[i] == k:
            return self._items[i]
        raise ValueError('No item found with key equal to: %r' % (k,))

    def find_le(self, k):
        if False:
            print('Hello World!')
        'Return last item with a key <= k.  Raise ValueError if not found.'
        i = bisect_right(self._keys, k)
        if i:
            return self._items[i - 1]
        raise ValueError('No item found with key at or below: %r' % (k,))

    def find_lt(self, k):
        if False:
            return 10
        'Return last item with a key < k.  Raise ValueError if not found.'
        i = bisect_left(self._keys, k)
        if i:
            return self._items[i - 1]
        raise ValueError('No item found with key below: %r' % (k,))

    def find_ge(self, k):
        if False:
            while True:
                i = 10
        'Return first item with a key >= equal to k.  Raise ValueError if not found'
        i = bisect_left(self._keys, k)
        if i != len(self):
            return self._items[i]
        raise ValueError('No item found with key at or above: %r' % (k,))

    def find_gt(self, k):
        if False:
            return 10
        'Return first item with a key > k.  Raise ValueError if not found'
        i = bisect_right(self._keys, k)
        if i != len(self):
            return self._items[i]
        raise ValueError('No item found with key above: %r' % (k,))
    add = insert

    def update(self, iterable):
        if False:
            print('Hello World!')
        for value in iterable:
            self.insert(value)

    def bisect(self, item):
        if False:
            while True:
                i = 10
        key = self._key(item)
        pos = bisect_left(self._keys, key)
        return pos

    def pop(self):
        if False:
            i = 10
            return i + 15
        self._keys.pop()
        return self._items.pop()

    def discard(self, item):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.remove(item)
        except ValueError:
            pass

    def __delitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        del self._keys[index]
        del self._items[index]
if __name__ == '__main__':

    def ve2no(f, *args):
        if False:
            return 10
        'Convert ValueError result to -1'
        try:
            return f(*args)
        except ValueError:
            return -1

    def slow_index(seq, k):
        if False:
            return 10
        'Location of match or -1 if not found'
        for (i, item) in enumerate(seq):
            if item == k:
                return i
        return -1

    def slow_find(seq, k):
        if False:
            print('Hello World!')
        'First item with a key equal to k. -1 if not found'
        for item in seq:
            if item == k:
                return item
        return -1

    def slow_find_le(seq, k):
        if False:
            print('Hello World!')
        'Last item with a key less-than or equal to k.'
        for item in reversed(seq):
            if item <= k:
                return item
        return -1

    def slow_find_lt(seq, k):
        if False:
            while True:
                i = 10
        'Last item with a key less-than k.'
        for item in reversed(seq):
            if item < k:
                return item
        return -1

    def slow_find_ge(seq, k):
        if False:
            i = 10
            return i + 15
        'First item with a key-value greater-than or equal to k.'
        for item in seq:
            if item >= k:
                return item
        return -1

    def slow_find_gt(seq, k):
        if False:
            while True:
                i = 10
        'First item with a key-value greater-than or equal to k.'
        for item in seq:
            if item > k:
                return item
        return -1
    from random import choice
    pool = [1.5, 2, 2.0, 3, 3.0, 3.5, 4, 4.0, 4.5]
    for i in range(500):
        for n in range(6):
            s = [choice(pool) for i in range(n)]
            sc = SortedCollection(s)
            s.sort()
            for probe in pool:
                assert repr(ve2no(sc.index, probe)) == repr(slow_index(s, probe))
                assert repr(ve2no(sc.find, probe)) == repr(slow_find(s, probe))
                assert repr(ve2no(sc.find_le, probe)) == repr(slow_find_le(s, probe))
                assert repr(ve2no(sc.find_lt, probe)) == repr(slow_find_lt(s, probe))
                assert repr(ve2no(sc.find_ge, probe)) == repr(slow_find_ge(s, probe))
                assert repr(ve2no(sc.find_gt, probe)) == repr(slow_find_gt(s, probe))
            for (i, item) in enumerate(s):
                assert repr(item) == repr(sc[i])
                assert item in sc
                assert s.count(item) == sc.count(item)
            assert len(sc) == n
            assert list(map(repr, reversed(sc))) == list(map(repr, reversed(s)))
            assert list(sc.copy()) == list(sc)
            sc.clear()
            assert len(sc) == 0
    sd = SortedCollection('The quick Brown Fox jumped'.split(), key=str.lower)
    assert sd._keys == ['brown', 'fox', 'jumped', 'quick', 'the']
    assert sd._items == ['Brown', 'Fox', 'jumped', 'quick', 'The']
    assert sd._key == str.lower
    assert repr(sd) == "SortedCollection(['Brown', 'Fox', 'jumped', 'quick', 'The'], key=lower)"
    sd.key = str.upper
    assert sd._key == str.upper
    assert len(sd) == 5
    assert list(reversed(sd)) == ['The', 'quick', 'jumped', 'Fox', 'Brown']
    for item in sd:
        assert item in sd
    for (i, item) in enumerate(sd):
        assert item == sd[i]
    sd.insert('jUmPeD')
    sd.insert_right('QuIcK')
    assert sd._keys == ['BROWN', 'FOX', 'JUMPED', 'JUMPED', 'QUICK', 'QUICK', 'THE']
    assert sd._items == ['Brown', 'Fox', 'jUmPeD', 'jumped', 'quick', 'QuIcK', 'The']
    assert sd.find_le('JUMPED') == 'jumped', sd.find_le('JUMPED')
    assert sd.find_ge('JUMPED') == 'jUmPeD'
    assert sd.find_le('GOAT') == 'Fox'
    assert sd.find_ge('GOAT') == 'jUmPeD'
    assert sd.find('FOX') == 'Fox'
    assert sd[3] == 'jumped'
    assert sd[3:5] == ['jumped', 'quick']
    assert sd[-2] == 'QuIcK'
    assert sd[-4:-2] == ['jumped', 'quick']
    for (i, item) in enumerate(sd):
        assert sd.index(item) == i
    try:
        sd.index('xyzpdq')
    except ValueError:
        pass
    else:
        assert 0, 'Oops, failed to notify of missing value'
    sd.remove('jumped')
    assert list(sd) == ['Brown', 'Fox', 'jUmPeD', 'quick', 'QuIcK', 'The']
    import doctest
    from operator import itemgetter
    print(doctest.testmod())