"""

Available at repository https://github.com/LuminosoInsight/ordered-set

    salt.utils.oset
    ~~~~~~~~~~~~~~~~

An OrderedSet is a custom MutableSet that remembers its order, so that every
entry has an index that can be looked up.

Based on a recipe originally posted to ActiveState Recipes by Raymond Hettiger,
and released under the MIT license.

Rob Speer's changes are as follows:

    - changed the content from a doubly-linked list to a regular Python list.
      Seriously, who wants O(1) deletes but O(N) lookups by index?
    - add() returns the index of the added item
    - index() just returns the index of an item
    - added a __getstate__ and __setstate__ so it can be pickled
    - added __getitem__
"""
from collections.abc import MutableSet
SLICE_ALL = slice(None)
__version__ = '2.0.1'

def is_iterable(obj):
    if False:
        return 10
    "\n    Are we being asked to look up a list of things, instead of a single thing?\n    We check for the `__iter__` attribute so that this can cover types that\n    don't have to be known by this module, such as NumPy arrays.\n\n    Strings, however, should be considered as atomic values to look up, not\n    iterables. The same goes for tuples, since they are immutable and therefore\n    valid entries.\n\n    We don't need to check for the Python 2 `unicode` type, because it doesn't\n    have an `__iter__` attribute anyway.\n    "
    return hasattr(obj, '__iter__') and (not isinstance(obj, str)) and (not isinstance(obj, tuple))

class OrderedSet(MutableSet):
    """
    An OrderedSet is a custom MutableSet that remembers its order, so that
    every entry has an index that can be looked up.
    """

    def __init__(self, iterable=None):
        if False:
            i = 10
            return i + 15
        self.items = []
        self.map = {}
        if iterable is not None:
            self |= iterable

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.items)

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        '\n        Get the item at a given index.\n\n        If `index` is a slice, you will get back that slice of items. If it\'s\n        the slice [:], exactly the same object is returned. (If you want an\n        independent copy of an OrderedSet, use `OrderedSet.copy()`.)\n\n        If `index` is an iterable, you\'ll get the OrderedSet of items\n        corresponding to those indices. This is similar to NumPy\'s\n        "fancy indexing".\n        '
        if index == SLICE_ALL:
            return self
        elif hasattr(index, '__index__') or isinstance(index, slice):
            result = self.items[index]
            if isinstance(result, list):
                return OrderedSet(result)
            else:
                return result
        elif is_iterable(index):
            return OrderedSet([self.items[i] for i in index])
        else:
            raise TypeError("Don't know how to index an OrderedSet by {}".format(repr(index)))

    def copy(self):
        if False:
            i = 10
            return i + 15
        return OrderedSet(self)

    def __getstate__(self):
        if False:
            while True:
                i = 10
        if not self.items:
            return (None,)
        else:
            return list(self)

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        if state == (None,):
            self.__init__([])
        else:
            self.__init__(state)

    def __contains__(self, key):
        if False:
            i = 10
            return i + 15
        return key in self.map

    def add(self, key):
        if False:
            return 10
        '\n        Add `key` as an item to this OrderedSet, then return its index.\n\n        If `key` is already in the OrderedSet, return the index it already\n        had.\n        '
        if key not in self.map:
            self.map[key] = len(self.items)
            self.items.append(key)
        return self.map[key]
    append = add

    def update(self, sequence):
        if False:
            return 10
        '\n        Update the set with the given iterable sequence, then return the index\n        of the last element inserted.\n        '
        item_index = None
        try:
            for item in sequence:
                item_index = self.add(item)
        except TypeError:
            raise ValueError('Argument needs to be an iterable, got {}'.format(type(sequence)))
        return item_index

    def index(self, key):
        if False:
            while True:
                i = 10
        "\n        Get the index of a given entry, raising an IndexError if it's not\n        present.\n\n        `key` can be an iterable of entries that is not a string, in which case\n        this returns a list of indices.\n        "
        if is_iterable(key):
            return [self.index(subkey) for subkey in key]
        return self.map[key]

    def pop(self):
        if False:
            return 10
        '\n        Remove and return the last element from the set.\n\n        Raises KeyError if the set is empty.\n        '
        if not self.items:
            raise KeyError('Set is empty')
        elem = self.items[-1]
        del self.items[-1]
        del self.map[elem]
        return elem

    def discard(self, key):
        if False:
            while True:
                i = 10
        '\n        Remove an element.  Do not raise an exception if absent.\n\n        The MutableSet mixin uses this to implement the .remove() method, which\n        *does* raise an error when asked to remove a non-existent item.\n        '
        if key in self:
            i = self.map[key]
            del self.items[i]
            del self.map[key]
            for (k, v) in self.map.items():
                if v >= i:
                    self.map[k] = v - 1

    def clear(self):
        if False:
            while True:
                i = 10
        '\n        Remove all items from this OrderedSet.\n        '
        del self.items[:]
        self.map.clear()

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.items)

    def __reversed__(self):
        if False:
            i = 10
            return i + 15
        return reversed(self.items)

    def __repr__(self):
        if False:
            while True:
                i = 10
        if not self:
            return '{}()'.format(self.__class__.__name__)
        return '{}({})'.format(self.__class__.__name__, repr(list(self)))

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and self.items == other.items
        try:
            other_as_set = set(other)
        except TypeError:
            return False
        else:
            return set(self) == other_as_set