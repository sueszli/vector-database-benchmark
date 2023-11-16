"""
Provides a very simple implementation of an ordered set. We use the
Python dictionaries as a basis because they are guaranteed to
be ordered since Python 3.6.
"""
from typing import Generic, Hashable, TypeVar
OrderedSetItem = TypeVar('OrderedSetItem')

class OrderedSet(Generic[OrderedSetItem]):
    """
    Set that saves the input order of elements.
    """
    __slots__ = ('ordered_set',)

    def __init__(self, elements: Hashable=None):
        if False:
            i = 10
            return i + 15
        self.ordered_set = {}
        if elements:
            self.update(elements)

    def add(self, elem: Hashable) -> None:
        if False:
            while True:
                i = 10
        '\n        Set-like add that calls append_right().\n        '
        self.append_right(elem)

    def append_left(self, elem: Hashable) -> None:
        if False:
            while True:
                i = 10
        '\n        Add an element to the front of the set.\n        '
        if elem not in self.ordered_set:
            temp_set = {elem: 0}
            for key in self.ordered_set:
                self.ordered_set[key] += 1
            temp_set.update(self.ordered_set)
            self.ordered_set = temp_set

    def append_right(self, elem: Hashable) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Add an element to the back of the set.\n        '
        if elem not in self.ordered_set:
            self.ordered_set[elem] = len(self)

    def discard(self, elem: Hashable) -> None:
        if False:
            print('Hello World!')
        '\n        Remove an element from the set.\n        '
        index = self.ordered_set.pop(elem, -1)
        if index > -1:
            for (key, value) in self.ordered_set.items():
                if value > index:
                    self.ordered_set[key] -= 1

    def get_list(self) -> list:
        if False:
            print('Hello World!')
        '\n        Returns a normal list containing the values from the ordered set.\n        '
        return list(self.ordered_set.keys())

    def index(self, elem: Hashable) -> int:
        if False:
            return 10
        '\n        Returns the index of the element in the set or\n        -1 if it is not in the set.\n        '
        if elem in self.ordered_set:
            return self.ordered_set[elem]
        return -1

    def intersection_update(self, other):
        if False:
            while True:
                i = 10
        '\n        Only keep elements that are both in self and other.\n        '
        keys_self = set(self.ordered_set.keys())
        keys_other = set(other.keys())
        intersection = keys_self & keys_other
        for elem in self:
            if elem not in intersection:
                self.discard(elem)

    def union(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Returns a new ordered set with the elements from self and other.\n        '
        element_list = self.get_list() + other.get_list()
        return OrderedSet(element_list)

    def update(self, other) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Append the elements of another iterable to the right of the\n        ordered set.\n        '
        for elem in other:
            self.append_right(elem)

    def __contains__(self, elem):
        if False:
            return 10
        return elem in self.ordered_set

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self.ordered_set.keys())

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.ordered_set)

    def __reversed__(self):
        if False:
            i = 10
            return i + 15
        return reversed(self.ordered_set.keys())

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'OrderedSet({list(self.ordered_set.keys())})'

    def __repr__(self):
        if False:
            print('Hello World!')
        return str(self)