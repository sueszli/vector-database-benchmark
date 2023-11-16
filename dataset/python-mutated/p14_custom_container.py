"""
Topic: 自定义容器
Desc : 
"""
import collections
import bisect

class SortedItems(collections.Sequence):

    def __init__(self, initial=None):
        if False:
            return 10
        self._items = sorted(initial) if initial is not None else []

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        return self._items[index]

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self._items)

    def add(self, item):
        if False:
            i = 10
            return i + 15
        bisect.insort(self._items, item)
items = SortedItems([5, 1, 3])
print(list(items))
print(items[0], items[-1])
items.add(2)
print(list(items))

class Items(collections.MutableSequence):

    def __init__(self, initial=None):
        if False:
            for i in range(10):
                print('nop')
        self._items = list(initial) if initial is not None else []

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        print('Getting:', index)
        return self._items[index]

    def __setitem__(self, index, value):
        if False:
            while True:
                i = 10
        print('Setting:', index, value)
        self._items[index] = value

    def __delitem__(self, index):
        if False:
            print('Hello World!')
        print('Deleting:', index)
        del self._items[index]

    def insert(self, index, value):
        if False:
            print('Hello World!')
        print('Inserting:', index, value)
        self._items.insert(index, value)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        print('Len')
        return len(self._items)
a = Items([1, 2, 3])
print(len(a))