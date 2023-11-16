""" This module is only an abstraction of OrderedSet which is not present in
Python at all.

It was originally downloaded from http://code.activestate.com/recipes/576694/
"""
from nuitka.__past__ import MutableSet

class OrderedSet(MutableSet):
    is_fallback = True

    def __init__(self, iterable=()):
        if False:
            i = 10
            return i + 15
        self.end = end = []
        end += (None, end, end)
        self.map = {}
        if iterable:
            self |= iterable

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.map)

    def __contains__(self, key):
        if False:
            while True:
                i = 10
        return key in self.map

    def add(self, key):
        if False:
            for i in range(10):
                print('nop')
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def update(self, keys):
        if False:
            return 10
        for key in keys:
            self.add(key)

    def discard(self, key):
        if False:
            print('Hello World!')
        if key in self.map:
            (key, prev, next) = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        if False:
            while True:
                i = 10
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        if False:
            print('Hello World!')
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if False:
            i = 10
            return i + 15
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if False:
            while True:
                i = 10
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

    def union(self, iterable):
        if False:
            return 10
        result = OrderedSet(self)
        for key in iterable:
            result.add(key)
        return result

    def index(self, key):
        if False:
            return 10
        if key in self.map:
            end = self.end
            curr = self.map[key]
            count = 0
            while curr is not end:
                curr = curr[1]
                count += 1
            return count - 1
        return None