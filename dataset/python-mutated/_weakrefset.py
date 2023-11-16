from _weakref import ref
from types import GenericAlias
__all__ = ['WeakSet']

class _IterationGuard:

    def __init__(self, weakcontainer):
        if False:
            while True:
                i = 10
        self.weakcontainer = ref(weakcontainer)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        w = self.weakcontainer()
        if w is not None:
            w._iterating.add(self)
        return self

    def __exit__(self, e, t, b):
        if False:
            print('Hello World!')
        w = self.weakcontainer()
        if w is not None:
            s = w._iterating
            s.remove(self)
            if not s:
                w._commit_removals()

class WeakSet:

    def __init__(self, data=None):
        if False:
            for i in range(10):
                print('nop')
        self.data = set()

        def _remove(item, selfref=ref(self)):
            if False:
                return 10
            self = selfref()
            if self is not None:
                if self._iterating:
                    self._pending_removals.append(item)
                else:
                    self.data.discard(item)
        self._remove = _remove
        self._pending_removals = []
        self._iterating = set()
        if data is not None:
            self.update(data)

    def _commit_removals(self):
        if False:
            i = 10
            return i + 15
        pop = self._pending_removals.pop
        discard = self.data.discard
        while True:
            try:
                item = pop()
            except IndexError:
                return
            discard(item)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        with _IterationGuard(self):
            for itemref in self.data:
                item = itemref()
                if item is not None:
                    yield item

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.data) - len(self._pending_removals)

    def __contains__(self, item):
        if False:
            print('Hello World!')
        try:
            wr = ref(item)
        except TypeError:
            return False
        return wr in self.data

    def __reduce__(self):
        if False:
            while True:
                i = 10
        return (self.__class__, (list(self),), getattr(self, '__dict__', None))

    def add(self, item):
        if False:
            i = 10
            return i + 15
        if self._pending_removals:
            self._commit_removals()
        self.data.add(ref(item, self._remove))

    def clear(self):
        if False:
            while True:
                i = 10
        if self._pending_removals:
            self._commit_removals()
        self.data.clear()

    def copy(self):
        if False:
            print('Hello World!')
        return self.__class__(self)

    def pop(self):
        if False:
            return 10
        if self._pending_removals:
            self._commit_removals()
        while True:
            try:
                itemref = self.data.pop()
            except KeyError:
                raise KeyError('pop from empty WeakSet') from None
            item = itemref()
            if item is not None:
                return item

    def remove(self, item):
        if False:
            print('Hello World!')
        if self._pending_removals:
            self._commit_removals()
        self.data.remove(ref(item))

    def discard(self, item):
        if False:
            return 10
        if self._pending_removals:
            self._commit_removals()
        self.data.discard(ref(item))

    def update(self, other):
        if False:
            print('Hello World!')
        if self._pending_removals:
            self._commit_removals()
        for element in other:
            self.add(element)

    def __ior__(self, other):
        if False:
            while True:
                i = 10
        self.update(other)
        return self

    def difference(self, other):
        if False:
            while True:
                i = 10
        newset = self.copy()
        newset.difference_update(other)
        return newset
    __sub__ = difference

    def difference_update(self, other):
        if False:
            while True:
                i = 10
        self.__isub__(other)

    def __isub__(self, other):
        if False:
            i = 10
            return i + 15
        if self._pending_removals:
            self._commit_removals()
        if self is other:
            self.data.clear()
        else:
            self.data.difference_update((ref(item) for item in other))
        return self

    def intersection(self, other):
        if False:
            i = 10
            return i + 15
        return self.__class__((item for item in other if item in self))
    __and__ = intersection

    def intersection_update(self, other):
        if False:
            i = 10
            return i + 15
        self.__iand__(other)

    def __iand__(self, other):
        if False:
            i = 10
            return i + 15
        if self._pending_removals:
            self._commit_removals()
        self.data.intersection_update((ref(item) for item in other))
        return self

    def issubset(self, other):
        if False:
            return 10
        return self.data.issubset((ref(item) for item in other))
    __le__ = issubset

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.data < set(map(ref, other))

    def issuperset(self, other):
        if False:
            i = 10
            return i + 15
        return self.data.issuperset((ref(item) for item in other))
    __ge__ = issuperset

    def __gt__(self, other):
        if False:
            while True:
                i = 10
        return self.data > set(map(ref, other))

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.data == set(map(ref, other))

    def symmetric_difference(self, other):
        if False:
            return 10
        newset = self.copy()
        newset.symmetric_difference_update(other)
        return newset
    __xor__ = symmetric_difference

    def symmetric_difference_update(self, other):
        if False:
            while True:
                i = 10
        self.__ixor__(other)

    def __ixor__(self, other):
        if False:
            i = 10
            return i + 15
        if self._pending_removals:
            self._commit_removals()
        if self is other:
            self.data.clear()
        else:
            self.data.symmetric_difference_update((ref(item, self._remove) for item in other))
        return self

    def union(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__((e for s in (self, other) for e in s))
    __or__ = union

    def isdisjoint(self, other):
        if False:
            for i in range(10):
                print('nop')
        return len(self.intersection(other)) == 0

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return repr(self.data)
    __class_getitem__ = classmethod(GenericAlias)