class WeakSet:

    def __init__(self, data=None):
        if False:
            for i in range(10):
                print('nop')
        self.data = set(data)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for item in self.data:
            if item is not None:
                yield item

    def union(self, other):
        if False:
            while True:
                i = 10
        return self.__class__((e for s in (self, other) for e in s))
a = WeakSet([1, 2, 3])
b = WeakSet([1, 3, 5])
assert list(a.union(b)) == [1, 2, 3, 5]