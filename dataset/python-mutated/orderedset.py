from collections.abc import MutableSet

class OrderedSet(MutableSet):

    def __init__(self, iterable=None):
        if False:
            for i in range(10):
                print('nop')
        self.end = end = []
        end += [None, end, end]
        self.map = {}
        if iterable is not None:
            self |= iterable

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.map)

    def __contains__(self, key):
        if False:
            print('Hello World!')
        return key in self.map

    def add(self, key):
        if False:
            for i in range(10):
                print('nop')
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if False:
            for i in range(10):
                print('nop')
        if key in self.map:
            (key, prev, next) = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        if False:
            return 10
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        if False:
            return 10
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if False:
            print('Hello World!')
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if False:
            print('Hello World!')
        if not self:
            return f'{self.__class__.__name__}()'
        return f'{self.__class__.__name__}({list(self)!r})'

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)
if __name__ == '__main__':
    s = OrderedSet('abracadaba')
    t = OrderedSet('simsalabim')
    print(s | t)
    print(s & t)
    print(s - t)