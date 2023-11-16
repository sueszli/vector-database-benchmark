from hypothesis.internal.conjecture.junkdrawer import find_integer
from hypothesis.internal.conjecture.shrinking.common import Shrinker

def identity(v):
    if False:
        for i in range(10):
            print('nop')
    return v

class Ordering(Shrinker):
    """A shrinker that tries to make a sequence more sorted.

    Will not change the length or the contents, only tries to reorder
    the elements of the sequence.
    """

    def setup(self, key=identity):
        if False:
            return 10
        self.key = key

    def make_immutable(self, value):
        if False:
            i = 10
            return i + 15
        return tuple(value)

    def short_circuit(self):
        if False:
            for i in range(10):
                print('nop')
        return self.consider(sorted(self.current, key=self.key))

    def left_is_better(self, left, right):
        if False:
            print('Hello World!')
        return tuple(map(self.key, left)) < tuple(map(self.key, right))

    def check_invariants(self, value):
        if False:
            while True:
                i = 10
        assert len(value) == len(self.current)
        assert sorted(value) == sorted(self.current)

    def run_step(self):
        if False:
            i = 10
            return i + 15
        self.sort_regions()
        self.sort_regions_with_gaps()

    def sort_regions(self):
        if False:
            return 10
        'Guarantees that for each i we have tried to swap index i with\n        index i + 1.\n\n        This uses an adaptive algorithm that works by sorting contiguous\n        regions starting from each element.\n        '
        i = 0
        while i + 1 < len(self.current):
            prefix = list(self.current[:i])
            k = find_integer(lambda k: i + k <= len(self.current) and self.consider(prefix + sorted(self.current[i:i + k], key=self.key) + list(self.current[i + k:])))
            i += k

    def sort_regions_with_gaps(self):
        if False:
            print('Hello World!')
        'Guarantees that for each i we have tried to swap index i with\n        index i + 2.\n\n        This uses an adaptive algorithm that works by sorting contiguous\n        regions centered on each element, where that element is treated as\n        fixed and the elements around it are sorted..\n        '
        for i in range(1, len(self.current) - 1):
            if self.current[i - 1] <= self.current[i] <= self.current[i + 1]:
                continue

            def can_sort(a, b):
                if False:
                    for i in range(10):
                        print('nop')
                if a < 0 or b > len(self.current):
                    return False
                assert a <= i < b
                split = i - a
                values = sorted(self.current[a:i] + self.current[i + 1:b])
                return self.consider(list(self.current[:a]) + values[:split] + [self.current[i]] + values[split:] + list(self.current[b:]))
            left = i
            right = i + 1
            right += find_integer(lambda k: can_sort(left, right + k))
            find_integer(lambda k: can_sort(left - k, right))