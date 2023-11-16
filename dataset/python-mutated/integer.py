from hypothesis.internal.conjecture.junkdrawer import find_integer
from hypothesis.internal.conjecture.shrinking.common import Shrinker
'\nThis module implements a shrinker for non-negative integers.\n'

class Integer(Shrinker):
    """Attempts to find a smaller integer. Guaranteed things to try ``0``,

    ``1``, ``initial - 1``, ``initial - 2``. Plenty of optimisations beyond
    that but those are the guaranteed ones.
    """

    def short_circuit(self):
        if False:
            print('Hello World!')
        for i in range(2):
            if self.consider(i):
                return True
        self.mask_high_bits()
        if self.size > 8:
            self.consider(self.current >> self.size - 8)
            self.consider(self.current & 255)
        return self.current == 2

    def check_invariants(self, value):
        if False:
            while True:
                i = 10
        assert value >= 0

    def left_is_better(self, left, right):
        if False:
            i = 10
            return i + 15
        return left < right

    def run_step(self):
        if False:
            print('Hello World!')
        self.shift_right()
        self.shrink_by_multiples(2)
        self.shrink_by_multiples(1)

    def shift_right(self):
        if False:
            print('Hello World!')
        base = self.current
        find_integer(lambda k: k <= self.size and self.consider(base >> k))

    def mask_high_bits(self):
        if False:
            return 10
        base = self.current
        n = base.bit_length()

        @find_integer
        def try_mask(k):
            if False:
                while True:
                    i = 10
            if k >= n:
                return False
            mask = (1 << n - k) - 1
            return self.consider(mask & base)

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        return self.current.bit_length()

    def shrink_by_multiples(self, k):
        if False:
            while True:
                i = 10
        base = self.current

        @find_integer
        def shrunk(n):
            if False:
                i = 10
                return i + 15
            attempt = base - n * k
            return attempt >= 0 and self.consider(attempt)
        return shrunk > 0