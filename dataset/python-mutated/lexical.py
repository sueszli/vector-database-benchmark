from hypothesis.internal.compat import int_from_bytes, int_to_bytes
from hypothesis.internal.conjecture.shrinking.common import Shrinker
from hypothesis.internal.conjecture.shrinking.integer import Integer
from hypothesis.internal.conjecture.shrinking.ordering import Ordering
'\nThis module implements a lexicographic minimizer for blocks of bytes.\n'

class Lexical(Shrinker):

    def make_immutable(self, value):
        if False:
            return 10
        return bytes(value)

    @property
    def size(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.current)

    def check_invariants(self, value):
        if False:
            while True:
                i = 10
        assert len(value) == self.size

    def left_is_better(self, left, right):
        if False:
            i = 10
            return i + 15
        return left < right

    def incorporate_int(self, i):
        if False:
            print('Hello World!')
        return self.incorporate(int_to_bytes(i, self.size))

    @property
    def current_int(self):
        if False:
            i = 10
            return i + 15
        return int_from_bytes(self.current)

    def minimize_as_integer(self):
        if False:
            while True:
                i = 10
        Integer.shrink(self.current_int, lambda c: c == self.current_int or self.incorporate_int(c), random=self.random)

    def partial_sort(self):
        if False:
            i = 10
            return i + 15
        Ordering.shrink(self.current, self.consider, random=self.random)

    def short_circuit(self):
        if False:
            for i in range(10):
                print('nop')
        'This is just an assemblage of other shrinkers, so we rely on their\n        short circuiting.'
        return False

    def run_step(self):
        if False:
            for i in range(10):
                print('nop')
        self.minimize_as_integer()
        self.partial_sort()