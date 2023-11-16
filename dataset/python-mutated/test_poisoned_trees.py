from random import Random
import pytest
from hypothesis import HealthCheck, settings
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.conjecture.engine import ConjectureData, ConjectureRunner
from hypothesis.strategies._internal import SearchStrategy
POISON = 'POISON'
MAX_INT = 2 ** 32 - 1

class PoisonedTree(SearchStrategy):
    """Generates variable sized tuples with an implicit tree structure.

    The actual result is flattened out, but the hierarchy is implicit in
    the data.
    """

    def __init__(self, p):
        if False:
            while True:
                i = 10
        super().__init__()
        self.__p = p

    def do_draw(self, data):
        if False:
            return 10
        if cu.biased_coin(data, self.__p):
            return data.draw(self) + data.draw(self)
        else:
            n = data.draw_bits(16) << 16 | data.draw_bits(16)
            if n == MAX_INT:
                return (POISON,)
            else:
                return (None,)
LOTS = 10 ** 6
TEST_SETTINGS = settings(database=None, suppress_health_check=list(HealthCheck), max_examples=LOTS, deadline=None)

@pytest.mark.parametrize('size', [2, 5, 10])
@pytest.mark.parametrize('seed', [0, 15993493061449915028])
def test_can_reduce_poison_from_any_subtree(size, seed):
    if False:
        return 10
    'This test validates that we can minimize to any leaf node of a binary\n    tree, regardless of where in the tree the leaf is.'
    random = Random(seed)
    p = 1.0 / (2.0 - 1.0 / size)
    strat = PoisonedTree(p)

    def test_function(data):
        if False:
            for i in range(10):
                print('nop')
        v = data.draw(strat)
        if len(v) >= size:
            data.mark_interesting()
    runner = ConjectureRunner(test_function, random=random, settings=TEST_SETTINGS)
    runner.generate_new_examples()
    runner.shrink_interesting_examples()
    (data,) = runner.interesting_examples.values()
    assert len(ConjectureData.for_buffer(data.buffer).draw(strat)) == size
    starts = [b.start for b in data.blocks if b.length == 2]
    assert len(starts) % 2 == 0
    marker = bytes([1, 2, 3, 4])
    for i in range(0, len(starts), 2):
        u = starts[i]

        def test_function_with_poison(data):
            if False:
                while True:
                    i = 10
            v = data.draw(strat)
            m = data.draw_bytes(len(marker))
            if POISON in v and m == marker:
                data.mark_interesting()
        runner = ConjectureRunner(test_function_with_poison, random=random, settings=TEST_SETTINGS)
        runner.cached_test_function(data.buffer[:u] + bytes([255]) * 4 + data.buffer[u + 4:] + marker)
        assert runner.interesting_examples
        runner.shrink_interesting_examples()
        (shrunk,) = runner.interesting_examples.values()
        assert ConjectureData.for_buffer(shrunk.buffer).draw(strat) == (POISON,)