from random import Random
import pytest
from hypothesis import settings, strategies as st
from hypothesis.internal.compat import ceil
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.conjecture.engine import ConjectureData, ConjectureRunner
from hypothesis.strategies._internal import SearchStrategy
POISON = 'POISON'

class Poisoned(SearchStrategy):

    def __init__(self, poison_chance):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.__poison_chance = poison_chance
        self.__ints = st.integers(0, 10)

    def do_draw(self, data):
        if False:
            i = 10
            return i + 15
        if cu.biased_coin(data, self.__poison_chance):
            return POISON
        else:
            return data.draw(self.__ints)

class LinearLists(SearchStrategy):

    def __init__(self, elements, size):
        if False:
            print('Hello World!')
        super().__init__()
        self.__length = st.integers(0, size)
        self.__elements = elements

    def do_draw(self, data):
        if False:
            for i in range(10):
                print('nop')
        return [data.draw(self.__elements) for _ in range(data.draw(self.__length))]

class Matrices(SearchStrategy):

    def __init__(self, elements, size):
        if False:
            return 10
        super().__init__()
        self.__length = st.integers(0, ceil(size ** 0.5))
        self.__elements = elements

    def do_draw(self, data):
        if False:
            return 10
        n = data.draw(self.__length)
        m = data.draw(self.__length)
        return [data.draw(self.__elements) for _ in range(n * m)]
LOTS = 10 ** 6
TRIAL_SETTINGS = settings(max_examples=LOTS, database=None)

@pytest.mark.parametrize('seed', [2282791295271755424, 1284235381287210546, 14202812238092722246, 26097])
@pytest.mark.parametrize('size', [5, 10, 20])
@pytest.mark.parametrize('p', [0.01, 0.1])
@pytest.mark.parametrize('strategy_class', [LinearLists, Matrices])
def test_minimal_poisoned_containers(seed, size, p, strategy_class, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    elements = Poisoned(p)
    strategy = strategy_class(elements, size)

    def test_function(data):
        if False:
            i = 10
            return i + 15
        v = data.draw(strategy)
        data.output = repr(v)
        if POISON in v:
            data.mark_interesting()
    runner = ConjectureRunner(test_function, random=Random(seed), settings=TRIAL_SETTINGS)
    runner.run()
    (v,) = runner.interesting_examples.values()
    result = ConjectureData.for_buffer(v.buffer).draw(strategy)
    assert len(result) == 1