from collections import Counter
import pytest
from hypothesis import given, settings
from hypothesis.strategies._internal import SearchStrategy

class Blocks(SearchStrategy):

    def __init__(self, n):
        if False:
            i = 10
            return i + 15
        self.n = n

    def do_draw(self, data):
        if False:
            return 10
        return data.draw_bytes(self.n)

@pytest.mark.parametrize('n', range(1, 5))
def test_does_not_duplicate_blocks(n):
    if False:
        while True:
            i = 10
    counts = Counter()

    @given(Blocks(n))
    @settings(database=None)
    def test(b):
        if False:
            print('Hello World!')
        counts[b] += 1
    test()
    assert set(counts.values()) == {1}

@pytest.mark.parametrize('n', range(1, 5))
def test_mostly_does_not_duplicate_blocks_even_when_failing(n):
    if False:
        print('Hello World!')
    counts = Counter()

    @settings(database=None)
    @given(Blocks(n))
    def test(b):
        if False:
            print('Hello World!')
        counts[b] += 1
        if len(counts) > 3:
            raise ValueError
    try:
        test()
    except ValueError:
        pass
    seen_counts = set(counts.values())
    assert seen_counts in ({1, 2}, {1, 3})
    assert len([k for (k, v) in counts.items() if v > 1]) <= 2