import pytest
from hypothesis import given, settings
from hypothesis.internal.conjecture.utils import integer_range
from hypothesis.strategies import integers
from hypothesis.strategies._internal.strategies import SearchStrategy
from tests.common.debug import minimal

class interval(SearchStrategy):

    def __init__(self, lower, upper, center=None):
        if False:
            i = 10
            return i + 15
        self.lower = lower
        self.upper = upper
        self.center = center

    def do_draw(self, data):
        if False:
            i = 10
            return i + 15
        return integer_range(data, self.lower, self.upper, center=self.center)

@pytest.mark.parametrize('lower_center_upper', [(0, 5, 10), (-10, 10, 10), (0, 1, 1), (1, 1, 2), (-10, 0, 10), (-10, 5, 10)], ids=repr)
def test_intervals_shrink_to_center(lower_center_upper):
    if False:
        while True:
            i = 10
    (lower, center, upper) = lower_center_upper
    s = interval(lower, upper, center)
    assert minimal(s, lambda x: True) == center
    if lower < center:
        assert minimal(s, lambda x: x < center) == center - 1
    if center < upper:
        assert minimal(s, lambda x: x > center) == center + 1
        assert minimal(s, lambda x: x != center) == center + 1

def test_bounded_integers_distribution_of_bit_width_issue_1387_regression():
    if False:
        for i in range(10):
            print('nop')
    values = []

    @settings(database=None, max_examples=1000)
    @given(integers(0, 1e+100))
    def test(x):
        if False:
            for i in range(10):
                print('nop')
        if 2 <= x <= int(1e+100) - 2:
            values.append(x)
    test()
    huge = sum((x > 1e+97 for x in values))
    assert huge != 0 or len(values) < 800
    assert huge <= 0.3 * len(values)