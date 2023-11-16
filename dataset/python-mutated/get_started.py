from hypothesis import given, strategies as st
from hypothesis.strategies import integers, floats
from statistics import mean

@given(st.lists(st.floats(allow_infinity=False, allow_nan=False), min_size=1))
def test_mean_is_in_bounds(ls):
    if False:
        return 10
    assert min(ls) <= mean(ls) <= max(ls)

@given(floats(), floats())
def test_floats_are_commutative(x, y):
    if False:
        i = 10
        return i + 15
    assert x + y == y + x