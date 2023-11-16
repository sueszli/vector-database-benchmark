import pytest
from hypothesis import HealthCheck, Verbosity, example, given, settings, strategies as st
from hypothesis.internal.compat import ceil
from tests.common.debug import minimal

def test_shrinks_to_simple_floats():
    if False:
        i = 10
        return i + 15
    assert minimal(st.floats(), lambda x: x > 1) == 2.0
    assert minimal(st.floats(), lambda x: x > 0) == 1.0

@pytest.mark.parametrize('n', [1, 2, 3, 8, 10])
def test_can_shrink_in_variable_sized_context(n):
    if False:
        for i in range(10):
            print('nop')
    x = minimal(st.lists(st.floats(), min_size=n), any)
    assert len(x) == n
    assert x.count(0.0) == n - 1
    assert 1 in x

@example(1.7976931348623157e+308)
@example(1.5)
@given(st.floats(min_value=0, allow_infinity=False, allow_nan=False))
@settings(deadline=None, suppress_health_check=list(HealthCheck))
def test_shrinks_downwards_to_integers(f):
    if False:
        return 10
    g = minimal(st.floats().filter(lambda x: x >= f), settings=settings(verbosity=Verbosity.quiet, max_examples=10 ** 6))
    assert g == ceil(f)

@example(1)
@given(st.integers(1, 2 ** 16 - 1))
@settings(deadline=None, suppress_health_check=list(HealthCheck), max_examples=10)
def test_shrinks_downwards_to_integers_when_fractional(b):
    if False:
        return 10
    g = minimal(st.floats().filter(lambda x: b < x < 2 ** 53 and int(x) != x), settings=settings(verbosity=Verbosity.quiet, max_examples=10 ** 6))
    assert g == b + 0.5