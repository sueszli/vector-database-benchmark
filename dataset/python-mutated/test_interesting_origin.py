import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.internal.compat import ExceptionGroup
from tests.common.utils import flaky

def go_wrong_naive(a, b):
    if False:
        while True:
            i = 10
    try:
        assert a + b < 100
        a / b
    except Exception:
        raise ValueError('Something went wrong') from None

def go_wrong_with_cause(a, b):
    if False:
        return 10
    try:
        assert a + b < 100
        a / b
    except Exception as err:
        raise ValueError('Something went wrong') from err

def go_wrong_coverup(a, b):
    if False:
        i = 10
        return i + 15
    try:
        assert a + b < 100
        a / b
    except Exception:
        raise ValueError('Something went wrong') from None

@pytest.mark.parametrize('function', [go_wrong_naive, go_wrong_with_cause, go_wrong_coverup], ids=lambda f: f.__name__)
@flaky(max_runs=3, min_passes=1)
def test_can_generate_specified_version(function):
    if False:
        return 10

    @given(st.integers(), st.integers())
    @settings(database=None)
    def test_fn(x, y):
        if False:
            return 10
        return function(x, y)
    with pytest.raises(ExceptionGroup):
        test_fn()