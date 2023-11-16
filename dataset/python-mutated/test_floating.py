"""Tests for being able to generate weird and wonderful floating point numbers."""
import math
import sys
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.internal.floats import float_to_int
from hypothesis.strategies import data, floats, lists
from tests.common.debug import find_any
from tests.common.utils import fails
TRY_HARDER = settings(max_examples=1000, suppress_health_check=[HealthCheck.filter_too_much])

@given(floats())
@TRY_HARDER
def test_is_float(x):
    if False:
        print('Hello World!')
    assert isinstance(x, float)

@fails
@given(floats())
@TRY_HARDER
def test_inversion_is_imperfect(x):
    if False:
        while True:
            i = 10
    assume(x != 0.0)
    y = 1.0 / x
    assert x * y == 1.0

@given(floats(-sys.float_info.max, sys.float_info.max))
def test_largest_range(x):
    if False:
        while True:
            i = 10
    assert not math.isinf(x)

@given(floats())
@TRY_HARDER
def test_negation_is_self_inverse(x):
    if False:
        return 10
    assume(not math.isnan(x))
    y = -x
    assert -y == x

@fails
@given(lists(floats()))
def test_is_not_nan(xs):
    if False:
        print('Hello World!')
    assert not any((math.isnan(x) for x in xs))

@fails
@given(floats())
@TRY_HARDER
def test_is_not_positive_infinite(x):
    if False:
        for i in range(10):
            print('nop')
    assume(x > 0)
    assert not math.isinf(x)

@fails
@given(floats())
@TRY_HARDER
def test_is_not_negative_infinite(x):
    if False:
        i = 10
        return i + 15
    assume(x < 0)
    assert not math.isinf(x)

@fails
@given(floats())
@TRY_HARDER
def test_is_int(x):
    if False:
        print('Hello World!')
    assume(math.isfinite(x))
    assert x == int(x)

@fails
@given(floats())
@TRY_HARDER
def test_is_not_int(x):
    if False:
        i = 10
        return i + 15
    assume(math.isfinite(x))
    assert x != int(x)

@fails
@given(floats())
@TRY_HARDER
def test_is_in_exact_int_range(x):
    if False:
        for i in range(10):
            print('nop')
    assume(math.isfinite(x))
    assert x + 1 != x

@fails
@given(floats())
@TRY_HARDER
def test_can_find_floats_that_do_not_round_trip_through_strings(x):
    if False:
        print('Hello World!')
    assert float(str(x)) == x

@fails
@given(floats())
@TRY_HARDER
def test_can_find_floats_that_do_not_round_trip_through_reprs(x):
    if False:
        for i in range(10):
            print('nop')
    assert float(repr(x)) == x
finite_floats = floats(allow_infinity=False, allow_nan=False)

@settings(deadline=None)
@given(finite_floats, finite_floats, data())
def test_floats_are_in_range(x, y, data):
    if False:
        return 10
    (x, y) = sorted((x, y))
    assume(x < y)
    t = data.draw(floats(x, y))
    assert x <= t <= y

@pytest.mark.parametrize('neg', [False, True])
@pytest.mark.parametrize('snan', [False, True])
def test_can_find_negative_and_signaling_nans(neg, snan):
    if False:
        while True:
            i = 10
    find_any(floats().filter(math.isnan), lambda x: snan is (float_to_int(abs(x)) != float_to_int(float('nan'))) and neg is (math.copysign(1, x) == -1), settings=TRY_HARDER)