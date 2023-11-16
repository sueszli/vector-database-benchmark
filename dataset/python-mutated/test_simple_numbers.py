import math
import sys
import pytest
from hypothesis import given
from hypothesis.strategies import floats, integers, lists
from tests.common.debug import minimal

def test_minimize_negative_int():
    if False:
        while True:
            i = 10
    assert minimal(integers(), lambda x: x < 0) == -1
    assert minimal(integers(), lambda x: x < -1) == -2

def test_positive_negative_int():
    if False:
        print('Hello World!')
    assert minimal(integers(), lambda x: x > 0) == 1
    assert minimal(integers(), lambda x: x > 1) == 2
boundaries = pytest.mark.parametrize('boundary', sorted([2 ** i for i in range(10)] + [2 ** i - 1 for i in range(10)] + [2 ** i + 1 for i in range(10)] + [10 ** i for i in range(6)]))

@boundaries
def test_minimizes_int_down_to_boundary(boundary):
    if False:
        print('Hello World!')
    assert minimal(integers(), lambda x: x >= boundary) == boundary

@boundaries
def test_minimizes_int_up_to_boundary(boundary):
    if False:
        return 10
    assert minimal(integers(), lambda x: x <= -boundary) == -boundary

@boundaries
def test_minimizes_ints_from_down_to_boundary(boundary):
    if False:
        for i in range(10):
            print('nop')

    def is_good(x):
        if False:
            for i in range(10):
                print('nop')
        assert x >= boundary - 10
        return x >= boundary
    assert minimal(integers(min_value=boundary - 10), is_good) == boundary
    assert minimal(integers(min_value=boundary), lambda x: True) == boundary

def test_minimizes_negative_integer_range_upwards():
    if False:
        print('Hello World!')
    assert minimal(integers(min_value=-10, max_value=-1)) == -1

@boundaries
def test_minimizes_integer_range_to_boundary(boundary):
    if False:
        return 10
    assert minimal(integers(boundary, boundary + 100), lambda x: True) == boundary

def test_single_integer_range_is_range():
    if False:
        print('Hello World!')
    assert minimal(integers(1, 1), lambda x: True) == 1

def test_minimal_small_number_in_large_range():
    if False:
        i = 10
        return i + 15
    assert minimal(integers(-2 ** 32, 2 ** 32), lambda x: x >= 101) == 101

def test_minimal_small_sum_float_list():
    if False:
        for i in range(10):
            print('nop')
    xs = minimal(lists(floats(), min_size=5), lambda x: sum(x) >= 1.0)
    assert xs == [0.0, 0.0, 0.0, 0.0, 1.0]

def test_minimals_boundary_floats():
    if False:
        for i in range(10):
            print('nop')

    def f(x):
        if False:
            while True:
                i = 10
        print(x)
        return True
    assert minimal(floats(min_value=-1, max_value=1), f) == 0

def test_minimal_non_boundary_float():
    if False:
        for i in range(10):
            print('nop')
    x = minimal(floats(min_value=1, max_value=9), lambda x: x > 2)
    assert x == 3

def test_minimal_float_is_zero():
    if False:
        i = 10
        return i + 15
    assert minimal(floats(), lambda x: True) == 0.0

def test_minimal_asymetric_bounded_float():
    if False:
        return 10
    assert minimal(floats(min_value=1.1, max_value=1.6), lambda x: True) == 1.5

def test_negative_floats_simplify_to_zero():
    if False:
        for i in range(10):
            print('nop')
    assert minimal(floats(), lambda x: x <= -1.0) == -1.0

def test_minimal_infinite_float_is_positive():
    if False:
        while True:
            i = 10
    assert minimal(floats(), math.isinf) == math.inf

def test_can_minimal_infinite_negative_float():
    if False:
        for i in range(10):
            print('nop')
    assert minimal(floats(), lambda x: x < -sys.float_info.max)

def test_can_minimal_float_on_boundary_of_representable():
    if False:
        for i in range(10):
            print('nop')
    minimal(floats(), lambda x: x + 1 == x and (not math.isinf(x)))

def test_minimize_nan():
    if False:
        print('Hello World!')
    assert math.isnan(minimal(floats(), math.isnan))

def test_minimize_very_large_float():
    if False:
        print('Hello World!')
    t = sys.float_info.max / 2
    assert minimal(floats(), lambda x: x >= t) == t

def is_integral(value):
    if False:
        return 10
    try:
        return int(value) == value
    except (OverflowError, ValueError):
        return False

def test_can_minimal_float_far_from_integral():
    if False:
        i = 10
        return i + 15
    minimal(floats(), lambda x: math.isfinite(x) and (not is_integral(x * 2 ** 32)))

def test_list_of_fractional_float():
    if False:
        return 10
    assert set(minimal(lists(floats(), min_size=5), lambda x: len([t for t in x if t >= 1.5]) >= 5, timeout_after=60)) == {2}

def test_minimal_fractional_float():
    if False:
        print('Hello World!')
    assert minimal(floats(), lambda x: x >= 1.5) == 2

def test_minimizes_lists_of_negative_ints_up_to_boundary():
    if False:
        return 10
    result = minimal(lists(integers(), min_size=10), lambda x: len([t for t in x if t <= -1]) >= 10, timeout_after=60)
    assert result == [-1] * 10

@pytest.mark.parametrize(('left', 'right'), [(0.0, 5e-324), (-5e-324, 0.0), (-5e-324, 5e-324), (5e-324, 1e-323)])
def test_floats_in_constrained_range(left, right):
    if False:
        print('Hello World!')

    @given(floats(left, right))
    def test_in_range(r):
        if False:
            i = 10
            return i + 15
        assert left <= r <= right
    test_in_range()

def test_bounds_are_respected():
    if False:
        i = 10
        return i + 15
    assert minimal(floats(min_value=1.0), lambda x: True) == 1.0
    assert minimal(floats(max_value=-1.0), lambda x: True) == -1.0

@pytest.mark.parametrize('k', range(10))
def test_floats_from_zero_have_reasonable_range(k):
    if False:
        i = 10
        return i + 15
    n = 10 ** k
    assert minimal(floats(min_value=0.0), lambda x: x >= n) == float(n)
    assert minimal(floats(max_value=0.0), lambda x: x <= -n) == float(-n)

def test_explicit_allow_nan():
    if False:
        for i in range(10):
            print('nop')
    minimal(floats(allow_nan=True), math.isnan)

def test_one_sided_contains_infinity():
    if False:
        return 10
    minimal(floats(min_value=1.0), math.isinf)
    minimal(floats(max_value=1.0), math.isinf)

@given(floats(min_value=0.0, allow_infinity=False))
def test_no_allow_infinity_upper(x):
    if False:
        i = 10
        return i + 15
    assert not math.isinf(x)

@given(floats(max_value=0.0, allow_infinity=False))
def test_no_allow_infinity_lower(x):
    if False:
        return 10
    assert not math.isinf(x)

class TestFloatsAreFloats:

    @given(floats())
    def test_unbounded(self, arg):
        if False:
            return 10
        assert isinstance(arg, float)

    @given(floats(min_value=0, max_value=float(2 ** 64 - 1)))
    def test_int_float(self, arg):
        if False:
            return 10
        assert isinstance(arg, float)

    @given(floats(min_value=float(0), max_value=float(2 ** 64 - 1)))
    def test_float_float(self, arg):
        if False:
            i = 10
            return i + 15
        assert isinstance(arg, float)