import math
import sys
import warnings
import pytest
from hypothesis import assume, given, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.internal.floats import float_of, float_to_int, int_to_float, is_negative, next_down, next_up
from tests.common.debug import find_any, minimal
try:
    import numpy
except ImportError:
    numpy = None

@pytest.mark.parametrize(('lower', 'upper'), [(9.9792015476736e+291, 1.7976931348623157e+308), (-sys.float_info.max, sys.float_info.max)])
def test_floats_are_in_range(lower, upper):
    if False:
        for i in range(10):
            print('nop')

    @given(st.floats(lower, upper))
    def test_is_in_range(t):
        if False:
            while True:
                i = 10
        assert lower <= t <= upper
    test_is_in_range()

@pytest.mark.parametrize('sign', [-1, 1])
def test_can_generate_both_zeros(sign):
    if False:
        for i in range(10):
            print('nop')
    assert minimal(st.floats(), lambda x: math.copysign(1, x) == sign) == sign * 0.0

@pytest.mark.parametrize(('l', 'r'), [(-1.0, 1.0), (-0.0, 1.0), (-1.0, 0.0), (-sys.float_info.min, sys.float_info.min)])
@pytest.mark.parametrize('sign', [-1, 1])
def test_can_generate_both_zeros_when_in_interval(l, r, sign):
    if False:
        print('Hello World!')
    assert minimal(st.floats(l, r), lambda x: math.copysign(1, x) == sign) == sign * 0.0

@given(st.floats(0.0, 1.0))
def test_does_not_generate_negative_if_right_boundary_is_positive(x):
    if False:
        while True:
            i = 10
    assert math.copysign(1, x) == 1

@given(st.floats(-1.0, -0.0))
def test_does_not_generate_positive_if_right_boundary_is_negative(x):
    if False:
        print('Hello World!')
    assert math.copysign(1, x) == -1

def test_half_bounded_generates_zero():
    if False:
        while True:
            i = 10
    find_any(st.floats(min_value=-1.0), lambda x: x == 0.0)
    find_any(st.floats(max_value=1.0), lambda x: x == 0.0)

@given(st.floats(max_value=-0.0))
def test_half_bounded_respects_sign_of_upper_bound(x):
    if False:
        return 10
    assert math.copysign(1, x) == -1

@given(st.floats(min_value=0.0))
def test_half_bounded_respects_sign_of_lower_bound(x):
    if False:
        i = 10
        return i + 15
    assert math.copysign(1, x) == 1

@given(st.floats(allow_nan=False))
def test_filter_nan(x):
    if False:
        print('Hello World!')
    assert not math.isnan(x)

@given(st.floats(allow_infinity=False))
def test_filter_infinity(x):
    if False:
        for i in range(10):
            print('nop')
    assert not math.isinf(x)

def test_can_guard_against_draws_of_nan():
    if False:
        for i in range(10):
            print('nop')
    'In this test we create a NaN value that naturally "tries" to shrink into\n    the first strategy, where it is not permitted. This tests a case that is\n    very unlikely to happen in random generation: When the unconstrained first\n    branch of generating a float just happens to produce a NaN value.\n\n    Here what happens is that we get a NaN from the *second* strategy,\n    but this then shrinks into its unconstrained branch. The natural\n    thing to happen is then to try to zero the branch parameter of the\n    one_of, but that will put an illegal value there, so it\'s not\n    allowed to happen.\n    '
    tagged_floats = st.one_of(st.tuples(st.just(0), st.floats(allow_nan=False)), st.tuples(st.just(1), st.floats(allow_nan=True)))
    (tag, f) = minimal(tagged_floats, lambda x: math.isnan(x[1]))
    assert tag == 1

def test_very_narrow_interval():
    if False:
        for i in range(10):
            print('nop')
    upper_bound = -1.0
    lower_bound = int_to_float(float_to_int(upper_bound) + 10)
    assert lower_bound < upper_bound

    @given(st.floats(lower_bound, upper_bound))
    def test(f):
        if False:
            print('Hello World!')
        assert lower_bound <= f <= upper_bound
    test()

@given(st.floats())
def test_up_means_greater(x):
    if False:
        while True:
            i = 10
    hi = next_up(x)
    if not x < hi:
        assert math.isnan(x) and math.isnan(hi) or (x > 0 and math.isinf(x)) or (x == hi == 0 and is_negative(x) and (not is_negative(hi)))

@given(st.floats())
def test_down_means_lesser(x):
    if False:
        i = 10
        return i + 15
    lo = next_down(x)
    if not x > lo:
        assert math.isnan(x) and math.isnan(lo) or (x < 0 and math.isinf(x)) or (x == lo == 0 and is_negative(lo) and (not is_negative(x)))

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_updown_roundtrip(val):
    if False:
        i = 10
        return i + 15
    assert val == next_up(next_down(val))
    assert val == next_down(next_up(val))

@given(st.floats(width=32, allow_infinity=False))
def test_float32_can_exclude_infinity(x):
    if False:
        i = 10
        return i + 15
    assert not math.isinf(x)

@given(st.floats(width=16, allow_infinity=False))
def test_float16_can_exclude_infinity(x):
    if False:
        while True:
            i = 10
    assert not math.isinf(x)

@pytest.mark.parametrize('kwargs', [{'min_value': 10 ** 5, 'width': 16}, {'max_value': 10 ** 5, 'width': 16}, {'min_value': 10 ** 40, 'width': 32}, {'max_value': 10 ** 40, 'width': 32}, {'min_value': 10 ** 400, 'width': 64}, {'max_value': 10 ** 400, 'width': 64}, {'min_value': 10 ** 400}, {'max_value': 10 ** 400}])
def test_out_of_range(kwargs):
    if False:
        print('Hello World!')
    with pytest.raises(OverflowError):
        st.floats(**kwargs).validate()

def test_disallowed_width():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InvalidArgument):
        st.floats(width=128).validate()

def test_no_single_floats_in_range():
    if False:
        return 10
    low = 2.0 ** 25 + 1
    high = low + 2
    st.floats(low, high).validate()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(InvalidArgument):
            st.floats(low, high, width=32).validate()

@given(st.floats(min_value=1e+304, allow_infinity=False))
def test_finite_min_bound_does_not_overflow(x):
    if False:
        while True:
            i = 10
    assert not math.isinf(x)

@given(st.floats(max_value=-1e+304, allow_infinity=False))
def test_finite_max_bound_does_not_overflow(x):
    if False:
        i = 10
        return i + 15
    assert not math.isinf(x)

@given(st.floats(0, 1, exclude_min=True, exclude_max=True))
def test_can_exclude_endpoints(x):
    if False:
        while True:
            i = 10
    assert 0 < x < 1

@given(st.floats(-math.inf, -1e+307, exclude_min=True))
def test_can_exclude_neg_infinite_endpoint(x):
    if False:
        while True:
            i = 10
    assert not math.isinf(x)

@given(st.floats(1e+307, math.inf, exclude_max=True))
def test_can_exclude_pos_infinite_endpoint(x):
    if False:
        i = 10
        return i + 15
    assert not math.isinf(x)

def test_exclude_infinite_endpoint_is_invalid():
    if False:
        while True:
            i = 10
    with pytest.raises(InvalidArgument):
        st.floats(min_value=math.inf, exclude_min=True).validate()
    with pytest.raises(InvalidArgument):
        st.floats(max_value=-math.inf, exclude_max=True).validate()

@pytest.mark.parametrize('lo,hi', [(True, False), (False, True), (True, True)])
@given(bound=st.floats(allow_nan=False, allow_infinity=False).filter(bool))
def test_exclude_entire_interval(lo, hi, bound):
    if False:
        i = 10
        return i + 15
    with pytest.raises(InvalidArgument, match='exclude_min=.+ and exclude_max='):
        st.floats(bound, bound, exclude_min=lo, exclude_max=hi).validate()

def test_zero_intervals_are_OK():
    if False:
        while True:
            i = 10
    st.floats(0.0, 0.0).validate()
    st.floats(-0.0, 0.0).validate()
    st.floats(-0.0, -0.0).validate()

@pytest.mark.parametrize('lo', [0.0, -0.0])
@pytest.mark.parametrize('hi', [0.0, -0.0])
@pytest.mark.parametrize('exmin,exmax', [(True, False), (False, True), (True, True)])
def test_cannot_exclude_endpoint_with_zero_interval(lo, hi, exmin, exmax):
    if False:
        print('Hello World!')
    with pytest.raises(InvalidArgument):
        st.floats(lo, hi, exclude_min=exmin, exclude_max=exmax).validate()
WIDTHS = (64, 32, 16)

@pytest.mark.parametrize('nonfloat', [st.nothing(), st.none()])
@given(data=st.data(), width=st.sampled_from(WIDTHS))
def test_fuzzing_floats_bounds(data, width, nonfloat):
    if False:
        print('Hello World!')
    lo = data.draw(nonfloat | st.floats(allow_nan=False, width=width), label='lo')
    hi = data.draw(nonfloat | st.floats(allow_nan=False, width=width), label='hi')
    if lo is not None and hi is not None and (lo > hi):
        (lo, hi) = (hi, lo)
    assume(lo != 0 or hi != 0)
    value = data.draw(st.floats(min_value=lo, max_value=hi, width=width, allow_nan=False), label='value')
    assert value == float_of(value, width=width)
    assert lo is None or lo <= value
    assert hi is None or value <= hi