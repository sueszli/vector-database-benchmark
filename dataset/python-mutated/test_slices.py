import pytest
from hypothesis import given, settings, strategies as st
from tests.common.debug import assert_all_examples, find_any, minimal
use_several_sizes = pytest.mark.parametrize('size', [1, 2, 5, 10, 100, 1000])

@use_several_sizes
def test_stop_stays_within_bounds(size):
    if False:
        return 10
    assert_all_examples(st.slices(size), lambda x: x.stop is None or (x.stop >= -size and x.stop <= size))

@use_several_sizes
def test_start_stay_within_bounds(size):
    if False:
        print('Hello World!')
    assert_all_examples(st.slices(size).filter(lambda x: x.start is not None), lambda x: range(size)[x.start] or True)

@use_several_sizes
def test_step_stays_within_bounds(size):
    if False:
        print('Hello World!')
    assert_all_examples(st.slices(size), lambda x: x.indices(size)[0] + x.indices(size)[2] <= size and x.indices(size)[0] + x.indices(size)[2] >= -size or x.start % size == x.stop % size)

@use_several_sizes
def test_step_will_not_be_zero(size):
    if False:
        for i in range(10):
            print('nop')
    assert_all_examples(st.slices(size), lambda x: x.step != 0)

@use_several_sizes
def test_slices_will_shrink(size):
    if False:
        print('Hello World!')
    sliced = minimal(st.slices(size))
    assert sliced.start == 0 or sliced.start is None
    assert sliced.stop == 0 or sliced.stop is None
    assert sliced.step is None

@given(st.integers(1, 1000))
@settings(deadline=None)
def test_step_will_be_negative(size):
    if False:
        for i in range(10):
            print('nop')
    find_any(st.slices(size), lambda x: (x.step or 1) < 0, settings(max_examples=10 ** 6))

@given(st.integers(1, 1000))
@settings(deadline=None)
def test_step_will_be_positive(size):
    if False:
        for i in range(10):
            print('nop')
    find_any(st.slices(size), lambda x: (x.step or 1) > 0)

@pytest.mark.parametrize('size', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_stop_will_equal_size(size):
    if False:
        print('Hello World!')
    find_any(st.slices(size), lambda x: x.stop == size, settings(max_examples=10 ** 6))

@pytest.mark.parametrize('size', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_start_will_equal_size(size):
    if False:
        i = 10
        return i + 15
    find_any(st.slices(size), lambda x: x.start == size - 1, settings(max_examples=10 ** 6))

@given(st.integers(1, 1000))
@settings(deadline=None)
def test_start_will_equal_0(size):
    if False:
        return 10
    find_any(st.slices(size), lambda x: x.start == 0)

@given(st.integers(1, 1000))
@settings(deadline=None)
def test_start_will_equal_stop(size):
    if False:
        print('Hello World!')
    find_any(st.slices(size), lambda x: x.start == x.stop)

def test_size_is_equal_0():
    if False:
        print('Hello World!')
    assert_all_examples(st.slices(0), lambda x: x.step != 0 and x.start is None and (x.stop is None))