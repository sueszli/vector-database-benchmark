import pytest
from pandas import interval_range
import pandas._testing as tm

@pytest.mark.parametrize('kwargs', [{'start': 0, 'periods': 4}, {'start': 1, 'periods': 5}, {'start': 5, 'end': 10, 'closed': 'left'}])
def test_interval_array_equal(kwargs):
    if False:
        while True:
            i = 10
    arr = interval_range(**kwargs).values
    tm.assert_interval_array_equal(arr, arr)

def test_interval_array_equal_closed_mismatch():
    if False:
        print('Hello World!')
    kwargs = {'start': 0, 'periods': 5}
    arr1 = interval_range(closed='left', **kwargs).values
    arr2 = interval_range(closed='right', **kwargs).values
    msg = 'IntervalArray are different\n\nAttribute "closed" are different\n\\[left\\]:  left\n\\[right\\]: right'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)

def test_interval_array_equal_periods_mismatch():
    if False:
        for i in range(10):
            print('nop')
    kwargs = {'start': 0}
    arr1 = interval_range(periods=5, **kwargs).values
    arr2 = interval_range(periods=6, **kwargs).values
    msg = 'IntervalArray.left are different\n\nIntervalArray.left shapes are different\n\\[left\\]:  \\(5,\\)\n\\[right\\]: \\(6,\\)'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)

def test_interval_array_equal_end_mismatch():
    if False:
        i = 10
        return i + 15
    kwargs = {'start': 0, 'periods': 5}
    arr1 = interval_range(end=10, **kwargs).values
    arr2 = interval_range(end=20, **kwargs).values
    msg = 'IntervalArray.left are different\n\nIntervalArray.left values are different \\(80.0 %\\)\n\\[left\\]:  \\[0, 2, 4, 6, 8\\]\n\\[right\\]: \\[0, 4, 8, 12, 16\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)

def test_interval_array_equal_start_mismatch():
    if False:
        print('Hello World!')
    kwargs = {'periods': 4}
    arr1 = interval_range(start=0, **kwargs).values
    arr2 = interval_range(start=1, **kwargs).values
    msg = 'IntervalArray.left are different\n\nIntervalArray.left values are different \\(100.0 %\\)\n\\[left\\]:  \\[0, 1, 2, 3\\]\n\\[right\\]: \\[1, 2, 3, 4\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)