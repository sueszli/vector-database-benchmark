"""Tests for Interval-Interval operations, such as overlaps, contains, etc."""
import numpy as np
import pytest
from pandas import Interval, IntervalIndex, Timedelta, Timestamp
import pandas._testing as tm
from pandas.core.arrays import IntervalArray

@pytest.fixture(params=[IntervalArray, IntervalIndex])
def constructor(request):
    if False:
        print('Hello World!')
    '\n    Fixture for testing both interval container classes.\n    '
    return request.param

@pytest.fixture(params=[(Timedelta('0 days'), Timedelta('1 day')), (Timestamp('2018-01-01'), Timedelta('1 day')), (0, 1)], ids=lambda x: type(x[0]).__name__)
def start_shift(request):
    if False:
        for i in range(10):
            print('nop')
    '\n    Fixture for generating intervals of different types from a start value\n    and a shift value that can be added to start to generate an endpoint.\n    '
    return request.param

class TestOverlaps:

    def test_overlaps_interval(self, constructor, start_shift, closed, other_closed):
        if False:
            print('Hello World!')
        (start, shift) = start_shift
        interval = Interval(start, start + 3 * shift, other_closed)
        tuples = [(start, start + 3 * shift), (start + shift, start + 2 * shift), (start - shift, start + 4 * shift), (start + 2 * shift, start + 4 * shift), (start + 3 * shift, start + 4 * shift), (start + 4 * shift, start + 5 * shift)]
        interval_container = constructor.from_tuples(tuples, closed)
        adjacent = interval.closed_right and interval_container.closed_left
        expected = np.array([True, True, True, True, adjacent, False])
        result = interval_container.overlaps(interval)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('other_constructor', [IntervalArray, IntervalIndex])
    def test_overlaps_interval_container(self, constructor, other_constructor):
        if False:
            for i in range(10):
                print('nop')
        interval_container = constructor.from_breaks(range(5))
        other_container = other_constructor.from_breaks(range(5))
        with pytest.raises(NotImplementedError, match='^$'):
            interval_container.overlaps(other_container)

    def test_overlaps_na(self, constructor, start_shift):
        if False:
            for i in range(10):
                print('nop')
        'NA values are marked as False'
        (start, shift) = start_shift
        interval = Interval(start, start + shift)
        tuples = [(start, start + shift), np.nan, (start + 2 * shift, start + 3 * shift)]
        interval_container = constructor.from_tuples(tuples)
        expected = np.array([True, False, False])
        result = interval_container.overlaps(interval)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('other', [10, True, 'foo', Timedelta('1 day'), Timestamp('2018-01-01')], ids=lambda x: type(x).__name__)
    def test_overlaps_invalid_type(self, constructor, other):
        if False:
            for i in range(10):
                print('nop')
        interval_container = constructor.from_breaks(range(5))
        msg = f'`other` must be Interval-like, got {type(other).__name__}'
        with pytest.raises(TypeError, match=msg):
            interval_container.overlaps(other)