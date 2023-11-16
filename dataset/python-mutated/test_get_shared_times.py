from math import gcd
import pandas as pd
import pytest
from darts.utils.data.tabularization import get_shared_times
from darts.utils.timeseries_generation import linear_timeseries

def lcm(*integers):
    if False:
        while True:
            i = 10
    a = integers[0]
    for b in integers[1:]:
        a = a * b // gcd(a, b)
    return a

class TestGetSharedTimes:
    """
    Tests `get_shared_times` function defined in `darts.utils.data.tabularization`.
    """

    def test_shared_times_equal_freq_range_idx(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that `get_shared_times` correctly handles range time\n        index series that are of equal frequency.\n        '
        series_1 = linear_timeseries(start=1, end=11, freq=2)
        series_2 = linear_timeseries(start=3, end=13, freq=2)
        series_3 = linear_timeseries(start=5, end=15, freq=2)
        assert series_1.time_index.equals(get_shared_times(series_1))
        assert series_2.time_index.equals(get_shared_times(series_2))
        assert series_3.time_index.equals(get_shared_times(series_3))
        expected_12 = linear_timeseries(start=series_2.start_time(), end=series_1.end_time(), freq=series_1.freq)
        assert expected_12.time_index.equals(get_shared_times(series_1, series_2))
        expected_23 = linear_timeseries(start=series_3.start_time(), end=series_2.end_time(), freq=series_2.freq)
        assert expected_23.time_index.equals(get_shared_times(series_2, series_3))
        expected_13 = linear_timeseries(start=series_3.start_time(), end=series_1.end_time(), freq=series_1.freq)
        assert expected_13.time_index.equals(get_shared_times(series_1, series_3))
        expected_123 = linear_timeseries(start=series_3.start_time(), end=series_1.end_time(), freq=series_1.freq)
        assert expected_123.time_index.equals(get_shared_times(series_1, series_2, series_3))

    def test_shared_times_equal_freq_datetime_idx(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that `get_shared_times` correctly handles datetime time\n        index series that are of equal frequency.\n        '
        series_1 = linear_timeseries(start=pd.Timestamp('1/1/2000'), end=pd.Timestamp('1/11/2000'), freq='2d')
        series_2 = linear_timeseries(start=pd.Timestamp('1/3/2000'), end=pd.Timestamp('1/13/2000'), freq='2d')
        series_3 = linear_timeseries(start=pd.Timestamp('1/5/2000'), end=pd.Timestamp('1/15/2000'), freq='2d')
        assert series_1.time_index.equals(get_shared_times(series_1))
        assert series_2.time_index.equals(get_shared_times(series_2))
        assert series_3.time_index.equals(get_shared_times(series_3))
        expected_12 = linear_timeseries(start=series_2.start_time(), end=series_1.end_time(), freq=series_1.freq)
        assert expected_12.time_index.equals(get_shared_times(series_1, series_2))
        expected_23 = linear_timeseries(start=series_3.start_time(), end=series_2.end_time(), freq=series_2.freq)
        assert expected_23.time_index.equals(get_shared_times(series_2, series_3))
        expected_13 = linear_timeseries(start=series_3.start_time(), end=series_1.end_time(), freq=series_1.freq)
        assert expected_13.time_index.equals(get_shared_times(series_1, series_3))
        expected_123 = linear_timeseries(start=series_3.start_time(), end=series_1.end_time(), freq=series_1.freq)
        assert expected_123.time_index.equals(get_shared_times(series_1, series_2, series_3))

    def test_shared_times_unequal_freq_range_idx(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `get_shared_times` correctly handles range time\n        index series that are of different frequencies.\n        '
        series_1 = linear_timeseries(start=1, end=11, freq=1)
        series_2 = linear_timeseries(start=3, end=13, freq=2)
        series_3 = linear_timeseries(start=5, end=17, freq=3)
        assert series_1.time_index.equals(get_shared_times(series_1))
        assert series_2.time_index.equals(get_shared_times(series_2))
        assert series_3.time_index.equals(get_shared_times(series_3))
        expected_12 = linear_timeseries(start=series_2.start_time(), end=series_1.end_time(), freq=lcm(series_1.freq, series_2.freq))
        if expected_12.time_index[-1] > series_1.end_time():
            expected_12 = expected_12.drop_after(expected_12.time_index[-1])
        assert expected_12.time_index.equals(get_shared_times(series_1, series_2))
        expected_23 = linear_timeseries(start=series_3.start_time(), end=series_2.end_time(), freq=lcm(series_2.freq, series_3.freq))
        if expected_23.time_index[-1] > series_2.end_time():
            expected_23 = expected_23.drop_after(expected_23.time_index[-1])
        assert expected_23.time_index.equals(get_shared_times(series_2, series_3))
        expected_13 = linear_timeseries(start=series_3.start_time(), end=series_1.end_time(), freq=lcm(series_1.freq, series_3.freq))
        if expected_13.time_index[-1] > series_1.end_time():
            expected_13 = expected_13.drop_after(expected_13.time_index[-1])
        assert expected_13.time_index.equals(get_shared_times(series_1, series_3))
        expected_123 = linear_timeseries(start=series_3.start_time(), end=series_1.end_time(), freq=lcm(series_1.freq, series_2.freq, series_3.freq))
        if expected_123.time_index[-1] > series_1.end_time():
            expected_123 = expected_123.drop_after(expected_123.time_index[-1])
        assert expected_123.time_index.equals(get_shared_times(series_1, series_2, series_3))

    def test_shared_times_unequal_freq_datetime_idx(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `get_shared_times` correctly handles range time\n        index series that are of different frequencies.\n        '
        series_1 = linear_timeseries(start=pd.Timestamp('1/1/2000'), end=pd.Timestamp('1/11/2000'), freq='2d')
        series_2 = linear_timeseries(start=pd.Timestamp('1/3/2000'), end=pd.Timestamp('1/13/2000'), freq='2d')
        series_3 = linear_timeseries(start=pd.Timestamp('1/5/2000'), end=pd.Timestamp('1/15/2000'), freq='2d')
        assert series_1.time_index.equals(get_shared_times(series_1))
        assert series_2.time_index.equals(get_shared_times(series_2))
        assert series_3.time_index.equals(get_shared_times(series_3))
        freq_12 = f'{lcm(series_1.freq.n, series_2.freq.n)}d'
        expected_12 = linear_timeseries(start=series_2.start_time(), end=series_1.end_time(), freq=freq_12)
        if expected_12.time_index[-1] > series_1.end_time():
            expected_12 = expected_12.drop_after(expected_12.time_index[-1])
        assert expected_12.time_index.equals(get_shared_times(series_1, series_2))
        freq_23 = f'{lcm(series_2.freq.n, series_3.freq.n)}d'
        expected_23 = linear_timeseries(start=series_3.start_time(), end=series_2.end_time(), freq=freq_23)
        if expected_23.time_index[-1] > series_2.end_time():
            expected_23 = expected_23.drop_after(expected_23.time_index[-1])
        assert expected_23.time_index.equals(get_shared_times(series_2, series_3))
        freq_13 = f'{lcm(series_1.freq.n, series_3.freq.n)}d'
        expected_13 = linear_timeseries(start=series_3.start_time(), end=series_1.end_time(), freq=freq_13)
        if expected_13.time_index[-1] > series_1.end_time():
            expected_13 = expected_13.drop_after(expected_13.time_index[-1])
        assert expected_13.time_index.equals(get_shared_times(series_1, series_3))
        freq_123 = f'{lcm(series_1.freq.n, series_2.freq.n, series_3.freq.n)}d'
        expected_123 = linear_timeseries(start=series_3.start_time(), end=series_1.end_time(), freq=freq_123)
        if expected_123.time_index[-1] > series_1.end_time():
            expected_123 = expected_123.drop_after(expected_123.time_index[-1])
        assert expected_123.time_index.equals(get_shared_times(series_1, series_2, series_3))

    def test_shared_times_no_overlap_range_idx(self):
        if False:
            print('Hello World!')
        '\n        Tests that `get_shared_times` returns `None` when\n        supplied range time index series share no temporal overlap.\n        '
        series_1 = linear_timeseries(start=1, end=11, freq=2)
        series_2 = linear_timeseries(start=series_1.end_time() + 1, length=5, freq=3)
        assert get_shared_times(series_1, series_2) is None
        assert get_shared_times(series_1, series_1, series_2) is None
        assert get_shared_times(series_1, series_2, series_2) is None
        assert get_shared_times(series_1, series_1, series_2, series_2) is None

    def test_shared_times_no_overlap_datetime_idx(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `get_shared_times` returns `None` when\n        supplied datetime time index series share no temporal overlap.\n        '
        series_1 = linear_timeseries(start=pd.Timestamp('1/1/2000'), end=pd.Timestamp('1/11/2000'), freq='2d')
        series_2 = linear_timeseries(start=series_1.end_time() + pd.Timedelta(1, 'd'), length=5, freq='3d')
        assert get_shared_times(series_1, series_2) is None
        assert get_shared_times(series_1, series_1, series_2) is None
        assert get_shared_times(series_1, series_2, series_2) is None
        assert get_shared_times(series_1, series_1, series_2, series_2) is None

    def test_shared_times_single_time_point_overlap_range_idx(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that `get_shared_times` returns correct bounds when\n        given range index series that overlap at a single time point.\n        '
        series_1 = linear_timeseries(start=1, end=11, freq=2)
        series_2 = linear_timeseries(start=series_1.end_time(), length=5, freq=3)
        overlap_val = series_1.end_time()
        assert get_shared_times(series_1, series_2) == overlap_val
        assert get_shared_times(series_1, series_1, series_2) == overlap_val
        assert get_shared_times(series_1, series_2, series_2) == overlap_val
        assert get_shared_times(series_1, series_1, series_2, series_2) == overlap_val

    def test_shared_times_single_time_point_overlap_datetime_idx(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that `get_shared_times` returns correct bounds when\n        given datetime index series that overlap at a single time point.\n        '
        series_1 = linear_timeseries(start=pd.Timestamp('1/1/2000'), end=pd.Timestamp('1/11/2000'), freq='2d')
        series_2 = linear_timeseries(start=series_1.end_time(), length=5, freq='3d')
        overlap_val = series_1.end_time()
        assert get_shared_times(series_1, series_2) == overlap_val
        assert get_shared_times(series_1, series_1, series_2) == overlap_val
        assert get_shared_times(series_1, series_2, series_2) == overlap_val
        assert get_shared_times(series_1, series_1, series_2, series_2) == overlap_val

    def test_shared_times_identical_inputs_range_idx(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that `get_shared_times` correctly handles case where\n        multiple copies of same range index timeseries is passed;\n        we expect that the unaltered time index of the series is returned.\n        '
        series = linear_timeseries(start=0, length=5, freq=1)
        assert series.time_index.equals(get_shared_times(series))
        assert series.time_index.equals(get_shared_times(series, series))
        assert series.time_index.equals(get_shared_times(series, series, series))

    def test_shared_times_identical_inputs_datetime_idx(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `get_shared_times` correctly handles case where\n        multiple copies of same datetime index timeseries is passed;\n        we expect that the unaltered time index of the series is returned.\n        '
        series = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=5, freq='d')
        assert series.time_index.equals(get_shared_times(series))
        assert series.time_index.equals(get_shared_times(series, series))
        assert series.time_index.equals(get_shared_times(series, series, series))

    def test_shared_times_unspecified_inputs(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that `get_shared_times` correctly handles unspecified\n        (i.e. `None` value) inputs. If `None` is passed with another\n        series/time index, then `None` should be ignored and the time\n        index of the other series should be returned. If only `None`\n        values are passed, `None` should be returned.\n        '
        series = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=5, freq='d')
        assert get_shared_times(None) is None
        assert series.time_index.equals(get_shared_times(series, None))
        assert series.time_index.equals(get_shared_times(None, series, None))
        assert series.time_index.equals(get_shared_times(None, series.time_index, None))
        assert get_shared_times(None) is None

    def test_shared_times_time_index_inputs(self):
        if False:
            print('Hello World!')
        '\n        Tests that `get_shared_times` can accept time index\n        inputs instead of `TimeSeries` inputs; combinations\n        of time index and `TimeSeries` inputs are also tested.\n        '
        series_1 = linear_timeseries(start=0, end=10, freq=1)
        series_2 = linear_timeseries(start=0, end=20, freq=2)
        intersection = pd.RangeIndex(start=series_2.start_time(), stop=series_1.end_time() + 1, step=2)
        assert intersection.equals(get_shared_times(series_1.time_index, series_2))
        assert intersection.equals(get_shared_times(series_1, series_2.time_index))
        assert intersection.equals(get_shared_times(series_1.time_index, series_2.time_index))

    def test_shared_times_empty_input(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `get_shared_times` returns `None` when\n        given a non-`None` input with no timesteps.\n        '
        series = linear_timeseries(start=0, length=0, freq=1)
        assert get_shared_times(series) is None
        assert get_shared_times(series.time_index) is None
        assert get_shared_times(series, series.time_index) is None

    def test_shared_times_different_time_index_types_error(self):
        if False:
            return 10
        '\n        Tests that `get_shared_times` throws correct error when\n        provided with series with different types of time indices.\n        '
        series_1 = linear_timeseries(start=1, length=5, freq=1)
        series_2 = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=5, freq='d')
        with pytest.raises(ValueError) as err:
            get_shared_times(series_1, series_2)
        assert 'Specified series and/or times must all have the same type of `time_index` (i.e. all `pd.RangeIndex` or all `pd.DatetimeIndex`).' == str(err.value)