import pandas as pd
import pytest
from darts.utils.data.tabularization import get_shared_times_bounds
from darts.utils.timeseries_generation import linear_timeseries

class TestGetSharedTimesBounds:
    """
    Tests `get_shared_times_bounds` function defined in `darts.utils.data.tabularization`.
    """

    def test_shared_times_bounds_overlapping_range_idx_series(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that `get_shared_times_bounds` correctly computes bounds\n        of two overlapping range index timeseries.\n        '
        series_1 = linear_timeseries(start=1, end=15, freq=3)
        series_2 = linear_timeseries(start=2, end=20, freq=2)
        expected_bounds = (series_2.start_time(), series_1.end_time())
        assert get_shared_times_bounds(series_1, series_2) == expected_bounds

    def test_shared_times_bounds_overlapping_datetime_idx_series(self):
        if False:
            return 10
        '\n        Tests that `get_shared_times_bounds` correctly computes bounds\n        of two overlapping datetime index timeseries.\n        '
        series_1 = linear_timeseries(start=pd.Timestamp('1/1/2000'), end=pd.Timestamp('1/15/2000'), freq='3d')
        series_2 = linear_timeseries(start=pd.Timestamp('1/2/2000'), end=pd.Timestamp('1/20/2000'), freq='2d')
        expected_bounds = (series_2.start_time(), series_1.end_time())
        assert get_shared_times_bounds(series_1, series_2) == expected_bounds

    def test_shared_times_bounds_time_idx_inputs(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `get_shared_times_bounds` behaves correctly\n        when passed `pd.Index` inputs instead of `TimeSeries`\n        inputs. Mixtures of `pd.Index` and `TimeSeries` inputs\n        are also checked.\n        '
        series_1 = linear_timeseries(start=0, end=10, freq=1)
        series_2 = linear_timeseries(start=2, end=16, freq=2)
        expected_bounds = (series_2.start_time(), series_1.end_time())
        assert get_shared_times_bounds(series_1.time_index) == (series_1.start_time(), series_1.end_time())
        assert get_shared_times_bounds(series_2.time_index) == (series_2.start_time(), series_2.end_time())
        assert get_shared_times_bounds(series_1.time_index, series_2) == expected_bounds
        assert get_shared_times_bounds(series_1, series_2.time_index) == expected_bounds
        assert get_shared_times_bounds(series_1.time_index, series_2.time_index) == expected_bounds

    def test_shared_times_bounds_subset_series_range_idx(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that `get_shared_times_bounds` correctly handles case where\n        the provided series are formed by taking successive subsets of an\n        initial series (i.e. `series_2` is formed by taking a subset of\n        `series_1`, and `series_3` is formed by taking a subset of `series_2`).\n        In such cases, the bounds are simply the start and end times of the\n        shortest series. This particular test uses range index series to\n        check this behaviour.\n        '
        series = linear_timeseries(start=0, length=10, freq=3)
        subseries = series.copy().drop_after(series.time_index[-1]).drop_before(series.time_index[1])
        subsubseries = subseries.copy().drop_after(subseries.time_index[-1]).drop_before(subseries.time_index[1])
        expected_bounds = (subsubseries.start_time(), subsubseries.end_time())
        assert get_shared_times_bounds(series, subseries, subsubseries) == expected_bounds

    def test_shared_times_bounds_subset_series_datetime_idx(self):
        if False:
            return 10
        '\n        Tests that `get_shared_times_bounds` correctly handles case where\n        the provided series are formed by taking successive subsets of an\n        initial series (i.e. `series_2` is formed by taking a subset of\n        `series_1`, and `series_3` is formed by taking a subset of `series_2`).\n        In such cases, the bounds are simply the start and end times of the\n        shortest series. This particular test uses datetime index series to\n        check this behaviour.\n        '
        series = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=10, freq='3d')
        subseries = series.copy().drop_after(series.time_index[-1]).drop_before(series.time_index[1])
        subsubseries = subseries.copy().drop_after(subseries.time_index[-1]).drop_before(subseries.time_index[1])
        expected_bounds = (subsubseries.start_time(), subsubseries.end_time())
        assert get_shared_times_bounds(series, subseries, subsubseries) == expected_bounds

    def test_shared_times_bounds_identical_inputs_range_idx(self):
        if False:
            print('Hello World!')
        '\n        Tests that `get_shared_times_bounds` correctly handles case where\n        multiple copies of the same series is passed as an input; we expect\n        the return bounds to just be the start and end times of that repeated\n        series. This particular test uses range index series to\n        check this behaviour.\n        '
        series = linear_timeseries(start=0, length=5, freq=1)
        expected = (series.start_time(), series.end_time())
        assert get_shared_times_bounds(series, series) == expected
        assert get_shared_times_bounds(series, series, series) == expected

    def test_shared_times_bounds_identical_inputs_datetime_idx(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that `get_shared_times_bounds` correctly handles case where\n        multiple copies of the same series is passed as an input; we expect\n        the return bounds to just be the start and end times of that repeated\n        series. This particular test uses datetime index series to\n        check this behaviour.\n        '
        series = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=5, freq='d')
        expected = (series.start_time(), series.end_time())
        assert get_shared_times_bounds(series) == expected
        assert get_shared_times_bounds(series, series) == expected
        assert get_shared_times_bounds(series, series, series) == expected

    def test_shared_times_bounds_unspecified_inputs(self):
        if False:
            return 10
        '\n        Tests that `get_shared_times_bounds` correctly handles case unspecified\n        inputs (i.e. `None`) are passed. If passed with a specified series, the\n        `None` input should be ignored, meaning that the returned bounds should\n        be the start and end times of the only specified series. If only `None`\n        inputs are passed, `None` should be returned.\n        '
        series = linear_timeseries(start=0, length=5, freq=1)
        expected = (series.start_time(), series.end_time())
        assert get_shared_times_bounds(series, None) == expected
        assert get_shared_times_bounds(None, series) == expected
        assert get_shared_times_bounds(None, series, None) == expected
        assert get_shared_times_bounds(None) is None
        assert get_shared_times_bounds(None, None, None) is None

    def test_shared_times_bounds_single_idx_overlap_range_idx(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that `get_shared_times_bounds` correctly handles cases\n        where the bounds contains a single time index value. This\n        particular test uses range time index series to check this\n        behaviour.\n        '
        series = linear_timeseries(start=0, length=1, freq=1)
        assert get_shared_times_bounds(series, series) == (series.start_time(), series.end_time())
        series_1 = linear_timeseries(start=0, length=3, freq=1)
        series_2 = linear_timeseries(start=series_1.end_time(), length=2, freq=2)
        assert get_shared_times_bounds(series_1, series_2) == (series_1.end_time(), series_2.start_time())

    def test_shared_times_bounds_single_idx_overlap_datetime_idx(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `get_shared_times_bounds` correctly handles cases\n        where the bounds contains a single time index value. This\n        particular test uses range time index series to check this\n        behaviour.\n        '
        series = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=1, freq='d')
        assert get_shared_times_bounds(series, series) == (series.start_time(), series.end_time())
        series_1 = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=3, freq='d')
        series_2 = linear_timeseries(start=series_1.end_time(), length=2, freq='2d')
        assert get_shared_times_bounds(series_1, series_2) == (series_1.end_time(), series_2.start_time())

    def test_shared_times_bounds_no_overlap_range_idx(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `get_shared_times_bounds` returns `None` when provided\n        with two series that share no overlap. This particular test uses\n        range index series to check this behaviour.\n        '
        series_1 = linear_timeseries(start=0, length=5, freq=1)
        series_2 = linear_timeseries(start=series_1.end_time() + 1, length=6, freq=2)
        assert get_shared_times_bounds(series_1, series_2) is None
        assert get_shared_times_bounds(series_2, series_1, series_2) is None

    def test_shared_times_bounds_no_overlap_datetime_idx(self):
        if False:
            while True:
                i = 10
        '\n        Tests that `get_shared_times_bounds` returns `None` when provided\n        with two series that share no overlap. This particular test uses\n        datetime index series to check this behaviour.\n        '
        series_1 = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=5, freq='d')
        series_2 = linear_timeseries(start=series_1.end_time() + pd.Timedelta('1d'), length=6, freq='2d')
        assert get_shared_times_bounds(series_1, series_2) is None
        assert get_shared_times_bounds(series_2, series_1, series_2) is None

    def test_shared_times_bounds_different_time_idx_types_error(self):
        if False:
            print('Hello World!')
        '\n        Tests that `get_shared_times_bounds` throws correct error\n        when a range time index series and a datetime index series\n        are specified as inputs together.\n        '
        series_1 = linear_timeseries(start=1, length=5, freq=1)
        series_2 = linear_timeseries(start=pd.Timestamp('1/1/2000'), length=5, freq='d')
        with pytest.raises(ValueError) as err:
            get_shared_times_bounds(series_1, series_2)
        assert 'Specified series and/or times must all have the same type of `time_index` (i.e. all `pd.RangeIndex` or all `pd.DatetimeIndex`).' == str(err.value)

    def test_shared_times_bounds_empty_input(self):
        if False:
            print('Hello World!')
        '\n        Tests that `get_shared_times_bounds` returns `None` when\n        handed a non-`None` input that has no timesteps.\n        '
        series = linear_timeseries(start=0, length=0, freq=1)
        assert get_shared_times_bounds(series) is None
        assert get_shared_times_bounds(series.time_index) is None
        assert get_shared_times_bounds(series, series.time_index) is None