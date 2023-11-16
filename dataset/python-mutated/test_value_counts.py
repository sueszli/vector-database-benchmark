import numpy as np
from pandas import DatetimeIndex, NaT, PeriodIndex, Series, TimedeltaIndex, date_range, period_range, timedelta_range
import pandas._testing as tm

class TestValueCounts:

    def test_value_counts_unique_datetimeindex(self, tz_naive_fixture):
        if False:
            for i in range(10):
                print('nop')
        tz = tz_naive_fixture
        orig = date_range('2011-01-01 09:00', freq='h', periods=10, tz=tz)
        self._check_value_counts_with_repeats(orig)

    def test_value_counts_unique_timedeltaindex(self):
        if False:
            return 10
        orig = timedelta_range('1 days 09:00:00', freq='h', periods=10)
        self._check_value_counts_with_repeats(orig)

    def test_value_counts_unique_periodindex(self):
        if False:
            i = 10
            return i + 15
        orig = period_range('2011-01-01 09:00', freq='h', periods=10)
        self._check_value_counts_with_repeats(orig)

    def _check_value_counts_with_repeats(self, orig):
        if False:
            return 10
        idx = type(orig)(np.repeat(orig._values, range(1, len(orig) + 1)), dtype=orig.dtype)
        exp_idx = orig[::-1]
        if not isinstance(exp_idx, PeriodIndex):
            exp_idx = exp_idx._with_freq(None)
        expected = Series(range(10, 0, -1), index=exp_idx, dtype='int64', name='count')
        for obj in [idx, Series(idx)]:
            tm.assert_series_equal(obj.value_counts(), expected)
        tm.assert_index_equal(idx.unique(), orig)

    def test_value_counts_unique_datetimeindex2(self, tz_naive_fixture):
        if False:
            i = 10
            return i + 15
        tz = tz_naive_fixture
        idx = DatetimeIndex(['2013-01-01 09:00', '2013-01-01 09:00', '2013-01-01 09:00', '2013-01-01 08:00', '2013-01-01 08:00', NaT], tz=tz)
        self._check_value_counts_dropna(idx)

    def test_value_counts_unique_timedeltaindex2(self):
        if False:
            i = 10
            return i + 15
        idx = TimedeltaIndex(['1 days 09:00:00', '1 days 09:00:00', '1 days 09:00:00', '1 days 08:00:00', '1 days 08:00:00', NaT])
        self._check_value_counts_dropna(idx)

    def test_value_counts_unique_periodindex2(self):
        if False:
            return 10
        idx = PeriodIndex(['2013-01-01 09:00', '2013-01-01 09:00', '2013-01-01 09:00', '2013-01-01 08:00', '2013-01-01 08:00', NaT], freq='h')
        self._check_value_counts_dropna(idx)

    def _check_value_counts_dropna(self, idx):
        if False:
            i = 10
            return i + 15
        exp_idx = idx[[2, 3]]
        expected = Series([3, 2], index=exp_idx, name='count')
        for obj in [idx, Series(idx)]:
            tm.assert_series_equal(obj.value_counts(), expected)
        exp_idx = idx[[2, 3, -1]]
        expected = Series([3, 2, 1], index=exp_idx, name='count')
        for obj in [idx, Series(idx)]:
            tm.assert_series_equal(obj.value_counts(dropna=False), expected)
        tm.assert_index_equal(idx.unique(), exp_idx)