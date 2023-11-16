"""
test date_range, bdate_range construction from the convenience range functions
"""
from datetime import datetime, time, timedelta
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import BDay, CDay, DateOffset, MonthEnd, prefix_mapping
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Series, Timedelta, Timestamp, bdate_range, date_range, offsets
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import FixedOffset, fixed_off_no_name
from pandas.tseries.holiday import USFederalHolidayCalendar
(START, END) = (datetime(2009, 1, 1), datetime(2010, 1, 1))

def _get_expected_range(begin_to_match, end_to_match, both_range, inclusive_endpoints):
    if False:
        return 10
    'Helper to get expected range from a both inclusive range'
    left_match = begin_to_match == both_range[0]
    right_match = end_to_match == both_range[-1]
    if inclusive_endpoints == 'left' and right_match:
        expected_range = both_range[:-1]
    elif inclusive_endpoints == 'right' and left_match:
        expected_range = both_range[1:]
    elif inclusive_endpoints == 'neither' and left_match and right_match:
        expected_range = both_range[1:-1]
    elif inclusive_endpoints == 'neither' and right_match:
        expected_range = both_range[:-1]
    elif inclusive_endpoints == 'neither' and left_match:
        expected_range = both_range[1:]
    elif inclusive_endpoints == 'both':
        expected_range = both_range[:]
    else:
        expected_range = both_range[:]
    return expected_range

class TestTimestampEquivDateRange:

    def test_date_range_timestamp_equiv(self):
        if False:
            print('Hello World!')
        rng = date_range('20090415', '20090519', tz='US/Eastern')
        stamp = rng[0]
        ts = Timestamp('20090415', tz='US/Eastern')
        assert ts == stamp

    def test_date_range_timestamp_equiv_dateutil(self):
        if False:
            while True:
                i = 10
        rng = date_range('20090415', '20090519', tz='dateutil/US/Eastern')
        stamp = rng[0]
        ts = Timestamp('20090415', tz='dateutil/US/Eastern')
        assert ts == stamp

    def test_date_range_timestamp_equiv_explicit_pytz(self):
        if False:
            return 10
        rng = date_range('20090415', '20090519', tz=pytz.timezone('US/Eastern'))
        stamp = rng[0]
        ts = Timestamp('20090415', tz=pytz.timezone('US/Eastern'))
        assert ts == stamp

    @td.skip_if_windows
    def test_date_range_timestamp_equiv_explicit_dateutil(self):
        if False:
            print('Hello World!')
        from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
        rng = date_range('20090415', '20090519', tz=gettz('US/Eastern'))
        stamp = rng[0]
        ts = Timestamp('20090415', tz=gettz('US/Eastern'))
        assert ts == stamp

    def test_date_range_timestamp_equiv_from_datetime_instance(self):
        if False:
            i = 10
            return i + 15
        datetime_instance = datetime(2014, 3, 4)
        timestamp_instance = date_range(datetime_instance, periods=1, freq='D')[0]
        ts = Timestamp(datetime_instance)
        assert ts == timestamp_instance

    def test_date_range_timestamp_equiv_preserve_frequency(self):
        if False:
            print('Hello World!')
        timestamp_instance = date_range('2014-03-05', periods=1, freq='D')[0]
        ts = Timestamp('2014-03-05')
        assert timestamp_instance == ts

class TestDateRanges:

    def test_date_range_name(self):
        if False:
            while True:
                i = 10
        idx = date_range(start='2000-01-01', periods=1, freq='YE', name='TEST')
        assert idx.name == 'TEST'

    def test_date_range_invalid_periods(self):
        if False:
            i = 10
            return i + 15
        msg = 'periods must be a number, got foo'
        with pytest.raises(TypeError, match=msg):
            date_range(start='1/1/2000', periods='foo', freq='D')

    def test_date_range_float_periods(self):
        if False:
            i = 10
            return i + 15
        rng = date_range('1/1/2000', periods=10.5)
        exp = date_range('1/1/2000', periods=10)
        tm.assert_index_equal(rng, exp)

    def test_date_range_frequency_M_deprecated(self):
        if False:
            i = 10
            return i + 15
        depr_msg = "'M' will be deprecated, please use 'ME' instead."
        expected = date_range('1/1/2000', periods=4, freq='2ME')
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            result = date_range('1/1/2000', periods=4, freq='2M')
        tm.assert_index_equal(result, expected)

    def test_date_range_tuple_freq_raises(self):
        if False:
            i = 10
            return i + 15
        edate = datetime(2000, 1, 1)
        with pytest.raises(TypeError, match='pass as a string instead'):
            date_range(end=edate, freq=('D', 5), periods=20)

    @pytest.mark.parametrize('freq', ['ns', 'us', 'ms', 'min', 's', 'h', 'D'])
    def test_date_range_edges(self, freq):
        if False:
            for i in range(10):
                print('nop')
        td = Timedelta(f'1{freq}')
        ts = Timestamp('1970-01-01')
        idx = date_range(start=ts + td, end=ts + 4 * td, freq=freq)
        exp = DatetimeIndex([ts + n * td for n in range(1, 5)], freq=freq)
        tm.assert_index_equal(idx, exp)
        idx = date_range(start=ts + 4 * td, end=ts + td, freq=freq)
        exp = DatetimeIndex([], freq=freq)
        tm.assert_index_equal(idx, exp)
        idx = date_range(start=ts + td, end=ts + td, freq=freq)
        exp = DatetimeIndex([ts + td], freq=freq)
        tm.assert_index_equal(idx, exp)

    def test_date_range_near_implementation_bound(self):
        if False:
            print('Hello World!')
        freq = Timedelta(1)
        with pytest.raises(OutOfBoundsDatetime, match='Cannot generate range with'):
            date_range(end=Timestamp.min, periods=2, freq=freq)

    def test_date_range_nat(self):
        if False:
            print('Hello World!')
        msg = 'Neither `start` nor `end` can be NaT'
        with pytest.raises(ValueError, match=msg):
            date_range(start='2016-01-01', end=pd.NaT, freq='D')
        with pytest.raises(ValueError, match=msg):
            date_range(start=pd.NaT, end='2016-01-01', freq='D')

    def test_date_range_multiplication_overflow(self):
        if False:
            print('Hello World!')
        with tm.assert_produces_warning(None):
            dti = date_range(start='1677-09-22', periods=213503, freq='D')
        assert dti[0] == Timestamp('1677-09-22')
        assert len(dti) == 213503
        msg = 'Cannot generate range with'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range('1969-05-04', periods=200000000, freq='30000D')

    def test_date_range_unsigned_overflow_handling(self):
        if False:
            return 10
        dti = date_range(start='1677-09-22', end='2262-04-11', freq='D')
        dti2 = date_range(start=dti[0], periods=len(dti), freq='D')
        assert dti2.equals(dti)
        dti3 = date_range(end=dti[-1], periods=len(dti), freq='D')
        assert dti3.equals(dti)

    def test_date_range_int64_overflow_non_recoverable(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Cannot generate range with'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range(start='1970-02-01', periods=106752 * 24, freq='h')
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range(end='1969-11-14', periods=106752 * 24, freq='h')

    @pytest.mark.slow
    @pytest.mark.parametrize('s_ts, e_ts', [('2262-02-23', '1969-11-14'), ('1970-02-01', '1677-10-22')])
    def test_date_range_int64_overflow_stride_endpoint_different_signs(self, s_ts, e_ts):
        if False:
            while True:
                i = 10
        start = Timestamp(s_ts)
        end = Timestamp(e_ts)
        expected = date_range(start=start, end=end, freq='-1h')
        assert expected[0] == start
        assert expected[-1] == end
        dti = date_range(end=end, periods=len(expected), freq='-1h')
        tm.assert_index_equal(dti, expected)

    def test_date_range_out_of_bounds(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Cannot generate range'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range('2016-01-01', periods=100000, freq='D')
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range(end='1763-10-12', periods=100000, freq='D')

    def test_date_range_gen_error(self):
        if False:
            print('Hello World!')
        rng = date_range('1/1/2000 00:00', '1/1/2000 00:18', freq='5min')
        assert len(rng) == 4

    def test_date_range_normalize(self):
        if False:
            return 10
        snap = datetime.today()
        n = 50
        rng = date_range(snap, periods=n, normalize=False, freq='2D')
        offset = timedelta(2)
        expected = DatetimeIndex([snap + i * offset for i in range(n)], dtype='M8[ns]', freq=offset)
        tm.assert_index_equal(rng, expected)
        rng = date_range('1/1/2000 08:15', periods=n, normalize=False, freq='B')
        the_time = time(8, 15)
        for val in rng:
            assert val.time() == the_time

    def test_date_range_ambiguous_arguments(self):
        if False:
            print('Hello World!')
        start = datetime(2011, 1, 1, 5, 3, 40)
        end = datetime(2011, 1, 1, 8, 9, 40)
        msg = 'Of the four parameters: start, end, periods, and freq, exactly three must be specified'
        with pytest.raises(ValueError, match=msg):
            date_range(start, end, periods=10, freq='s')

    def test_date_range_convenience_periods(self, unit):
        if False:
            return 10
        result = date_range('2018-04-24', '2018-04-27', periods=3, unit=unit)
        expected = DatetimeIndex(['2018-04-24 00:00:00', '2018-04-25 12:00:00', '2018-04-27 00:00:00'], dtype=f'M8[{unit}]', freq=None)
        tm.assert_index_equal(result, expected)
        result = date_range('2018-04-01 01:00:00', '2018-04-01 04:00:00', tz='Australia/Sydney', periods=3, unit=unit)
        expected = DatetimeIndex([Timestamp('2018-04-01 01:00:00+1100', tz='Australia/Sydney'), Timestamp('2018-04-01 02:00:00+1000', tz='Australia/Sydney'), Timestamp('2018-04-01 04:00:00+1000', tz='Australia/Sydney')]).as_unit(unit)
        tm.assert_index_equal(result, expected)

    def test_date_range_index_comparison(self):
        if False:
            print('Hello World!')
        rng = date_range('2011-01-01', periods=3, tz='US/Eastern')
        df = Series(rng).to_frame()
        arr = np.array([rng.to_list()]).T
        arr2 = np.array([rng]).T
        with pytest.raises(ValueError, match='Unable to coerce to Series'):
            rng == df
        with pytest.raises(ValueError, match='Unable to coerce to Series'):
            df == rng
        expected = DataFrame([True, True, True])
        results = df == arr2
        tm.assert_frame_equal(results, expected)
        expected = Series([True, True, True], name=0)
        results = df[0] == arr2[:, 0]
        tm.assert_series_equal(results, expected)
        expected = np.array([[True, False, False], [False, True, False], [False, False, True]])
        results = rng == arr
        tm.assert_numpy_array_equal(results, expected)

    @pytest.mark.parametrize('start,end,result_tz', [['20180101', '20180103', 'US/Eastern'], [datetime(2018, 1, 1), datetime(2018, 1, 3), 'US/Eastern'], [Timestamp('20180101'), Timestamp('20180103'), 'US/Eastern'], [Timestamp('20180101', tz='US/Eastern'), Timestamp('20180103', tz='US/Eastern'), 'US/Eastern'], [Timestamp('20180101', tz='US/Eastern'), Timestamp('20180103', tz='US/Eastern'), None]])
    def test_date_range_linspacing_tz(self, start, end, result_tz):
        if False:
            return 10
        result = date_range(start, end, periods=3, tz=result_tz)
        expected = date_range('20180101', periods=3, freq='D', tz='US/Eastern')
        tm.assert_index_equal(result, expected)

    def test_date_range_timedelta(self):
        if False:
            print('Hello World!')
        start = '2020-01-01'
        end = '2020-01-11'
        rng1 = date_range(start, end, freq='3D')
        rng2 = date_range(start, end, freq=timedelta(days=3))
        tm.assert_index_equal(rng1, rng2)

    def test_range_misspecified(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Of the four parameters: start, end, periods, and freq, exactly three must be specified'
        with pytest.raises(ValueError, match=msg):
            date_range(start='1/1/2000')
        with pytest.raises(ValueError, match=msg):
            date_range(end='1/1/2000')
        with pytest.raises(ValueError, match=msg):
            date_range(periods=10)
        with pytest.raises(ValueError, match=msg):
            date_range(start='1/1/2000', freq='h')
        with pytest.raises(ValueError, match=msg):
            date_range(end='1/1/2000', freq='h')
        with pytest.raises(ValueError, match=msg):
            date_range(periods=10, freq='h')
        with pytest.raises(ValueError, match=msg):
            date_range()

    def test_compat_replace(self):
        if False:
            return 10
        result = date_range(Timestamp('1960-04-01 00:00:00'), periods=76, freq='QS-JAN')
        assert len(result) == 76

    def test_catch_infinite_loop(self):
        if False:
            for i in range(10):
                print('nop')
        offset = offsets.DateOffset(minute=5)
        msg = 'Offset <DateOffset: minute=5> did not increment date'
        with pytest.raises(ValueError, match=msg):
            date_range(datetime(2011, 11, 11), datetime(2011, 11, 12), freq=offset)

    def test_construct_over_dst(self, unit):
        if False:
            while True:
                i = 10
        pre_dst = Timestamp('2010-11-07 01:00:00').tz_localize('US/Pacific', ambiguous=True)
        pst_dst = Timestamp('2010-11-07 01:00:00').tz_localize('US/Pacific', ambiguous=False)
        expect_data = [Timestamp('2010-11-07 00:00:00', tz='US/Pacific'), pre_dst, pst_dst]
        expected = DatetimeIndex(expect_data, freq='h').as_unit(unit)
        result = date_range(start='2010-11-7', periods=3, freq='h', tz='US/Pacific', unit=unit)
        tm.assert_index_equal(result, expected)

    def test_construct_with_different_start_end_string_format(self, unit):
        if False:
            for i in range(10):
                print('nop')
        result = date_range('2013-01-01 00:00:00+09:00', '2013/01/01 02:00:00+09:00', freq='h', unit=unit)
        expected = DatetimeIndex([Timestamp('2013-01-01 00:00:00+09:00'), Timestamp('2013-01-01 01:00:00+09:00'), Timestamp('2013-01-01 02:00:00+09:00')], freq='h').as_unit(unit)
        tm.assert_index_equal(result, expected)

    def test_error_with_zero_monthends(self):
        if False:
            print('Hello World!')
        msg = 'Offset <0 \\* MonthEnds> did not increment date'
        with pytest.raises(ValueError, match=msg):
            date_range('1/1/2000', '1/1/2001', freq=MonthEnd(0))

    def test_range_bug(self, unit):
        if False:
            for i in range(10):
                print('nop')
        offset = DateOffset(months=3)
        result = date_range('2011-1-1', '2012-1-31', freq=offset, unit=unit)
        start = datetime(2011, 1, 1)
        expected = DatetimeIndex([start + i * offset for i in range(5)], dtype=f'M8[{unit}]', freq=offset)
        tm.assert_index_equal(result, expected)

    def test_range_tz_pytz(self):
        if False:
            return 10
        tz = timezone('US/Eastern')
        start = tz.localize(datetime(2011, 1, 1))
        end = tz.localize(datetime(2011, 1, 3))
        dr = date_range(start=start, periods=3)
        assert dr.tz.zone == tz.zone
        assert dr[0] == start
        assert dr[2] == end
        dr = date_range(end=end, periods=3)
        assert dr.tz.zone == tz.zone
        assert dr[0] == start
        assert dr[2] == end
        dr = date_range(start=start, end=end)
        assert dr.tz.zone == tz.zone
        assert dr[0] == start
        assert dr[2] == end

    @pytest.mark.parametrize('start, end', [[Timestamp(datetime(2014, 3, 6), tz='US/Eastern'), Timestamp(datetime(2014, 3, 12), tz='US/Eastern')], [Timestamp(datetime(2013, 11, 1), tz='US/Eastern'), Timestamp(datetime(2013, 11, 6), tz='US/Eastern')]])
    def test_range_tz_dst_straddle_pytz(self, start, end):
        if False:
            print('Hello World!')
        dr = date_range(start, end, freq='D')
        assert dr[0] == start
        assert dr[-1] == end
        assert np.all(dr.hour == 0)
        dr = date_range(start, end, freq='D', tz='US/Eastern')
        assert dr[0] == start
        assert dr[-1] == end
        assert np.all(dr.hour == 0)
        dr = date_range(start.replace(tzinfo=None), end.replace(tzinfo=None), freq='D', tz='US/Eastern')
        assert dr[0] == start
        assert dr[-1] == end
        assert np.all(dr.hour == 0)

    def test_range_tz_dateutil(self):
        if False:
            while True:
                i = 10
        from pandas._libs.tslibs.timezones import maybe_get_tz
        tz = lambda x: maybe_get_tz('dateutil/' + x)
        start = datetime(2011, 1, 1, tzinfo=tz('US/Eastern'))
        end = datetime(2011, 1, 3, tzinfo=tz('US/Eastern'))
        dr = date_range(start=start, periods=3)
        assert dr.tz == tz('US/Eastern')
        assert dr[0] == start
        assert dr[2] == end
        dr = date_range(end=end, periods=3)
        assert dr.tz == tz('US/Eastern')
        assert dr[0] == start
        assert dr[2] == end
        dr = date_range(start=start, end=end)
        assert dr.tz == tz('US/Eastern')
        assert dr[0] == start
        assert dr[2] == end

    @pytest.mark.parametrize('freq', ['1D', '3D', '2ME', '7W', '3h', 'YE'])
    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    def test_range_closed(self, freq, tz, inclusive_endpoints_fixture):
        if False:
            print('Hello World!')
        begin = Timestamp('2011/1/1', tz=tz)
        end = Timestamp('2014/1/1', tz=tz)
        result_range = date_range(begin, end, inclusive=inclusive_endpoints_fixture, freq=freq)
        both_range = date_range(begin, end, inclusive='both', freq=freq)
        expected_range = _get_expected_range(begin, end, both_range, inclusive_endpoints_fixture)
        tm.assert_index_equal(expected_range, result_range)

    @pytest.mark.parametrize('freq', ['1D', '3D', '2ME', '7W', '3h', 'YE'])
    def test_range_with_tz_closed_with_tz_aware_start_end(self, freq, inclusive_endpoints_fixture):
        if False:
            for i in range(10):
                print('nop')
        begin = Timestamp('2011/1/1')
        end = Timestamp('2014/1/1')
        begintz = Timestamp('2011/1/1', tz='US/Eastern')
        endtz = Timestamp('2014/1/1', tz='US/Eastern')
        result_range = date_range(begin, end, inclusive=inclusive_endpoints_fixture, freq=freq, tz='US/Eastern')
        both_range = date_range(begin, end, inclusive='both', freq=freq, tz='US/Eastern')
        expected_range = _get_expected_range(begintz, endtz, both_range, inclusive_endpoints_fixture)
        tm.assert_index_equal(expected_range, result_range)

    def test_range_closed_boundary(self, inclusive_endpoints_fixture):
        if False:
            return 10
        right_boundary = date_range('2015-09-12', '2015-12-01', freq='QS-MAR', inclusive=inclusive_endpoints_fixture)
        left_boundary = date_range('2015-09-01', '2015-09-12', freq='QS-MAR', inclusive=inclusive_endpoints_fixture)
        both_boundary = date_range('2015-09-01', '2015-12-01', freq='QS-MAR', inclusive=inclusive_endpoints_fixture)
        neither_boundary = date_range('2015-09-11', '2015-09-12', freq='QS-MAR', inclusive=inclusive_endpoints_fixture)
        expected_right = both_boundary
        expected_left = both_boundary
        expected_both = both_boundary
        if inclusive_endpoints_fixture == 'right':
            expected_left = both_boundary[1:]
        elif inclusive_endpoints_fixture == 'left':
            expected_right = both_boundary[:-1]
        elif inclusive_endpoints_fixture == 'both':
            expected_right = both_boundary[1:]
            expected_left = both_boundary[:-1]
        expected_neither = both_boundary[1:-1]
        tm.assert_index_equal(right_boundary, expected_right)
        tm.assert_index_equal(left_boundary, expected_left)
        tm.assert_index_equal(both_boundary, expected_both)
        tm.assert_index_equal(neither_boundary, expected_neither)

    def test_date_range_years_only(self, tz_naive_fixture):
        if False:
            for i in range(10):
                print('nop')
        tz = tz_naive_fixture
        rng1 = date_range('2014', '2015', freq='ME', tz=tz)
        expected1 = date_range('2014-01-31', '2014-12-31', freq='ME', tz=tz)
        tm.assert_index_equal(rng1, expected1)
        rng2 = date_range('2014', '2015', freq='MS', tz=tz)
        expected2 = date_range('2014-01-01', '2015-01-01', freq='MS', tz=tz)
        tm.assert_index_equal(rng2, expected2)
        rng3 = date_range('2014', '2020', freq='YE', tz=tz)
        expected3 = date_range('2014-12-31', '2019-12-31', freq='YE', tz=tz)
        tm.assert_index_equal(rng3, expected3)
        rng4 = date_range('2014', '2020', freq='YS', tz=tz)
        expected4 = date_range('2014-01-01', '2020-01-01', freq='YS', tz=tz)
        tm.assert_index_equal(rng4, expected4)

    def test_freq_divides_end_in_nanos(self):
        if False:
            print('Hello World!')
        result_1 = date_range('2005-01-12 10:00', '2005-01-12 16:00', freq='345min')
        result_2 = date_range('2005-01-13 10:00', '2005-01-13 16:00', freq='345min')
        expected_1 = DatetimeIndex(['2005-01-12 10:00:00', '2005-01-12 15:45:00'], dtype='datetime64[ns]', freq='345min', tz=None)
        expected_2 = DatetimeIndex(['2005-01-13 10:00:00', '2005-01-13 15:45:00'], dtype='datetime64[ns]', freq='345min', tz=None)
        tm.assert_index_equal(result_1, expected_1)
        tm.assert_index_equal(result_2, expected_2)

    def test_cached_range_bug(self):
        if False:
            for i in range(10):
                print('nop')
        rng = date_range('2010-09-01 05:00:00', periods=50, freq=DateOffset(hours=6))
        assert len(rng) == 50
        assert rng[0] == datetime(2010, 9, 1, 5)

    def test_timezone_comparison_bug(self):
        if False:
            while True:
                i = 10
        start = Timestamp('20130220 10:00', tz='US/Eastern')
        result = date_range(start, periods=2, tz='US/Eastern')
        assert len(result) == 2

    def test_timezone_comparison_assert(self):
        if False:
            print('Hello World!')
        start = Timestamp('20130220 10:00', tz='US/Eastern')
        msg = 'Inferred time zone not equal to passed time zone'
        with pytest.raises(AssertionError, match=msg):
            date_range(start, periods=2, tz='Europe/Berlin')

    def test_negative_non_tick_frequency_descending_dates(self, tz_aware_fixture):
        if False:
            for i in range(10):
                print('nop')
        tz = tz_aware_fixture
        result = date_range(start='2011-06-01', end='2011-01-01', freq='-1MS', tz=tz)
        expected = date_range(end='2011-06-01', start='2011-01-01', freq='1MS', tz=tz)[::-1]
        tm.assert_index_equal(result, expected)

    def test_range_where_start_equal_end(self, inclusive_endpoints_fixture):
        if False:
            while True:
                i = 10
        start = '2021-09-02'
        end = '2021-09-02'
        result = date_range(start=start, end=end, freq='D', inclusive=inclusive_endpoints_fixture)
        both_range = date_range(start=start, end=end, freq='D', inclusive='both')
        if inclusive_endpoints_fixture == 'neither':
            expected = both_range[1:-1]
        elif inclusive_endpoints_fixture in ('left', 'right', 'both'):
            expected = both_range[:]
        tm.assert_index_equal(result, expected)

    def test_freq_dateoffset_with_relateivedelta_nanos(self):
        if False:
            i = 10
            return i + 15
        freq = DateOffset(hours=10, days=57, nanoseconds=3)
        result = date_range(end='1970-01-01 00:00:00', periods=10, freq=freq, name='a')
        expected = DatetimeIndex(['1968-08-02T05:59:59.999999973', '1968-09-28T15:59:59.999999976', '1968-11-25T01:59:59.999999979', '1969-01-21T11:59:59.999999982', '1969-03-19T21:59:59.999999985', '1969-05-16T07:59:59.999999988', '1969-07-12T17:59:59.999999991', '1969-09-08T03:59:59.999999994', '1969-11-04T13:59:59.999999997', '1970-01-01T00:00:00.000000000'], name='a')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('freq,freq_depr', [('h', 'H'), ('2min', '2T'), ('1s', '1S'), ('2ms', '2L'), ('1us', '1U'), ('2ns', '2N')])
    def test_frequencies_H_T_S_L_U_N_deprecated(self, freq, freq_depr):
        if False:
            print('Hello World!')
        freq_msg = re.split('[0-9]*', freq_depr, maxsplit=1)[1]
        msg = f"'{freq_msg}' is deprecated and will be removed in a future version."
        expected = date_range('1/1/2000', periods=2, freq=freq)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = date_range('1/1/2000', periods=2, freq=freq_depr)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('freq,freq_depr', [('200YE', '200A'), ('YE', 'Y'), ('2YE-MAY', '2A-MAY'), ('YE-MAY', 'Y-MAY')])
    def test_frequencies_A_deprecated_Y_renamed(self, freq, freq_depr):
        if False:
            i = 10
            return i + 15
        freq_msg = re.split('[0-9]*', freq, maxsplit=1)[1]
        freq_depr_msg = re.split('[0-9]*', freq_depr, maxsplit=1)[1]
        msg = f"'{freq_depr_msg}' will be deprecated, please use '{freq_msg}' instead."
        expected = date_range('1/1/2000', periods=2, freq=freq)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = date_range('1/1/2000', periods=2, freq=freq_depr)
        tm.assert_index_equal(result, expected)

    def test_date_range_bday(self):
        if False:
            for i in range(10):
                print('nop')
        sdate = datetime(1999, 12, 25)
        idx = date_range(start=sdate, freq='1B', periods=20)
        assert len(idx) == 20
        assert idx[0] == sdate + 0 * offsets.BDay()
        assert idx.freq == 'B'

class TestDateRangeTZ:
    """Tests for date_range with timezones"""

    def test_hongkong_tz_convert(self):
        if False:
            print('Hello World!')
        dr = date_range('2012-01-01', '2012-01-10', freq='D', tz='Hongkong')
        dr.hour

    @pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_date_range_span_dst_transition(self, tzstr):
        if False:
            return 10
        dr = date_range('03/06/2012 00:00', periods=200, freq='W-FRI', tz='US/Eastern')
        assert (dr.hour == 0).all()
        dr = date_range('2012-11-02', periods=10, tz=tzstr)
        result = dr.hour
        expected = pd.Index([0] * 10, dtype='int32')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_date_range_timezone_str_argument(self, tzstr):
        if False:
            return 10
        tz = timezones.maybe_get_tz(tzstr)
        result = date_range('1/1/2000', periods=10, tz=tzstr)
        expected = date_range('1/1/2000', periods=10, tz=tz)
        tm.assert_index_equal(result, expected)

    def test_date_range_with_fixed_tz(self):
        if False:
            while True:
                i = 10
        off = FixedOffset(420, '+07:00')
        start = datetime(2012, 3, 11, 5, 0, 0, tzinfo=off)
        end = datetime(2012, 6, 11, 5, 0, 0, tzinfo=off)
        rng = date_range(start=start, end=end)
        assert off == rng.tz
        rng2 = date_range(start, periods=len(rng), tz=off)
        tm.assert_index_equal(rng, rng2)
        rng3 = date_range('3/11/2012 05:00:00+07:00', '6/11/2012 05:00:00+07:00')
        assert (rng.values == rng3.values).all()

    def test_date_range_with_fixedoffset_noname(self):
        if False:
            return 10
        off = fixed_off_no_name
        start = datetime(2012, 3, 11, 5, 0, 0, tzinfo=off)
        end = datetime(2012, 6, 11, 5, 0, 0, tzinfo=off)
        rng = date_range(start=start, end=end)
        assert off == rng.tz
        idx = pd.Index([start, end])
        assert off == idx.tz

    @pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_date_range_with_tz(self, tzstr):
        if False:
            while True:
                i = 10
        stamp = Timestamp('3/11/2012 05:00', tz=tzstr)
        assert stamp.hour == 5
        rng = date_range('3/11/2012 04:00', periods=10, freq='h', tz=tzstr)
        assert stamp == rng[1]

    @pytest.mark.parametrize('tz', ['Europe/London', 'dateutil/Europe/London'])
    def test_date_range_ambiguous_endpoint(self, tz):
        if False:
            i = 10
            return i + 15
        with pytest.raises(pytz.AmbiguousTimeError, match='Cannot infer dst time'):
            date_range('2013-10-26 23:00', '2013-10-27 01:00', tz='Europe/London', freq='h')
        times = date_range('2013-10-26 23:00', '2013-10-27 01:00', freq='h', tz=tz, ambiguous='infer')
        assert times[0] == Timestamp('2013-10-26 23:00', tz=tz)
        assert times[-1] == Timestamp('2013-10-27 01:00:00+0000', tz=tz)

    @pytest.mark.parametrize('tz, option, expected', [['US/Pacific', 'shift_forward', '2019-03-10 03:00'], ['dateutil/US/Pacific', 'shift_forward', '2019-03-10 03:00'], ['US/Pacific', 'shift_backward', '2019-03-10 01:00'], ['dateutil/US/Pacific', 'shift_backward', '2019-03-10 01:00'], ['US/Pacific', timedelta(hours=1), '2019-03-10 03:00']])
    def test_date_range_nonexistent_endpoint(self, tz, option, expected):
        if False:
            print('Hello World!')
        with pytest.raises(pytz.NonExistentTimeError, match='2019-03-10 02:00:00'):
            date_range('2019-03-10 00:00', '2019-03-10 02:00', tz='US/Pacific', freq='h')
        times = date_range('2019-03-10 00:00', '2019-03-10 02:00', freq='h', tz=tz, nonexistent=option)
        assert times[-1] == Timestamp(expected, tz=tz)

class TestGenRangeGeneration:

    @pytest.mark.parametrize('freqstr,offset', [('B', BDay()), ('C', CDay())])
    def test_generate(self, freqstr, offset):
        if False:
            i = 10
            return i + 15
        rng1 = list(generate_range(START, END, periods=None, offset=offset, unit='ns'))
        rng2 = list(generate_range(START, END, periods=None, offset=freqstr, unit='ns'))
        assert rng1 == rng2

    def test_1(self):
        if False:
            for i in range(10):
                print('nop')
        rng = list(generate_range(start=datetime(2009, 3, 25), end=None, periods=2, offset=BDay(), unit='ns'))
        expected = [datetime(2009, 3, 25), datetime(2009, 3, 26)]
        assert rng == expected

    def test_2(self):
        if False:
            while True:
                i = 10
        rng = list(generate_range(start=datetime(2008, 1, 1), end=datetime(2008, 1, 3), periods=None, offset=BDay(), unit='ns'))
        expected = [datetime(2008, 1, 1), datetime(2008, 1, 2), datetime(2008, 1, 3)]
        assert rng == expected

    def test_3(self):
        if False:
            while True:
                i = 10
        rng = list(generate_range(start=datetime(2008, 1, 5), end=datetime(2008, 1, 6), periods=None, offset=BDay(), unit='ns'))
        expected = []
        assert rng == expected

    def test_precision_finer_than_offset(self):
        if False:
            return 10
        result1 = date_range(start='2015-04-15 00:00:03', end='2016-04-22 00:00:00', freq='QE')
        result2 = date_range(start='2015-04-15 00:00:03', end='2015-06-22 00:00:04', freq='W')
        expected1_list = ['2015-06-30 00:00:03', '2015-09-30 00:00:03', '2015-12-31 00:00:03', '2016-03-31 00:00:03']
        expected2_list = ['2015-04-19 00:00:03', '2015-04-26 00:00:03', '2015-05-03 00:00:03', '2015-05-10 00:00:03', '2015-05-17 00:00:03', '2015-05-24 00:00:03', '2015-05-31 00:00:03', '2015-06-07 00:00:03', '2015-06-14 00:00:03', '2015-06-21 00:00:03']
        expected1 = DatetimeIndex(expected1_list, dtype='datetime64[ns]', freq='QE-DEC', tz=None)
        expected2 = DatetimeIndex(expected2_list, dtype='datetime64[ns]', freq='W-SUN', tz=None)
        tm.assert_index_equal(result1, expected1)
        tm.assert_index_equal(result2, expected2)
    (dt1, dt2) = ('2017-01-01', '2017-01-01')
    (tz1, tz2) = ('US/Eastern', 'Europe/London')

    @pytest.mark.parametrize('start,end', [(Timestamp(dt1, tz=tz1), Timestamp(dt2)), (Timestamp(dt1), Timestamp(dt2, tz=tz2)), (Timestamp(dt1, tz=tz1), Timestamp(dt2, tz=tz2)), (Timestamp(dt1, tz=tz2), Timestamp(dt2, tz=tz1))])
    def test_mismatching_tz_raises_err(self, start, end):
        if False:
            print('Hello World!')
        msg = 'Start and end cannot both be tz-aware with different timezones'
        with pytest.raises(TypeError, match=msg):
            date_range(start, end)
        with pytest.raises(TypeError, match=msg):
            date_range(start, end, freq=BDay())

class TestBusinessDateRange:

    def test_constructor(self):
        if False:
            print('Hello World!')
        bdate_range(START, END, freq=BDay())
        bdate_range(START, periods=20, freq=BDay())
        bdate_range(end=START, periods=20, freq=BDay())
        msg = 'periods must be a number, got B'
        with pytest.raises(TypeError, match=msg):
            date_range('2011-1-1', '2012-1-1', 'B')
        with pytest.raises(TypeError, match=msg):
            bdate_range('2011-1-1', '2012-1-1', 'B')
        msg = 'freq must be specified for bdate_range; use date_range instead'
        with pytest.raises(TypeError, match=msg):
            bdate_range(START, END, periods=10, freq=None)

    def test_misc(self):
        if False:
            i = 10
            return i + 15
        end = datetime(2009, 5, 13)
        dr = bdate_range(end=end, periods=20)
        firstDate = end - 19 * BDay()
        assert len(dr) == 20
        assert dr[0] == firstDate
        assert dr[-1] == end

    def test_date_parse_failure(self):
        if False:
            while True:
                i = 10
        badly_formed_date = '2007/100/1'
        msg = 'Unknown datetime string format, unable to parse: 2007/100/1'
        with pytest.raises(ValueError, match=msg):
            Timestamp(badly_formed_date)
        with pytest.raises(ValueError, match=msg):
            bdate_range(start=badly_formed_date, periods=10)
        with pytest.raises(ValueError, match=msg):
            bdate_range(end=badly_formed_date, periods=10)
        with pytest.raises(ValueError, match=msg):
            bdate_range(badly_formed_date, badly_formed_date)

    def test_daterange_bug_456(self):
        if False:
            while True:
                i = 10
        rng1 = bdate_range('12/5/2011', '12/5/2011')
        rng2 = bdate_range('12/2/2011', '12/5/2011')
        assert rng2._data.freq == BDay()
        result = rng1.union(rng2)
        assert isinstance(result, DatetimeIndex)

    @pytest.mark.parametrize('inclusive', ['left', 'right', 'neither', 'both'])
    def test_bdays_and_open_boundaries(self, inclusive):
        if False:
            i = 10
            return i + 15
        start = '2018-07-21'
        end = '2018-07-29'
        result = date_range(start, end, freq='B', inclusive=inclusive)
        bday_start = '2018-07-23'
        bday_end = '2018-07-27'
        expected = date_range(bday_start, bday_end, freq='D')
        tm.assert_index_equal(result, expected)

    def test_bday_near_overflow(self):
        if False:
            for i in range(10):
                print('nop')
        start = Timestamp.max.floor('D').to_pydatetime()
        rng = date_range(start, end=None, periods=1, freq='B')
        expected = DatetimeIndex([start], freq='B').as_unit('ns')
        tm.assert_index_equal(rng, expected)

    def test_bday_overflow_error(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Out of bounds nanosecond timestamp'
        start = Timestamp.max.floor('D').to_pydatetime()
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range(start, periods=2, freq='B')

class TestCustomDateRange:

    def test_constructor(self):
        if False:
            i = 10
            return i + 15
        bdate_range(START, END, freq=CDay())
        bdate_range(START, periods=20, freq=CDay())
        bdate_range(end=START, periods=20, freq=CDay())
        msg = 'periods must be a number, got C'
        with pytest.raises(TypeError, match=msg):
            date_range('2011-1-1', '2012-1-1', 'C')
        with pytest.raises(TypeError, match=msg):
            bdate_range('2011-1-1', '2012-1-1', 'C')

    def test_misc(self):
        if False:
            for i in range(10):
                print('nop')
        end = datetime(2009, 5, 13)
        dr = bdate_range(end=end, periods=20, freq='C')
        firstDate = end - 19 * CDay()
        assert len(dr) == 20
        assert dr[0] == firstDate
        assert dr[-1] == end

    def test_daterange_bug_456(self):
        if False:
            i = 10
            return i + 15
        rng1 = bdate_range('12/5/2011', '12/5/2011', freq='C')
        rng2 = bdate_range('12/2/2011', '12/5/2011', freq='C')
        assert rng2._data.freq == CDay()
        result = rng1.union(rng2)
        assert isinstance(result, DatetimeIndex)

    def test_cdaterange(self, unit):
        if False:
            return 10
        result = bdate_range('2013-05-01', periods=3, freq='C', unit=unit)
        expected = DatetimeIndex(['2013-05-01', '2013-05-02', '2013-05-03'], dtype=f'M8[{unit}]', freq='C')
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

    def test_cdaterange_weekmask(self, unit):
        if False:
            print('Hello World!')
        result = bdate_range('2013-05-01', periods=3, freq='C', weekmask='Sun Mon Tue Wed Thu', unit=unit)
        expected = DatetimeIndex(['2013-05-01', '2013-05-02', '2013-05-05'], dtype=f'M8[{unit}]', freq=result.freq)
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq
        msg = 'a custom frequency string is required when holidays or weekmask are passed, got frequency B'
        with pytest.raises(ValueError, match=msg):
            bdate_range('2013-05-01', periods=3, weekmask='Sun Mon Tue Wed Thu')

    def test_cdaterange_holidays(self, unit):
        if False:
            for i in range(10):
                print('nop')
        result = bdate_range('2013-05-01', periods=3, freq='C', holidays=['2013-05-01'], unit=unit)
        expected = DatetimeIndex(['2013-05-02', '2013-05-03', '2013-05-06'], dtype=f'M8[{unit}]', freq=result.freq)
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq
        msg = 'a custom frequency string is required when holidays or weekmask are passed, got frequency B'
        with pytest.raises(ValueError, match=msg):
            bdate_range('2013-05-01', periods=3, holidays=['2013-05-01'])

    def test_cdaterange_weekmask_and_holidays(self, unit):
        if False:
            while True:
                i = 10
        result = bdate_range('2013-05-01', periods=3, freq='C', weekmask='Sun Mon Tue Wed Thu', holidays=['2013-05-01'], unit=unit)
        expected = DatetimeIndex(['2013-05-02', '2013-05-05', '2013-05-06'], dtype=f'M8[{unit}]', freq=result.freq)
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

    def test_cdaterange_holidays_weekmask_requires_freqstr(self):
        if False:
            i = 10
            return i + 15
        msg = 'a custom frequency string is required when holidays or weekmask are passed, got frequency B'
        with pytest.raises(ValueError, match=msg):
            bdate_range('2013-05-01', periods=3, weekmask='Sun Mon Tue Wed Thu', holidays=['2013-05-01'])

    @pytest.mark.parametrize('freq', [freq for freq in prefix_mapping if freq.startswith('C')])
    def test_all_custom_freq(self, freq):
        if False:
            print('Hello World!')
        bdate_range(START, END, freq=freq, weekmask='Mon Wed Fri', holidays=['2009-03-14'])
        bad_freq = freq + 'FOO'
        msg = f'invalid custom frequency string: {bad_freq}'
        with pytest.raises(ValueError, match=msg):
            bdate_range(START, END, freq=bad_freq)

    @pytest.mark.parametrize('start_end', [('2018-01-01T00:00:01.000Z', '2018-01-03T00:00:01.000Z'), ('2018-01-01T00:00:00.010Z', '2018-01-03T00:00:00.010Z'), ('2001-01-01T00:00:00.010Z', '2001-01-03T00:00:00.010Z')])
    def test_range_with_millisecond_resolution(self, start_end):
        if False:
            return 10
        (start, end) = start_end
        result = date_range(start=start, end=end, periods=2, inclusive='left')
        expected = DatetimeIndex([start], dtype='M8[ns, UTC]')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('start,period,expected', [('2022-07-23 00:00:00+02:00', 1, ['2022-07-25 00:00:00+02:00']), ('2022-07-22 00:00:00+02:00', 1, ['2022-07-22 00:00:00+02:00']), ('2022-07-22 00:00:00+02:00', 2, ['2022-07-22 00:00:00+02:00', '2022-07-25 00:00:00+02:00'])])
    def test_range_with_timezone_and_custombusinessday(self, start, period, expected):
        if False:
            for i in range(10):
                print('nop')
        result = date_range(start=start, periods=period, freq='C')
        expected = DatetimeIndex(expected).as_unit('ns')
        tm.assert_index_equal(result, expected)

class TestDateRangeNonNano:

    def test_date_range_reso_validation(self):
        if False:
            for i in range(10):
                print('nop')
        msg = "'unit' must be one of 's', 'ms', 'us', 'ns'"
        with pytest.raises(ValueError, match=msg):
            date_range('2016-01-01', '2016-03-04', periods=3, unit='h')

    def test_date_range_freq_higher_than_reso(self):
        if False:
            i = 10
            return i + 15
        msg = 'Use a lower freq or a higher unit instead'
        with pytest.raises(ValueError, match=msg):
            date_range('2016-01-01', '2016-01-02', freq='ns', unit='ms')

    def test_date_range_freq_matches_reso(self):
        if False:
            i = 10
            return i + 15
        dti = date_range('2016-01-01', '2016-01-01 00:00:01', freq='ms', unit='ms')
        rng = np.arange(1451606400000, 1451606401001, dtype=np.int64)
        expected = DatetimeIndex(rng.view('M8[ms]'), freq='ms')
        tm.assert_index_equal(dti, expected)
        dti = date_range('2016-01-01', '2016-01-01 00:00:01', freq='us', unit='us')
        rng = np.arange(1451606400000000, 1451606401000001, dtype=np.int64)
        expected = DatetimeIndex(rng.view('M8[us]'), freq='us')
        tm.assert_index_equal(dti, expected)
        dti = date_range('2016-01-01', '2016-01-01 00:00:00.001', freq='ns', unit='ns')
        rng = np.arange(1451606400000000000, 1451606400001000001, dtype=np.int64)
        expected = DatetimeIndex(rng.view('M8[ns]'), freq='ns')
        tm.assert_index_equal(dti, expected)

    def test_date_range_freq_lower_than_endpoints(self):
        if False:
            for i in range(10):
                print('nop')
        start = Timestamp('2022-10-19 11:50:44.719781')
        end = Timestamp('2022-10-19 11:50:47.066458')
        with pytest.raises(ValueError, match='Cannot losslessly convert units'):
            date_range(start, end, periods=3, unit='s')
        dti = date_range(start, end, periods=2, unit='us')
        rng = np.array([start.as_unit('us')._value, end.as_unit('us')._value], dtype=np.int64)
        expected = DatetimeIndex(rng.view('M8[us]'))
        tm.assert_index_equal(dti, expected)

    def test_date_range_non_nano(self):
        if False:
            while True:
                i = 10
        start = np.datetime64('1066-10-14')
        end = np.datetime64('2305-07-13')
        dti = date_range(start, end, freq='D', unit='s')
        assert dti.freq == 'D'
        assert dti.dtype == 'M8[s]'
        exp = np.arange(start.astype('M8[s]').view('i8'), (end + 1).astype('M8[s]').view('i8'), 24 * 3600).view('M8[s]')
        tm.assert_numpy_array_equal(dti.to_numpy(), exp)

class TestDateRangeNonTickFreq:

    def test_date_range_custom_business_month_begin(self, unit):
        if False:
            return 10
        hcal = USFederalHolidayCalendar()
        freq = offsets.CBMonthBegin(calendar=hcal)
        dti = date_range(start='20120101', end='20130101', freq=freq, unit=unit)
        assert all((freq.is_on_offset(x) for x in dti))
        expected = DatetimeIndex(['2012-01-03', '2012-02-01', '2012-03-01', '2012-04-02', '2012-05-01', '2012-06-01', '2012-07-02', '2012-08-01', '2012-09-04', '2012-10-01', '2012-11-01', '2012-12-03'], dtype=f'M8[{unit}]', freq=freq)
        tm.assert_index_equal(dti, expected)

    def test_date_range_custom_business_month_end(self, unit):
        if False:
            while True:
                i = 10
        hcal = USFederalHolidayCalendar()
        freq = offsets.CBMonthEnd(calendar=hcal)
        dti = date_range(start='20120101', end='20130101', freq=freq, unit=unit)
        assert all((freq.is_on_offset(x) for x in dti))
        expected = DatetimeIndex(['2012-01-31', '2012-02-29', '2012-03-30', '2012-04-30', '2012-05-31', '2012-06-29', '2012-07-31', '2012-08-31', '2012-09-28', '2012-10-31', '2012-11-30', '2012-12-31'], dtype=f'M8[{unit}]', freq=freq)
        tm.assert_index_equal(dti, expected)

    def test_date_range_with_custom_holidays(self, unit):
        if False:
            while True:
                i = 10
        freq = offsets.CustomBusinessHour(start='15:00', holidays=['2020-11-26'])
        result = date_range(start='2020-11-25 15:00', periods=4, freq=freq, unit=unit)
        expected = DatetimeIndex(['2020-11-25 15:00:00', '2020-11-25 16:00:00', '2020-11-27 15:00:00', '2020-11-27 16:00:00'], dtype=f'M8[{unit}]', freq=freq)
        tm.assert_index_equal(result, expected)

    def test_date_range_businesshour(self, unit):
        if False:
            for i in range(10):
                print('nop')
        idx = DatetimeIndex(['2014-07-04 09:00', '2014-07-04 10:00', '2014-07-04 11:00', '2014-07-04 12:00', '2014-07-04 13:00', '2014-07-04 14:00', '2014-07-04 15:00', '2014-07-04 16:00'], dtype=f'M8[{unit}]', freq='bh')
        rng = date_range('2014-07-04 09:00', '2014-07-04 16:00', freq='bh', unit=unit)
        tm.assert_index_equal(idx, rng)
        idx = DatetimeIndex(['2014-07-04 16:00', '2014-07-07 09:00'], dtype=f'M8[{unit}]', freq='bh')
        rng = date_range('2014-07-04 16:00', '2014-07-07 09:00', freq='bh', unit=unit)
        tm.assert_index_equal(idx, rng)
        idx = DatetimeIndex(['2014-07-04 09:00', '2014-07-04 10:00', '2014-07-04 11:00', '2014-07-04 12:00', '2014-07-04 13:00', '2014-07-04 14:00', '2014-07-04 15:00', '2014-07-04 16:00', '2014-07-07 09:00', '2014-07-07 10:00', '2014-07-07 11:00', '2014-07-07 12:00', '2014-07-07 13:00', '2014-07-07 14:00', '2014-07-07 15:00', '2014-07-07 16:00', '2014-07-08 09:00', '2014-07-08 10:00', '2014-07-08 11:00', '2014-07-08 12:00', '2014-07-08 13:00', '2014-07-08 14:00', '2014-07-08 15:00', '2014-07-08 16:00'], dtype=f'M8[{unit}]', freq='bh')
        rng = date_range('2014-07-04 09:00', '2014-07-08 16:00', freq='bh', unit=unit)
        tm.assert_index_equal(idx, rng)

    def test_date_range_business_hour2(self, unit):
        if False:
            print('Hello World!')
        idx1 = date_range(start='2014-07-04 15:00', end='2014-07-08 10:00', freq='bh', unit=unit)
        idx2 = date_range(start='2014-07-04 15:00', periods=12, freq='bh', unit=unit)
        idx3 = date_range(end='2014-07-08 10:00', periods=12, freq='bh', unit=unit)
        expected = DatetimeIndex(['2014-07-04 15:00', '2014-07-04 16:00', '2014-07-07 09:00', '2014-07-07 10:00', '2014-07-07 11:00', '2014-07-07 12:00', '2014-07-07 13:00', '2014-07-07 14:00', '2014-07-07 15:00', '2014-07-07 16:00', '2014-07-08 09:00', '2014-07-08 10:00'], dtype=f'M8[{unit}]', freq='bh')
        tm.assert_index_equal(idx1, expected)
        tm.assert_index_equal(idx2, expected)
        tm.assert_index_equal(idx3, expected)
        idx4 = date_range(start='2014-07-04 15:45', end='2014-07-08 10:45', freq='bh', unit=unit)
        idx5 = date_range(start='2014-07-04 15:45', periods=12, freq='bh', unit=unit)
        idx6 = date_range(end='2014-07-08 10:45', periods=12, freq='bh', unit=unit)
        expected2 = expected + Timedelta(minutes=45).as_unit(unit)
        expected2.freq = 'bh'
        tm.assert_index_equal(idx4, expected2)
        tm.assert_index_equal(idx5, expected2)
        tm.assert_index_equal(idx6, expected2)

    def test_date_range_business_hour_short(self, unit):
        if False:
            while True:
                i = 10
        idx4 = date_range(start='2014-07-01 10:00', freq='bh', periods=1, unit=unit)
        expected4 = DatetimeIndex(['2014-07-01 10:00'], dtype=f'M8[{unit}]', freq='bh')
        tm.assert_index_equal(idx4, expected4)

    def test_date_range_year_start(self, unit):
        if False:
            while True:
                i = 10
        rng = date_range('1/1/2013', '7/1/2017', freq='YS', unit=unit)
        exp = DatetimeIndex(['2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01'], dtype=f'M8[{unit}]', freq='YS')
        tm.assert_index_equal(rng, exp)

    def test_date_range_year_end(self, unit):
        if False:
            while True:
                i = 10
        rng = date_range('1/1/2013', '7/1/2017', freq='YE', unit=unit)
        exp = DatetimeIndex(['2013-12-31', '2014-12-31', '2015-12-31', '2016-12-31'], dtype=f'M8[{unit}]', freq='YE')
        tm.assert_index_equal(rng, exp)

    def test_date_range_negative_freq_year_end(self, unit):
        if False:
            while True:
                i = 10
        rng = date_range('2011-12-31', freq='-2YE', periods=3, unit=unit)
        exp = DatetimeIndex(['2011-12-31', '2009-12-31', '2007-12-31'], dtype=f'M8[{unit}]', freq='-2YE')
        tm.assert_index_equal(rng, exp)
        assert rng.freq == '-2YE'

    def test_date_range_business_year_end_year(self, unit):
        if False:
            i = 10
            return i + 15
        rng = date_range('1/1/2013', '7/1/2017', freq='BY', unit=unit)
        exp = DatetimeIndex(['2013-12-31', '2014-12-31', '2015-12-31', '2016-12-30'], dtype=f'M8[{unit}]', freq='BY')
        tm.assert_index_equal(rng, exp)

    def test_date_range_bms(self, unit):
        if False:
            return 10
        result = date_range('1/1/2000', periods=10, freq='BMS', unit=unit)
        expected = DatetimeIndex(['2000-01-03', '2000-02-01', '2000-03-01', '2000-04-03', '2000-05-01', '2000-06-01', '2000-07-03', '2000-08-01', '2000-09-01', '2000-10-02'], dtype=f'M8[{unit}]', freq='BMS')
        tm.assert_index_equal(result, expected)

    def test_date_range_semi_month_begin(self, unit):
        if False:
            print('Hello World!')
        dates = [datetime(2007, 12, 15), datetime(2008, 1, 1), datetime(2008, 1, 15), datetime(2008, 2, 1), datetime(2008, 2, 15), datetime(2008, 3, 1), datetime(2008, 3, 15), datetime(2008, 4, 1), datetime(2008, 4, 15), datetime(2008, 5, 1), datetime(2008, 5, 15), datetime(2008, 6, 1), datetime(2008, 6, 15), datetime(2008, 7, 1), datetime(2008, 7, 15), datetime(2008, 8, 1), datetime(2008, 8, 15), datetime(2008, 9, 1), datetime(2008, 9, 15), datetime(2008, 10, 1), datetime(2008, 10, 15), datetime(2008, 11, 1), datetime(2008, 11, 15), datetime(2008, 12, 1), datetime(2008, 12, 15)]
        result = date_range(start=dates[0], end=dates[-1], freq='SMS', unit=unit)
        exp = DatetimeIndex(dates, dtype=f'M8[{unit}]', freq='SMS')
        tm.assert_index_equal(result, exp)

    def test_date_range_semi_month_end(self, unit):
        if False:
            i = 10
            return i + 15
        dates = [datetime(2007, 12, 31), datetime(2008, 1, 15), datetime(2008, 1, 31), datetime(2008, 2, 15), datetime(2008, 2, 29), datetime(2008, 3, 15), datetime(2008, 3, 31), datetime(2008, 4, 15), datetime(2008, 4, 30), datetime(2008, 5, 15), datetime(2008, 5, 31), datetime(2008, 6, 15), datetime(2008, 6, 30), datetime(2008, 7, 15), datetime(2008, 7, 31), datetime(2008, 8, 15), datetime(2008, 8, 31), datetime(2008, 9, 15), datetime(2008, 9, 30), datetime(2008, 10, 15), datetime(2008, 10, 31), datetime(2008, 11, 15), datetime(2008, 11, 30), datetime(2008, 12, 15), datetime(2008, 12, 31)]
        result = date_range(start=dates[0], end=dates[-1], freq='SM', unit=unit)
        exp = DatetimeIndex(dates, dtype=f'M8[{unit}]', freq='SM')
        tm.assert_index_equal(result, exp)

    def test_date_range_week_of_month(self, unit):
        if False:
            print('Hello World!')
        result = date_range(start='20110101', periods=1, freq='WOM-1MON', unit=unit)
        expected = DatetimeIndex(['2011-01-03'], dtype=f'M8[{unit}]', freq='WOM-1MON')
        tm.assert_index_equal(result, expected)
        result2 = date_range(start='20110101', periods=2, freq='WOM-1MON', unit=unit)
        expected2 = DatetimeIndex(['2011-01-03', '2011-02-07'], dtype=f'M8[{unit}]', freq='WOM-1MON')
        tm.assert_index_equal(result2, expected2)

    def test_date_range_week_of_month2(self, unit):
        if False:
            while True:
                i = 10
        result = date_range('2013-1-1', periods=4, freq='WOM-1SAT', unit=unit)
        expected = DatetimeIndex(['2013-01-05', '2013-02-02', '2013-03-02', '2013-04-06'], dtype=f'M8[{unit}]', freq='WOM-1SAT')
        tm.assert_index_equal(result, expected)

    def test_date_range_negative_freq_month_end(self, unit):
        if False:
            return 10
        rng = date_range('2011-01-31', freq='-2ME', periods=3, unit=unit)
        exp = DatetimeIndex(['2011-01-31', '2010-11-30', '2010-09-30'], dtype=f'M8[{unit}]', freq='-2ME')
        tm.assert_index_equal(rng, exp)
        assert rng.freq == '-2ME'

    def test_date_range_fy5253(self, unit):
        if False:
            return 10
        freq = offsets.FY5253(startingMonth=1, weekday=3, variation='nearest')
        dti = date_range(start='2013-01-01', periods=2, freq=freq, unit=unit)
        expected = DatetimeIndex(['2013-01-31', '2014-01-30'], dtype=f'M8[{unit}]', freq=freq)
        tm.assert_index_equal(dti, expected)

    @pytest.mark.parametrize('freqstr,offset', [('QS', offsets.QuarterBegin(startingMonth=1)), ('BQ', offsets.BQuarterEnd(startingMonth=12)), ('W-SUN', offsets.Week(weekday=6))])
    def test_date_range_freqstr_matches_offset(self, freqstr, offset):
        if False:
            for i in range(10):
                print('nop')
        sdate = datetime(1999, 12, 25)
        edate = datetime(2000, 1, 1)
        idx1 = date_range(start=sdate, end=edate, freq=freqstr)
        idx2 = date_range(start=sdate, end=edate, freq=offset)
        assert len(idx1) == len(idx2)
        assert idx1.freq == idx2.freq