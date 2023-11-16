from datetime import datetime, timedelta
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import DatetimeIndex, Timestamp, bdate_range, date_range, offsets, to_datetime
import pandas._testing as tm
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None
easts = [pytz.timezone('US/Eastern'), gettz('US/Eastern')]
if ZoneInfo is not None:
    try:
        tz = ZoneInfo('US/Eastern')
    except KeyError:
        pass
    else:
        easts.append(tz)

class TestTZLocalize:

    def test_tz_localize_invalidates_freq(self):
        if False:
            i = 10
            return i + 15
        dti = date_range('2014-03-08 23:00', '2014-03-09 09:00', freq='h')
        assert dti.freq == 'h'
        result = dti.tz_localize(None)
        assert result.freq == 'h'
        result = dti.tz_localize('UTC')
        assert result.freq == 'h'
        result = dti.tz_localize('US/Eastern', nonexistent='shift_forward')
        assert result.freq is None
        assert result.inferred_freq is None
        dti2 = dti[:1]
        result = dti2.tz_localize('US/Eastern')
        assert result.freq == 'h'

    def test_tz_localize_utc_copies(self, utc_fixture):
        if False:
            for i in range(10):
                print('nop')
        times = ['2015-03-08 01:00', '2015-03-08 02:00', '2015-03-08 03:00']
        index = DatetimeIndex(times)
        res = index.tz_localize(utc_fixture)
        assert not tm.shares_memory(res, index)
        res2 = index._data.tz_localize(utc_fixture)
        assert not tm.shares_memory(index._data, res2)

    def test_dti_tz_localize_nonexistent_raise_coerce(self):
        if False:
            print('Hello World!')
        times = ['2015-03-08 01:00', '2015-03-08 02:00', '2015-03-08 03:00']
        index = DatetimeIndex(times)
        tz = 'US/Eastern'
        with pytest.raises(pytz.NonExistentTimeError, match='|'.join(times)):
            index.tz_localize(tz=tz)
        with pytest.raises(pytz.NonExistentTimeError, match='|'.join(times)):
            index.tz_localize(tz=tz, nonexistent='raise')
        result = index.tz_localize(tz=tz, nonexistent='NaT')
        test_times = ['2015-03-08 01:00-05:00', 'NaT', '2015-03-08 03:00-04:00']
        dti = to_datetime(test_times, utc=True)
        expected = dti.tz_convert('US/Eastern')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('tz', easts)
    def test_dti_tz_localize_ambiguous_infer(self, tz):
        if False:
            return 10
        dr = date_range(datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour())
        with pytest.raises(pytz.AmbiguousTimeError, match='Cannot infer dst time'):
            dr.tz_localize(tz)

    @pytest.mark.parametrize('tz', easts)
    def test_dti_tz_localize_ambiguous_infer2(self, tz, unit):
        if False:
            while True:
                i = 10
        dr = date_range(datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour(), tz=tz, unit=unit)
        times = ['11/06/2011 00:00', '11/06/2011 01:00', '11/06/2011 01:00', '11/06/2011 02:00', '11/06/2011 03:00']
        di = DatetimeIndex(times).as_unit(unit)
        result = di.tz_localize(tz, ambiguous='infer')
        expected = dr._with_freq(None)
        tm.assert_index_equal(result, expected)
        result2 = DatetimeIndex(times, tz=tz, ambiguous='infer').as_unit(unit)
        tm.assert_index_equal(result2, expected)

    @pytest.mark.parametrize('tz', easts)
    def test_dti_tz_localize_ambiguous_infer3(self, tz):
        if False:
            return 10
        dr = date_range(datetime(2011, 6, 1, 0), periods=10, freq=offsets.Hour())
        localized = dr.tz_localize(tz)
        localized_infer = dr.tz_localize(tz, ambiguous='infer')
        tm.assert_index_equal(localized, localized_infer)

    @pytest.mark.parametrize('tz', easts)
    def test_dti_tz_localize_ambiguous_times(self, tz):
        if False:
            print('Hello World!')
        dr = date_range(datetime(2011, 3, 13, 1, 30), periods=3, freq=offsets.Hour())
        with pytest.raises(pytz.NonExistentTimeError, match='2011-03-13 02:30:00'):
            dr.tz_localize(tz)
        dr = date_range(datetime(2011, 3, 13, 3, 30), periods=3, freq=offsets.Hour(), tz=tz)
        dr = date_range(datetime(2011, 11, 6, 1, 30), periods=3, freq=offsets.Hour())
        with pytest.raises(pytz.AmbiguousTimeError, match='Cannot infer dst time'):
            dr.tz_localize(tz)
        dr = date_range(datetime(2011, 3, 13), periods=48, freq=offsets.Minute(30), tz=pytz.utc)

    @pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_dti_tz_localize_pass_dates_to_utc(self, tzstr):
        if False:
            for i in range(10):
                print('nop')
        strdates = ['1/1/2012', '3/1/2012', '4/1/2012']
        idx = DatetimeIndex(strdates)
        conv = idx.tz_localize(tzstr)
        fromdates = DatetimeIndex(strdates, tz=tzstr)
        assert conv.tz == fromdates.tz
        tm.assert_numpy_array_equal(conv.values, fromdates.values)

    @pytest.mark.parametrize('prefix', ['', 'dateutil/'])
    def test_dti_tz_localize(self, prefix):
        if False:
            i = 10
            return i + 15
        tzstr = prefix + 'US/Eastern'
        dti = date_range(start='1/1/2005', end='1/1/2005 0:00:30.256', freq='ms')
        dti2 = dti.tz_localize(tzstr)
        dti_utc = date_range(start='1/1/2005 05:00', end='1/1/2005 5:00:30.256', freq='ms', tz='utc')
        tm.assert_numpy_array_equal(dti2.values, dti_utc.values)
        dti3 = dti2.tz_convert(prefix + 'US/Pacific')
        tm.assert_numpy_array_equal(dti3.values, dti_utc.values)
        dti = date_range(start='11/6/2011 1:59', end='11/6/2011 2:00', freq='ms')
        with pytest.raises(pytz.AmbiguousTimeError, match='Cannot infer dst time'):
            dti.tz_localize(tzstr)
        dti = date_range(start='3/13/2011 1:59', end='3/13/2011 2:00', freq='ms')
        with pytest.raises(pytz.NonExistentTimeError, match='2011-03-13 02:00:00'):
            dti.tz_localize(tzstr)

    @pytest.mark.parametrize('tz', ['US/Eastern', 'dateutil/US/Eastern', pytz.timezone('US/Eastern'), gettz('US/Eastern')])
    def test_dti_tz_localize_utc_conversion(self, tz):
        if False:
            for i in range(10):
                print('nop')
        rng = date_range('3/10/2012', '3/11/2012', freq='30min')
        converted = rng.tz_localize(tz)
        expected_naive = rng + offsets.Hour(5)
        tm.assert_numpy_array_equal(converted.asi8, expected_naive.asi8)
        rng = date_range('3/11/2012', '3/12/2012', freq='30min')
        with pytest.raises(pytz.NonExistentTimeError, match='2012-03-11 02:00:00'):
            rng.tz_localize(tz)

    def test_dti_tz_localize_roundtrip(self, tz_aware_fixture):
        if False:
            for i in range(10):
                print('nop')
        idx = date_range(start='2014-06-01', end='2014-08-30', freq='15min')
        tz = tz_aware_fixture
        localized = idx.tz_localize(tz)
        with pytest.raises(TypeError, match='Already tz-aware, use tz_convert to convert'):
            localized.tz_localize(tz)
        reset = localized.tz_localize(None)
        assert reset.tzinfo is None
        expected = idx._with_freq(None)
        tm.assert_index_equal(reset, expected)

    def test_dti_tz_localize_naive(self):
        if False:
            return 10
        rng = date_range('1/1/2011', periods=100, freq='h')
        conv = rng.tz_localize('US/Pacific')
        exp = date_range('1/1/2011', periods=100, freq='h', tz='US/Pacific')
        tm.assert_index_equal(conv, exp._with_freq(None))

    def test_dti_tz_localize_tzlocal(self):
        if False:
            print('Hello World!')
        offset = dateutil.tz.tzlocal().utcoffset(datetime(2011, 1, 1))
        offset = int(offset.total_seconds() * 1000000000)
        dti = date_range(start='2001-01-01', end='2001-03-01')
        dti2 = dti.tz_localize(dateutil.tz.tzlocal())
        tm.assert_numpy_array_equal(dti2.asi8 + offset, dti.asi8)
        dti = date_range(start='2001-01-01', end='2001-03-01', tz=dateutil.tz.tzlocal())
        dti2 = dti.tz_localize(None)
        tm.assert_numpy_array_equal(dti2.asi8 - offset, dti.asi8)

    @pytest.mark.parametrize('tz', easts)
    def test_dti_tz_localize_ambiguous_nat(self, tz):
        if False:
            for i in range(10):
                print('nop')
        times = ['11/06/2011 00:00', '11/06/2011 01:00', '11/06/2011 01:00', '11/06/2011 02:00', '11/06/2011 03:00']
        di = DatetimeIndex(times)
        localized = di.tz_localize(tz, ambiguous='NaT')
        times = ['11/06/2011 00:00', np.nan, np.nan, '11/06/2011 02:00', '11/06/2011 03:00']
        di_test = DatetimeIndex(times, tz='US/Eastern')
        tm.assert_numpy_array_equal(di_test.values, localized.values)

    @pytest.mark.parametrize('tz', easts)
    def test_dti_tz_localize_ambiguous_flags(self, tz, unit):
        if False:
            return 10
        dr = date_range(datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour(), tz=tz, unit=unit)
        times = ['11/06/2011 00:00', '11/06/2011 01:00', '11/06/2011 01:00', '11/06/2011 02:00', '11/06/2011 03:00']
        di = DatetimeIndex(times).as_unit(unit)
        is_dst = [1, 1, 0, 0, 0]
        localized = di.tz_localize(tz, ambiguous=is_dst)
        expected = dr._with_freq(None)
        tm.assert_index_equal(expected, localized)
        result = DatetimeIndex(times, tz=tz, ambiguous=is_dst).as_unit(unit)
        tm.assert_index_equal(result, expected)
        localized = di.tz_localize(tz, ambiguous=np.array(is_dst))
        tm.assert_index_equal(dr, localized)
        localized = di.tz_localize(tz, ambiguous=np.array(is_dst).astype('bool'))
        tm.assert_index_equal(dr, localized)
        localized = DatetimeIndex(times, tz=tz, ambiguous=is_dst).as_unit(unit)
        tm.assert_index_equal(dr, localized)
        times += times
        di = DatetimeIndex(times).as_unit(unit)
        msg = 'Length of ambiguous bool-array must be the same size as vals'
        with pytest.raises(Exception, match=msg):
            di.tz_localize(tz, ambiguous=is_dst)
        is_dst = np.hstack((is_dst, is_dst))
        localized = di.tz_localize(tz, ambiguous=is_dst)
        dr = dr.append(dr)
        tm.assert_index_equal(dr, localized)

    @pytest.mark.parametrize('tz', easts)
    def test_dti_tz_localize_ambiguous_flags2(self, tz, unit):
        if False:
            i = 10
            return i + 15
        dr = date_range(datetime(2011, 6, 1, 0), periods=10, freq=offsets.Hour())
        is_dst = np.array([1] * 10)
        localized = dr.tz_localize(tz)
        localized_is_dst = dr.tz_localize(tz, ambiguous=is_dst)
        tm.assert_index_equal(localized, localized_is_dst)

    def test_dti_tz_localize_bdate_range(self):
        if False:
            print('Hello World!')
        dr = bdate_range('1/1/2009', '1/1/2010')
        dr_utc = bdate_range('1/1/2009', '1/1/2010', tz=pytz.utc)
        localized = dr.tz_localize(pytz.utc)
        tm.assert_index_equal(dr_utc, localized)

    @pytest.mark.parametrize('start_ts, tz, end_ts, shift', [['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 03:00:00', 'forward'], ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 01:59:59.999999999', 'backward'], ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 03:20:00', timedelta(hours=1)], ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 01:20:00', timedelta(hours=-1)], ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 03:00:00', 'forward'], ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 01:59:59.999999999', 'backward'], ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 03:33:00', timedelta(hours=1)], ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 01:33:00', timedelta(hours=-1)]])
    @pytest.mark.parametrize('tz_type', ['', 'dateutil/'])
    def test_dti_tz_localize_nonexistent_shift(self, start_ts, tz, end_ts, shift, tz_type, unit):
        if False:
            while True:
                i = 10
        tz = tz_type + tz
        if isinstance(shift, str):
            shift = 'shift_' + shift
        dti = DatetimeIndex([Timestamp(start_ts)]).as_unit(unit)
        result = dti.tz_localize(tz, nonexistent=shift)
        expected = DatetimeIndex([Timestamp(end_ts)]).tz_localize(tz).as_unit(unit)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('offset', [-1, 1])
    def test_dti_tz_localize_nonexistent_shift_invalid(self, offset, warsaw):
        if False:
            while True:
                i = 10
        tz = warsaw
        dti = DatetimeIndex([Timestamp('2015-03-29 02:20:00')])
        msg = 'The provided timedelta will relocalize on a nonexistent time'
        with pytest.raises(ValueError, match=msg):
            dti.tz_localize(tz, nonexistent=timedelta(seconds=offset))