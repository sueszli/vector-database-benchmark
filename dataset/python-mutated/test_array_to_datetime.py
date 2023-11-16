from datetime import date, datetime, timedelta, timezone
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import NaT, iNaT, tslib
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
creso_infer = NpyDatetimeUnit.NPY_FR_GENERIC.value

class TestArrayToDatetimeResolutionInference:

    def test_infer_homogeoneous_datetimes(self):
        if False:
            while True:
                i = 10
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        arr = np.array([dt, dt, dt], dtype=object)
        (result, tz) = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([dt, dt, dt], dtype='M8[us]')
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_date_objects(self):
        if False:
            for i in range(10):
                print('nop')
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        dt2 = dt.date()
        arr = np.array([None, dt2, dt2, dt2], dtype=object)
        (result, tz) = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([np.datetime64('NaT'), dt2, dt2, dt2], dtype='M8[s]')
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_dt64(self):
        if False:
            print('Hello World!')
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        dt64 = np.datetime64(dt, 'ms')
        arr = np.array([None, dt64, dt64, dt64], dtype=object)
        (result, tz) = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([np.datetime64('NaT'), dt64, dt64, dt64], dtype='M8[ms]')
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_timestamps(self):
        if False:
            for i in range(10):
                print('nop')
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        ts = Timestamp(dt).as_unit('ns')
        arr = np.array([None, ts, ts, ts], dtype=object)
        (result, tz) = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([np.datetime64('NaT')] + [ts.asm8] * 3, dtype='M8[ns]')
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_datetimes_strings(self):
        if False:
            while True:
                i = 10
        item = '2023-10-27 18:03:05.678000'
        arr = np.array([None, item, item, item], dtype=object)
        (result, tz) = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([np.datetime64('NaT'), item, item, item], dtype='M8[us]')
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_heterogeneous(self):
        if False:
            return 10
        dtstr = '2023-10-27 18:03:05.678000'
        arr = np.array([dtstr, dtstr[:-3], dtstr[:-7], None], dtype=object)
        (result, tz) = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array(arr, dtype='M8[us]')
        tm.assert_numpy_array_equal(result, expected)
        (result, tz) = tslib.array_to_datetime(arr[::-1], creso=creso_infer)
        assert tz is None
        tm.assert_numpy_array_equal(result, expected[::-1])

class TestArrayToDatetimeWithTZResolutionInference:

    def test_array_to_datetime_with_tz_resolution(self):
        if False:
            print('Hello World!')
        tz = tzoffset('custom', 3600)
        vals = np.array(['2016-01-01 02:03:04.567', NaT], dtype=object)
        res = tslib.array_to_datetime_with_tz(vals, tz, False, False, creso_infer)
        assert res.dtype == 'M8[ms]'
        vals2 = np.array([datetime(2016, 1, 1, 2, 3, 4), NaT], dtype=object)
        res2 = tslib.array_to_datetime_with_tz(vals2, tz, False, False, creso_infer)
        assert res2.dtype == 'M8[us]'
        vals3 = np.array([NaT, np.datetime64(12345, 's')], dtype=object)
        res3 = tslib.array_to_datetime_with_tz(vals3, tz, False, False, creso_infer)
        assert res3.dtype == 'M8[s]'

    def test_array_to_datetime_with_tz_resolution_all_nat(self):
        if False:
            i = 10
            return i + 15
        tz = tzoffset('custom', 3600)
        vals = np.array(['NaT'], dtype=object)
        res = tslib.array_to_datetime_with_tz(vals, tz, False, False, creso_infer)
        assert res.dtype == 'M8[ns]'
        vals2 = np.array([NaT, NaT], dtype=object)
        res2 = tslib.array_to_datetime_with_tz(vals2, tz, False, False, creso_infer)
        assert res2.dtype == 'M8[ns]'

@pytest.mark.parametrize('data,expected', [(['01-01-2013', '01-02-2013'], ['2013-01-01T00:00:00.000000000', '2013-01-02T00:00:00.000000000']), (['Mon Sep 16 2013', 'Tue Sep 17 2013'], ['2013-09-16T00:00:00.000000000', '2013-09-17T00:00:00.000000000'])])
def test_parsing_valid_dates(data, expected):
    if False:
        return 10
    arr = np.array(data, dtype=object)
    (result, _) = tslib.array_to_datetime(arr)
    expected = np.array(expected, dtype='M8[ns]')
    tm.assert_numpy_array_equal(result, expected)

@pytest.mark.parametrize('dt_string, expected_tz', [['01-01-2013 08:00:00+08:00', 480], ['2013-01-01T08:00:00.000000000+0800', 480], ['2012-12-31T16:00:00.000000000-0800', -480], ['12-31-2012 23:00:00-01:00', -60]])
def test_parsing_timezone_offsets(dt_string, expected_tz):
    if False:
        print('Hello World!')
    arr = np.array(['01-01-2013 00:00:00'], dtype=object)
    (expected, _) = tslib.array_to_datetime(arr)
    arr = np.array([dt_string], dtype=object)
    (result, result_tz) = tslib.array_to_datetime(arr)
    tm.assert_numpy_array_equal(result, expected)
    assert result_tz == timezone(timedelta(minutes=expected_tz))

def test_parsing_non_iso_timezone_offset():
    if False:
        i = 10
        return i + 15
    dt_string = '01-01-2013T00:00:00.000000000+0000'
    arr = np.array([dt_string], dtype=object)
    with tm.assert_produces_warning(None):
        (result, result_tz) = tslib.array_to_datetime(arr)
    expected = np.array([np.datetime64('2013-01-01 00:00:00.000000000')])
    tm.assert_numpy_array_equal(result, expected)
    assert result_tz is timezone.utc

def test_parsing_different_timezone_offsets():
    if False:
        while True:
            i = 10
    data = ['2015-11-18 15:30:00+05:30', '2015-11-18 15:30:00+06:30']
    data = np.array(data, dtype=object)
    msg = 'parsing datetimes with mixed time zones will raise an error'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        (result, result_tz) = tslib.array_to_datetime(data)
    expected = np.array([datetime(2015, 11, 18, 15, 30, tzinfo=tzoffset(None, 19800)), datetime(2015, 11, 18, 15, 30, tzinfo=tzoffset(None, 23400))], dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    assert result_tz is None

@pytest.mark.parametrize('data', [['-352.737091', '183.575577'], ['1', '2', '3', '4', '5']])
def test_number_looking_strings_not_into_datetime(data):
    if False:
        print('Hello World!')
    arr = np.array(data, dtype=object)
    (result, _) = tslib.array_to_datetime(arr, errors='ignore')
    tm.assert_numpy_array_equal(result, arr)

@pytest.mark.parametrize('invalid_date', [date(1000, 1, 1), datetime(1000, 1, 1), '1000-01-01', 'Jan 1, 1000', np.datetime64('1000-01-01')])
@pytest.mark.parametrize('errors', ['coerce', 'raise'])
def test_coerce_outside_ns_bounds(invalid_date, errors):
    if False:
        return 10
    arr = np.array([invalid_date], dtype='object')
    kwargs = {'values': arr, 'errors': errors}
    if errors == 'raise':
        msg = '^Out of bounds nanosecond timestamp: .*, at position 0$'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            tslib.array_to_datetime(**kwargs)
    else:
        (result, _) = tslib.array_to_datetime(**kwargs)
        expected = np.array([iNaT], dtype='M8[ns]')
        tm.assert_numpy_array_equal(result, expected)

def test_coerce_outside_ns_bounds_one_valid():
    if False:
        for i in range(10):
            print('nop')
    arr = np.array(['1/1/1000', '1/1/2000'], dtype=object)
    (result, _) = tslib.array_to_datetime(arr, errors='coerce')
    expected = [iNaT, '2000-01-01T00:00:00.000000000']
    expected = np.array(expected, dtype='M8[ns]')
    tm.assert_numpy_array_equal(result, expected)

@pytest.mark.parametrize('errors', ['ignore', 'coerce'])
def test_coerce_of_invalid_datetimes(errors):
    if False:
        i = 10
        return i + 15
    arr = np.array(['01-01-2013', 'not_a_date', '1'], dtype=object)
    kwargs = {'values': arr, 'errors': errors}
    if errors == 'ignore':
        (result, _) = tslib.array_to_datetime(**kwargs)
        tm.assert_numpy_array_equal(result, arr)
    else:
        (result, _) = tslib.array_to_datetime(arr, errors='coerce')
        expected = ['2013-01-01T00:00:00.000000000', iNaT, iNaT]
        tm.assert_numpy_array_equal(result, np.array(expected, dtype='M8[ns]'))

def test_to_datetime_barely_out_of_bounds():
    if False:
        print('Hello World!')
    arr = np.array(['2262-04-11 23:47:16.854775808'], dtype=object)
    msg = '^Out of bounds nanosecond timestamp: 2262-04-11 23:47:16, at position 0$'
    with pytest.raises(tslib.OutOfBoundsDatetime, match=msg):
        tslib.array_to_datetime(arr)

class SubDatetime(datetime):
    pass

@pytest.mark.parametrize('data,expected', [([SubDatetime(2000, 1, 1)], ['2000-01-01T00:00:00.000000000']), ([datetime(2000, 1, 1)], ['2000-01-01T00:00:00.000000000']), ([Timestamp(2000, 1, 1)], ['2000-01-01T00:00:00.000000000'])])
def test_datetime_subclass(data, expected):
    if False:
        print('Hello World!')
    arr = np.array(data, dtype=object)
    (result, _) = tslib.array_to_datetime(arr)
    expected = np.array(expected, dtype='M8[ns]')
    tm.assert_numpy_array_equal(result, expected)