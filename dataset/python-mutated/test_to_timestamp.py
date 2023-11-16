from datetime import datetime
import numpy as np
import pytest
from pandas import DatetimeIndex, NaT, PeriodIndex, Timedelta, Timestamp, date_range, period_range
import pandas._testing as tm

class TestToTimestamp:

    def test_to_timestamp_non_contiguous(self):
        if False:
            print('Hello World!')
        dti = date_range('2021-10-18', periods=9, freq='D')
        pi = dti.to_period()
        result = pi[::2].to_timestamp()
        expected = dti[::2]
        tm.assert_index_equal(result, expected)
        result = pi._data[::2].to_timestamp()
        expected = dti._data[::2]
        tm.assert_datetime_array_equal(result, expected, check_freq=False)
        result = pi[::-1].to_timestamp()
        expected = dti[::-1]
        tm.assert_index_equal(result, expected)
        result = pi._data[::-1].to_timestamp()
        expected = dti._data[::-1]
        tm.assert_datetime_array_equal(result, expected, check_freq=False)
        result = pi[::2][::-1].to_timestamp()
        expected = dti[::2][::-1]
        tm.assert_index_equal(result, expected)
        result = pi._data[::2][::-1].to_timestamp()
        expected = dti._data[::2][::-1]
        tm.assert_datetime_array_equal(result, expected, check_freq=False)

    def test_to_timestamp_freq(self):
        if False:
            for i in range(10):
                print('nop')
        idx = period_range('2017', periods=12, freq='Y-DEC')
        result = idx.to_timestamp()
        expected = date_range('2017', periods=12, freq='YS-JAN')
        tm.assert_index_equal(result, expected)

    def test_to_timestamp_pi_nat(self):
        if False:
            i = 10
            return i + 15
        index = PeriodIndex(['NaT', '2011-01', '2011-02'], freq='M', name='idx')
        result = index.to_timestamp('D')
        expected = DatetimeIndex([NaT, datetime(2011, 1, 1), datetime(2011, 2, 1)], name='idx')
        tm.assert_index_equal(result, expected)
        assert result.name == 'idx'
        result2 = result.to_period(freq='M')
        tm.assert_index_equal(result2, index)
        assert result2.name == 'idx'
        result3 = result.to_period(freq='3M')
        exp = PeriodIndex(['NaT', '2011-01', '2011-02'], freq='3M', name='idx')
        tm.assert_index_equal(result3, exp)
        assert result3.freqstr == '3M'
        msg = 'Frequency must be positive, because it represents span: -2Y'
        with pytest.raises(ValueError, match=msg):
            result.to_period(freq='-2Y')

    def test_to_timestamp_preserve_name(self):
        if False:
            for i in range(10):
                print('nop')
        index = period_range(freq='Y', start='1/1/2001', end='12/1/2009', name='foo')
        assert index.name == 'foo'
        conv = index.to_timestamp('D')
        assert conv.name == 'foo'

    def test_to_timestamp_quarterly_bug(self):
        if False:
            i = 10
            return i + 15
        years = np.arange(1960, 2000).repeat(4)
        quarters = np.tile(list(range(1, 5)), 40)
        pindex = PeriodIndex.from_fields(year=years, quarter=quarters)
        stamps = pindex.to_timestamp('D', 'end')
        expected = DatetimeIndex([x.to_timestamp('D', 'end') for x in pindex])
        tm.assert_index_equal(stamps, expected)
        assert stamps.freq == expected.freq

    def test_to_timestamp_pi_mult(self):
        if False:
            print('Hello World!')
        idx = PeriodIndex(['2011-01', 'NaT', '2011-02'], freq='2M', name='idx')
        result = idx.to_timestamp()
        expected = DatetimeIndex(['2011-01-01', 'NaT', '2011-02-01'], name='idx')
        tm.assert_index_equal(result, expected)
        result = idx.to_timestamp(how='E')
        expected = DatetimeIndex(['2011-02-28', 'NaT', '2011-03-31'], name='idx')
        expected = expected + Timedelta(1, 'D') - Timedelta(1, 'ns')
        tm.assert_index_equal(result, expected)

    def test_to_timestamp_pi_combined(self):
        if False:
            while True:
                i = 10
        idx = period_range(start='2011', periods=2, freq='1D1h', name='idx')
        result = idx.to_timestamp()
        expected = DatetimeIndex(['2011-01-01 00:00', '2011-01-02 01:00'], name='idx')
        tm.assert_index_equal(result, expected)
        result = idx.to_timestamp(how='E')
        expected = DatetimeIndex(['2011-01-02 00:59:59', '2011-01-03 01:59:59'], name='idx')
        expected = expected + Timedelta(1, 's') - Timedelta(1, 'ns')
        tm.assert_index_equal(result, expected)
        result = idx.to_timestamp(how='E', freq='h')
        expected = DatetimeIndex(['2011-01-02 00:00', '2011-01-03 01:00'], name='idx')
        expected = expected + Timedelta(1, 'h') - Timedelta(1, 'ns')
        tm.assert_index_equal(result, expected)

    def test_to_timestamp_1703(self):
        if False:
            return 10
        index = period_range('1/1/2012', periods=4, freq='D')
        result = index.to_timestamp()
        assert result[0] == Timestamp('1/1/2012')