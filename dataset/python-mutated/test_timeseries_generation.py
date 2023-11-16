from typing import Union
import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries
from darts.utils.timeseries_generation import autoregressive_timeseries, constant_timeseries, datetime_attribute_timeseries, gaussian_timeseries, generate_index, holidays_timeseries, linear_timeseries, random_walk_timeseries, sine_timeseries

class TestTimeSeriesGeneration:

    def test_constant_timeseries(self):
        if False:
            i = 10
            return i + 15
        value = 5

        def test_routine(start, end=None, length=None):
            if False:
                print('Hello World!')
            constant_ts = constant_timeseries(start=start, end=end, value=value, length=length)
            value_set = set(constant_ts.values().flatten())
            assert len(value_set) == 1
            assert len(constant_ts) == length_assert
        for length_assert in [1, 2, 5, 10, 100]:
            test_routine(start=0, length=length_assert)
            test_routine(start=0, end=length_assert - 1)
            test_routine(start=pd.Timestamp('2000-01-01'), length=length_assert)
            end_date = generate_index(start=pd.Timestamp('2000-01-01'), length=length_assert)[-1]
            test_routine(start=pd.Timestamp('2000-01-01'), end=end_date)

    def test_linear_timeseries(self):
        if False:
            while True:
                i = 10
        start_value = 5
        end_value = 12

        def test_routine(start, end=None, length=None):
            if False:
                return 10
            linear_ts = linear_timeseries(start=start, end=end, length=length, start_value=start_value, end_value=end_value)
            assert linear_ts.values()[0][0] == start_value
            assert linear_ts.values()[-1][0] == end_value
            assert round(abs(linear_ts.values()[-1][0] - linear_ts.values()[-2][0] - (end_value - start_value) / (length_assert - 1)), 7) == 0
            assert len(linear_ts) == length_assert
        for length_assert in [2, 5, 10, 100]:
            test_routine(start=0, length=length_assert)
            test_routine(start=0, end=length_assert - 1)
            test_routine(start=pd.Timestamp('2000-01-01'), length=length_assert)
            end_date = generate_index(start=pd.Timestamp('2000-01-01'), length=length_assert)[-1]
            test_routine(start=pd.Timestamp('2000-01-01'), end=end_date)

    def test_sine_timeseries(self):
        if False:
            for i in range(10):
                print('nop')
        value_amplitude = 5
        value_y_offset = -3

        def test_routine(start, end=None, length=None):
            if False:
                for i in range(10):
                    print('nop')
            sine_ts = sine_timeseries(start=start, end=end, length=length, value_amplitude=value_amplitude, value_y_offset=value_y_offset)
            assert (sine_ts <= value_y_offset + value_amplitude).all().all()
            assert (sine_ts >= value_y_offset - value_amplitude).all().all()
            assert len(sine_ts) == length_assert
        for length_assert in [1, 2, 5, 10, 100]:
            test_routine(start=0, length=length_assert)
            test_routine(start=0, end=length_assert - 1)
            test_routine(start=pd.Timestamp('2000-01-01'), length=length_assert)
            end_date = generate_index(start=pd.Timestamp('2000-01-01'), length=length_assert)[-1]
            test_routine(start=pd.Timestamp('2000-01-01'), end=end_date)

    def test_gaussian_timeseries(self):
        if False:
            return 10

        def test_routine(start, end=None, length=None):
            if False:
                return 10
            gaussian_ts = gaussian_timeseries(start=start, end=end, length=length)
            assert len(gaussian_ts) == length_assert
        for length_assert in [1, 2, 5, 10, 100]:
            test_routine(start=0, length=length_assert)
            test_routine(start=0, end=length_assert - 1)
            test_routine(start=pd.Timestamp('2000-01-01'), length=length_assert)
            end_date = generate_index(start=pd.Timestamp('2000-01-01'), length=length_assert)[-1]
            test_routine(start=pd.Timestamp('2000-01-01'), end=end_date)

    def test_random_walk_timeseries(self):
        if False:
            return 10

        def test_routine(start, end=None, length=None):
            if False:
                while True:
                    i = 10
            random_walk_ts = random_walk_timeseries(start=start, end=end, length=length)
            assert len(random_walk_ts) == length_assert
        for length_assert in [1, 2, 5, 10, 100]:
            test_routine(start=0, length=length_assert)
            test_routine(start=0, end=length_assert - 1)
            test_routine(start=pd.Timestamp('2000-01-01'), length=length_assert)
            end_date = generate_index(start=pd.Timestamp('2000-01-01'), length=length_assert)[-1]
            test_routine(start=pd.Timestamp('2000-01-01'), end=end_date)

    def test_holidays_timeseries(self):
        if False:
            while True:
                i = 10
        time_index_1 = pd.date_range(periods=365 * 3, freq='D', start=pd.Timestamp('2012-01-01'))
        time_index_2 = pd.date_range(periods=365 * 3, freq='D', start=pd.Timestamp('2014-12-24'))
        time_index_3 = pd.date_range(periods=10, freq='Y', start=pd.Timestamp('1950-01-01')) + pd.Timedelta(days=1)

        def test_routine(time_index, country_code, until: Union[int, pd.Timestamp, str]=0, add_length=0):
            if False:
                print('Hello World!')
            ts = holidays_timeseries(time_index, country_code, until=until, add_length=add_length)
            assert all(ts.pd_dataframe().groupby(pd.Grouper(freq='y')).sum().values)
        for time_index in [time_index_1, time_index_2, time_index_3]:
            for country_code in ['US', 'CH', 'AR']:
                test_routine(time_index, country_code)
        test_routine(time_index_1, 'US', add_length=365)
        test_routine(time_index_1, 'CH', until='2016-01-01')
        test_routine(time_index_1, 'CH', until='20160101')
        test_routine(time_index_1, 'AR', until=pd.Timestamp('2016-01-01'))
        with pytest.raises(ValueError):
            holidays_timeseries(time_index_1, 'US', add_length=99999)
        with pytest.raises(ValueError):
            holidays_timeseries(time_index_2, 'US', until='2016-01-01')
        with pytest.raises(ValueError):
            holidays_timeseries(time_index_3, 'US', until=163)
        with pytest.raises(ValueError):
            holidays_timeseries(time_index_3.tz_localize('UTC'), 'US', until=163)
        idx = generate_index(start=pd.Timestamp('2000-07-31 22:00:00'), length=3, freq='h')
        ts = holidays_timeseries(idx, country_code='CH')
        np.testing.assert_array_almost_equal(ts.values()[:, 0], np.array([0, 0, 1]))
        ts = holidays_timeseries(idx, country_code='CH', tz='CET')
        np.testing.assert_array_almost_equal(ts.values()[:, 0], np.array([1, 1, 1]))
        series = TimeSeries.from_times_and_values(times=idx, values=np.arange(len(idx)))
        ts = holidays_timeseries(series, country_code='CH', tz='CET')
        np.testing.assert_array_almost_equal(ts.values()[:, 0], np.array([1, 1, 1]))

    def test_generate_index(self):
        if False:
            return 10

        def test_routine(expected_length, expected_start, expected_end, start, end=None, length=None, freq=None):
            if False:
                return 10
            index = generate_index(start=start, end=end, length=length, freq=freq)
            assert len(index) == expected_length
            assert index[0] == expected_start
            assert index[-1] == expected_end
        for length in [1, 2, 5, 50]:
            for start in [0, 1, 9]:
                for step in [1, 2, 4]:
                    expected_start = start
                    expected_end = start + (length - 1) * step
                    freq = None if step == 1 else step
                    test_routine(expected_length=length, expected_start=expected_start, expected_end=expected_end, start=start, length=length, freq=freq)
                    test_routine(expected_length=length, expected_start=expected_start, expected_end=expected_end, start=start, end=expected_end, freq=step)
                    test_routine(expected_length=length, expected_start=expected_start, expected_end=expected_end, start=None, end=expected_end, length=length, freq=step)
                    if start == 0:
                        continue
                    start_date = pd.Timestamp(f'2000-01-0{start}')
                    dates = generate_index(start=start_date, length=length, freq='D' if step == 1 else f'{step}D')
                    (start_assert, end_assert) = (dates[0], dates[-1])
                    test_routine(expected_length=length, expected_start=start_assert, expected_end=end_assert, start=start_assert, length=length, freq='D' if step == 1 else f'{step}D')
                    test_routine(expected_length=length, expected_start=start_assert, expected_end=end_assert, start=start_assert, end=end_assert, freq='D' if step == 1 else f'{step}D')
                    test_routine(expected_length=length, expected_start=start_assert, expected_end=end_assert, start=None, end=end_assert, length=length, freq='D' if step == 1 else f'{step}D')
        with pytest.raises(ValueError):
            generate_index(start=0, end=9, length=10)
        with pytest.raises(ValueError):
            linear_timeseries(end=9, length=10)
        with pytest.raises(ValueError):
            generate_index(start=0)
        with pytest.raises(ValueError):
            generate_index(start=None, end=1)
        with pytest.raises(ValueError):
            generate_index(start=None, end=None, length=10)
        with pytest.raises(ValueError):
            generate_index(start=0, end=pd.Timestamp('2000-01-01'))
        with pytest.raises(ValueError):
            generate_index(start=pd.Timestamp('2000-01-01'), end=10)

    def test_autoregressive_timeseries(self):
        if False:
            i = 10
            return i + 15

        def test_length(start, end=None, length=None):
            if False:
                print('Hello World!')
            autoregressive_ts = autoregressive_timeseries(coef=[-1, 1.618], start=start, end=end, length=length)
            assert len(autoregressive_ts) == length_assert

        def test_calculation(coef):
            if False:
                return 10
            autoregressive_values = autoregressive_timeseries(coef=coef, length=100).values()
            for (idx, val) in enumerate(autoregressive_values[len(coef):]):
                assert val == np.dot(coef, autoregressive_values[idx:idx + len(coef)].ravel())
        for length_assert in [1, 2, 5, 10, 100]:
            test_length(start=0, length=length_assert)
            test_length(start=0, end=length_assert - 1)
            test_length(start=pd.Timestamp('2000-01-01'), length=length_assert)
            end_date = generate_index(start=pd.Timestamp('2000-01-01'), length=length_assert)[-1]
            test_length(start=pd.Timestamp('2000-01-01'), end=end_date)
        for coef_assert in [[-1], [-1, 1.618], [1, 2, 3], list(range(10))]:
            test_calculation(coef=coef_assert)

    def test_datetime_attribute_timeseries(self):
        if False:
            while True:
                i = 10
        idx = generate_index(start=pd.Timestamp('2000-01-01'), length=48, freq='h')

        def helper_routine(idx, attr, vals_exp, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            ts = datetime_attribute_timeseries(idx, attribute=attr, **kwargs)
            vals_exp = np.array(vals_exp, dtype=ts.dtype)
            if len(vals_exp.shape) == 1:
                vals_act = ts.values()[:, 0]
            else:
                vals_act = ts.values()
            np.testing.assert_array_almost_equal(vals_act, vals_exp)
        with pytest.raises(ValueError) as err:
            helper_routine(pd.RangeIndex(start=0, stop=len(idx)), 'h', vals_exp=np.arange(len(idx)))
        assert str(err.value).startswith('`time_index` must be a pandas `DatetimeIndex`')
        with pytest.raises(ValueError) as err:
            helper_routine(idx, 'h', vals_exp=np.arange(len(idx)))
        assert str(err.value).startswith('attribute `h` needs to be an attribute of pd.DatetimeIndex.')
        with pytest.raises(ValueError) as err:
            helper_routine(idx.tz_localize('UTC'), 'h', vals_exp=np.arange(len(idx)))
        assert '`time_index` must be time zone naive.' == str(err.value)
        vals = [i for i in range(24)] * 2
        helper_routine(idx, 'hour', vals_exp=vals)
        helper_routine(TimeSeries.from_times_and_values(times=idx, values=np.arange(len(idx))), 'hour', vals_exp=vals)
        vals = vals[1:] + [0]
        helper_routine(idx, 'hour', vals_exp=vals, tz='CET')
        vals = [1] * 24 + [2] * 24
        helper_routine(idx, 'day', vals_exp=vals)
        vals = [5] * 24 + [6] * 24
        helper_routine(idx, 'dayofweek', vals_exp=vals)
        vals = [1] * 48
        helper_routine(idx, 'month', vals_exp=vals)
        vals = [1] + [0] * 11
        vals = [vals for _ in range(48)]
        helper_routine(idx, 'month', vals_exp=vals, one_hot=True)
        vals = [1] + [0] * 11
        vals = [vals for _ in range(48)]
        helper_routine(idx, 'month', vals_exp=vals, tz='CET', one_hot=True)
        period = 24
        freq = 2 * np.pi / period
        vals_dta = [i for i in range(24)] * 2
        vals = np.array(vals_dta)
        sin_vals = np.sin(freq * vals)[:, None]
        cos_vals = np.cos(freq * vals)[:, None]
        vals = np.concatenate([sin_vals, cos_vals], axis=1)
        helper_routine(idx, 'hour', vals_exp=vals, cyclic=True)
        vals = np.array(vals_dta[1:] + [0])
        sin_vals = np.sin(freq * vals)[:, None]
        cos_vals = np.cos(freq * vals)[:, None]
        vals = np.concatenate([sin_vals, cos_vals], axis=1)
        helper_routine(idx, 'hour', vals_exp=vals, tz='CET', cyclic=True)