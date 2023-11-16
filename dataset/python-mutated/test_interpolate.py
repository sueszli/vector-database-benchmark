import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import Index, MultiIndex, Series, date_range, isna
import pandas._testing as tm

@pytest.fixture(params=['linear', 'index', 'values', 'nearest', 'slinear', 'zero', 'quadratic', 'cubic', 'barycentric', 'krogh', 'polynomial', 'spline', 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima', 'cubicspline'])
def nontemporal_method(request):
    if False:
        while True:
            i = 10
    "Fixture that returns an (method name, required kwargs) pair.\n\n    This fixture does not include method 'time' as a parameterization; that\n    method requires a Series with a DatetimeIndex, and is generally tested\n    separately from these non-temporal methods.\n    "
    method = request.param
    kwargs = {'order': 1} if method in ('spline', 'polynomial') else {}
    return (method, kwargs)

@pytest.fixture(params=['linear', 'slinear', 'zero', 'quadratic', 'cubic', 'barycentric', 'krogh', 'polynomial', 'spline', 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima', 'cubicspline'])
def interp_methods_ind(request):
    if False:
        return 10
    "Fixture that returns a (method name, required kwargs) pair to\n    be tested for various Index types.\n\n    This fixture does not include methods - 'time', 'index', 'nearest',\n    'values' as a parameterization\n    "
    method = request.param
    kwargs = {'order': 1} if method in ('spline', 'polynomial') else {}
    return (method, kwargs)

class TestSeriesInterpolateData:

    @pytest.mark.xfail(reason="EA.fillna does not handle 'linear' method")
    def test_interpolate_period_values(self):
        if False:
            i = 10
            return i + 15
        orig = Series(date_range('2012-01-01', periods=5))
        ser = orig.copy()
        ser[2] = pd.NaT
        ser_per = ser.dt.to_period('D')
        res_per = ser_per.interpolate()
        expected_per = orig.dt.to_period('D')
        tm.assert_series_equal(res_per, expected_per)

    def test_interpolate(self, datetime_series):
        if False:
            while True:
                i = 10
        ts = Series(np.arange(len(datetime_series), dtype=float), datetime_series.index)
        ts_copy = ts.copy()
        ts_copy[5:10] = np.nan
        linear_interp = ts_copy.interpolate(method='linear')
        tm.assert_series_equal(linear_interp, ts)
        ord_ts = Series([d.toordinal() for d in datetime_series.index], index=datetime_series.index).astype(float)
        ord_ts_copy = ord_ts.copy()
        ord_ts_copy[5:10] = np.nan
        time_interp = ord_ts_copy.interpolate(method='time')
        tm.assert_series_equal(time_interp, ord_ts)

    def test_interpolate_time_raises_for_non_timeseries(self):
        if False:
            print('Hello World!')
        non_ts = Series([0, 1, 2, np.nan])
        msg = 'time-weighted interpolation only works on Series.* with a DatetimeIndex'
        with pytest.raises(ValueError, match=msg):
            non_ts.interpolate(method='time')

    def test_interpolate_cubicspline(self):
        if False:
            for i in range(10):
                print('nop')
        pytest.importorskip('scipy')
        ser = Series([10, 11, 12, 13])
        expected = Series([11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0], index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]))
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(float)
        result = ser.reindex(new_index).interpolate(method='cubicspline').loc[1:3]
        tm.assert_series_equal(result, expected)

    def test_interpolate_pchip(self):
        if False:
            print('Hello World!')
        pytest.importorskip('scipy')
        ser = Series(np.sort(np.random.default_rng(2).uniform(size=100)))
        new_index = ser.index.union(Index([49.25, 49.5, 49.75, 50.25, 50.5, 50.75])).astype(float)
        interp_s = ser.reindex(new_index).interpolate(method='pchip')
        interp_s.loc[49:51]

    def test_interpolate_akima(self):
        if False:
            return 10
        pytest.importorskip('scipy')
        ser = Series([10, 11, 12, 13])
        expected = Series([11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0], index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]))
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(float)
        interp_s = ser.reindex(new_index).interpolate(method='akima')
        tm.assert_series_equal(interp_s.loc[1:3], expected)
        expected = Series([11.0, 1.0, 1.0, 1.0, 12.0, 1.0, 1.0, 1.0, 13.0], index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]))
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(float)
        interp_s = ser.reindex(new_index).interpolate(method='akima', der=1)
        tm.assert_series_equal(interp_s.loc[1:3], expected)

    def test_interpolate_piecewise_polynomial(self):
        if False:
            i = 10
            return i + 15
        pytest.importorskip('scipy')
        ser = Series([10, 11, 12, 13])
        expected = Series([11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0], index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]))
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(float)
        interp_s = ser.reindex(new_index).interpolate(method='piecewise_polynomial')
        tm.assert_series_equal(interp_s.loc[1:3], expected)

    def test_interpolate_from_derivatives(self):
        if False:
            return 10
        pytest.importorskip('scipy')
        ser = Series([10, 11, 12, 13])
        expected = Series([11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0], index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]))
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(float)
        interp_s = ser.reindex(new_index).interpolate(method='from_derivatives')
        tm.assert_series_equal(interp_s.loc[1:3], expected)

    @pytest.mark.parametrize('kwargs', [{}, pytest.param({'method': 'polynomial', 'order': 1}, marks=td.skip_if_no_scipy)])
    def test_interpolate_corners(self, kwargs):
        if False:
            while True:
                i = 10
        s = Series([np.nan, np.nan])
        tm.assert_series_equal(s.interpolate(**kwargs), s)
        s = Series([], dtype=object).interpolate()
        tm.assert_series_equal(s.interpolate(**kwargs), s)

    def test_interpolate_index_values(self):
        if False:
            return 10
        s = Series(np.nan, index=np.sort(np.random.default_rng(2).random(30)))
        s.loc[::3] = np.random.default_rng(2).standard_normal(10)
        vals = s.index.values.astype(float)
        result = s.interpolate(method='index')
        expected = s.copy()
        bad = isna(expected.values)
        good = ~bad
        expected = Series(np.interp(vals[bad], vals[good], s.values[good]), index=s.index[bad])
        tm.assert_series_equal(result[bad], expected)
        other_result = s.interpolate(method='values')
        tm.assert_series_equal(other_result, result)
        tm.assert_series_equal(other_result[bad], expected)

    def test_interpolate_non_ts(self):
        if False:
            print('Hello World!')
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])
        msg = 'time-weighted interpolation only works on Series or DataFrames with a DatetimeIndex'
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='time')

    @pytest.mark.parametrize('kwargs', [{}, pytest.param({'method': 'polynomial', 'order': 1}, marks=td.skip_if_no_scipy)])
    def test_nan_interpolate(self, kwargs):
        if False:
            i = 10
            return i + 15
        s = Series([0, 1, np.nan, 3])
        result = s.interpolate(**kwargs)
        expected = Series([0.0, 1.0, 2.0, 3.0])
        tm.assert_series_equal(result, expected)

    def test_nan_irregular_index(self):
        if False:
            i = 10
            return i + 15
        s = Series([1, 2, np.nan, 4], index=[1, 3, 5, 9])
        result = s.interpolate()
        expected = Series([1.0, 2.0, 3.0, 4.0], index=[1, 3, 5, 9])
        tm.assert_series_equal(result, expected)

    def test_nan_str_index(self):
        if False:
            for i in range(10):
                print('nop')
        s = Series([0, 1, 2, np.nan], index=list('abcd'))
        result = s.interpolate()
        expected = Series([0.0, 1.0, 2.0, 2.0], index=list('abcd'))
        tm.assert_series_equal(result, expected)

    def test_interp_quad(self):
        if False:
            print('Hello World!')
        pytest.importorskip('scipy')
        sq = Series([1, 4, np.nan, 16], index=[1, 2, 3, 4])
        result = sq.interpolate(method='quadratic')
        expected = Series([1.0, 4.0, 9.0, 16.0], index=[1, 2, 3, 4])
        tm.assert_series_equal(result, expected)

    def test_interp_scipy_basic(self):
        if False:
            print('Hello World!')
        pytest.importorskip('scipy')
        s = Series([1, 3, np.nan, 12, np.nan, 25])
        expected = Series([1.0, 3.0, 7.5, 12.0, 18.5, 25.0])
        result = s.interpolate(method='slinear')
        tm.assert_series_equal(result, expected)
        msg = "The 'downcast' keyword in Series.interpolate is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.interpolate(method='slinear', downcast='infer')
        tm.assert_series_equal(result, expected)
        expected = Series([1, 3, 3, 12, 12, 25])
        result = s.interpolate(method='nearest')
        tm.assert_series_equal(result, expected.astype('float'))
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.interpolate(method='nearest', downcast='infer')
        tm.assert_series_equal(result, expected)
        expected = Series([1, 3, 3, 12, 12, 25])
        result = s.interpolate(method='zero')
        tm.assert_series_equal(result, expected.astype('float'))
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.interpolate(method='zero', downcast='infer')
        tm.assert_series_equal(result, expected)
        expected = Series([1, 3.0, 6.823529, 12.0, 18.058824, 25.0])
        result = s.interpolate(method='quadratic')
        tm.assert_series_equal(result, expected)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.interpolate(method='quadratic', downcast='infer')
        tm.assert_series_equal(result, expected)
        expected = Series([1.0, 3.0, 6.8, 12.0, 18.2, 25.0])
        result = s.interpolate(method='cubic')
        tm.assert_series_equal(result, expected)

    def test_interp_limit(self):
        if False:
            for i in range(10):
                print('nop')
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])
        expected = Series([1.0, 3.0, 5.0, 7.0, np.nan, 11.0])
        result = s.interpolate(method='linear', limit=2)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('limit', [-1, 0])
    def test_interpolate_invalid_nonpositive_limit(self, nontemporal_method, limit):
        if False:
            return 10
        s = Series([1, 2, np.nan, 4])
        (method, kwargs) = nontemporal_method
        with pytest.raises(ValueError, match='Limit must be greater than 0'):
            s.interpolate(limit=limit, method=method, **kwargs)

    def test_interpolate_invalid_float_limit(self, nontemporal_method):
        if False:
            while True:
                i = 10
        s = Series([1, 2, np.nan, 4])
        (method, kwargs) = nontemporal_method
        limit = 2.0
        with pytest.raises(ValueError, match='Limit must be an integer'):
            s.interpolate(limit=limit, method=method, **kwargs)

    @pytest.mark.parametrize('invalid_method', [None, 'nonexistent_method'])
    def test_interp_invalid_method(self, invalid_method):
        if False:
            for i in range(10):
                print('nop')
        s = Series([1, 3, np.nan, 12, np.nan, 25])
        msg = f"method must be one of.* Got '{invalid_method}' instead"
        if invalid_method is None:
            msg = "'method' should be a string, not None"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method=invalid_method)
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method=invalid_method, limit=-1)

    def test_interp_invalid_method_and_value(self):
        if False:
            return 10
        ser = Series([1, 3, np.nan, 12, np.nan, 25])
        msg = "'fill_value' is not a valid keyword for Series.interpolate"
        msg2 = 'Series.interpolate with method=pad'
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                ser.interpolate(fill_value=3, method='pad')

    def test_interp_limit_forward(self):
        if False:
            i = 10
            return i + 15
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])
        expected = Series([1.0, 3.0, 5.0, 7.0, np.nan, 11.0])
        result = s.interpolate(method='linear', limit=2, limit_direction='forward')
        tm.assert_series_equal(result, expected)
        result = s.interpolate(method='linear', limit=2, limit_direction='FORWARD')
        tm.assert_series_equal(result, expected)

    def test_interp_unlimited(self):
        if False:
            return 10
        s = Series([np.nan, 1.0, 3.0, np.nan, np.nan, np.nan, 11.0, np.nan])
        expected = Series([1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 11.0])
        result = s.interpolate(method='linear', limit_direction='both')
        tm.assert_series_equal(result, expected)
        expected = Series([np.nan, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 11.0])
        result = s.interpolate(method='linear', limit_direction='forward')
        tm.assert_series_equal(result, expected)
        expected = Series([1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, np.nan])
        result = s.interpolate(method='linear', limit_direction='backward')
        tm.assert_series_equal(result, expected)

    def test_interp_limit_bad_direction(self):
        if False:
            i = 10
            return i + 15
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])
        msg = "Invalid limit_direction: expecting one of \\['forward', 'backward', 'both'\\], got 'abc'"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='linear', limit=2, limit_direction='abc')
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='linear', limit_direction='abc')

    def test_interp_limit_area(self):
        if False:
            return 10
        s = Series([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan])
        expected = Series([np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan, np.nan])
        result = s.interpolate(method='linear', limit_area='inside')
        tm.assert_series_equal(result, expected)
        expected = Series([np.nan, np.nan, 3.0, 4.0, np.nan, np.nan, 7.0, np.nan, np.nan])
        result = s.interpolate(method='linear', limit_area='inside', limit=1)
        tm.assert_series_equal(result, expected)
        expected = Series([np.nan, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, np.nan, np.nan])
        result = s.interpolate(method='linear', limit_area='inside', limit_direction='both', limit=1)
        tm.assert_series_equal(result, expected)
        expected = Series([np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, 7.0])
        result = s.interpolate(method='linear', limit_area='outside')
        tm.assert_series_equal(result, expected)
        expected = Series([np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan])
        result = s.interpolate(method='linear', limit_area='outside', limit=1)
        tm.assert_series_equal(result, expected)
        expected = Series([np.nan, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan])
        result = s.interpolate(method='linear', limit_area='outside', limit_direction='both', limit=1)
        tm.assert_series_equal(result, expected)
        expected = Series([3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan])
        result = s.interpolate(method='linear', limit_area='outside', limit_direction='backward')
        tm.assert_series_equal(result, expected)
        msg = "Invalid limit_area: expecting one of \\['inside', 'outside'\\], got abc"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='linear', limit_area='abc')

    @pytest.mark.parametrize('method, limit_direction, expected', [('pad', 'backward', 'forward'), ('ffill', 'backward', 'forward'), ('backfill', 'forward', 'backward'), ('bfill', 'forward', 'backward'), ('pad', 'both', 'forward'), ('ffill', 'both', 'forward'), ('backfill', 'both', 'backward'), ('bfill', 'both', 'backward')])
    def test_interp_limit_direction_raises(self, method, limit_direction, expected):
        if False:
            while True:
                i = 10
        s = Series([1, 2, 3])
        msg = f"`limit_direction` must be '{expected}' for method `{method}`"
        msg2 = 'Series.interpolate with method='
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                s.interpolate(method=method, limit_direction=limit_direction)

    @pytest.mark.parametrize('data, expected_data, kwargs', (([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, 3.0, 3.0, 3.0, 7.0, np.nan, np.nan], {'method': 'pad', 'limit_area': 'inside'}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, 3.0, np.nan, np.nan, 7.0, np.nan, np.nan], {'method': 'pad', 'limit_area': 'inside', 'limit': 1}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, 7.0], {'method': 'pad', 'limit_area': 'outside'}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan], {'method': 'pad', 'limit_area': 'outside', 'limit': 1}), ([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], {'method': 'pad', 'limit_area': 'outside', 'limit': 1}), (range(5), range(5), {'method': 'pad', 'limit_area': 'outside', 'limit': 1})))
    def test_interp_limit_area_with_pad(self, data, expected_data, kwargs):
        if False:
            for i in range(10):
                print('nop')
        s = Series(data)
        expected = Series(expected_data)
        msg = 'Series.interpolate with method=pad'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.interpolate(**kwargs)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('data, expected_data, kwargs', (([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, 7.0, 7.0, 7.0, 7.0, np.nan, np.nan], {'method': 'bfill', 'limit_area': 'inside'}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, np.nan, np.nan, 7.0, 7.0, np.nan, np.nan], {'method': 'bfill', 'limit_area': 'inside', 'limit': 1}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan], {'method': 'bfill', 'limit_area': 'outside'}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan], {'method': 'bfill', 'limit_area': 'outside', 'limit': 1})))
    def test_interp_limit_area_with_backfill(self, data, expected_data, kwargs):
        if False:
            print('Hello World!')
        s = Series(data)
        expected = Series(expected_data)
        msg = 'Series.interpolate with method=bfill'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.interpolate(**kwargs)
        tm.assert_series_equal(result, expected)

    def test_interp_limit_direction(self):
        if False:
            for i in range(10):
                print('nop')
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])
        expected = Series([1.0, 3.0, np.nan, 7.0, 9.0, 11.0])
        result = s.interpolate(method='linear', limit=2, limit_direction='backward')
        tm.assert_series_equal(result, expected)
        expected = Series([1.0, 3.0, 5.0, np.nan, 9.0, 11.0])
        result = s.interpolate(method='linear', limit=1, limit_direction='both')
        tm.assert_series_equal(result, expected)
        s = Series([1, 3, np.nan, np.nan, np.nan, 7, 9, np.nan, np.nan, 12, np.nan])
        expected = Series([1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 12.0])
        result = s.interpolate(method='linear', limit=2, limit_direction='both')
        tm.assert_series_equal(result, expected)
        expected = Series([1.0, 3.0, 4.0, np.nan, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 12.0])
        result = s.interpolate(method='linear', limit=1, limit_direction='both')
        tm.assert_series_equal(result, expected)

    def test_interp_limit_to_ends(self):
        if False:
            print('Hello World!')
        s = Series([np.nan, np.nan, 5, 7, 9, np.nan])
        expected = Series([5.0, 5.0, 5.0, 7.0, 9.0, np.nan])
        result = s.interpolate(method='linear', limit=2, limit_direction='backward')
        tm.assert_series_equal(result, expected)
        expected = Series([5.0, 5.0, 5.0, 7.0, 9.0, 9.0])
        result = s.interpolate(method='linear', limit=2, limit_direction='both')
        tm.assert_series_equal(result, expected)

    def test_interp_limit_before_ends(self):
        if False:
            i = 10
            return i + 15
        s = Series([np.nan, np.nan, 5, 7, np.nan, np.nan])
        expected = Series([np.nan, np.nan, 5.0, 7.0, 7.0, np.nan])
        result = s.interpolate(method='linear', limit=1, limit_direction='forward')
        tm.assert_series_equal(result, expected)
        expected = Series([np.nan, 5.0, 5.0, 7.0, np.nan, np.nan])
        result = s.interpolate(method='linear', limit=1, limit_direction='backward')
        tm.assert_series_equal(result, expected)
        expected = Series([np.nan, 5.0, 5.0, 7.0, 7.0, np.nan])
        result = s.interpolate(method='linear', limit=1, limit_direction='both')
        tm.assert_series_equal(result, expected)

    def test_interp_all_good(self):
        if False:
            for i in range(10):
                print('nop')
        pytest.importorskip('scipy')
        s = Series([1, 2, 3])
        result = s.interpolate(method='polynomial', order=1)
        tm.assert_series_equal(result, s)
        result = s.interpolate()
        tm.assert_series_equal(result, s)

    @pytest.mark.parametrize('check_scipy', [False, pytest.param(True, marks=td.skip_if_no_scipy)])
    def test_interp_multiIndex(self, check_scipy):
        if False:
            while True:
                i = 10
        idx = MultiIndex.from_tuples([(0, 'a'), (1, 'b'), (2, 'c')])
        s = Series([1, 2, np.nan], index=idx)
        expected = s.copy()
        expected.loc[2] = 2
        result = s.interpolate()
        tm.assert_series_equal(result, expected)
        msg = 'Only `method=linear` interpolation is supported on MultiIndexes'
        if check_scipy:
            with pytest.raises(ValueError, match=msg):
                s.interpolate(method='polynomial', order=1)

    def test_interp_nonmono_raise(self):
        if False:
            while True:
                i = 10
        pytest.importorskip('scipy')
        s = Series([1, np.nan, 3], index=[0, 2, 1])
        msg = 'krogh interpolation requires that the index be monotonic'
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='krogh')

    @pytest.mark.parametrize('method', ['nearest', 'pad'])
    def test_interp_datetime64(self, method, tz_naive_fixture):
        if False:
            return 10
        pytest.importorskip('scipy')
        df = Series([1, np.nan, 3], index=date_range('1/1/2000', periods=3, tz=tz_naive_fixture))
        warn = None if method == 'nearest' else FutureWarning
        msg = 'Series.interpolate with method=pad is deprecated'
        with tm.assert_produces_warning(warn, match=msg):
            result = df.interpolate(method=method)
        if warn is not None:
            alt = df.ffill()
            tm.assert_series_equal(result, alt)
        expected = Series([1.0, 1.0, 3.0], index=date_range('1/1/2000', periods=3, tz=tz_naive_fixture))
        tm.assert_series_equal(result, expected)

    def test_interp_pad_datetime64tz_values(self):
        if False:
            for i in range(10):
                print('nop')
        dti = date_range('2015-04-05', periods=3, tz='US/Central')
        ser = Series(dti)
        ser[1] = pd.NaT
        msg = 'Series.interpolate with method=pad is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser.interpolate(method='pad')
        alt = ser.ffill()
        tm.assert_series_equal(result, alt)
        expected = Series(dti)
        expected[1] = expected[0]
        tm.assert_series_equal(result, expected)

    def test_interp_limit_no_nans(self):
        if False:
            i = 10
            return i + 15
        s = Series([1.0, 2.0, 3.0])
        result = s.interpolate(limit=1)
        expected = s
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('method', ['polynomial', 'spline'])
    def test_no_order(self, method):
        if False:
            for i in range(10):
                print('nop')
        pytest.importorskip('scipy')
        s = Series([0, 1, np.nan, 3])
        msg = 'You must specify the order of the spline or polynomial'
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method=method)

    @pytest.mark.parametrize('order', [-1, -1.0, 0, 0.0, np.nan])
    def test_interpolate_spline_invalid_order(self, order):
        if False:
            print('Hello World!')
        pytest.importorskip('scipy')
        s = Series([0, 1, np.nan, 3])
        msg = 'order needs to be specified and greater than 0'
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='spline', order=order)

    def test_spline(self):
        if False:
            print('Hello World!')
        pytest.importorskip('scipy')
        s = Series([1, 2, np.nan, 4, 5, np.nan, 7])
        result = s.interpolate(method='spline', order=1)
        expected = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        tm.assert_series_equal(result, expected)

    def test_spline_extrapolate(self):
        if False:
            i = 10
            return i + 15
        pytest.importorskip('scipy')
        s = Series([1, 2, 3, 4, np.nan, 6, np.nan])
        result3 = s.interpolate(method='spline', order=1, ext=3)
        expected3 = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0])
        tm.assert_series_equal(result3, expected3)
        result1 = s.interpolate(method='spline', order=1, ext=0)
        expected1 = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        tm.assert_series_equal(result1, expected1)

    def test_spline_smooth(self):
        if False:
            i = 10
            return i + 15
        pytest.importorskip('scipy')
        s = Series([1, 2, np.nan, 4, 5.1, np.nan, 7])
        assert s.interpolate(method='spline', order=3, s=0)[5] != s.interpolate(method='spline', order=3)[5]

    def test_spline_interpolation(self):
        if False:
            i = 10
            return i + 15
        pytest.importorskip('scipy')
        s = Series(np.arange(10) ** 2, dtype='float')
        s[np.random.default_rng(2).integers(0, 9, 3)] = np.nan
        result1 = s.interpolate(method='spline', order=1)
        expected1 = s.interpolate(method='spline', order=1)
        tm.assert_series_equal(result1, expected1)

    def test_interp_timedelta64(self):
        if False:
            return 10
        df = Series([1, np.nan, 3], index=pd.to_timedelta([1, 2, 3]))
        result = df.interpolate(method='time')
        expected = Series([1.0, 2.0, 3.0], index=pd.to_timedelta([1, 2, 3]))
        tm.assert_series_equal(result, expected)
        df = Series([1, np.nan, 3], index=pd.to_timedelta([1, 2, 4]))
        result = df.interpolate(method='time')
        expected = Series([1.0, 1.666667, 3.0], index=pd.to_timedelta([1, 2, 4]))
        tm.assert_series_equal(result, expected)

    def test_series_interpolate_method_values(self):
        if False:
            return 10
        rng = date_range('1/1/2000', '1/20/2000', freq='D')
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        ts[::2] = np.nan
        result = ts.interpolate(method='values')
        exp = ts.interpolate()
        tm.assert_series_equal(result, exp)

    def test_series_interpolate_intraday(self):
        if False:
            print('Hello World!')
        index = date_range('1/1/2012', periods=4, freq='12D')
        ts = Series([0, 12, 24, 36], index)
        new_index = index.append(index + pd.DateOffset(days=1)).sort_values()
        exp = ts.reindex(new_index).interpolate(method='time')
        index = date_range('1/1/2012', periods=4, freq='12h')
        ts = Series([0, 12, 24, 36], index)
        new_index = index.append(index + pd.DateOffset(hours=1)).sort_values()
        result = ts.reindex(new_index).interpolate(method='time')
        tm.assert_numpy_array_equal(result.values, exp.values)

    @pytest.mark.parametrize('ind', [['a', 'b', 'c', 'd'], pd.period_range(start='2019-01-01', periods=4), pd.interval_range(start=0, end=4)])
    def test_interp_non_timedelta_index(self, interp_methods_ind, ind):
        if False:
            while True:
                i = 10
        df = pd.DataFrame([0, 1, np.nan, 3], index=ind)
        (method, kwargs) = interp_methods_ind
        if method == 'pchip':
            pytest.importorskip('scipy')
        if method == 'linear':
            result = df[0].interpolate(**kwargs)
            expected = Series([0.0, 1.0, 2.0, 3.0], name=0, index=ind)
            tm.assert_series_equal(result, expected)
        else:
            expected_error = f'Index column must be numeric or datetime type when using {method} method other than linear. Try setting a numeric or datetime index column before interpolating.'
            with pytest.raises(ValueError, match=expected_error):
                df[0].interpolate(method=method, **kwargs)

    def test_interpolate_timedelta_index(self, request, interp_methods_ind):
        if False:
            print('Hello World!')
        '\n        Tests for non numerical index types  - object, period, timedelta\n        Note that all methods except time, index, nearest and values\n        are tested here.\n        '
        pytest.importorskip('scipy')
        ind = pd.timedelta_range(start=1, periods=4)
        df = pd.DataFrame([0, 1, np.nan, 3], index=ind)
        (method, kwargs) = interp_methods_ind
        if method in {'cubic', 'zero'}:
            request.applymarker(pytest.mark.xfail(reason=f'{method} interpolation is not supported for TimedeltaIndex'))
        result = df[0].interpolate(method=method, **kwargs)
        expected = Series([0.0, 1.0, 2.0, 3.0], name=0, index=ind)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('ascending, expected_values', [(True, [1, 2, 3, 9, 10]), (False, [10, 9, 3, 2, 1])])
    def test_interpolate_unsorted_index(self, ascending, expected_values):
        if False:
            print('Hello World!')
        ts = Series(data=[10, 9, np.nan, 2, 1], index=[10, 9, 3, 2, 1])
        result = ts.sort_index(ascending=ascending).interpolate(method='index')
        expected = Series(data=expected_values, index=expected_values, dtype=float)
        tm.assert_series_equal(result, expected)

    def test_interpolate_asfreq_raises(self):
        if False:
            for i in range(10):
                print('nop')
        ser = Series(['a', None, 'b'], dtype=object)
        msg2 = 'Series.interpolate with object dtype'
        msg = 'Invalid fill method'
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                ser.interpolate(method='asfreq')

    def test_interpolate_fill_value(self):
        if False:
            for i in range(10):
                print('nop')
        pytest.importorskip('scipy')
        ser = Series([np.nan, 0, 1, np.nan, 3, np.nan])
        result = ser.interpolate(method='nearest', fill_value=0)
        expected = Series([np.nan, 0, 1, 1, 3, 0])
        tm.assert_series_equal(result, expected)