"""
Test index support in time series models

1. Test support for passing / constructing the underlying index in __init__
2. Test wrapping of output using the underlying index
3. Test wrapping of prediction / forecasting using the underlying index or
   extensions of it.

Author: Chad Fulton
License: BSD-3
"""
from statsmodels.compat.pandas import is_int_index
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base import tsa_model
nobs = 5
base_dta = np.arange(nobs)
dta = [base_dta.tolist(), base_dta, pd.Series(base_dta), pd.DataFrame(base_dta)]
base_date_indexes = [pd.date_range(start='1950-01-01', periods=nobs, freq='D'), pd.date_range(start='1950-01-01', periods=nobs, freq='W'), pd.date_range(start='1950-01-01', periods=nobs, freq='MS'), pd.date_range(start='1950-01-01', periods=nobs, freq='QS'), pd.date_range(start='1950-01-01', periods=nobs, freq='Y'), pd.date_range(start='1950-01-01', periods=nobs, freq='2Q-DEC'), pd.date_range(start='1950-01-01', periods=nobs, freq='2QS'), pd.date_range(start='1950-01-01', periods=nobs, freq='5s'), pd.date_range(start='1950-01-01', periods=nobs, freq='1D10min')]
base_period_indexes = [pd.period_range(start='1950-01-01', periods=nobs, freq='D'), pd.period_range(start='1950-01-01', periods=nobs, freq='W'), pd.period_range(start='1950-01-01', periods=nobs, freq='M'), pd.period_range(start='1950-01-01', periods=nobs, freq='Q'), pd.period_range(start='1950-01-01', periods=nobs, freq='Y')]
try:
    base_period_indexes += [pd.period_range(start='1950-01-01', periods=nobs, freq='2Q'), pd.period_range(start='1950-01-01', periods=nobs, freq='5s'), pd.period_range(start='1950-01-01', periods=nobs, freq='1D10min')]
except AttributeError:
    pass
date_indexes = [(x, None) for x in base_date_indexes]
period_indexes = [(x, None) for x in base_period_indexes]
numpy_datestr_indexes = [(x.map(str), x.freq) for x in base_date_indexes]
list_datestr_indexes = [(x.tolist(), y) for (x, y) in numpy_datestr_indexes]
series_datestr_indexes = [(pd.Series(x), y) for (x, y) in list_datestr_indexes]
numpy_datetime_indexes = [(pd.to_datetime(x).to_pydatetime(), x.freq) for x in base_date_indexes]
list_datetime_indexes = [(x.tolist(), y) for (x, y) in numpy_datetime_indexes]
series_datetime_indexes = [(pd.Series(x, dtype=object), y) for (x, y) in list_datetime_indexes]
series_timestamp_indexes = [(pd.Series(x), x.freq) for x in base_date_indexes]
supported_increment_indexes = [(pd.Index(np.arange(nobs)), None), (pd.RangeIndex(start=0, stop=nobs, step=1), None), (pd.RangeIndex(start=-5, stop=nobs - 5, step=1), None), (pd.RangeIndex(start=0, stop=nobs * 6, step=6), None)]
supported_date_indexes = numpy_datestr_indexes + list_datestr_indexes + series_datestr_indexes + numpy_datetime_indexes + list_datetime_indexes + series_datetime_indexes + series_timestamp_indexes
unsupported_indexes = [(np.arange(1, nobs + 1), None), (np.arange(nobs)[::-1], None), (np.arange(nobs) * 1.0, None), ([x for x in 'abcde'], None), ([str, 1, 'a', -30.1, {}], None)]
unsupported_date_indexes = [(['1950', '1952', '1941', '1954', '1991'], None), (['1950-01-01', '1950-01-02', '1950-01-03', '1950-01-04', '1950-01-06'], None)]

def test_instantiation_valid():
    if False:
        return 10
    tsa_model.__warningregistry__ = {}
    for endog in dta[:2]:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            mod = tsa_model.TimeSeriesModel(endog)
            assert isinstance(mod._index, pd.RangeIndex) or np.issubdtype(mod._index.dtype, np.integer)
            assert_equal(mod._index_none, True)
            assert_equal(mod._index_dates, False)
            assert_equal(mod._index_generated, True)
            assert_equal(mod.data.dates, None)
            assert_equal(mod.data.freq, None)
    for endog in dta:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for (ix, freq) in date_indexes + period_indexes:
                mod = tsa_model.TimeSeriesModel(endog, dates=ix)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert_equal(isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for (ix, freq) in date_indexes + period_indexes:
                mod = tsa_model.TimeSeriesModel(endog, dates=ix, freq=freq)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert_equal(isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for (ix, freq) in supported_date_indexes:
                mod = tsa_model.TimeSeriesModel(endog, dates=ix, freq=freq)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert_equal(isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)
        for (ix, freq) in supported_increment_indexes + unsupported_indexes:
            assert_raises(ValueError, tsa_model.TimeSeriesModel, endog, dates=ix)
    for base_endog in dta[2:4]:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for (ix, freq) in date_indexes + period_indexes:
                endog = base_endog.copy()
                endog.index = ix
                mod = tsa_model.TimeSeriesModel(endog)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert_equal(isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)
        endog = base_endog.copy()
        endog.index = supported_increment_indexes[0][0]
        mod = tsa_model.TimeSeriesModel(endog)
        assert is_int_index(mod._index)
        assert_equal(mod._index_none, False)
        assert_equal(mod._index_dates, False)
        assert_equal(mod._index_generated, False)
        assert_equal(mod._index_freq, None)
        assert_equal(mod.data.dates, None)
        assert_equal(mod.data.freq, None)
        endog = base_endog.copy()
        endog.index = supported_increment_indexes[1][0]
        mod = tsa_model.TimeSeriesModel(endog)
        assert type(mod._index) is pd.RangeIndex
        assert not mod._index_none
        assert not mod._index_dates
        assert not mod._index_generated
        assert mod._index_freq is None
        assert mod.data.dates is None
        assert mod.data.freq is None
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for (ix, freq) in supported_date_indexes:
                endog = base_endog.copy()
                endog.index = ix
                mod = tsa_model.TimeSeriesModel(endog, freq=freq)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert_equal(isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for (ix, freq) in supported_date_indexes:
                endog = base_endog.copy()
                endog.index = unsupported_indexes[0][0]
                mod = tsa_model.TimeSeriesModel(endog, dates=ix, freq=freq)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert_equal(isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)
        message = 'No frequency information was provided, so inferred frequency %s will be used.'
        last_len = 0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            for (ix, freq) in supported_date_indexes:
                endog = base_endog.copy()
                endog.index = ix
                mod = tsa_model.TimeSeriesModel(endog)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert type(mod._index) is pd.DatetimeIndex
                assert not mod._index_none
                assert mod._index_dates
                assert not mod._index_generated
                assert_equal(mod._index.freq, mod._index_freq)
                assert mod.data.dates.equals(mod._index)
                if len(w) == last_len:
                    continue
                assert_equal(mod.data.freq.split('-')[0], freq.split('-')[0])
                assert_equal(str(w[-1].message), message % mod.data.freq)
                last_len = len(w)
        message = 'An unsupported index was provided and will be ignored when e.g. forecasting.'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            for (ix, freq) in unsupported_indexes:
                endog = base_endog.copy()
                endog.index = ix
                mod = tsa_model.TimeSeriesModel(endog)
                assert_equal(isinstance(mod._index, (pd.Index, pd.RangeIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, False)
                assert_equal(mod._index_generated, True)
                assert_equal(mod._index_freq, None)
                assert_equal(mod.data.dates, None)
                assert_equal(mod.data.freq, None)
                assert_equal(str(w[0].message), message)
        message = 'A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            for (ix, freq) in unsupported_date_indexes:
                endog = base_endog.copy()
                endog.index = ix
                mod = tsa_model.TimeSeriesModel(endog)
                assert isinstance(mod._index, pd.RangeIndex) or is_int_index(mod._index)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, False)
                assert_equal(mod._index_generated, True)
                assert_equal(mod._index_freq, None)
                assert_equal(mod.data.dates, None)
                assert_equal(mod.data.freq, None)
                assert_equal(str(w[0].message), message)
    endog = dta[0]
    assert_raises(ValueError, tsa_model.TimeSeriesModel, endog, freq=date_indexes[1][0].freq)
    endog = dta[2].copy()
    endog.index = date_indexes[0][0]
    assert_raises(ValueError, tsa_model.TimeSeriesModel, endog, freq=date_indexes[1][0].freq)
    endog = dta[2].copy()
    endog.index = unsupported_indexes[0][0]
    assert_raises(ValueError, tsa_model.TimeSeriesModel, endog, freq=date_indexes[1][0].freq)
    endog = dta[2].copy()
    endog.index = numpy_datestr_indexes[0][0]
    assert_raises(ValueError, tsa_model.TimeSeriesModel, endog, freq=date_indexes[1][0].freq)

def test_prediction_increment_unsupported():
    if False:
        i = 10
        return i + 15
    endog = dta[2].copy()
    endog.index = unsupported_indexes[-2][0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')
        mod = tsa_model.TimeSeriesModel(endog)
    start_key = 0
    end_key = None
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index.equals(mod.data.row_labels), True)
    start_key = -2
    end_key = -1
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index.equals(mod.data.row_labels[3:]), True)
    start_key = 1
    end_key = nobs
    message = 'No supported index is available. Prediction results will be given with an integer index beginning at `start`.'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
        assert_equal(str(w[0].message), message)
    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    assert_equal(prediction_index.equals(pd.Index(np.arange(1, 6))), True)
    (loc, index, index_was_expanded) = mod._get_index_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.RangeIndex(start=0, stop=3, step=1)
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)
    (loc, index, index_was_expanded) = mod._get_index_label_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.RangeIndex(start=0, stop=3, step=1)
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)
    (loc, index, index_was_expanded) = mod._get_index_label_loc('c')
    assert_equal(loc, 2)
    desired_index = mod.data.row_labels[:3]
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)

def test_prediction_increment_nonpandas():
    if False:
        for i in range(10):
            print('nop')
    endog = dta[0]
    mod = tsa_model.TimeSeriesModel(endog)
    start_key = 0
    end_key = None
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index is None, True)
    start_key = -2
    end_key = -1
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index is None, True)
    start_key = 1
    end_key = nobs
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    assert_equal(prediction_index is None, True)
    (loc, index, index_was_expanded) = mod._get_index_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.RangeIndex(start=0, stop=3, step=1)
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)
    (loc, index, index_was_expanded) = mod._get_index_label_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.RangeIndex(start=0, stop=3, step=1)
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)

def test_prediction_increment_pandas_noindex():
    if False:
        for i in range(10):
            print('nop')
    endog = dta[2].copy()
    mod = tsa_model.TimeSeriesModel(endog)
    start_key = 0
    end_key = None
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index.equals(mod._index), True)
    start_key = -2
    end_key = -1
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index.equals(mod._index[3:]), True)
    start_key = 1
    end_key = nobs
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    assert_equal(prediction_index.equals(pd.Index(np.arange(1, 6))), True)

def test_prediction_increment_pandas_dates_daily():
    if False:
        i = 10
        return i + 15
    endog = dta[2].copy()
    endog.index = date_indexes[0][0]
    mod = tsa_model.TimeSeriesModel(endog)
    start_key = 0
    end_key = None
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index), True)
    start_key = 0
    end_key = 3
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, 3)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[:4]), True)
    start_key = -2
    end_key = -1
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[3:]), True)
    start_key = 1
    end_key = nobs
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    desired_index = pd.date_range(start='1950-01-02', periods=5, freq='D')
    assert_equal(prediction_index.equals(desired_index), True)
    start_key = '1950-01-02'
    end_key = '1950-01-04'
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 1)
    assert_equal(end, 3)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[1:4]), True)
    start_key = '1950-01-01'
    end_key = '1950-01-08'
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 3)
    desired_index = pd.date_range(start='1950-01-01', periods=8, freq='D')
    assert_equal(prediction_index.equals(desired_index), True)
    (loc, index, index_was_expanded) = mod._get_index_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.date_range(start='1950-01-01', periods=3, freq='D')
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)
    (loc, index, index_was_expanded) = mod._get_index_label_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.date_range(start='1950-01-01', periods=3, freq='D')
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)
    (loc, index, index_was_expanded) = mod._get_index_label_loc('1950-01-03')
    assert_equal(loc, 2)
    desired_index = mod.data.row_labels[:3]
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)

def test_prediction_increment_pandas_dates_monthly():
    if False:
        print('Hello World!')
    endog = dta[2].copy()
    endog.index = date_indexes[2][0]
    mod = tsa_model.TimeSeriesModel(endog)
    start_key = 0
    end_key = None
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index), True)
    start_key = 0
    end_key = 3
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, 3)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[:4]), True)
    start_key = -2
    end_key = -1
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[3:]), True)
    start_key = 1
    end_key = nobs
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    desired_index = pd.date_range(start='1950-02', periods=5, freq='MS')
    assert_equal(prediction_index.equals(desired_index), True)
    start_key = '1950-02'
    end_key = '1950-04'
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 1)
    assert_equal(end, 3)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[1:4]), True)
    start_key = '1950-01'
    end_key = '1950-08'
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 3)
    desired_index = pd.date_range(start='1950-01', periods=8, freq='MS')
    assert_equal(prediction_index.equals(desired_index), True)
    (loc, index, index_was_expanded) = mod._get_index_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.date_range(start='1950-01', periods=3, freq='MS')
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)
    (loc, index, index_was_expanded) = mod._get_index_label_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.date_range(start='1950-01', periods=3, freq='MS')
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)
    (loc, index, index_was_expanded) = mod._get_index_label_loc('1950-03')
    assert_equal(loc, slice(2, 3, None))
    desired_index = mod.data.row_labels[:3]
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)

def test_prediction_increment_pandas_dates_nanosecond():
    if False:
        while True:
            i = 10
    endog = dta[2].copy()
    endog.index = pd.date_range(start='1970-01-01', periods=len(endog), freq='ns')
    mod = tsa_model.TimeSeriesModel(endog)
    start_key = 0
    end_key = None
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index), True)
    start_key = -2
    end_key = -1
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[3:]), True)
    start_key = 1
    end_key = nobs
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    desired_index = pd.date_range(start='1970-01-01', periods=6, freq='ns')[1:]
    assert_equal(prediction_index.equals(desired_index), True)
    start_key = pd.Timestamp('1970-01-01')
    end_key = pd.Timestamp(start_key.value + 7)
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 3)
    desired_index = pd.date_range(start='1970-01-01', periods=8, freq='ns')
    assert_equal(prediction_index.equals(desired_index), True)

def test_range_index():
    if False:
        return 10
    tsa_model.__warningregistry__ = {}
    endog = pd.Series(np.random.normal(size=5))
    assert_equal(isinstance(endog.index, pd.RangeIndex), True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        mod = tsa_model.TimeSeriesModel(endog)
        assert_equal(len(w), 0)

def test_prediction_rangeindex():
    if False:
        while True:
            i = 10
    index = supported_increment_indexes[2][0]
    endog = pd.Series(dta[0], index=index)
    mod = tsa_model.TimeSeriesModel(endog)
    start_key = 0
    end_key = None
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    desired_index = pd.RangeIndex(start=-5, stop=0, step=1)
    assert_equal(prediction_index.equals(desired_index), True)
    start_key = -2
    end_key = -1
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    desired_index = pd.RangeIndex(start=-2, stop=0, step=1)
    assert_equal(prediction_index.equals(desired_index), True)
    start_key = 1
    end_key = nobs
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    desired_index = pd.RangeIndex(start=-4, stop=1, step=1)
    assert_equal(prediction_index.equals(desired_index), True)

def test_prediction_rangeindex_withstep():
    if False:
        i = 10
        return i + 15
    index = supported_increment_indexes[3][0]
    endog = pd.Series(dta[0], index=index)
    mod = tsa_model.TimeSeriesModel(endog)
    start_key = 0
    end_key = None
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    desired_index = pd.RangeIndex(start=0, stop=nobs * 6, step=6)
    assert_equal(prediction_index.equals(desired_index), True)
    start_key = -2
    end_key = -1
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    desired_index = pd.RangeIndex(start=3 * 6, stop=nobs * 6, step=6)
    assert_equal(prediction_index.equals(desired_index), True)
    start_key = 1
    end_key = nobs
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    desired_index = pd.RangeIndex(start=1 * 6, stop=(nobs + 1) * 6, step=6)
    assert_equal(prediction_index.equals(desired_index), True)
    (loc, index, index_was_expanded) = mod._get_index_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.RangeIndex(start=0, stop=3 * 6, step=6)
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)

def test_custom_index():
    if False:
        print('Hello World!')
    tsa_model.__warningregistry__ = {}
    endog = pd.Series(np.random.normal(size=5), index=['a', 'b', 'c', 'd', 'e'])
    message = 'An unsupported index was provided and will be ignored when e.g. forecasting.'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        mod = tsa_model.TimeSeriesModel(endog)
        messages = [str(warn.message) for warn in w]
        assert message in messages
    start_key = -2
    end_key = -1
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
    assert_equal(prediction_index.equals(pd.Index(['d', 'e'])), True)
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key, index=['f', 'g'])
    assert_equal(prediction_index.equals(pd.Index(['f', 'g'])), True)
    (loc, index, index_was_expanded) = mod._get_index_loc(2)
    assert_equal(loc, 2)
    assert_equal(index.equals(pd.RangeIndex(0, 3)), True)
    assert_equal(index_was_expanded, False)
    assert_equal(index_was_expanded, False)
    with pytest.raises(KeyError):
        mod._get_index_loc('c')
    (loc, index, index_was_expanded) = mod._get_index_label_loc('c')
    assert_equal(loc, 2)
    assert_equal(index.equals(pd.Index(['a', 'b', 'c'])), True)
    assert_equal(index_was_expanded, False)
    with pytest.raises(KeyError):
        mod._get_index_label_loc('aa')
    start_key = 4
    end_key = 5
    message = 'No supported index is available. Prediction results will be given with an integer index beginning at `start`.'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key)
        assert_equal(prediction_index.equals(pd.Index([4, 5])), True)
        assert_equal(str(w[0].message), message)
    (start, end, out_of_sample, prediction_index) = mod._get_prediction_index(start_key, end_key, index=['f', 'g'])
    assert_equal(prediction_index.equals(pd.Index(['f', 'g'])), True)
    assert_raises(ValueError, mod._get_prediction_index, start_key, end_key, index=['f', 'g', 'h'])

def test_nonmonotonic_periodindex():
    if False:
        while True:
            i = 10
    tmp = pd.period_range(start=2000, end=2002, freq='Y')
    index = tmp.tolist() + tmp.tolist()
    endog = pd.Series(np.zeros(len(index)), index=index)
    message = 'A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.'
    with pytest.warns(ValueWarning, match=message):
        tsa_model.TimeSeriesModel(endog)

@pytest.mark.xfail(reason='Pandas PeriodIndex.is_full does not yet work for all frequencies (e.g. frequencies with a multiplier, like "2Q").')
def test_nonfull_periodindex():
    if False:
        while True:
            i = 10
    index = pd.PeriodIndex(['2000-01', '2000-03'], freq='M')
    endog = pd.Series(np.zeros(len(index)), index=index)
    message = 'A Period index has been provided, but it is not full and so will be ignored when e.g. forecasting.'
    with pytest.warns(ValueWarning, match=message):
        tsa_model.TimeSeriesModel(endog)

def test_get_index_loc_quarterly():
    if False:
        while True:
            i = 10
    ix = pd.date_range('2000Q1', periods=8, freq='QS')
    endog = pd.Series(np.zeros(8), index=ix)
    mod = tsa_model.TimeSeriesModel(endog)
    (loc, index, _) = mod._get_index_loc('2003Q2')
    assert_equal(index[loc], pd.Timestamp('2003Q2'))