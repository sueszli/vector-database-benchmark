"""Module to test the datetime utility functions
"""
import numpy as np
import pandas as pd
from pycaret.datasets import get_data
from pycaret.utils.datetime import coerce_datetime_to_period_index, coerce_period_to_datetime_index

def test_coerce_period_to_datetime_index():
    if False:
        i = 10
        return i + 15
    'Tests coercion of PeriodIndex to DatetimeIndex'
    data = get_data('airline')
    orig_freq = data.index.freq
    new_data = coerce_period_to_datetime_index(data=data)
    assert isinstance(new_data.index, pd.DatetimeIndex)
    assert new_data.index.freq == orig_freq
    data_diff_freq = data.copy()
    data_diff_freq = data_diff_freq.asfreq('D')
    new_data = coerce_period_to_datetime_index(data=data_diff_freq, freq=orig_freq)
    assert isinstance(new_data.index, pd.DatetimeIndex)
    assert new_data.index.freq == orig_freq
    assert isinstance(data.index, pd.PeriodIndex)
    coerce_period_to_datetime_index(data=data, inplace=True)
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.freq == orig_freq
    data_np = np.array(data.values)
    assert isinstance(data_np, np.ndarray)
    data_np_new = coerce_period_to_datetime_index(data=data_np)
    assert isinstance(data_np_new, np.ndarray)
    data = get_data('uschange')
    original_index_type = type(data.index)
    new_data = coerce_period_to_datetime_index(data=data)
    assert isinstance(new_data.index, original_index_type)
    orig_freq = 'Q-DEC'
    data = pd.DataFrame([1, 2], index=pd.PeriodIndex(['2018Q2', '2018Q3'], freq=orig_freq))
    new_data = coerce_period_to_datetime_index(data=data)
    assert isinstance(new_data.index, pd.DatetimeIndex)
    assert new_data.index.freq == orig_freq

def test_coerce_datetime_to_period_index():
    if False:
        while True:
            i = 10
    'Tests coercion of DatetimeIndex to PeriodIndex\n    Note since we are converting from a period to Datetime,\n    there is no guarantee of the frequency unless we explicitly\n    pass it. e.g. DateTime could be MonthStart, but Period will\n    represent Month.\n    '
    data = get_data('airline')
    data.index = data.index.to_timestamp()
    new_data = coerce_datetime_to_period_index(data=data)
    assert isinstance(new_data.index, pd.PeriodIndex)
    new_data = coerce_datetime_to_period_index(data=data, freq='D')
    assert isinstance(new_data.index, pd.PeriodIndex)
    assert new_data.index.freq == 'D'
    assert isinstance(data.index, pd.DatetimeIndex)
    coerce_datetime_to_period_index(data=data, inplace=True)
    assert isinstance(data.index, pd.PeriodIndex)
    data_np = np.array(data.values)
    assert isinstance(data_np, np.ndarray)
    data_np_new = coerce_datetime_to_period_index(data=data_np)
    assert isinstance(data_np_new, np.ndarray)
    data = get_data('uschange')
    original_index_type = type(data.index)
    new_data = coerce_datetime_to_period_index(data=data)
    assert isinstance(new_data.index, original_index_type)