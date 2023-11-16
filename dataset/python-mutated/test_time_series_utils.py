import numpy as np
import pandas as pd
import pytest
from pycaret.utils.time_series import SeasonalPeriod, clean_time_index, remove_harmonics_from_sp

def test_harmonic_removal():
    if False:
        for i in range(10):
            print('nop')
    'Tests the removal of harmonics'
    results = remove_harmonics_from_sp([2, 51, 5])
    assert results == [2, 51, 5]
    results = remove_harmonics_from_sp([2, 52, 3])
    assert results == [52, 3]
    results = remove_harmonics_from_sp([50, 3, 11, 100, 39])
    assert results == [11, 100, 39]
    results = remove_harmonics_from_sp([2, 3, 4, 50])
    assert results == [3, 4, 50]
    results = remove_harmonics_from_sp([2, 3, 4, 50], harmonic_order_method='harmonic_max')
    assert results == [50, 3, 4]
    results = remove_harmonics_from_sp([2, 3, 4, 50], harmonic_order_method='harmonic_strength')
    assert results == [4, 3, 50]
    results = remove_harmonics_from_sp([3, 2, 6, 50])
    assert results == [6, 50]
    results = remove_harmonics_from_sp([3, 2, 6, 50], harmonic_order_method='harmonic_max')
    assert results == [6, 50]
    results = remove_harmonics_from_sp([2, 3, 6, 50], harmonic_order_method='harmonic_max')
    assert results == [50, 6]
    results = remove_harmonics_from_sp([3, 2, 6, 50], harmonic_order_method='harmonic_strength')
    assert results == [6, 50]
    results = remove_harmonics_from_sp([2, 3, 6, 50], harmonic_order_method='harmonic_strength')
    assert results == [6, 50]
    results = remove_harmonics_from_sp([10, 20, 30, 40, 50, 60], harmonic_order_method='harmonic_strength')
    assert results == [20, 40, 60, 50]
    results = remove_harmonics_from_sp([10, 20, 30, 40, 50, 60], harmonic_order_method='harmonic_max')
    assert results == [60, 40, 50]
    results = remove_harmonics_from_sp([50, 100, 150, 49, 200, 51, 23, 27, 10, 250])
    assert results == [150, 49, 200, 51, 23, 27, 250]
    results = remove_harmonics_from_sp([49, 98, 18])
    assert results == [98, 18]
    results = remove_harmonics_from_sp([50, 16, 15, 17, 34, 2, 33, 49, 18, 100, 32])
    assert results == [15, 34, 33, 49, 18, 100, 32]

def _get_seasonal_keys():
    if False:
        return 10
    return [freq for (freq, _) in SeasonalPeriod.__members__.items()]

@pytest.mark.parametrize('freq', _get_seasonal_keys())
@pytest.mark.parametrize('index', [True, False])
def test_clean_time_index_datetime(freq, index):
    if False:
        for i in range(10):
            print('nop')
    'Test clean_time_index utility when index/column is of type DateTime'
    dates = pd.date_range('2019-01-01', '2022-01-30', freq=freq)
    if len(dates) > 100:
        dates = dates[:100]
    assert len(dates) >= 3
    data = pd.DataFrame({'date': dates, 'value': np.random.rand(len(dates))})
    if index:
        data.set_index('date', inplace=True)
        index_col = None
    else:
        index_col = 'date'
    try:
        cleaned = clean_time_index(data=data, index_col=index_col, freq=freq)
    except AttributeError:
        return
    assert len(cleaned) == len(data)
    data_missing = data.copy()
    data_missing = data_missing.drop(data_missing.index[1])
    cleaned = clean_time_index(data=data_missing, index_col=index_col, freq=freq)
    assert len(cleaned) == len(data)

@pytest.mark.parametrize('freq', _get_seasonal_keys())
@pytest.mark.parametrize('index', [False])
def test_clean_time_index_str_datetime(freq, index):
    if False:
        while True:
            i = 10
    'Test clean_time_index utility when index/column is of type str in format\n    acceptable to DatetimeIndex\n\n    NOTE: Index can not be string (only column). Code unchanges, just parameter\n    restricted to False\n    '
    dates = pd.date_range('2019-01-01 00:00:00', '2022-01-30 00:00:00', freq=freq)
    if len(dates) > 100:
        dates = dates[:100]
    assert len(dates) >= 3
    dates = dates.strftime('%Y-%m-%d %H:%M:%S')
    data = pd.DataFrame({'date': dates, 'value': np.random.rand(len(dates))})
    if index:
        data.set_index('date', inplace=True)
        index_col = None
    else:
        index_col = 'date'
    try:
        cleaned = clean_time_index(data=data, index_col=index_col, freq=freq)
    except AttributeError:
        return
    assert len(cleaned) == len(data)
    data_missing = data.copy()
    data_missing = data_missing.drop(data_missing.index[1])
    cleaned = clean_time_index(data=data_missing, index_col=index_col, freq=freq)
    assert len(cleaned) == len(data)

@pytest.mark.parametrize('freq', _get_seasonal_keys())
@pytest.mark.parametrize('index', [True, False])
def test_clean_time_index_period(freq, index):
    if False:
        return 10
    'Test clean_time_index utility when index/column is of type Period'
    try:
        dates = pd.period_range('2019-01-01', '2022-01-30', freq=freq)
    except ValueError:
        return
    if len(dates) > 100:
        dates = dates[:100]
    assert len(dates) >= 3
    data = pd.DataFrame({'date': dates, 'value': np.random.rand(len(dates))})
    if index:
        data.set_index('date', inplace=True)
        index_col = None
    else:
        index_col = 'date'
    cleaned = clean_time_index(data=data, index_col=index_col, freq=freq)
    assert len(cleaned) == len(data)
    data_missing = data.copy()
    data_missing = data_missing.drop(data_missing.index[1])
    cleaned = clean_time_index(data=data_missing, index_col=index_col, freq=freq)
    assert len(cleaned) == len(data)

@pytest.mark.parametrize('freq', _get_seasonal_keys())
@pytest.mark.parametrize('index', [False])
def test_clean_time_index_str_period(freq, index):
    if False:
        for i in range(10):
            print('nop')
    'Test clean_time_index utility when index/column is of type str in format\n    acceptable to PeriodIndex\n\n    NOTE: Index can not be string (only column). Code unchanges, just parameter\n    restricted to False\n    '
    try:
        dates = pd.period_range('2019-01-01', '2022-01-30', freq=freq)
    except ValueError:
        return
    if len(dates) > 100:
        dates = dates[:100]
    assert len(dates) >= 3
    dates = dates.astype(str)
    data = pd.DataFrame({'date': dates, 'value': np.random.rand(len(dates))})
    if index:
        data.set_index('date', inplace=True)
        index_col = None
    else:
        index_col = 'date'
    cleaned = clean_time_index(data=data, index_col=index_col, freq=freq)
    assert len(cleaned) == len(data)
    data_missing = data.copy()
    data_missing = data_missing.drop(data_missing.index[1])
    cleaned = clean_time_index(data=data_missing, index_col=index_col, freq=freq)
    assert len(cleaned) == len(data)

@pytest.mark.parametrize('freq', _get_seasonal_keys())
@pytest.mark.parametrize('index', [True, False])
def test_clean_time_index_int(freq, index):
    if False:
        while True:
            i = 10
    'Test clean_time_index utility when index/column is of type Int'
    dates = np.arange(100)
    if len(dates) > 100:
        dates = dates[:100]
    assert len(dates) >= 3
    data = pd.DataFrame({'date': dates, 'value': np.random.rand(len(dates))})
    if index:
        data.set_index('date', inplace=True)
        index_col = None
    else:
        index_col = 'date'
    cleaned = clean_time_index(data=data, index_col=index_col, freq=freq)
    assert len(cleaned) == len(data)
    data_missing = data.copy()
    data_missing = data_missing.drop(data_missing.index[1])
    cleaned = clean_time_index(data=data_missing, index_col=index_col, freq=freq)
    assert len(cleaned) == len(data) - 1