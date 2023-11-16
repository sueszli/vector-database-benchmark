__author__ = 'saeedamen'
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from finmarketpy.economics.techindicator import TechParams, TechIndicator
tech_params = TechParams(fillna=True, atr_period=14, sma_period=3, green_n=4, green_count=9, red_n=2, red_count=13)
tech_ind = TechIndicator()
dates = pd.date_range(start='1/1/2018', end='1/08/2018')

def get_cols_name(n):
    if False:
        print('Hello World!')
    return ['Asset%d.close' % x for x in range(1, n + 1)]

def test_sma():
    if False:
        while True:
            i = 10
    indicator_name = 'SMA'
    cols = get_cols_name(1)
    data_df = pd.DataFrame(index=dates, columns=cols, data=1)
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=-1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=1)
    expected_signal_df.iloc[:tech_params.sma_period] = np.nan
    expected_df.iloc[:tech_params.sma_period - 1] = np.nan
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(1, 9)))
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=data_df.shift().values)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    expected_signal_df.iloc[:tech_params.sma_period] = np.nan
    expected_df.iloc[:tech_params.sma_period - 1] = np.nan
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    cols = get_cols_name(10)
    col_prices = np.array(range(1, 9))
    data_df = pd.DataFrame(index=dates, columns=cols, data=np.tile(col_prices, (len(cols), 1)).T)
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=data_df.shift().values)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    expected_signal_df.iloc[:tech_params.sma_period] = np.nan
    expected_df.iloc[:tech_params.sma_period - 1] = np.nan
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    cols = get_cols_name(10)
    col_prices = np.array(range(8, 0, -1))
    data_df = pd.DataFrame(index=dates, columns=cols, data=np.tile(col_prices, (len(cols), 1)).T)
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=-1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=data_df.shift().values)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    expected_signal_df.iloc[:tech_params.sma_period] = np.nan
    expected_df.iloc[:tech_params.sma_period - 1] = np.nan
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    cols = get_cols_name(1)
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(1, 9)))
    data_df.iloc[3] = np.nan
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=[np.nan, np.nan, 2, 2.67, 3.67, 4.67, 6, 7])
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    expected_signal_df.iloc[:tech_params.sma_period] = np.nan
    assert_frame_equal(df.apply(lambda x: round(x, 2)), expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(1, 9)))
    tech_params.sma_period = 20
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=np.nan)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=np.nan)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)

def test_roc():
    if False:
        print('Hello World!')
    indicator_name = 'ROC'
    cols = get_cols_name(1)
    tech_params.roc_period = 3
    data_df = pd.DataFrame(index=dates, columns=cols, data=1)
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=-1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=0)
    expected_signal_df.iloc[:tech_params.roc_period] = np.nan
    expected_df.iloc[:tech_params.roc_period] = np.nan
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    cols = get_cols_name(1)
    tech_params.roc_period = 1
    data_df = pd.DataFrame(index=dates, columns=cols, data=[1, 2, 4, 8, 16, 32, 64, 128])
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=1)
    expected_signal_df.iloc[:tech_params.roc_period] = np.nan
    expected_df.iloc[:tech_params.roc_period] = np.nan
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    tech_params.roc_period = 2
    data_df = pd.DataFrame(index=dates, columns=cols, data=[1, 2, 4, 8, 16, 32, 64, 128])
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=3)
    expected_signal_df.iloc[:tech_params.roc_period] = np.nan
    expected_df.iloc[:tech_params.roc_period] = np.nan
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    tech_params.roc_period = 3
    data_df = pd.DataFrame(index=dates, columns=cols, data=[1, 2, 4, 8, 16, 32, 64, 128])
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=7)
    expected_signal_df.iloc[:tech_params.roc_period] = np.nan
    expected_df.iloc[:tech_params.roc_period] = np.nan
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    cols = get_cols_name(1)
    tech_params.roc_period = 1
    data_df = pd.DataFrame(index=dates, columns=cols, data=[1, 2, 4, 8, 16, 32, 64, 128])
    data_df.iloc[3] = np.nan
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=[1, 1, 1, -1, 1, 1, 1, 1])
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=[1, 1, 1, 0, 3, 1, 1, 1])
    expected_signal_df.iloc[:tech_params.roc_period] = np.nan
    expected_df.iloc[:tech_params.roc_period] = np.nan
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)

def test_sma2():
    if False:
        return 10
    indicator_name = 'SMA2'
    tech_params.sma_period = 2
    tech_params.sma2_period = 3
    cols = get_cols_name(1)
    signals = ['SMA', 'SMA2']
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(1, 9)))
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, sig]) for col in cols for sig in signals], data=[[np.nan, np.nan], [1.5, np.nan], [2.5, 2.0], [3.5, 3.0], [4.5, 4.0], [5.5, 5.0], [6.5, 6.0], [7.5, 7.0]])
    expected_signal_df.iloc[:tech_params.sma2_period] = np.nan
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    cols = get_cols_name(1)
    signals = ['SMA', 'SMA2']
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(8, 0, -1)))
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=-1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, sig]) for col in cols for sig in signals], data=[[np.nan, np.nan], [7.5, np.nan], [6.5, 7.0], [5.5, 6.0], [4.5, 5.0], [3.5, 4.0], [2.5, 3.0], [1.5, 2.0]])
    expected_signal_df.iloc[:tech_params.sma2_period] = np.nan
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    cols = get_cols_name(1)
    signals = ['SMA', 'SMA2']
    data_df = pd.DataFrame(index=dates, columns=cols, data=1)
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=-1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, sig]) for col in cols for sig in signals], data=np.tile(np.ones(len(dates)), (2, 1)).T)
    expected_signal_df.iloc[:tech_params.sma2_period] = np.nan
    expected_df.iloc[:tech_params.sma2_period - 1] = np.nan
    expected_df.set_value('2018-01-02', 'Asset1.close SMA', 1.0)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)

def test_polarity():
    if False:
        for i in range(10):
            print('nop')
    indicator_name = 'polarity'
    cols = get_cols_name(1)
    data_df = pd.DataFrame(index=dates, columns=cols, data=1)
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, 'Polarity', 'Signal']) for col in cols], data=1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, 'Polarity']) for col in cols], data=1)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df, check_dtype=False)
    cols = get_cols_name(1)
    data_df = pd.DataFrame(index=dates, columns=cols, data=[1, 2, 4, 8, 16, 32, 64, 128])
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, 'Polarity', 'Signal']) for col in cols], data=1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, 'Polarity']) for col in cols], data=data_df.values)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df, check_dtype=False)
    cols = get_cols_name(1)
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(-1, -9, -1)))
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, 'Polarity', 'Signal']) for col in cols], data=-1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, 'Polarity']) for col in cols], data=data_df.values)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df, check_dtype=False)
    cols = get_cols_name(1)
    tech_params.roc_period = 1
    data_df = pd.DataFrame(index=dates, columns=cols, data=[1, 2, 4, 8, 16, 32, 64, 128])
    data_df.iloc[3] = np.nan
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, 'Polarity', 'Signal']) for col in cols], data=1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, 'Polarity']) for col in cols], data=[1.0, 2.0, 4.0, 4.0, 16.0, 32.0, 64.0, 128.0])
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df, check_dtype=False)

def test_attr():
    if False:
        print('Hello World!')
    'Testing of attributes such as long only\n    '
    indicator_name = 'SMA'
    tech_params = TechParams()
    cols = get_cols_name(1)
    tech_params.sma_period = 3
    tech_params.only_allow_longs = True
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(1, 9)))
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=data_df.shift().values)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    expected_signal_df.iloc[:tech_params.sma_period] = np.nan
    expected_df.iloc[:tech_params.sma_period - 1] = np.nan
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(8, 0, -1)))
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=0)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=data_df.shift().values)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    expected_signal_df.iloc[:tech_params.sma_period] = np.nan
    expected_df.iloc[:tech_params.sma_period - 1] = np.nan
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    tech_params = TechParams()
    tech_params.sma_period = 3
    tech_params.only_allow_shorts = True
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(1, 9)))
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=0)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=data_df.shift().values)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    expected_signal_df.iloc[:tech_params.sma_period] = np.nan
    expected_df.iloc[:tech_params.sma_period - 1] = np.nan
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(8, 0, -1)))
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=-1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=data_df.shift().values)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    expected_signal_df.iloc[:tech_params.sma_period] = np.nan
    expected_df.iloc[:tech_params.sma_period - 1] = np.nan
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    tech_params = TechParams()
    tech_params.only_allow_shorts = True
    with pytest.raises(Exception):
        tech_params.only_allow_longs = True
    tech_params = TechParams()
    tech_params.only_allow_longs = True
    with pytest.raises(Exception):
        tech_params.only_allow_shorts = True
    tech_params = TechParams()
    tech_params.sma_period = 3
    tech_params.strip_signal_name = True
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(1, 9)))
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=cols, data=1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=data_df.shift().values)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    expected_signal_df.iloc[:tech_params.sma_period] = np.nan
    expected_df.iloc[:tech_params.sma_period - 1] = np.nan
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    tech_params = TechParams()
    tech_params.sma_period = 3
    tech_params.signal_mult = -1
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(1, 9)))
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=-1)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=data_df.shift().values)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    expected_signal_df.iloc[:tech_params.sma_period] = np.nan
    expected_df.iloc[:tech_params.sma_period - 1] = np.nan
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    tech_params.signal_mult = 0
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(1, 9)))
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=0)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=data_df.shift().values)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    expected_signal_df.iloc[:tech_params.sma_period] = np.nan
    expected_df.iloc[:tech_params.sma_period - 1] = np.nan
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)
    tech_params.signal_mult = 3
    data_df = pd.DataFrame(index=dates, columns=cols, data=list(range(1, 9)))
    tech_ind.create_tech_ind(data_df, indicator_name, tech_params)
    expected_signal_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name, 'Signal']) for col in cols], data=3)
    expected_df = pd.DataFrame(index=dates, columns=[' '.join([col, indicator_name]) for col in cols], data=data_df.shift().values)
    df = tech_ind.get_techind()
    signal_df = tech_ind.get_signal()
    expected_signal_df.iloc[:tech_params.sma_period] = np.nan
    expected_df.iloc[:tech_params.sma_period - 1] = np.nan
    assert_frame_equal(df, expected_df)
    assert_frame_equal(signal_df, expected_signal_df)