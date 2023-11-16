import itertools
from datetime import timedelta
import numpy as np
import pandas as pd
import pytest
from pandas import to_datetime as dt
from pandas.testing import assert_frame_equal
from arctic.multi_index import groupby_asof, fancy_group_by, insert_at
from tests.util import multi_index_df_from_arrs

def get_bitemporal_test_data():
    if False:
        for i in range(10):
            print('nop')
    sample_dates = pd.date_range('1/1/2014', periods=4, freq='D')
    sample_dates = pd.DatetimeIndex(data=sorted(itertools.chain(sample_dates, sample_dates)))
    insert_dates = pd.date_range('1/1/2015', periods=8, freq='D')
    index = pd.MultiIndex.from_arrays([sample_dates, insert_dates], names=['sample_dt', 'observed_dt'])
    prices = [[1.0, 10.0], [1.1, 10.1], [2.0, 20.0], [2.1, 20.1], [3.0, 30.0], [3.1, 30.1], [4.0, 40.0], [4.1, 40.1]]
    df = pd.DataFrame(prices, index=index, columns=['OPEN', 'CLOSE'])
    return df

def test__can_create_df_with_multiple_index():
    if False:
        for i in range(10):
            print('nop')
    ' I can make a Pandas DF with an multi-index on sampled_dt and observed_dt\n    '
    df = get_bitemporal_test_data()
    assert df.index.names == ['sample_dt', 'observed_dt']
    assert all(df.columns == ['OPEN', 'CLOSE'])
    assert len(df) == 8
    assert len(df.groupby(level='sample_dt').sum()) == 4

def test__get_ts__asof_latest():
    if False:
        for i in range(10):
            print('nop')
    ' I can get the latest known value for each sample date\n    '
    df = groupby_asof(get_bitemporal_test_data())
    assert len(df) == 4
    assert all(df['OPEN'] == [1.1, 2.1, 3.1, 4.1])
    assert all(df['CLOSE'] == [10.1, 20.1, 30.1, 40.1])

def test__get_ts__asof_datetime():
    if False:
        print('Hello World!')
    '  I can get a timeseries as-of a particular point in time\n    '
    df = groupby_asof(get_bitemporal_test_data(), as_of=dt('2015-01-05'))
    assert len(df) == 3
    assert all(df['OPEN'] == [1.1, 2.1, 3.0])
    assert all(df['CLOSE'] == [10.1, 20.1, 30.0])

def test__get_ts__unsorted_index():
    if False:
        for i in range(10):
            print('nop')
    " I can get a timeseries as-of a date when the index isn't sorted properly\n    "
    df = get_bitemporal_test_data()
    df = df.reindex(df.index[[0, 1, 3, 2, 4, 5, 6, 7]])
    df = groupby_asof(df)
    assert len(df) == 4
    assert all(df['OPEN'] == [1.1, 2.1, 3.1, 4.1])
    assert all(df['CLOSE'] == [10.1, 20.1, 30.1, 40.1])

def test_fancy_group_by_multi_index():
    if False:
        print('Hello World!')
    ts = multi_index_df_from_arrs(index_headers=('index 1', 'index 2', 'observed_dt'), index_arrs=[['2012-09-08 17:06:11.040', '2012-09-08 17:06:11.040', '2012-10-08 17:06:11.040', '2012-10-08 17:06:11.040', '2012-10-08 17:06:11.040', '2012-10-09 17:06:11.040', '2012-10-09 17:06:11.040', '2012-11-08 17:06:11.040'], ['SPAM Index', 'EGG Index', 'SPAM Index', 'SPAM Index'] + ['EGG Index', 'SPAM Index'] * 2, ['2015-01-01'] * 3 + ['2015-01-05'] + ['2015-01-01'] * 4], data_dict={'near': [1.0, 1.6, 2.0, 4.2, 2.1, 2.5, 2.6, 3.0]})
    expected_ts = multi_index_df_from_arrs(index_headers=('index 1', 'index 2'), index_arrs=[['2012-09-08 17:06:11.040', '2012-09-08 17:06:11.040', '2012-10-08 17:06:11.040', '2012-10-08 17:06:11.040', '2012-10-09 17:06:11.040', '2012-10-09 17:06:11.040', '2012-11-08 17:06:11.040'], ['EGG Index', 'SPAM Index'] * 3 + ['SPAM Index']], data_dict={'near': [1.6, 1.0, 2.1, 4.2, 2.6, 2.5, 3.0]})
    assert_frame_equal(expected_ts, groupby_asof(ts, dt_col=['index 1', 'index 2'], asof_col='observed_dt'))

def get_numeric_index_test_data():
    if False:
        while True:
            i = 10
    group_idx = sorted(4 * list(range(4)))
    agg_idx = list(range(16))
    prices = np.arange(32).reshape(16, 2) * 10
    df = pd.DataFrame(prices, index=[group_idx, agg_idx], columns=['OPEN', 'CLOSE'])
    return df

def test__minmax_last():
    if False:
        for i in range(10):
            print('nop')
    df = get_numeric_index_test_data()
    df = fancy_group_by(df, min_=3, max_=10, method='last')
    assert all(df['OPEN'] == [60, 140, 200])
    assert all(df['CLOSE'] == [70, 150, 210])

def test__minmax_first():
    if False:
        while True:
            i = 10
    df = get_numeric_index_test_data()
    df = fancy_group_by(df, min_=3, max_=10, method='first')
    assert all(df['OPEN'] == [60, 80, 160])
    assert all(df['CLOSE'] == [70, 90, 170])

def test__within_numeric_first():
    if False:
        for i in range(10):
            print('nop')
    df = get_numeric_index_test_data()
    df = fancy_group_by(df, within=5, method='first')
    assert all(df['OPEN'] == [0, 80])
    assert all(df['CLOSE'] == [10, 90])

def test__within_numeric_last():
    if False:
        i = 10
        return i + 15
    df = get_numeric_index_test_data()
    df = fancy_group_by(df, within=5, method='last')
    assert all(df['OPEN'] == [60, 120])
    assert all(df['CLOSE'] == [70, 130])

def get_datetime_index_test_data():
    if False:
        print('Hello World!')
    sample_dates = pd.DatetimeIndex(4 * [dt('1/1/2014 21:30')] + 4 * [dt('2/1/2014 21:30')] + 4 * [dt('3/1/2014 21:30')])
    observed_dates = [dt('1/1/2014 22:00'), dt('1/1/2014 22:30'), dt('2/1/2014 00:00'), dt('1/1/2015 21:30'), dt('2/1/2014 23:00'), dt('2/1/2014 23:30'), dt('3/1/2014 00:00'), dt('2/1/2015 21:30'), dt('3/1/2014 21:30'), dt('3/1/2014 22:30'), dt('4/1/2014 00:00'), dt('3/1/2015 21:30')]
    index = pd.MultiIndex.from_arrays([sample_dates, observed_dates], names=['sample_dt', 'observed_dt'])
    prices = np.arange(24).reshape(12, 2) * 10
    df = pd.DataFrame(prices, index=index, columns=['OPEN', 'CLOSE'])
    return df

def test__first_within_datetime():
    if False:
        print('Hello World!')
    " This shows the groupby function can give you a timeseries of points that were observed\n        within a rolling window of the sample time.\n        This is like saying 'give me the timeseries as it was on the day'.\n        It usually makes sense I think for the window to be the same as the sample period.\n    "
    df = get_datetime_index_test_data()
    df = fancy_group_by(df, within=timedelta(hours=1), method='first')
    assert all(df['OPEN'] == [0, 160])
    assert all(df['CLOSE'] == [10, 170])

def test__last_within_datetime():
    if False:
        i = 10
        return i + 15
    ' Last-observed variant of the above.\n    '
    df = get_datetime_index_test_data()
    df = fancy_group_by(df, within=timedelta(hours=1), method='last')
    assert all(df['OPEN'] == [20, 180])
    assert all(df['CLOSE'] == [30, 190])

def test__can_insert_row():
    if False:
        return 10
    ' I can insert a new row into a bitemp ts and it comes back when selecting the latest data\n    '
    df = get_bitemporal_test_data()
    df = insert_at(df, dt('2014-01-03'), [[9, 90]])
    assert len(df) == 9
    df = groupby_asof(df)
    assert len(df) == 4
    assert df.loc[dt('2014-01-03')]['OPEN'] == 9
    assert df.loc[dt('2014-01-03')]['CLOSE'] == 90

def test__can_append_row():
    if False:
        for i in range(10):
            print('nop')
    ' I can append a new row to a bitemp ts and it comes back when selecting the latest data\n    '
    df = get_bitemporal_test_data()
    df = insert_at(df, dt('2014-01-05'), [[9, 90]])
    assert len(df) == 9
    df = groupby_asof(df)
    assert len(df) == 5
    assert df.loc[dt('2014-01-05')]['OPEN'] == 9
    assert df.loc[dt('2014-01-05')]['CLOSE'] == 90

def test_fancy_group_by_raises():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):
        assert fancy_group_by(None, method=None)