from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import DataFrame, Index, Series, TimedeltaIndex, Timestamp
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range

@pytest.fixture
def test_frame():
    if False:
        return 10
    return DataFrame({'A': [1] * 20 + [2] * 12 + [3] * 8, 'B': np.arange(40)}, index=date_range('1/1/2000', freq='s', periods=40))

def test_tab_complete_ipython6_warning(ip):
    if False:
        print('Hello World!')
    from IPython.core.completer import provisionalcompleter
    code = dedent('    import pandas._testing as tm\n    s = tm.makeTimeSeries()\n    rs = s.resample("D")\n    ')
    ip.run_cell(code)
    with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
        with provisionalcompleter('ignore'):
            list(ip.Completer.completions('rs.', 1))

def test_deferred_with_groupby():
    if False:
        while True:
            i = 10
    data = [['2010-01-01', 'A', 2], ['2010-01-02', 'A', 3], ['2010-01-05', 'A', 8], ['2010-01-10', 'A', 7], ['2010-01-13', 'A', 3], ['2010-01-01', 'B', 5], ['2010-01-03', 'B', 2], ['2010-01-04', 'B', 1], ['2010-01-11', 'B', 7], ['2010-01-14', 'B', 3]]
    df = DataFrame(data, columns=['date', 'id', 'score'])
    df.date = pd.to_datetime(df.date)

    def f_0(x):
        if False:
            print('Hello World!')
        return x.set_index('date').resample('D').asfreq()
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby('id').apply(f_0)
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.set_index('date').groupby('id').resample('D').asfreq()
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'date': date_range(start='2016-01-01', periods=4, freq='W'), 'group': [1, 1, 2, 2], 'val': [5, 6, 7, 8]}).set_index('date')

    def f_1(x):
        if False:
            print('Hello World!')
        return x.resample('1D').ffill()
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby('group').apply(f_1)
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby('group').resample('1D').ffill()
    tm.assert_frame_equal(result, expected)

def test_getitem(test_frame):
    if False:
        for i in range(10):
            print('nop')
    g = test_frame.groupby('A')
    expected = g.B.apply(lambda x: x.resample('2s').mean())
    result = g.resample('2s').B.mean()
    tm.assert_series_equal(result, expected)
    result = g.B.resample('2s').mean()
    tm.assert_series_equal(result, expected)
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = g.resample('2s').mean().B
    tm.assert_series_equal(result, expected)

def test_getitem_multiple():
    if False:
        i = 10
        return i + 15
    data = [{'id': 1, 'buyer': 'A'}, {'id': 2, 'buyer': 'B'}]
    df = DataFrame(data, index=date_range('2016-01-01', periods=2))
    r = df.groupby('id').resample('1D')
    result = r['buyer'].count()
    expected = Series([1, 1], index=pd.MultiIndex.from_tuples([(1, Timestamp('2016-01-01')), (2, Timestamp('2016-01-02'))], names=['id', None]), name='buyer')
    tm.assert_series_equal(result, expected)
    result = r['buyer'].count()
    tm.assert_series_equal(result, expected)

def test_groupby_resample_on_api_with_getitem():
    if False:
        return 10
    df = DataFrame({'id': list('aabbb'), 'date': date_range('1-1-2016', periods=5), 'data': 1})
    exp = df.set_index('date').groupby('id').resample('2D')['data'].sum()
    result = df.groupby('id').resample('2D', on='date')['data'].sum()
    tm.assert_series_equal(result, exp)

def test_groupby_with_origin():
    if False:
        return 10
    freq = '1399min'
    (start, end) = ('1/1/2000 00:00:00', '1/31/2000 00:00')
    middle = '1/15/2000 00:00:00'
    rng = date_range(start, end, freq='1231min')
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts2 = ts[middle:end]
    simple_grouper = pd.Grouper(freq=freq)
    count_ts = ts.groupby(simple_grouper).agg('count')
    count_ts = count_ts[middle:end]
    count_ts2 = ts2.groupby(simple_grouper).agg('count')
    with pytest.raises(AssertionError, match='Index are different'):
        tm.assert_index_equal(count_ts.index, count_ts2.index)
    origin = Timestamp(0)
    adjusted_grouper = pd.Grouper(freq=freq, origin=origin)
    adjusted_count_ts = ts.groupby(adjusted_grouper).agg('count')
    adjusted_count_ts = adjusted_count_ts[middle:end]
    adjusted_count_ts2 = ts2.groupby(adjusted_grouper).agg('count')
    tm.assert_series_equal(adjusted_count_ts, adjusted_count_ts2)
    origin_future = Timestamp(0) + pd.Timedelta('1399min') * 30000
    adjusted_grouper2 = pd.Grouper(freq=freq, origin=origin_future)
    adjusted2_count_ts = ts.groupby(adjusted_grouper2).agg('count')
    adjusted2_count_ts = adjusted2_count_ts[middle:end]
    adjusted2_count_ts2 = ts2.groupby(adjusted_grouper2).agg('count')
    tm.assert_series_equal(adjusted2_count_ts, adjusted2_count_ts2)
    tm.assert_series_equal(adjusted_count_ts, adjusted2_count_ts2)

def test_nearest():
    if False:
        return 10
    index = date_range('1/1/2000', periods=3, freq='min')
    result = Series(range(3), index=index).resample('20s').nearest()
    expected = Series([0, 0, 1, 1, 1, 2, 2], index=pd.DatetimeIndex(['2000-01-01 00:00:00', '2000-01-01 00:00:20', '2000-01-01 00:00:40', '2000-01-01 00:01:00', '2000-01-01 00:01:20', '2000-01-01 00:01:40', '2000-01-01 00:02:00'], dtype='datetime64[ns]', freq='20s'))
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('f', ['first', 'last', 'median', 'sem', 'sum', 'mean', 'min', 'max', 'size', 'count', 'nearest', 'bfill', 'ffill', 'asfreq', 'ohlc'])
def test_methods(f, test_frame):
    if False:
        i = 10
        return i + 15
    g = test_frame.groupby('A')
    r = g.resample('2s')
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(r, f)()
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = g.apply(lambda x: getattr(x.resample('2s'), f)())
    tm.assert_equal(result, expected)

def test_methods_nunique(test_frame):
    if False:
        print('Hello World!')
    g = test_frame.groupby('A')
    r = g.resample('2s')
    result = r.B.nunique()
    expected = g.B.apply(lambda x: x.resample('2s').nunique())
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('f', ['std', 'var'])
def test_methods_std_var(f, test_frame):
    if False:
        print('Hello World!')
    g = test_frame.groupby('A')
    r = g.resample('2s')
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(r, f)(ddof=1)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = g.apply(lambda x: getattr(x.resample('2s'), f)(ddof=1))
    tm.assert_frame_equal(result, expected)

def test_apply(test_frame):
    if False:
        while True:
            i = 10
    g = test_frame.groupby('A')
    r = g.resample('2s')
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = g.resample('2s').sum()

    def f_0(x):
        if False:
            for i in range(10):
                print('nop')
        return x.resample('2s').sum()
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = r.apply(f_0)
    tm.assert_frame_equal(result, expected)

    def f_1(x):
        if False:
            for i in range(10):
                print('nop')
        return x.resample('2s').apply(lambda y: y.sum())
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = g.apply(f_1)
    expected = expected.astype('int64')
    tm.assert_frame_equal(result, expected)

def test_apply_with_mutated_index():
    if False:
        while True:
            i = 10
    index = date_range('1-1-2015', '12-31-15', freq='D')
    df = DataFrame(data={'col1': np.random.default_rng(2).random(len(index))}, index=index)

    def f(x):
        if False:
            for i in range(10):
                print('nop')
        s = Series([1, 2], index=['a', 'b'])
        return s
    expected = df.groupby(pd.Grouper(freq='ME')).apply(f)
    result = df.resample('ME').apply(f)
    tm.assert_frame_equal(result, expected)
    expected = df['col1'].groupby(pd.Grouper(freq='ME'), group_keys=False).apply(f)
    result = df['col1'].resample('ME').apply(f)
    tm.assert_series_equal(result, expected)

def test_apply_columns_multilevel():
    if False:
        i = 10
        return i + 15
    cols = pd.MultiIndex.from_tuples([('A', 'a', '', 'one'), ('B', 'b', 'i', 'two')])
    ind = date_range(start='2017-01-01', freq='15Min', periods=8)
    df = DataFrame(np.array([0] * 16).reshape(8, 2), index=ind, columns=cols)
    agg_dict = {col: np.sum if col[3] == 'one' else np.mean for col in df.columns}
    result = df.resample('h').apply(lambda x: agg_dict[x.name](x))
    expected = DataFrame(2 * [[0, 0.0]], index=date_range(start='2017-01-01', freq='1h', periods=2), columns=pd.MultiIndex.from_tuples([('A', 'a', '', 'one'), ('B', 'b', 'i', 'two')]))
    tm.assert_frame_equal(result, expected)

def test_apply_non_naive_index():
    if False:
        i = 10
        return i + 15

    def weighted_quantile(series, weights, q):
        if False:
            while True:
                i = 10
        series = series.sort_values()
        cumsum = weights.reindex(series.index).fillna(0).cumsum()
        cutoff = cumsum.iloc[-1] * q
        return series[cumsum >= cutoff].iloc[0]
    times = date_range('2017-6-23 18:00', periods=8, freq='15min', tz='UTC')
    data = Series([1.0, 1, 1, 1, 1, 2, 2, 0], index=times)
    weights = Series([160.0, 91, 65, 43, 24, 10, 1, 0], index=times)
    result = data.resample('D').apply(weighted_quantile, weights=weights, q=0.5)
    ind = date_range('2017-06-23 00:00:00+00:00', '2017-06-23 00:00:00+00:00', freq='D', tz='UTC')
    expected = Series([1.0], index=ind)
    tm.assert_series_equal(result, expected)

def test_resample_groupby_with_label():
    if False:
        for i in range(10):
            print('nop')
    index = date_range('2000-01-01', freq='2D', periods=5)
    df = DataFrame(index=index, data={'col0': [0, 0, 1, 1, 2], 'col1': [1, 1, 1, 1, 1]})
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby('col0').resample('1W', label='left').sum()
    mi = [np.array([0, 0, 1, 2], dtype=np.int64), pd.to_datetime(np.array(['1999-12-26', '2000-01-02', '2000-01-02', '2000-01-02']))]
    mindex = pd.MultiIndex.from_arrays(mi, names=['col0', None])
    expected = DataFrame(data={'col0': [0, 0, 2, 2], 'col1': [1, 1, 2, 1]}, index=mindex)
    tm.assert_frame_equal(result, expected)

def test_consistency_with_window(test_frame):
    if False:
        i = 10
        return i + 15
    df = test_frame
    expected = Index([1, 2, 3], name='A')
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby('A').resample('2s').mean()
    assert result.index.nlevels == 2
    tm.assert_index_equal(result.index.levels[0], expected)
    result = df.groupby('A').rolling(20).mean()
    assert result.index.nlevels == 2
    tm.assert_index_equal(result.index.levels[0], expected)

def test_median_duplicate_columns():
    if False:
        return 10
    df = DataFrame(np.random.default_rng(2).standard_normal((20, 3)), columns=list('aaa'), index=date_range('2012-01-01', periods=20, freq='s'))
    df2 = df.copy()
    df2.columns = ['a', 'b', 'c']
    expected = df2.resample('5s').median()
    result = df.resample('5s').median()
    expected.columns = result.columns
    tm.assert_frame_equal(result, expected)

def test_apply_to_one_column_of_df():
    if False:
        return 10
    df = DataFrame({'col': range(10), 'col1': range(10, 20)}, index=date_range('2012-01-01', periods=10, freq='20min'))
    result = df.resample('h').apply(lambda group: group.col.sum())
    expected = Series([3, 12, 21, 9], index=date_range('2012-01-01', periods=4, freq='h'))
    tm.assert_series_equal(result, expected)
    result = df.resample('h').apply(lambda group: group['col'].sum())
    tm.assert_series_equal(result, expected)

def test_resample_groupby_agg():
    if False:
        i = 10
        return i + 15
    df = DataFrame({'cat': ['cat_1', 'cat_1', 'cat_2', 'cat_1', 'cat_2', 'cat_1', 'cat_2', 'cat_1'], 'num': [5, 20, 22, 3, 4, 30, 10, 50], 'date': ['2019-2-1', '2018-02-03', '2020-3-11', '2019-2-2', '2019-2-2', '2018-12-4', '2020-3-11', '2020-12-12']})
    df['date'] = pd.to_datetime(df['date'])
    resampled = df.groupby('cat').resample('YE', on='date')
    expected = resampled[['num']].sum()
    result = resampled.agg({'num': 'sum'})
    tm.assert_frame_equal(result, expected)

def test_resample_groupby_agg_listlike():
    if False:
        while True:
            i = 10
    ts = Timestamp('2021-02-28 00:00:00')
    df = DataFrame({'class': ['beta'], 'value': [69]}, index=Index([ts], name='date'))
    resampled = df.groupby('class').resample('ME')['value']
    result = resampled.agg(['sum', 'size'])
    expected = DataFrame([[69, 1]], index=pd.MultiIndex.from_tuples([('beta', ts)], names=['class', 'date']), columns=['sum', 'size'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('keys', [['a'], ['a', 'b']])
def test_empty(keys):
    if False:
        return 10
    df = DataFrame([], columns=['a', 'b'], index=TimedeltaIndex([]))
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(keys).resample(rule=pd.to_timedelta('00:00:01')).mean()
    expected = DataFrame(columns=['a', 'b']).set_index(keys, drop=False).set_index(TimedeltaIndex([]), append=True)
    if len(keys) == 1:
        expected.index.name = keys[0]
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('consolidate', [True, False])
def test_resample_groupby_agg_object_dtype_all_nan(consolidate):
    if False:
        return 10
    dates = date_range('2020-01-01', periods=15, freq='D')
    df1 = DataFrame({'key': 'A', 'date': dates, 'col1': range(15), 'col_object': 'val'})
    df2 = DataFrame({'key': 'B', 'date': dates, 'col1': range(15)})
    df = pd.concat([df1, df2], ignore_index=True)
    if consolidate:
        df = df._consolidate()
    with tm.assert_produces_warning(FutureWarning):
        result = df.groupby(['key']).resample('W', on='date').min()
    idx = pd.MultiIndex.from_arrays([['A'] * 3 + ['B'] * 3, pd.to_datetime(['2020-01-05', '2020-01-12', '2020-01-19'] * 2)], names=['key', 'date'])
    expected = DataFrame({'key': ['A'] * 3 + ['B'] * 3, 'col1': [0, 5, 12] * 2, 'col_object': ['val'] * 3 + [np.nan] * 3}, index=idx)
    tm.assert_frame_equal(result, expected)

def test_groupby_resample_with_list_of_keys():
    if False:
        for i in range(10):
            print('nop')
    df = DataFrame(data={'date': date_range(start='2016-01-01', periods=8), 'group': [0, 0, 0, 0, 1, 1, 1, 1], 'val': [1, 7, 5, 2, 3, 10, 5, 1]})
    result = df.groupby('group').resample('2D', on='date')[['val']].mean()
    expected = DataFrame(data={'val': [4.0, 3.5, 6.5, 3.0]}, index=Index(data=[(0, Timestamp('2016-01-01')), (0, Timestamp('2016-01-03')), (1, Timestamp('2016-01-05')), (1, Timestamp('2016-01-07'))], name=('group', 'date')))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('keys', [['a'], ['a', 'b']])
def test_resample_no_index(keys):
    if False:
        i = 10
        return i + 15
    df = DataFrame([], columns=['a', 'b', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(keys).resample(rule=pd.to_timedelta('00:00:01')).mean()
    expected = DataFrame(columns=['a', 'b', 'date']).set_index(keys, drop=False)
    expected['date'] = pd.to_datetime(expected['date'])
    expected = expected.set_index('date', append=True, drop=True)
    if len(keys) == 1:
        expected.index.name = keys[0]
    tm.assert_frame_equal(result, expected)

def test_resample_no_columns():
    if False:
        for i in range(10):
            print('nop')
    df = DataFrame(index=Index(pd.to_datetime(['2018-01-01 00:00:00', '2018-01-01 12:00:00', '2018-01-02 00:00:00']), name='date'))
    result = df.groupby([0, 0, 1]).resample(rule=pd.to_timedelta('06:00:00')).mean()
    index = pd.to_datetime(['2018-01-01 00:00:00', '2018-01-01 06:00:00', '2018-01-01 12:00:00', '2018-01-02 00:00:00'])
    expected = DataFrame(index=pd.MultiIndex(levels=[np.array([0, 1], dtype=np.intp), index], codes=[[0, 0, 0, 1], [0, 1, 2, 3]], names=[None, 'date']))
    tm.assert_frame_equal(result, expected, check_index_type=not is_platform_windows())

def test_groupby_resample_size_all_index_same():
    if False:
        i = 10
        return i + 15
    df = DataFrame({'A': [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3, 'B': np.arange(12)}, index=date_range('31/12/2000 18:00', freq='h', periods=12))
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby('A').resample('D').size()
    expected = Series(3, index=pd.MultiIndex.from_tuples([(1, Timestamp('2000-12-31')), (1, Timestamp('2001-01-01')), (2, Timestamp('2000-12-31')), (2, Timestamp('2001-01-01'))], names=['A', None]))
    tm.assert_series_equal(result, expected)

def test_groupby_resample_on_index_with_list_of_keys():
    if False:
        for i in range(10):
            print('nop')
    df = DataFrame(data={'group': [0, 0, 0, 0, 1, 1, 1, 1], 'val': [3, 1, 4, 1, 5, 9, 2, 6]}, index=Series(date_range(start='2016-01-01', periods=8), name='date'))
    result = df.groupby('group').resample('2D')[['val']].mean()
    expected = DataFrame(data={'val': [2.0, 2.5, 7.0, 4.0]}, index=Index(data=[(0, Timestamp('2016-01-01')), (0, Timestamp('2016-01-03')), (1, Timestamp('2016-01-05')), (1, Timestamp('2016-01-07'))], name=('group', 'date')))
    tm.assert_frame_equal(result, expected)

def test_groupby_resample_on_index_with_list_of_keys_multi_columns():
    if False:
        print('Hello World!')
    df = DataFrame(data={'group': [0, 0, 0, 0, 1, 1, 1, 1], 'first_val': [3, 1, 4, 1, 5, 9, 2, 6], 'second_val': [2, 7, 1, 8, 2, 8, 1, 8], 'third_val': [1, 4, 1, 4, 2, 1, 3, 5]}, index=Series(date_range(start='2016-01-01', periods=8), name='date'))
    result = df.groupby('group').resample('2D')[['first_val', 'second_val']].mean()
    expected = DataFrame(data={'first_val': [2.0, 2.5, 7.0, 4.0], 'second_val': [4.5, 4.5, 5.0, 4.5]}, index=Index(data=[(0, Timestamp('2016-01-01')), (0, Timestamp('2016-01-03')), (1, Timestamp('2016-01-05')), (1, Timestamp('2016-01-07'))], name=('group', 'date')))
    tm.assert_frame_equal(result, expected)

def test_groupby_resample_on_index_with_list_of_keys_missing_column():
    if False:
        print('Hello World!')
    df = DataFrame(data={'group': [0, 0, 0, 0, 1, 1, 1, 1], 'val': [3, 1, 4, 1, 5, 9, 2, 6]}, index=Series(date_range(start='2016-01-01', periods=8), name='date'))
    gb = df.groupby('group')
    rs = gb.resample('2D')
    with pytest.raises(KeyError, match='Columns not found'):
        rs[['val_not_in_dataframe']]

@pytest.mark.parametrize('kind', ['datetime', 'period'])
def test_groupby_resample_kind(kind):
    if False:
        while True:
            i = 10
    df = DataFrame({'datetime': pd.to_datetime(['20181101 1100', '20181101 1200', '20181102 1300', '20181102 1400']), 'group': ['A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4]})
    df = df.set_index('datetime')
    result = df.groupby('group')['value'].resample('D', kind=kind).last()
    dt_level = pd.DatetimeIndex(['2018-11-01', '2018-11-02'])
    if kind == 'period':
        dt_level = dt_level.to_period(freq='D')
    expected_index = pd.MultiIndex.from_product([['A', 'B'], dt_level], names=['group', 'datetime'])
    expected = Series([1, 3, 2, 4], index=expected_index, name='value')
    tm.assert_series_equal(result, expected)