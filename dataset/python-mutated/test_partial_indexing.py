import numpy as np
import pytest
from pandas import DataFrame, IndexSlice, MultiIndex, date_range
import pandas._testing as tm

@pytest.fixture
def df():
    if False:
        while True:
            i = 10
    dr = date_range('2016-01-01', '2016-01-03', freq='12h')
    abc = ['a', 'b', 'c']
    mi = MultiIndex.from_product([dr, abc])
    frame = DataFrame({'c1': range(15)}, index=mi)
    return frame

def test_partial_string_matching_single_index(df):
    if False:
        return 10
    for df_swap in [df.swaplevel(), df.swaplevel(0), df.swaplevel(0, 1)]:
        df_swap = df_swap.sort_index()
        just_a = df_swap.loc['a']
        result = just_a.loc['2016-01-01']
        expected = df.loc[IndexSlice[:, 'a'], :].iloc[0:2]
        expected.index = expected.index.droplevel(1)
        tm.assert_frame_equal(result, expected)

def test_get_loc_partial_timestamp_multiindex(df):
    if False:
        for i in range(10):
            print('nop')
    mi = df.index
    key = ('2016-01-01', 'a')
    loc = mi.get_loc(key)
    expected = np.zeros(len(mi), dtype=bool)
    expected[[0, 3]] = True
    tm.assert_numpy_array_equal(loc, expected)
    key2 = ('2016-01-02', 'a')
    loc2 = mi.get_loc(key2)
    expected2 = np.zeros(len(mi), dtype=bool)
    expected2[[6, 9]] = True
    tm.assert_numpy_array_equal(loc2, expected2)
    key3 = ('2016-01', 'a')
    loc3 = mi.get_loc(key3)
    expected3 = np.zeros(len(mi), dtype=bool)
    expected3[mi.get_level_values(1).get_loc('a')] = True
    tm.assert_numpy_array_equal(loc3, expected3)
    key4 = ('2016', 'a')
    loc4 = mi.get_loc(key4)
    expected4 = expected3
    tm.assert_numpy_array_equal(loc4, expected4)
    taker = np.arange(len(mi), dtype=np.intp)
    taker[::2] = taker[::-2]
    mi2 = mi.take(taker)
    loc5 = mi2.get_loc(key)
    expected5 = np.zeros(len(mi2), dtype=bool)
    expected5[[3, 14]] = True
    tm.assert_numpy_array_equal(loc5, expected5)

def test_partial_string_timestamp_multiindex(df):
    if False:
        print('Hello World!')
    df_swap = df.swaplevel(0, 1).sort_index()
    SLC = IndexSlice
    result = df.loc[SLC['2016-01-01':'2016-02-01', :], :]
    expected = df
    tm.assert_frame_equal(result, expected)
    result = df_swap.loc[SLC[:, '2016-01-01':'2016-01-01'], :]
    expected = df_swap.iloc[[0, 1, 5, 6, 10, 11]]
    tm.assert_frame_equal(result, expected)
    result = df.loc['2016']
    expected = df
    tm.assert_frame_equal(result, expected)
    result = df.loc['2016-01-01']
    expected = df.iloc[0:6]
    tm.assert_frame_equal(result, expected)
    result = df.loc['2016-01-02 12']
    expected = df.iloc[9:12].droplevel(0)
    tm.assert_frame_equal(result, expected)
    result = df_swap.loc[SLC[:, '2016-01-02'], :]
    expected = df_swap.iloc[[2, 3, 7, 8, 12, 13]]
    tm.assert_frame_equal(result, expected)
    result = df.loc[('2016-01-01', 'a'), :]
    expected = df.iloc[[0, 3]]
    expected = df.iloc[[0, 3]].droplevel(1)
    tm.assert_frame_equal(result, expected)
    with pytest.raises(KeyError, match="'2016-01-01'"):
        df_swap.loc['2016-01-01']

def test_partial_string_timestamp_multiindex_str_key_raises(df):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(KeyError, match="'2016-01-01'"):
        df['2016-01-01']

def test_partial_string_timestamp_multiindex_daily_resolution(df):
    if False:
        return 10
    result = df.loc[IndexSlice['2013-03':'2013-03', :], :]
    expected = df.iloc[118:180]
    tm.assert_frame_equal(result, expected)