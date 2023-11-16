import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array

@td.skip_array_manager_invalid_test
def test_consolidate(using_copy_on_write):
    if False:
        return 10
    df = DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
    df['c'] = [4, 5, 6]
    subset = df[:]
    assert all((blk.refs.has_reference() for blk in subset._mgr.blocks))
    subset._consolidate_inplace()
    assert subset._mgr.blocks[0].refs.has_reference()
    assert np.shares_memory(get_array(df, 'b'), get_array(subset, 'b'))
    assert not subset._mgr.blocks[1].refs.has_reference()
    assert not df._mgr.blocks[0].refs.has_reference()
    assert df._mgr.blocks[1].refs.has_reference()
    assert not df._mgr.blocks[2].refs.has_reference()
    if using_copy_on_write:
        subset.iloc[0, 1] = 0.0
        assert not df._mgr.blocks[1].refs.has_reference()
        assert df.loc[0, 'b'] == 0.1

@pytest.mark.single_cpu
@td.skip_array_manager_invalid_test
def test_switch_options():
    if False:
        for i in range(10):
            print('nop')
    with pd.option_context('mode.copy_on_write', False):
        df = DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
        subset = df[:]
        subset.iloc[0, 0] = 0
        assert df.iloc[0, 0] == 0
        pd.options.mode.copy_on_write = True
        df = DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
        subset = df[:]
        subset.iloc[0, 0] = 0
        assert df.iloc[0, 0] == 1
        pd.options.mode.copy_on_write = False
        df = DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
        subset = df[:]
        subset.iloc[0, 0] = 0
        assert df.iloc[0, 0] == 0

@td.skip_array_manager_invalid_test
@pytest.mark.parametrize('dtype', [np.intp, np.int8])
@pytest.mark.parametrize('locs, arr', [([0], np.array([-1, -2, -3])), ([1], np.array([-1, -2, -3])), ([5], np.array([-1, -2, -3])), ([0, 1], np.array([[-1, -2, -3], [-4, -5, -6]]).T), ([0, 2], np.array([[-1, -2, -3], [-4, -5, -6]]).T), ([0, 1, 2], np.array([[-1, -2, -3], [-4, -5, -6], [-4, -5, -6]]).T), ([1, 2], np.array([[-1, -2, -3], [-4, -5, -6]]).T), ([1, 3], np.array([[-1, -2, -3], [-4, -5, -6]]).T), ([1, 3], np.array([[-1, -2, -3], [-4, -5, -6]]).T)])
def test_iset_splits_blocks_inplace(using_copy_on_write, locs, arr, dtype):
    if False:
        return 10
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'd': [10, 11, 12], 'e': [13, 14, 15], 'f': ['a', 'b', 'c']})
    arr = arr.astype(dtype)
    df_orig = df.copy()
    df2 = df.copy(deep=None)
    df2._mgr.iset(locs, arr, inplace=True)
    tm.assert_frame_equal(df, df_orig)
    if using_copy_on_write:
        for (i, col) in enumerate(df.columns):
            if i not in locs:
                assert np.shares_memory(get_array(df, col), get_array(df2, col))
    else:
        for col in df.columns:
            assert not np.shares_memory(get_array(df, col), get_array(df2, col))

def test_exponential_backoff():
    if False:
        i = 10
        return i + 15
    df = DataFrame({'a': [1, 2, 3]})
    for i in range(490):
        df.copy(deep=False)
    assert len(df._mgr.blocks[0].refs.referenced_blocks) == 491
    df = DataFrame({'a': [1, 2, 3]})
    dfs = [df.copy(deep=False) for i in range(510)]
    for i in range(20):
        df.copy(deep=False)
    assert len(df._mgr.blocks[0].refs.referenced_blocks) == 531
    assert df._mgr.blocks[0].refs.clear_counter == 1000
    for i in range(500):
        df.copy(deep=False)
    assert df._mgr.blocks[0].refs.clear_counter == 1000
    dfs = dfs[:300]
    for i in range(500):
        df.copy(deep=False)
    assert df._mgr.blocks[0].refs.clear_counter == 500