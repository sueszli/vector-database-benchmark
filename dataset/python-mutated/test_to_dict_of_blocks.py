import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import DataFrame, MultiIndex
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
pytestmark = td.skip_array_manager_invalid_test

class TestToDictOfBlocks:

    def test_copy_blocks(self, float_frame):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame(float_frame, copy=True)
        column = df.columns[0]
        _last_df = None
        blocks = df._to_dict_of_blocks(copy=True)
        for _df in blocks.values():
            _last_df = _df
            if column in _df:
                _df.loc[:, column] = _df[column] + 1
        assert _last_df is not None and (not _last_df[column].equals(df[column]))

    def test_no_copy_blocks(self, float_frame, using_copy_on_write):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame(float_frame, copy=True)
        column = df.columns[0]
        _last_df = None
        blocks = df._to_dict_of_blocks(copy=False)
        for _df in blocks.values():
            _last_df = _df
            if column in _df:
                _df.loc[:, column] = _df[column] + 1
        if not using_copy_on_write:
            assert _last_df is not None and _last_df[column].equals(df[column])
        else:
            assert _last_df is not None and (not _last_df[column].equals(df[column]))

def test_to_dict_of_blocks_item_cache(request, using_copy_on_write, warn_copy_on_write):
    if False:
        print('Hello World!')
    if using_copy_on_write:
        request.applymarker(pytest.mark.xfail(reason='CoW - not yet implemented'))
    df = DataFrame({'a': [1, 2, 3, 4], 'b': ['a', 'b', 'c', 'd']})
    df['c'] = NumpyExtensionArray(np.array([1, 2, None, 3], dtype=object))
    mgr = df._mgr
    assert len(mgr.blocks) == 3
    ser = df['b']
    df._to_dict_of_blocks()
    if using_copy_on_write:
        ser.values[0] = 'foo'
        assert df.loc[0, 'b'] == 'a'
    elif warn_copy_on_write:
        ser.values[0] = 'foo'
        assert df.loc[0, 'b'] == 'foo'
        assert df['b'] is not ser
    else:
        ser.values[0] = 'foo'
        assert df.loc[0, 'b'] == 'foo'
        assert df['b'] is ser

def test_set_change_dtype_slice():
    if False:
        print('Hello World!')
    cols = MultiIndex.from_tuples([('1st', 'a'), ('2nd', 'b'), ('3rd', 'c')])
    df = DataFrame([[1.0, 2, 3], [4.0, 5, 6]], columns=cols)
    df['2nd'] = df['2nd'] * 2.0
    blocks = df._to_dict_of_blocks()
    assert sorted(blocks.keys()) == ['float64', 'int64']
    tm.assert_frame_equal(blocks['float64'], DataFrame([[1.0, 4.0], [4.0, 10.0]], columns=cols[:2]))
    tm.assert_frame_equal(blocks['int64'], DataFrame([[3], [6]], columns=cols[2:]))