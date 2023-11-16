"""
Includes test for last_valid_index.
"""
import numpy as np
import pytest
from pandas import DataFrame, Series
import pandas._testing as tm

class TestFirstValidIndex:

    def test_first_valid_index_single_nan(self, frame_or_series):
        if False:
            print('Hello World!')
        obj = frame_or_series([np.nan])
        assert obj.first_valid_index() is None
        assert obj.iloc[:0].first_valid_index() is None

    @pytest.mark.parametrize('empty', [DataFrame(), Series(dtype=object), Series([], index=[], dtype=object)])
    def test_first_valid_index_empty(self, empty):
        if False:
            i = 10
            return i + 15
        assert empty.last_valid_index() is None
        assert empty.first_valid_index() is None

    @pytest.mark.parametrize('data,idx,expected_first,expected_last', [({'A': [1, 2, 3]}, [1, 1, 2], 1, 2), ({'A': [1, 2, 3]}, [1, 2, 2], 1, 2), ({'A': [1, 2, 3, 4]}, ['d', 'd', 'd', 'd'], 'd', 'd'), ({'A': [1, np.nan, 3]}, [1, 1, 2], 1, 2), ({'A': [np.nan, np.nan, 3]}, [1, 1, 2], 2, 2), ({'A': [1, np.nan, 3]}, [1, 2, 2], 1, 2)])
    def test_first_last_valid_frame(self, data, idx, expected_first, expected_last):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame(data, index=idx)
        assert expected_first == df.first_valid_index()
        assert expected_last == df.last_valid_index()

    @pytest.mark.parametrize('index_func', [tm.makeStringIndex, tm.makeDateIndex])
    def test_first_last_valid(self, index_func):
        if False:
            print('Hello World!')
        N = 30
        index = index_func(N)
        mat = np.random.default_rng(2).standard_normal(N)
        mat[:5] = np.nan
        mat[-5:] = np.nan
        frame = DataFrame({'foo': mat}, index=index)
        assert frame.first_valid_index() == frame.index[5]
        assert frame.last_valid_index() == frame.index[-6]
        ser = frame['foo']
        assert ser.first_valid_index() == frame.index[5]
        assert ser.last_valid_index() == frame.index[-6]

    @pytest.mark.parametrize('index_func', [tm.makeStringIndex, tm.makeDateIndex])
    def test_first_last_valid_all_nan(self, index_func):
        if False:
            print('Hello World!')
        index = index_func(30)
        frame = DataFrame(np.nan, columns=['foo'], index=index)
        assert frame.last_valid_index() is None
        assert frame.first_valid_index() is None
        ser = frame['foo']
        assert ser.first_valid_index() is None
        assert ser.last_valid_index() is None