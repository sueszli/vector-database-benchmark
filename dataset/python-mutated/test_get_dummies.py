import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series, _testing as tm

def test_get_dummies(any_string_dtype):
    if False:
        return 10
    s = Series(['a|b', 'a|c', np.nan], dtype=any_string_dtype)
    result = s.str.get_dummies('|')
    expected = DataFrame([[1, 1, 0], [1, 0, 1], [0, 0, 0]], columns=list('abc'))
    tm.assert_frame_equal(result, expected)
    s = Series(['a;b', 'a', 7], dtype=any_string_dtype)
    result = s.str.get_dummies(';')
    expected = DataFrame([[0, 1, 1], [0, 1, 0], [1, 0, 0]], columns=list('7ab'))
    tm.assert_frame_equal(result, expected)

def test_get_dummies_index():
    if False:
        while True:
            i = 10
    idx = Index(['a|b', 'a|c', 'b|c'])
    result = idx.str.get_dummies('|')
    expected = MultiIndex.from_tuples([(1, 1, 0), (1, 0, 1), (0, 1, 1)], names=('a', 'b', 'c'))
    tm.assert_index_equal(result, expected)

def test_get_dummies_with_name_dummy(any_string_dtype):
    if False:
        print('Hello World!')
    s = Series(['a', 'b,name', 'b'], dtype=any_string_dtype)
    result = s.str.get_dummies(',')
    expected = DataFrame([[1, 0, 0], [0, 1, 1], [0, 1, 0]], columns=['a', 'b', 'name'])
    tm.assert_frame_equal(result, expected)

def test_get_dummies_with_name_dummy_index():
    if False:
        while True:
            i = 10
    idx = Index(['a|b', 'name|c', 'b|name'])
    result = idx.str.get_dummies('|')
    expected = MultiIndex.from_tuples([(1, 1, 0, 0), (0, 0, 1, 1), (0, 1, 0, 1)], names=('a', 'b', 'c', 'name'))
    tm.assert_index_equal(result, expected)