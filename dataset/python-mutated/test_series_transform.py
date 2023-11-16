import numpy as np
import pytest
from pandas import DataFrame, MultiIndex, Series, concat
import pandas._testing as tm

@pytest.mark.parametrize('args, kwargs, increment', [((), {}, 0), ((), {'a': 1}, 1), ((2, 3), {}, 32), ((1,), {'c': 2}, 201)])
def test_agg_args(args, kwargs, increment):
    if False:
        i = 10
        return i + 15

    def f(x, a=0, b=0, c=0):
        if False:
            while True:
                i = 10
        return x + a + 10 * b + 100 * c
    s = Series([1, 2])
    result = s.transform(f, 0, *args, **kwargs)
    expected = s + increment
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('ops, names', [([np.sqrt], ['sqrt']), ([np.abs, np.sqrt], ['absolute', 'sqrt']), (np.array([np.sqrt]), ['sqrt']), (np.array([np.abs, np.sqrt]), ['absolute', 'sqrt'])])
def test_transform_listlike(string_series, ops, names):
    if False:
        return 10
    with np.errstate(all='ignore'):
        expected = concat([op(string_series) for op in ops], axis=1)
        expected.columns = names
        result = string_series.transform(ops)
        tm.assert_frame_equal(result, expected)

def test_transform_listlike_func_with_args():
    if False:
        i = 10
        return i + 15
    s = Series([1, 2, 3])

    def foo1(x, a=1, c=0):
        if False:
            print('Hello World!')
        return x + a + c

    def foo2(x, b=2, c=0):
        if False:
            for i in range(10):
                print('nop')
        return x + b + c
    msg = "foo1\\(\\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        s.transform([foo1, foo2], 0, 3, b=3, c=4)
    result = s.transform([foo1, foo2], 0, 3, c=4)
    expected = DataFrame({'foo1': [8, 9, 10], 'foo2': [8, 9, 10]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('box', [dict, Series])
def test_transform_dictlike(string_series, box):
    if False:
        for i in range(10):
            print('nop')
    with np.errstate(all='ignore'):
        expected = concat([np.sqrt(string_series), np.abs(string_series)], axis=1)
    expected.columns = ['foo', 'bar']
    result = string_series.transform(box({'foo': np.sqrt, 'bar': np.abs}))
    tm.assert_frame_equal(result, expected)

def test_transform_dictlike_mixed():
    if False:
        print('Hello World!')
    df = Series([1, 4])
    result = df.transform({'b': ['sqrt', 'abs'], 'c': 'sqrt'})
    expected = DataFrame([[1.0, 1, 1.0], [2.0, 4, 2.0]], columns=MultiIndex([('b', 'c'), ('sqrt', 'abs')], [(0, 0, 1), (0, 1, 0)]))
    tm.assert_frame_equal(result, expected)