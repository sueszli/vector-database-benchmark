import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array

def test_assigning_to_same_variable_removes_references(using_copy_on_write):
    if False:
        for i in range(10):
            print('nop')
    df = DataFrame({'a': [1, 2, 3]})
    df = df.reset_index()
    if using_copy_on_write:
        assert df._mgr._has_no_reference(1)
    arr = get_array(df, 'a')
    df.iloc[0, 1] = 100
    assert np.shares_memory(arr, get_array(df, 'a'))

def test_setitem_dont_track_unnecessary_references(using_copy_on_write):
    if False:
        while True:
            i = 10
    df = DataFrame({'a': [1, 2, 3], 'b': 1, 'c': 1})
    df['b'] = 100
    arr = get_array(df, 'a')
    df.iloc[0, 0] = 100
    assert np.shares_memory(arr, get_array(df, 'a'))

def test_setitem_with_view_copies(using_copy_on_write):
    if False:
        i = 10
        return i + 15
    df = DataFrame({'a': [1, 2, 3], 'b': 1, 'c': 1})
    view = df[:]
    expected = df.copy()
    df['b'] = 100
    arr = get_array(df, 'a')
    df.iloc[0, 0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(arr, get_array(df, 'a'))
        tm.assert_frame_equal(view, expected)

def test_setitem_with_view_invalidated_does_not_copy(using_copy_on_write, request):
    if False:
        while True:
            i = 10
    df = DataFrame({'a': [1, 2, 3], 'b': 1, 'c': 1})
    view = df[:]
    df['b'] = 100
    arr = get_array(df, 'a')
    view = None
    df.iloc[0, 0] = 100
    if using_copy_on_write:
        mark = pytest.mark.xfail(reason='blk.delete does not track references correctly')
        request.applymarker(mark)
        assert np.shares_memory(arr, get_array(df, 'a'))

def test_out_of_scope(using_copy_on_write):
    if False:
        while True:
            i = 10

    def func():
        if False:
            while True:
                i = 10
        df = DataFrame({'a': [1, 2], 'b': 1.5, 'c': 1})
        result = df[['a', 'b']]
        return result
    result = func()
    if using_copy_on_write:
        assert not result._mgr.blocks[0].refs.has_reference()
        assert not result._mgr.blocks[1].refs.has_reference()

def test_delete(using_copy_on_write):
    if False:
        i = 10
        return i + 15
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)), columns=['a', 'b', 'c'])
    del df['b']
    if using_copy_on_write:
        assert not df._mgr.blocks[0].refs.has_reference()
        assert not df._mgr.blocks[1].refs.has_reference()
    df = df[['a']]
    if using_copy_on_write:
        assert not df._mgr.blocks[0].refs.has_reference()

def test_delete_reference(using_copy_on_write):
    if False:
        while True:
            i = 10
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)), columns=['a', 'b', 'c'])
    x = df[:]
    del df['b']
    if using_copy_on_write:
        assert df._mgr.blocks[0].refs.has_reference()
        assert df._mgr.blocks[1].refs.has_reference()
        assert x._mgr.blocks[0].refs.has_reference()