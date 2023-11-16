import copy
import numpy as np
import pytest
import pandas as pd
from pandas import Timestamp
import pandas._testing as tm

def test_assert_numpy_array_equal_shape_mismatch():
    if False:
        while True:
            i = 10
    msg = 'numpy array are different\n\nnumpy array shapes are different\n\\[left\\]:  \\(2L*,\\)\n\\[right\\]: \\(3L*,\\)'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1, 2]), np.array([3, 4, 5]))

def test_assert_numpy_array_equal_bad_type():
    if False:
        return 10
    expected = 'Expected type'
    with pytest.raises(AssertionError, match=expected):
        tm.assert_numpy_array_equal(1, 2)

@pytest.mark.parametrize('a,b,klass1,klass2', [(np.array([1]), 1, 'ndarray', 'int'), (1, np.array([1]), 'int', 'ndarray')])
def test_assert_numpy_array_equal_class_mismatch(a, b, klass1, klass2):
    if False:
        while True:
            i = 10
    msg = f'numpy array are different\n\nnumpy array classes are different\n\\[left\\]:  {klass1}\n\\[right\\]: {klass2}'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)

def test_assert_numpy_array_equal_value_mismatch1():
    if False:
        for i in range(10):
            print('nop')
    msg = 'numpy array are different\n\nnumpy array values are different \\(66\\.66667 %\\)\n\\[left\\]:  \\[nan, 2\\.0, 3\\.0\\]\n\\[right\\]: \\[1\\.0, nan, 3\\.0\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([np.nan, 2, 3]), np.array([1, np.nan, 3]))

def test_assert_numpy_array_equal_value_mismatch2():
    if False:
        for i in range(10):
            print('nop')
    msg = 'numpy array are different\n\nnumpy array values are different \\(50\\.0 %\\)\n\\[left\\]:  \\[1, 2\\]\n\\[right\\]: \\[1, 3\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1, 2]), np.array([1, 3]))

def test_assert_numpy_array_equal_value_mismatch3():
    if False:
        return 10
    msg = 'numpy array are different\n\nnumpy array values are different \\(16\\.66667 %\\)\n\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]\n\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3], [3, 4], [5, 6]]))

def test_assert_numpy_array_equal_value_mismatch4():
    if False:
        print('Hello World!')
    msg = 'numpy array are different\n\nnumpy array values are different \\(50\\.0 %\\)\n\\[left\\]:  \\[1\\.1, 2\\.000001\\]\n\\[right\\]: \\[1\\.1, 2.0\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1.1, 2.000001]), np.array([1.1, 2.0]))

def test_assert_numpy_array_equal_value_mismatch5():
    if False:
        print('Hello World!')
    msg = 'numpy array are different\n\nnumpy array values are different \\(16\\.66667 %\\)\n\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]\n\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3], [3, 4], [5, 6]]))

def test_assert_numpy_array_equal_value_mismatch6():
    if False:
        while True:
            i = 10
    msg = 'numpy array are different\n\nnumpy array values are different \\(25\\.0 %\\)\n\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\]\\]\n\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\]\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([[1, 2], [3, 4]]), np.array([[1, 3], [3, 4]]))

def test_assert_numpy_array_equal_shape_mismatch_override():
    if False:
        i = 10
        return i + 15
    msg = 'Index are different\n\nIndex shapes are different\n\\[left\\]:  \\(2L*,\\)\n\\[right\\]: \\(3L*,\\)'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1, 2]), np.array([3, 4, 5]), obj='Index')

def test_numpy_array_equal_unicode():
    if False:
        i = 10
        return i + 15
    msg = 'numpy array are different\n\nnumpy array values are different \\(33\\.33333 %\\)\n\\[left\\]:  \\[á, à, ä\\]\n\\[right\\]: \\[á, à, å\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array(['á', 'à', 'ä']), np.array(['á', 'à', 'å']))

def test_numpy_array_equal_object():
    if False:
        while True:
            i = 10
    a = np.array([Timestamp('2011-01-01'), Timestamp('2011-01-01')])
    b = np.array([Timestamp('2011-01-01'), Timestamp('2011-01-02')])
    msg = 'numpy array are different\n\nnumpy array values are different \\(50\\.0 %\\)\n\\[left\\]:  \\[2011-01-01 00:00:00, 2011-01-01 00:00:00\\]\n\\[right\\]: \\[2011-01-01 00:00:00, 2011-01-02 00:00:00\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)

@pytest.mark.parametrize('other_type', ['same', 'copy'])
@pytest.mark.parametrize('check_same', ['same', 'copy'])
def test_numpy_array_equal_copy_flag(other_type, check_same):
    if False:
        while True:
            i = 10
    a = np.array([1, 2, 3])
    msg = None
    if other_type == 'same':
        other = a.view()
    else:
        other = a.copy()
    if check_same != other_type:
        msg = 'array\\(\\[1, 2, 3\\]\\) is not array\\(\\[1, 2, 3\\]\\)' if check_same == 'same' else 'array\\(\\[1, 2, 3\\]\\) is array\\(\\[1, 2, 3\\]\\)'
    if msg is not None:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_numpy_array_equal(a, other, check_same=check_same)
    else:
        tm.assert_numpy_array_equal(a, other, check_same=check_same)

def test_numpy_array_equal_contains_na():
    if False:
        for i in range(10):
            print('nop')
    a = np.array([True, False])
    b = np.array([True, pd.NA], dtype=object)
    msg = 'numpy array are different\n\nnumpy array values are different \\(50.0 %\\)\n\\[left\\]:  \\[True, False\\]\n\\[right\\]: \\[True, <NA>\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)

def test_numpy_array_equal_identical_na(nulls_fixture):
    if False:
        print('Hello World!')
    a = np.array([nulls_fixture], dtype=object)
    tm.assert_numpy_array_equal(a, a)
    if hasattr(nulls_fixture, 'copy'):
        other = nulls_fixture.copy()
    else:
        other = copy.copy(nulls_fixture)
    b = np.array([other], dtype=object)
    tm.assert_numpy_array_equal(a, b)

def test_numpy_array_equal_different_na():
    if False:
        print('Hello World!')
    a = np.array([np.nan], dtype=object)
    b = np.array([pd.NA], dtype=object)
    msg = 'numpy array are different\n\nnumpy array values are different \\(100.0 %\\)\n\\[left\\]:  \\[nan\\]\n\\[right\\]: \\[<NA>\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)