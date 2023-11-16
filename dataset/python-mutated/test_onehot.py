import numpy as np
import pytest
from mlxtend.preprocessing import one_hot

def test_default():
    if False:
        return 10
    y = np.array([0, 1, 2, 3, 4, 2])
    expect = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0]], dtype='float')
    out = one_hot(y)
    np.testing.assert_array_equal(expect, out)

def test_autoguessing():
    if False:
        print('Hello World!')
    y = np.array([0, 4, 0, 4])
    expect = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]], dtype='float')
    out = one_hot(y)
    np.testing.assert_array_equal(expect, out)

def test_list():
    if False:
        return 10
    y = [0, 1, 2, 3, 4, 2]
    expect = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0]], dtype='float')
    out = one_hot(y)
    np.testing.assert_array_equal(expect, out)

def test_multidim_list():
    if False:
        for i in range(10):
            print('nop')
    y = [[0, 1, 2, 3, 4, 2]]
    with pytest.raises(AttributeError):
        one_hot(y)

def test_multidim_array():
    if False:
        print('Hello World!')
    y = np.array([[0], [1], [2], [3], [4], [2]])
    with pytest.raises(AttributeError):
        one_hot(y)

def test_oneclass():
    if False:
        for i in range(10):
            print('nop')
    np.testing.assert_array_equal(one_hot([0]), np.array([[0.0]], dtype='float'))

def test_list_morelabels():
    if False:
        i = 10
        return i + 15
    y = [0, 1]
    expect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='float')
    out = one_hot(y, num_labels=3)
    np.testing.assert_array_equal(expect, out)