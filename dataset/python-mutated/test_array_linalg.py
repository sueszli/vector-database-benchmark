import numpy
import numpy.linalg as NLA
import pytest
import modin.numpy as np
import modin.numpy.linalg as LA
import modin.pandas as pd
from .utils import assert_scalar_or_array_equal

def test_dot_from_pandas_reindex():
    if False:
        i = 10
        return i + 15
    df = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
    s = pd.Series([1, 1, 2, 1])
    result1 = np.dot(df, s)
    s2 = s.reindex([1, 0, 2, 3])
    result2 = np.dot(df, s2)
    assert_scalar_or_array_equal(result1, result2)

def test_dot_1d():
    if False:
        while True:
            i = 10
    x1 = numpy.random.randint(-100, 100, size=100)
    x2 = numpy.random.randint(-100, 100, size=100)
    numpy_result = numpy.dot(x1, x2)
    (x1, x2) = (np.array(x1), np.array(x2))
    modin_result = np.dot(x1, x2)
    assert_scalar_or_array_equal(modin_result, numpy_result)

def test_dot_2d():
    if False:
        while True:
            i = 10
    x1 = numpy.random.randint(-100, 100, size=(100, 3))
    x2 = numpy.random.randint(-100, 100, size=(3, 50))
    numpy_result = numpy.dot(x1, x2)
    (x1, x2) = (np.array(x1), np.array(x2))
    modin_result = np.dot(x1, x2)
    assert_scalar_or_array_equal(modin_result, numpy_result)

def test_dot_scalar():
    if False:
        print('Hello World!')
    x1 = numpy.random.randint(-100, 100, size=(100, 3))
    x2 = numpy.random.randint(-100, 100)
    numpy_result = numpy.dot(x1, x2)
    x1 = np.array(x1)
    modin_result = np.dot(x1, x2)
    assert_scalar_or_array_equal(modin_result, numpy_result)

def test_matmul_scalar():
    if False:
        while True:
            i = 10
    x1 = numpy.random.randint(-100, 100, size=(100, 3))
    x2 = numpy.random.randint(-100, 100)
    x1 = np.array(x1)
    with pytest.raises(ValueError):
        x1 @ x2

def test_dot_broadcast():
    if False:
        i = 10
        return i + 15
    x1 = numpy.random.randint(-100, 100, size=(100, 3))
    x2 = numpy.random.randint(-100, 100, size=(3,))
    numpy_result = numpy.dot(x1, x2)
    (x1, x2) = (np.array(x1), np.array(x2))
    modin_result = np.dot(x1, x2)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    x1 = numpy.random.randint(-100, 100, size=(100,))
    x2 = numpy.random.randint(-100, 100, size=(100, 3))
    numpy_result = numpy.dot(x1, x2)
    (x1, x2) = (np.array(x1), np.array(x2))
    modin_result = np.dot(x1, x2)
    assert_scalar_or_array_equal(modin_result, numpy_result)

@pytest.mark.parametrize('axis', [None, 0, 1], ids=['axis=None', 'axis=0', 'axis=1'])
def test_norm_fro_2d(axis):
    if False:
        for i in range(10):
            print('nop')
    x1 = numpy.random.randint(-10, 10, size=(100, 3))
    numpy_result = NLA.norm(x1, axis=axis)
    x1 = np.array(x1)
    modin_result = LA.norm(x1, axis=axis)
    if isinstance(modin_result, np.array):
        modin_result = modin_result._to_numpy()
    numpy.testing.assert_allclose(modin_result, numpy_result, rtol=1e-12)

def test_norm_fro_1d():
    if False:
        for i in range(10):
            print('nop')
    x1 = numpy.random.randint(-10, 10, size=100)
    numpy_result = NLA.norm(x1)
    x1 = np.array(x1)
    modin_result = LA.norm(x1)
    numpy.testing.assert_allclose(modin_result, numpy_result, rtol=1e-12)