import numpy as np
import numpy.matlib
from numpy.testing import assert_array_equal, assert_

def test_empty():
    if False:
        while True:
            i = 10
    x = numpy.matlib.empty((2,))
    assert_(isinstance(x, np.matrix))
    assert_(x.shape, (1, 2))

def test_ones():
    if False:
        while True:
            i = 10
    assert_array_equal(numpy.matlib.ones((2, 3)), np.matrix([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    assert_array_equal(numpy.matlib.ones(2), np.matrix([[1.0, 1.0]]))

def test_zeros():
    if False:
        return 10
    assert_array_equal(numpy.matlib.zeros((2, 3)), np.matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert_array_equal(numpy.matlib.zeros(2), np.matrix([[0.0, 0.0]]))

def test_identity():
    if False:
        i = 10
        return i + 15
    x = numpy.matlib.identity(2, dtype=int)
    assert_array_equal(x, np.matrix([[1, 0], [0, 1]]))

def test_eye():
    if False:
        for i in range(10):
            print('nop')
    xc = numpy.matlib.eye(3, k=1, dtype=int)
    assert_array_equal(xc, np.matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))
    assert xc.flags.c_contiguous
    assert not xc.flags.f_contiguous
    xf = numpy.matlib.eye(3, 4, dtype=int, order='F')
    assert_array_equal(xf, np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
    assert not xf.flags.c_contiguous
    assert xf.flags.f_contiguous

def test_rand():
    if False:
        while True:
            i = 10
    x = numpy.matlib.rand(3)
    assert_(x.ndim == 2)

def test_randn():
    if False:
        while True:
            i = 10
    x = np.matlib.randn(3)
    assert_(x.ndim == 2)

def test_repmat():
    if False:
        i = 10
        return i + 15
    a1 = np.arange(4)
    x = numpy.matlib.repmat(a1, 2, 2)
    y = np.array([[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3]])
    assert_array_equal(x, y)