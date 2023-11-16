"""unit tests for sparse utility functions"""
import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils
from scipy.sparse._sputils import matrix

class TestSparseUtils:

    def test_upcast(self):
        if False:
            while True:
                i = 10
        assert_equal(sputils.upcast('intc'), np.intc)
        assert_equal(sputils.upcast('int32', 'float32'), np.float64)
        assert_equal(sputils.upcast('bool', complex, float), np.complex128)
        assert_equal(sputils.upcast('i', 'd'), np.float64)

    def test_getdtype(self):
        if False:
            for i in range(10):
                print('nop')
        A = np.array([1], dtype='int8')
        assert_equal(sputils.getdtype(None, default=float), float)
        assert_equal(sputils.getdtype(None, a=A), np.int8)
        with assert_raises(ValueError, match='object dtype is not supported by sparse matrices'):
            sputils.getdtype('O')

    def test_isscalarlike(self):
        if False:
            print('Hello World!')
        assert_equal(sputils.isscalarlike(3.0), True)
        assert_equal(sputils.isscalarlike(-4), True)
        assert_equal(sputils.isscalarlike(2.5), True)
        assert_equal(sputils.isscalarlike(1 + 3j), True)
        assert_equal(sputils.isscalarlike(np.array(3)), True)
        assert_equal(sputils.isscalarlike('16'), True)
        assert_equal(sputils.isscalarlike(np.array([3])), False)
        assert_equal(sputils.isscalarlike([[3]]), False)
        assert_equal(sputils.isscalarlike((1,)), False)
        assert_equal(sputils.isscalarlike((1, 2)), False)

    def test_isintlike(self):
        if False:
            return 10
        assert_equal(sputils.isintlike(-4), True)
        assert_equal(sputils.isintlike(np.array(3)), True)
        assert_equal(sputils.isintlike(np.array([3])), False)
        with assert_raises(ValueError, match='Inexact indices into sparse matrices are not allowed'):
            sputils.isintlike(3.0)
        assert_equal(sputils.isintlike(2.5), False)
        assert_equal(sputils.isintlike(1 + 3j), False)
        assert_equal(sputils.isintlike((1,)), False)
        assert_equal(sputils.isintlike((1, 2)), False)

    def test_isshape(self):
        if False:
            i = 10
            return i + 15
        assert_equal(sputils.isshape((1, 2)), True)
        assert_equal(sputils.isshape((5, 2)), True)
        assert_equal(sputils.isshape((1.5, 2)), False)
        assert_equal(sputils.isshape((2, 2, 2)), False)
        assert_equal(sputils.isshape(([2], 2)), False)
        assert_equal(sputils.isshape((-1, 2), nonneg=False), True)
        assert_equal(sputils.isshape((2, -1), nonneg=False), True)
        assert_equal(sputils.isshape((-1, 2), nonneg=True), False)
        assert_equal(sputils.isshape((2, -1), nonneg=True), False)
        assert_equal(sputils.isshape((1.5, 2), allow_ndim=True), False)
        assert_equal(sputils.isshape(([2], 2), allow_ndim=True), False)
        assert_equal(sputils.isshape((2, 2, -2), nonneg=True, allow_ndim=True), False)
        assert_equal(sputils.isshape((2,), allow_ndim=True), True)
        assert_equal(sputils.isshape((2, 2), allow_ndim=True), True)
        assert_equal(sputils.isshape((2, 2, 2), allow_ndim=True), True)

    def test_issequence(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(sputils.issequence((1,)), True)
        assert_equal(sputils.issequence((1, 2, 3)), True)
        assert_equal(sputils.issequence([1]), True)
        assert_equal(sputils.issequence([1, 2, 3]), True)
        assert_equal(sputils.issequence(np.array([1, 2, 3])), True)
        assert_equal(sputils.issequence(np.array([[1], [2], [3]])), False)
        assert_equal(sputils.issequence(3), False)

    def test_ismatrix(self):
        if False:
            i = 10
            return i + 15
        assert_equal(sputils.ismatrix(((),)), True)
        assert_equal(sputils.ismatrix([[1], [2]]), True)
        assert_equal(sputils.ismatrix(np.arange(3)[None]), True)
        assert_equal(sputils.ismatrix([1, 2]), False)
        assert_equal(sputils.ismatrix(np.arange(3)), False)
        assert_equal(sputils.ismatrix([[[1]]]), False)
        assert_equal(sputils.ismatrix(3), False)

    def test_isdense(self):
        if False:
            print('Hello World!')
        assert_equal(sputils.isdense(np.array([1])), True)
        assert_equal(sputils.isdense(matrix([1])), True)

    def test_validateaxis(self):
        if False:
            return 10
        assert_raises(TypeError, sputils.validateaxis, (0, 1))
        assert_raises(TypeError, sputils.validateaxis, 1.5)
        assert_raises(ValueError, sputils.validateaxis, 3)
        for axis in (-2, -1, 0, 1, None):
            sputils.validateaxis(axis)

    def test_get_index_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        imax = np.int64(np.iinfo(np.int32).max)
        too_big = imax + 1
        a1 = np.ones(90, dtype='uint32')
        a2 = np.ones(90, dtype='uint32')
        assert_equal(np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)), np.dtype('int32'))
        a1[-1] = imax
        assert_equal(np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)), np.dtype('int32'))
        a1[-1] = too_big
        assert_equal(np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)), np.dtype('int64'))
        a1 = np.ones(89, dtype='uint32')
        a2 = np.ones(89, dtype='uint32')
        assert_equal(np.dtype(sputils.get_index_dtype((a1, a2))), np.dtype('int64'))
        a1 = np.ones(12, dtype='uint32')
        a2 = np.ones(12, dtype='uint32')
        assert_equal(np.dtype(sputils.get_index_dtype((a1, a2), maxval=too_big, check_contents=True)), np.dtype('int64'))
        a1[-1] = too_big
        assert_equal(np.dtype(sputils.get_index_dtype((a1, a2), maxval=too_big)), np.dtype('int64'))

    def test_check_shape_overflow(self):
        if False:
            print('Hello World!')
        new_shape = sputils.check_shape([(10, -1)], (65535, 131070))
        assert_equal(new_shape, (10, 858967245))

    def test_matrix(self):
        if False:
            for i in range(10):
                print('nop')
        a = [[1, 2, 3]]
        b = np.array(a)
        assert isinstance(sputils.matrix(a), np.matrix)
        assert isinstance(sputils.matrix(b), np.matrix)
        c = sputils.matrix(b)
        c[:, :] = 123
        assert_equal(b, a)
        c = sputils.matrix(b, copy=False)
        c[:, :] = 123
        assert_equal(b, [[123, 123, 123]])

    def test_asmatrix(self):
        if False:
            return 10
        a = [[1, 2, 3]]
        b = np.array(a)
        assert isinstance(sputils.asmatrix(a), np.matrix)
        assert isinstance(sputils.asmatrix(b), np.matrix)
        c = sputils.asmatrix(b)
        c[:, :] = 123
        assert_equal(b, [[123, 123, 123]])