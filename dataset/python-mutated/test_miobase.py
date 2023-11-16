""" Testing miobase module
"""
import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.io.matlab._miobase import matdims

def test_matdims():
    if False:
        return 10
    assert_equal(matdims(np.array(1)), (1, 1))
    assert_equal(matdims(np.array([1])), (1, 1))
    assert_equal(matdims(np.array([1, 2])), (2, 1))
    assert_equal(matdims(np.array([[2], [3]])), (2, 1))
    assert_equal(matdims(np.array([[2, 3]])), (1, 2))
    assert_equal(matdims(np.array([[[2, 3]]])), (1, 1, 2))
    assert_equal(matdims(np.array([])), (0, 0))
    assert_equal(matdims(np.array([[]])), (1, 0))
    assert_equal(matdims(np.array([[[]]])), (1, 1, 0))
    assert_equal(matdims(np.empty((1, 0, 1))), (1, 0, 1))
    assert_equal(matdims(np.array([1, 2]), 'row'), (1, 2))
    assert_raises(ValueError, matdims, np.array([1, 2]), 'bizarre')
    from scipy.sparse import csr_matrix, csc_matrix
    assert_equal(matdims(csr_matrix(np.zeros((3, 3)))), (3, 3))
    assert_equal(matdims(csc_matrix(np.zeros((2, 2)))), (2, 2))