"""
Unit test for Linear Programming via Simplex Algorithm.
"""
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from .test_linprog import magic_square
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id
from scipy.sparse import csc_matrix

def setup_module():
    if False:
        i = 10
        return i + 15
    np.random.seed(2017)

def redundancy_removed(A, B):
    if False:
        return 10
    'Checks whether a matrix contains only independent rows of another'
    for rowA in A:
        for rowB in B:
            if np.all(rowA == rowB):
                break
        else:
            return False
    return A.shape[0] == np.linalg.matrix_rank(A) == np.linalg.matrix_rank(B)

class RRCommonTests:

    def test_no_redundancy(self):
        if False:
            return 10
        (m, n) = (10, 10)
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        (A1, b1, status, message) = self.rr(A0, b0)
        assert_allclose(A0, A1)
        assert_allclose(b0, b1)
        assert_equal(status, 0)

    def test_infeasible_zero_row(self):
        if False:
            while True:
                i = 10
        A = np.eye(3)
        A[1, :] = 0
        b = np.random.rand(3)
        (A1, b1, status, message) = self.rr(A, b)
        assert_equal(status, 2)

    def test_remove_zero_row(self):
        if False:
            while True:
                i = 10
        A = np.eye(3)
        A[1, :] = 0
        b = np.random.rand(3)
        b[1] = 0
        (A1, b1, status, message) = self.rr(A, b)
        assert_equal(status, 0)
        assert_allclose(A1, A[[0, 2], :])
        assert_allclose(b1, b[[0, 2]])

    def test_infeasible_m_gt_n(self):
        if False:
            i = 10
            return i + 15
        (m, n) = (20, 10)
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        (A1, b1, status, message) = self.rr(A0, b0)
        assert_equal(status, 2)

    def test_infeasible_m_eq_n(self):
        if False:
            for i in range(10):
                print('nop')
        (m, n) = (10, 10)
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        A0[-1, :] = 2 * A0[-2, :]
        (A1, b1, status, message) = self.rr(A0, b0)
        assert_equal(status, 2)

    def test_infeasible_m_lt_n(self):
        if False:
            print('Hello World!')
        (m, n) = (9, 10)
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        A0[-1, :] = np.arange(m - 1).dot(A0[:-1])
        (A1, b1, status, message) = self.rr(A0, b0)
        assert_equal(status, 2)

    def test_m_gt_n(self):
        if False:
            while True:
                i = 10
        np.random.seed(2032)
        (m, n) = (20, 10)
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        x = np.linalg.solve(A0[:n, :], b0[:n])
        b0[n:] = A0[n:, :].dot(x)
        (A1, b1, status, message) = self.rr(A0, b0)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], n)
        assert_equal(np.linalg.matrix_rank(A1), n)

    def test_m_gt_n_rank_deficient(self):
        if False:
            i = 10
            return i + 15
        (m, n) = (20, 10)
        A0 = np.zeros((m, n))
        A0[:, 0] = 1
        b0 = np.ones(m)
        (A1, b1, status, message) = self.rr(A0, b0)
        assert_equal(status, 0)
        assert_allclose(A1, A0[0:1, :])
        assert_allclose(b1, b0[0])

    def test_m_lt_n_rank_deficient(self):
        if False:
            return 10
        (m, n) = (9, 10)
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        A0[-1, :] = np.arange(m - 1).dot(A0[:-1])
        b0[-1] = np.arange(m - 1).dot(b0[:-1])
        (A1, b1, status, message) = self.rr(A0, b0)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], 8)
        assert_equal(np.linalg.matrix_rank(A1), 8)

    def test_dense1(self):
        if False:
            i = 10
            return i + 15
        A = np.ones((6, 6))
        A[0, :3] = 0
        A[1, 3:] = 0
        A[3:, ::2] = -1
        A[3, :2] = 0
        A[4, 2:] = 0
        b = np.zeros(A.shape[0])
        (A1, b1, status, message) = self.rr(A, b)
        assert_(redundancy_removed(A1, A))
        assert_equal(status, 0)

    def test_dense2(self):
        if False:
            return 10
        A = np.eye(6)
        A[-2, -1] = 1
        A[-1, :] = 1
        b = np.zeros(A.shape[0])
        (A1, b1, status, message) = self.rr(A, b)
        assert_(redundancy_removed(A1, A))
        assert_equal(status, 0)

    def test_dense3(self):
        if False:
            print('Hello World!')
        A = np.eye(6)
        A[-2, -1] = 1
        A[-1, :] = 1
        b = np.random.rand(A.shape[0])
        b[-1] = np.sum(b[:-1])
        (A1, b1, status, message) = self.rr(A, b)
        assert_(redundancy_removed(A1, A))
        assert_equal(status, 0)

    def test_m_gt_n_sparse(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(2013)
        (m, n) = (20, 5)
        p = 0.1
        A = np.random.rand(m, n)
        A[np.random.rand(m, n) > p] = 0
        rank = np.linalg.matrix_rank(A)
        b = np.zeros(A.shape[0])
        (A1, b1, status, message) = self.rr(A, b)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], rank)
        assert_equal(np.linalg.matrix_rank(A1), rank)

    def test_m_lt_n_sparse(self):
        if False:
            while True:
                i = 10
        np.random.seed(2017)
        (m, n) = (20, 50)
        p = 0.05
        A = np.random.rand(m, n)
        A[np.random.rand(m, n) > p] = 0
        rank = np.linalg.matrix_rank(A)
        b = np.zeros(A.shape[0])
        (A1, b1, status, message) = self.rr(A, b)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], rank)
        assert_equal(np.linalg.matrix_rank(A1), rank)

    def test_m_eq_n_sparse(self):
        if False:
            return 10
        np.random.seed(2017)
        (m, n) = (100, 100)
        p = 0.01
        A = np.random.rand(m, n)
        A[np.random.rand(m, n) > p] = 0
        rank = np.linalg.matrix_rank(A)
        b = np.zeros(A.shape[0])
        (A1, b1, status, message) = self.rr(A, b)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], rank)
        assert_equal(np.linalg.matrix_rank(A1), rank)

    def test_magic_square(self):
        if False:
            i = 10
            return i + 15
        (A, b, c, numbers, _) = magic_square(3)
        (A1, b1, status, message) = self.rr(A, b)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], 23)
        assert_equal(np.linalg.matrix_rank(A1), 23)

    def test_magic_square2(self):
        if False:
            for i in range(10):
                print('nop')
        (A, b, c, numbers, _) = magic_square(4)
        (A1, b1, status, message) = self.rr(A, b)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], 39)
        assert_equal(np.linalg.matrix_rank(A1), 39)

class TestRRSVD(RRCommonTests):

    def rr(self, A, b):
        if False:
            return 10
        return _remove_redundancy_svd(A, b)

class TestRRPivotDense(RRCommonTests):

    def rr(self, A, b):
        if False:
            i = 10
            return i + 15
        return _remove_redundancy_pivot_dense(A, b)

class TestRRID(RRCommonTests):

    def rr(self, A, b):
        if False:
            for i in range(10):
                print('nop')
        return _remove_redundancy_id(A, b)

class TestRRPivotSparse(RRCommonTests):

    def rr(self, A, b):
        if False:
            for i in range(10):
                print('nop')
        rr_res = _remove_redundancy_pivot_sparse(csc_matrix(A), b)
        (A1, b1, status, message) = rr_res
        return (A1.toarray(), b1, status, message)