"""
Copyright (C) 2010 David Fong and Michael Saunders
Distributed under the same license as SciPy

Testing Code for LSMR.

03 Jun 2010: First version release with lsmr.py

David Chin-lung Fong            clfong@stanford.edu
Institute for Computational and Mathematical Engineering
Stanford University

Michael Saunders                saunders@stanford.edu
Systems Optimization Laboratory
Dept of MS&E, Stanford University.

"""
from numpy import array, arange, eye, zeros, ones, transpose, hstack
from numpy.linalg import norm
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg._interface import aslinearoperator
from scipy.sparse.linalg import lsmr
from .test_lsqr import G, b

class TestLSMR:

    def setup_method(self):
        if False:
            return 10
        self.n = 10
        self.m = 10

    def assertCompatibleSystem(self, A, xtrue):
        if False:
            for i in range(10):
                print('nop')
        Afun = aslinearoperator(A)
        b = Afun.matvec(xtrue)
        x = lsmr(A, b)[0]
        assert norm(x - xtrue) == pytest.approx(0, abs=1e-05)

    def testIdentityACase1(self):
        if False:
            print('Hello World!')
        A = eye(self.n)
        xtrue = zeros((self.n, 1))
        self.assertCompatibleSystem(A, xtrue)

    def testIdentityACase2(self):
        if False:
            for i in range(10):
                print('nop')
        A = eye(self.n)
        xtrue = ones((self.n, 1))
        self.assertCompatibleSystem(A, xtrue)

    def testIdentityACase3(self):
        if False:
            return 10
        A = eye(self.n)
        xtrue = transpose(arange(self.n, 0, -1))
        self.assertCompatibleSystem(A, xtrue)

    def testBidiagonalA(self):
        if False:
            while True:
                i = 10
        A = lowerBidiagonalMatrix(20, self.n)
        xtrue = transpose(arange(self.n, 0, -1))
        self.assertCompatibleSystem(A, xtrue)

    def testScalarB(self):
        if False:
            return 10
        A = array([[1.0, 2.0]])
        b = 3.0
        x = lsmr(A, b)[0]
        assert norm(A.dot(x) - b) == pytest.approx(0)

    def testComplexX(self):
        if False:
            i = 10
            return i + 15
        A = eye(self.n)
        xtrue = transpose(arange(self.n, 0, -1) * (1 + 1j))
        self.assertCompatibleSystem(A, xtrue)

    def testComplexX0(self):
        if False:
            i = 10
            return i + 15
        A = 4 * eye(self.n) + ones((self.n, self.n))
        xtrue = transpose(arange(self.n, 0, -1))
        b = aslinearoperator(A).matvec(xtrue)
        x0 = zeros(self.n, dtype=complex)
        x = lsmr(A, b, x0=x0)[0]
        assert norm(x - xtrue) == pytest.approx(0, abs=1e-05)

    def testComplexA(self):
        if False:
            return 10
        A = 4 * eye(self.n) + 1j * ones((self.n, self.n))
        xtrue = transpose(arange(self.n, 0, -1).astype(complex))
        self.assertCompatibleSystem(A, xtrue)

    def testComplexB(self):
        if False:
            while True:
                i = 10
        A = 4 * eye(self.n) + ones((self.n, self.n))
        xtrue = transpose(arange(self.n, 0, -1) * (1 + 1j))
        b = aslinearoperator(A).matvec(xtrue)
        x = lsmr(A, b)[0]
        assert norm(x - xtrue) == pytest.approx(0, abs=1e-05)

    def testColumnB(self):
        if False:
            return 10
        A = eye(self.n)
        b = ones((self.n, 1))
        x = lsmr(A, b)[0]
        assert norm(A.dot(x) - b.ravel()) == pytest.approx(0)

    def testInitialization(self):
        if False:
            for i in range(10):
                print('nop')
        (x_ref, _, itn_ref, normr_ref, *_) = lsmr(G, b)
        assert_allclose(norm(b - G @ x_ref), normr_ref, atol=1e-06)
        x0 = zeros(b.shape)
        x = lsmr(G, b, x0=x0)[0]
        assert_allclose(x, x_ref)
        x0 = lsmr(G, b, maxiter=1)[0]
        (x, _, itn, normr, *_) = lsmr(G, b, x0=x0)
        assert_allclose(norm(b - G @ x), normr, atol=1e-06)
        assert itn - itn_ref in (0, 1)
        assert normr < normr_ref * (1 + 1e-06)

class TestLSMRReturns:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.n = 10
        self.A = lowerBidiagonalMatrix(20, self.n)
        self.xtrue = transpose(arange(self.n, 0, -1))
        self.Afun = aslinearoperator(self.A)
        self.b = self.Afun.matvec(self.xtrue)
        self.x0 = ones(self.n)
        self.x00 = self.x0.copy()
        self.returnValues = lsmr(self.A, self.b)
        self.returnValuesX0 = lsmr(self.A, self.b, x0=self.x0)

    def test_unchanged_x0(self):
        if False:
            return 10
        (x, istop, itn, normr, normar, normA, condA, normx) = self.returnValuesX0
        assert_allclose(self.x00, self.x0)

    def testNormr(self):
        if False:
            print('Hello World!')
        (x, istop, itn, normr, normar, normA, condA, normx) = self.returnValues
        assert norm(self.b - self.Afun.matvec(x)) == pytest.approx(normr)

    def testNormar(self):
        if False:
            for i in range(10):
                print('nop')
        (x, istop, itn, normr, normar, normA, condA, normx) = self.returnValues
        assert norm(self.Afun.rmatvec(self.b - self.Afun.matvec(x))) == pytest.approx(normar)

    def testNormx(self):
        if False:
            while True:
                i = 10
        (x, istop, itn, normr, normar, normA, condA, normx) = self.returnValues
        assert norm(x) == pytest.approx(normx)

def lowerBidiagonalMatrix(m, n):
    if False:
        return 10
    if m <= n:
        row = hstack((arange(m, dtype=int), arange(1, m, dtype=int)))
        col = hstack((arange(m, dtype=int), arange(m - 1, dtype=int)))
        data = hstack((arange(1, m + 1, dtype=float), arange(1, m, dtype=float)))
        return coo_matrix((data, (row, col)), shape=(m, n))
    else:
        row = hstack((arange(n, dtype=int), arange(1, n + 1, dtype=int)))
        col = hstack((arange(n, dtype=int), arange(n, dtype=int)))
        data = hstack((arange(1, n + 1, dtype=float), arange(1, n + 1, dtype=float)))
        return coo_matrix((data, (row, col)), shape=(m, n))