from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest import raises as assert_raises
from numpy import array, transpose, dot, conjugate, zeros_like, empty
from numpy.random import random
from scipy.linalg import cholesky, cholesky_banded, cho_solve_banded, cho_factor, cho_solve
from scipy.linalg._testutils import assert_no_overwrite

class TestCholesky:

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        a = [[8, 2, 3], [2, 9, 3], [3, 3, 6]]
        c = cholesky(a)
        assert_array_almost_equal(dot(transpose(c), c), a)
        c = transpose(c)
        a = dot(c, transpose(c))
        assert_array_almost_equal(cholesky(a, lower=1), c)

    def test_check_finite(self):
        if False:
            print('Hello World!')
        a = [[8, 2, 3], [2, 9, 3], [3, 3, 6]]
        c = cholesky(a, check_finite=False)
        assert_array_almost_equal(dot(transpose(c), c), a)
        c = transpose(c)
        a = dot(c, transpose(c))
        assert_array_almost_equal(cholesky(a, lower=1, check_finite=False), c)

    def test_simple_complex(self):
        if False:
            for i in range(10):
                print('nop')
        m = array([[3 + 1j, 3 + 4j, 5], [0, 2 + 2j, 2 + 7j], [0, 0, 7 + 4j]])
        a = dot(transpose(conjugate(m)), m)
        c = cholesky(a)
        a1 = dot(transpose(conjugate(c)), c)
        assert_array_almost_equal(a, a1)
        c = transpose(c)
        a = dot(c, transpose(conjugate(c)))
        assert_array_almost_equal(cholesky(a, lower=1), c)

    def test_random(self):
        if False:
            return 10
        n = 20
        for k in range(2):
            m = random([n, n])
            for i in range(n):
                m[i, i] = 20 * (0.1 + m[i, i])
            a = dot(transpose(m), m)
            c = cholesky(a)
            a1 = dot(transpose(c), c)
            assert_array_almost_equal(a, a1)
            c = transpose(c)
            a = dot(c, transpose(c))
            assert_array_almost_equal(cholesky(a, lower=1), c)

    def test_random_complex(self):
        if False:
            while True:
                i = 10
        n = 20
        for k in range(2):
            m = random([n, n]) + 1j * random([n, n])
            for i in range(n):
                m[i, i] = 20 * (0.1 + abs(m[i, i]))
            a = dot(transpose(conjugate(m)), m)
            c = cholesky(a)
            a1 = dot(transpose(conjugate(c)), c)
            assert_array_almost_equal(a, a1)
            c = transpose(c)
            a = dot(c, transpose(conjugate(c)))
            assert_array_almost_equal(cholesky(a, lower=1), c)

class TestCholeskyBanded:
    """Tests for cholesky_banded() and cho_solve_banded."""

    def test_check_finite(self):
        if False:
            print('Hello World!')
        a = array([[4.0, 1.0, 0.0, 0.0], [1.0, 4.0, 0.5, 0.0], [0.0, 0.5, 4.0, 0.2], [0.0, 0.0, 0.2, 4.0]])
        ab = array([[-1.0, 1.0, 0.5, 0.2], [4.0, 4.0, 4.0, 4.0]])
        c = cholesky_banded(ab, lower=False, check_finite=False)
        ufac = zeros_like(a)
        ufac[list(range(4)), list(range(4))] = c[-1]
        ufac[(0, 1, 2), (1, 2, 3)] = c[0, 1:]
        assert_array_almost_equal(a, dot(ufac.T, ufac))
        b = array([0.0, 0.5, 4.2, 4.2])
        x = cho_solve_banded((c, False), b, check_finite=False)
        assert_array_almost_equal(x, [0.0, 0.0, 1.0, 1.0])

    def test_upper_real(self):
        if False:
            return 10
        a = array([[4.0, 1.0, 0.0, 0.0], [1.0, 4.0, 0.5, 0.0], [0.0, 0.5, 4.0, 0.2], [0.0, 0.0, 0.2, 4.0]])
        ab = array([[-1.0, 1.0, 0.5, 0.2], [4.0, 4.0, 4.0, 4.0]])
        c = cholesky_banded(ab, lower=False)
        ufac = zeros_like(a)
        ufac[list(range(4)), list(range(4))] = c[-1]
        ufac[(0, 1, 2), (1, 2, 3)] = c[0, 1:]
        assert_array_almost_equal(a, dot(ufac.T, ufac))
        b = array([0.0, 0.5, 4.2, 4.2])
        x = cho_solve_banded((c, False), b)
        assert_array_almost_equal(x, [0.0, 0.0, 1.0, 1.0])

    def test_upper_complex(self):
        if False:
            i = 10
            return i + 15
        a = array([[4.0, 1.0, 0.0, 0.0], [1.0, 4.0, 0.5, 0.0], [0.0, 0.5, 4.0, -0.2j], [0.0, 0.0, 0.2j, 4.0]])
        ab = array([[-1.0, 1.0, 0.5, -0.2j], [4.0, 4.0, 4.0, 4.0]])
        c = cholesky_banded(ab, lower=False)
        ufac = zeros_like(a)
        ufac[list(range(4)), list(range(4))] = c[-1]
        ufac[(0, 1, 2), (1, 2, 3)] = c[0, 1:]
        assert_array_almost_equal(a, dot(ufac.conj().T, ufac))
        b = array([0.0, 0.5, 4.0 - 0.2j, 0.2j + 4.0])
        x = cho_solve_banded((c, False), b)
        assert_array_almost_equal(x, [0.0, 0.0, 1.0, 1.0])

    def test_lower_real(self):
        if False:
            while True:
                i = 10
        a = array([[4.0, 1.0, 0.0, 0.0], [1.0, 4.0, 0.5, 0.0], [0.0, 0.5, 4.0, 0.2], [0.0, 0.0, 0.2, 4.0]])
        ab = array([[4.0, 4.0, 4.0, 4.0], [1.0, 0.5, 0.2, -1.0]])
        c = cholesky_banded(ab, lower=True)
        lfac = zeros_like(a)
        lfac[list(range(4)), list(range(4))] = c[0]
        lfac[(1, 2, 3), (0, 1, 2)] = c[1, :3]
        assert_array_almost_equal(a, dot(lfac, lfac.T))
        b = array([0.0, 0.5, 4.2, 4.2])
        x = cho_solve_banded((c, True), b)
        assert_array_almost_equal(x, [0.0, 0.0, 1.0, 1.0])

    def test_lower_complex(self):
        if False:
            while True:
                i = 10
        a = array([[4.0, 1.0, 0.0, 0.0], [1.0, 4.0, 0.5, 0.0], [0.0, 0.5, 4.0, -0.2j], [0.0, 0.0, 0.2j, 4.0]])
        ab = array([[4.0, 4.0, 4.0, 4.0], [1.0, 0.5, 0.2j, -1.0]])
        c = cholesky_banded(ab, lower=True)
        lfac = zeros_like(a)
        lfac[list(range(4)), list(range(4))] = c[0]
        lfac[(1, 2, 3), (0, 1, 2)] = c[1, :3]
        assert_array_almost_equal(a, dot(lfac, lfac.conj().T))
        b = array([0.0, 0.5j, 3.8j, 3.8])
        x = cho_solve_banded((c, True), b)
        assert_array_almost_equal(x, [0.0, 0.0, 1j, 1.0])

class TestOverwrite:

    def test_cholesky(self):
        if False:
            for i in range(10):
                print('nop')
        assert_no_overwrite(cholesky, [(3, 3)])

    def test_cho_factor(self):
        if False:
            i = 10
            return i + 15
        assert_no_overwrite(cho_factor, [(3, 3)])

    def test_cho_solve(self):
        if False:
            return 10
        x = array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        xcho = cho_factor(x)
        assert_no_overwrite(lambda b: cho_solve(xcho, b), [(3,)])

    def test_cholesky_banded(self):
        if False:
            while True:
                i = 10
        assert_no_overwrite(cholesky_banded, [(2, 3)])

    def test_cho_solve_banded(self):
        if False:
            return 10
        x = array([[0, -1, -1], [2, 2, 2]])
        xcho = cholesky_banded(x)
        assert_no_overwrite(lambda b: cho_solve_banded((xcho, False), b), [(3,)])

class TestEmptyArray:

    def test_cho_factor_empty_square(self):
        if False:
            return 10
        a = empty((0, 0))
        b = array([])
        c = array([[]])
        d = []
        e = [[]]
        (x, _) = cho_factor(a)
        assert_array_equal(x, a)
        for x in [b, c, d, e]:
            assert_raises(ValueError, cho_factor, x)