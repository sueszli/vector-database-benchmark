from itertools import product, permutations
import numpy as np
from numpy.testing import assert_array_less, assert_allclose
from pytest import raises as assert_raises
from scipy.linalg import inv, eigh, norm
from scipy.linalg import orthogonal_procrustes
from scipy.sparse._sputils import matrix

def test_orthogonal_procrustes_ndim_too_large():
    if False:
        while True:
            i = 10
    np.random.seed(1234)
    A = np.random.randn(3, 4, 5)
    B = np.random.randn(3, 4, 5)
    assert_raises(ValueError, orthogonal_procrustes, A, B)

def test_orthogonal_procrustes_ndim_too_small():
    if False:
        i = 10
        return i + 15
    np.random.seed(1234)
    A = np.random.randn(3)
    B = np.random.randn(3)
    assert_raises(ValueError, orthogonal_procrustes, A, B)

def test_orthogonal_procrustes_shape_mismatch():
    if False:
        print('Hello World!')
    np.random.seed(1234)
    shapes = ((3, 3), (3, 4), (4, 3), (4, 4))
    for (a, b) in permutations(shapes, 2):
        A = np.random.randn(*a)
        B = np.random.randn(*b)
        assert_raises(ValueError, orthogonal_procrustes, A, B)

def test_orthogonal_procrustes_checkfinite_exception():
    if False:
        while True:
            i = 10
    np.random.seed(1234)
    (m, n) = (2, 3)
    A_good = np.random.randn(m, n)
    B_good = np.random.randn(m, n)
    for bad_value in (np.inf, -np.inf, np.nan):
        A_bad = A_good.copy()
        A_bad[1, 2] = bad_value
        B_bad = B_good.copy()
        B_bad[1, 2] = bad_value
        for (A, B) in ((A_good, B_bad), (A_bad, B_good), (A_bad, B_bad)):
            assert_raises(ValueError, orthogonal_procrustes, A, B)

def test_orthogonal_procrustes_scale_invariance():
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(1234)
    (m, n) = (4, 3)
    for i in range(3):
        A_orig = np.random.randn(m, n)
        B_orig = np.random.randn(m, n)
        (R_orig, s) = orthogonal_procrustes(A_orig, B_orig)
        for A_scale in np.square(np.random.randn(3)):
            for B_scale in np.square(np.random.randn(3)):
                (R, s) = orthogonal_procrustes(A_orig * A_scale, B_orig * B_scale)
                assert_allclose(R, R_orig)

def test_orthogonal_procrustes_array_conversion():
    if False:
        i = 10
        return i + 15
    np.random.seed(1234)
    for (m, n) in ((6, 4), (4, 4), (4, 6)):
        A_arr = np.random.randn(m, n)
        B_arr = np.random.randn(m, n)
        As = (A_arr, A_arr.tolist(), matrix(A_arr))
        Bs = (B_arr, B_arr.tolist(), matrix(B_arr))
        (R_arr, s) = orthogonal_procrustes(A_arr, B_arr)
        AR_arr = A_arr.dot(R_arr)
        for (A, B) in product(As, Bs):
            (R, s) = orthogonal_procrustes(A, B)
            AR = A_arr.dot(R)
            assert_allclose(AR, AR_arr)

def test_orthogonal_procrustes():
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(1234)
    for (m, n) in ((6, 4), (4, 4), (4, 6)):
        B = np.random.randn(m, n)
        X = np.random.randn(n, n)
        (w, V) = eigh(X.T + X)
        assert_allclose(inv(V), V.T)
        A = np.dot(B, V.T)
        (R, s) = orthogonal_procrustes(A, B)
        assert_allclose(inv(R), R.T)
        assert_allclose(A.dot(R), B)
        A_perturbed = A + 0.01 * np.random.randn(m, n)
        (R_prime, s) = orthogonal_procrustes(A_perturbed, B)
        assert_allclose(inv(R_prime), R_prime.T)
        naive_approx = A_perturbed.dot(R)
        optim_approx = A_perturbed.dot(R_prime)
        naive_approx_error = norm(naive_approx - B, ord='fro')
        optim_approx_error = norm(optim_approx - B, ord='fro')
        assert_array_less(optim_approx_error, naive_approx_error)

def _centered(A):
    if False:
        print('Hello World!')
    mu = A.mean(axis=0)
    return (A - mu, mu)

def test_orthogonal_procrustes_exact_example():
    if False:
        print('Hello World!')
    A_orig = np.array([[-3, 3], [-2, 3], [-2, 2], [-3, 2]], dtype=float)
    B_orig = np.array([[3, 2], [1, 0], [3, -2], [5, 0]], dtype=float)
    (A, A_mu) = _centered(A_orig)
    (B, B_mu) = _centered(B_orig)
    (R, s) = orthogonal_procrustes(A, B)
    scale = s / np.square(norm(A))
    B_approx = scale * np.dot(A, R) + B_mu
    assert_allclose(B_approx, B_orig, atol=1e-08)

def test_orthogonal_procrustes_stretched_example():
    if False:
        print('Hello World!')
    A_orig = np.array([[-3, 3], [-2, 3], [-2, 2], [-3, 2]], dtype=float)
    B_orig = np.array([[3, 40], [1, 0], [3, -40], [5, 0]], dtype=float)
    (A, A_mu) = _centered(A_orig)
    (B, B_mu) = _centered(B_orig)
    (R, s) = orthogonal_procrustes(A, B)
    scale = s / np.square(norm(A))
    B_approx = scale * np.dot(A, R) + B_mu
    expected = np.array([[3, 21], [-18, 0], [3, -21], [24, 0]], dtype=float)
    assert_allclose(B_approx, expected, atol=1e-08)
    expected_disparity = 0.4501246882793018
    AB_disparity = np.square(norm(B_approx - B_orig) / norm(B))
    assert_allclose(AB_disparity, expected_disparity)
    (R, s) = orthogonal_procrustes(B, A)
    scale = s / np.square(norm(B))
    A_approx = scale * np.dot(B, R) + A_mu
    BA_disparity = np.square(norm(A_approx - A_orig) / norm(A))
    assert_allclose(BA_disparity, expected_disparity)

def test_orthogonal_procrustes_skbio_example():
    if False:
        return 10
    A_orig = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], dtype=float)
    B_orig = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], dtype=float)
    B_standardized = np.array([[-0.13363062, 0.6681531], [-0.13363062, 0.13363062], [-0.13363062, -0.40089186], [0.40089186, -0.40089186]])
    (A, A_mu) = _centered(A_orig)
    (B, B_mu) = _centered(B_orig)
    (R, s) = orthogonal_procrustes(A, B)
    scale = s / np.square(norm(A))
    B_approx = scale * np.dot(A, R) + B_mu
    assert_allclose(B_approx, B_orig)
    assert_allclose(B / norm(B), B_standardized)