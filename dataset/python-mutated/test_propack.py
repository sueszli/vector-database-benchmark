import os
import pytest
import sys
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.linalg._svdp import _svdp
from scipy.sparse import csr_matrix, csc_matrix
TOLS = {np.float32: 0.0001, np.float64: 1e-08, np.complex64: 0.0001, np.complex128: 1e-08}

def is_complex_type(dtype):
    if False:
        return 10
    return np.dtype(dtype).kind == 'c'

def is_32bit():
    if False:
        for i in range(10):
            print('nop')
    return sys.maxsize <= 2 ** 32

def is_windows():
    if False:
        i = 10
        return i + 15
    return 'win32' in sys.platform
_dtypes = []
for dtype_flavour in TOLS.keys():
    marks = []
    if is_complex_type(dtype_flavour):
        if is_32bit():
            marks = [pytest.mark.skip]
        elif is_windows() and np.dtype(dtype_flavour).itemsize == 16:
            marks = [pytest.mark.skip]
        else:
            marks = [pytest.mark.slow]
    _dtypes.append(pytest.param(dtype_flavour, marks=marks, id=dtype_flavour.__name__))
_dtypes = tuple(_dtypes)

def generate_matrix(constructor, n, m, f, dtype=float, rseed=0, **kwargs):
    if False:
        print('Hello World!')
    'Generate a random sparse matrix'
    rng = np.random.RandomState(rseed)
    if is_complex_type(dtype):
        M = (-5 + 10 * rng.rand(n, m) - 5j + 10j * rng.rand(n, m)).astype(dtype)
    else:
        M = (-5 + 10 * rng.rand(n, m)).astype(dtype)
    M[M.real > 10 * f - 5] = 0
    return constructor(M, **kwargs)

def assert_orthogonal(u1, u2, rtol, atol):
    if False:
        print('Hello World!')
    'Check that the first k rows of u1 and u2 are orthogonal'
    A = abs(np.dot(u1.conj().T, u2))
    assert_allclose(A, np.eye(u1.shape[1], u2.shape[1]), rtol=rtol, atol=atol)

def check_svdp(n, m, constructor, dtype, k, irl_mode, which, f=0.8):
    if False:
        print('Hello World!')
    tol = TOLS[dtype]
    M = generate_matrix(np.asarray, n, m, f, dtype)
    Msp = constructor(M)
    (u1, sigma1, vt1) = np.linalg.svd(M, full_matrices=False)
    (u2, sigma2, vt2, _) = _svdp(Msp, k=k, which=which, irl_mode=irl_mode, tol=tol)
    if which.upper() == 'SM':
        u1 = np.roll(u1, k, 1)
        vt1 = np.roll(vt1, k, 0)
        sigma1 = np.roll(sigma1, k)
    assert_allclose(sigma1[:k], sigma2, rtol=tol, atol=tol)
    assert_orthogonal(u1, u2, rtol=tol, atol=tol)
    assert_orthogonal(vt1.T, vt2.T, rtol=tol, atol=tol)

@pytest.mark.parametrize('ctor', (np.array, csr_matrix, csc_matrix))
@pytest.mark.parametrize('dtype', _dtypes)
@pytest.mark.parametrize('irl', (True, False))
@pytest.mark.parametrize('which', ('LM', 'SM'))
def test_svdp(ctor, dtype, irl, which):
    if False:
        while True:
            i = 10
    np.random.seed(0)
    (n, m, k) = (10, 20, 3)
    if which == 'SM' and (not irl):
        message = "`which`='SM' requires irl_mode=True"
        with assert_raises(ValueError, match=message):
            check_svdp(n, m, ctor, dtype, k, irl, which)
    elif is_32bit() and is_complex_type(dtype):
        message = 'PROPACK complex-valued SVD methods not available '
        with assert_raises(TypeError, match=message):
            check_svdp(n, m, ctor, dtype, k, irl, which)
    else:
        check_svdp(n, m, ctor, dtype, k, irl, which)

@pytest.mark.parametrize('dtype', _dtypes)
@pytest.mark.parametrize('irl', (False, True))
@pytest.mark.timeout(120)
def test_examples(dtype, irl):
    if False:
        return 10
    atol = {np.float32: 0.00013, np.float64: 1e-09, np.complex64: 0.001, np.complex128: 1e-09}[dtype]
    path_prefix = os.path.dirname(__file__)
    relative_path = 'propack_test_data.npz'
    filename = os.path.join(path_prefix, relative_path)
    data = np.load(filename, allow_pickle=True)
    if is_complex_type(dtype):
        A = data['A_complex'].item().astype(dtype)
    else:
        A = data['A_real'].item().astype(dtype)
    k = 200
    (u, s, vh, _) = _svdp(A, k, irl_mode=irl, random_state=0)
    sv_check = 27 if is_complex_type(dtype) else k
    u = u[:, :sv_check]
    vh = vh[:sv_check, :]
    s = s[:sv_check]
    assert_allclose(np.eye(u.shape[1]), u.conj().T @ u, atol=atol)
    assert_allclose(np.eye(vh.shape[0]), vh @ vh.conj().T, atol=atol)
    (u3, s3, vh3) = np.linalg.svd(A.todense())
    u3 = u3[:, :sv_check]
    s3 = s3[:sv_check]
    vh3 = vh3[:sv_check, :]
    A3 = u3 @ np.diag(s3) @ vh3
    recon = u @ np.diag(s) @ vh
    assert_allclose(np.linalg.norm(A3 - recon), 0, atol=atol)

@pytest.mark.parametrize('shifts', (None, -10, 0, 1, 10, 70))
@pytest.mark.parametrize('dtype', _dtypes[:2])
def test_shifts(shifts, dtype):
    if False:
        return 10
    np.random.seed(0)
    (n, k) = (70, 10)
    A = np.random.random((n, n))
    if shifts is not None and (shifts < 0 or k > min(n - 1 - shifts, n)):
        with pytest.raises(ValueError):
            _svdp(A, k, shifts=shifts, kmax=5 * k, irl_mode=True)
    else:
        _svdp(A, k, shifts=shifts, kmax=5 * k, irl_mode=True)

@pytest.mark.slow
@pytest.mark.xfail()
def test_shifts_accuracy():
    if False:
        return 10
    np.random.seed(0)
    (n, k) = (70, 10)
    A = np.random.random((n, n)).astype(np.float64)
    (u1, s1, vt1, _) = _svdp(A, k, shifts=None, which='SM', irl_mode=True)
    (u2, s2, vt2, _) = _svdp(A, k, shifts=32, which='SM', irl_mode=True)
    assert_allclose(s1, s2)