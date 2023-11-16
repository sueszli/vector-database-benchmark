import numpy as np
from scipy.linalg import bandwidth, issymmetric, ishermitian
import pytest
from pytest import raises

def test_bandwidth_dtypes():
    if False:
        i = 10
        return i + 15
    n = 5
    for t in np.typecodes['All']:
        A = np.zeros([n, n], dtype=t)
        if t in 'eUVOMm':
            raises(TypeError, bandwidth, A)
        elif t == 'G':
            pass
        else:
            _ = bandwidth(A)

def test_bandwidth_non2d_input():
    if False:
        for i in range(10):
            print('nop')
    A = np.array([1, 2, 3])
    raises(ValueError, bandwidth, A)
    A = np.array([[[1, 2, 3], [4, 5, 6]]])
    raises(ValueError, bandwidth, A)

@pytest.mark.parametrize('T', [x for x in np.typecodes['All'] if x not in 'eGUVOMm'])
def test_bandwidth_square_inputs(T):
    if False:
        print('Hello World!')
    n = 20
    k = 4
    R = np.zeros([n, n], dtype=T, order='F')
    R[[x for x in range(n)], [x for x in range(n)]] = 1
    R[[x for x in range(n - k)], [x for x in range(k, n)]] = 1
    R[[x for x in range(1, n)], [x for x in range(n - 1)]] = 1
    R[[x for x in range(k, n)], [x for x in range(n - k)]] = 1
    assert bandwidth(R) == (k, k)

@pytest.mark.parametrize('T', [x for x in np.typecodes['All'] if x not in 'eGUVOMm'])
def test_bandwidth_rect_inputs(T):
    if False:
        return 10
    (n, m) = (10, 20)
    k = 5
    R = np.zeros([n, m], dtype=T, order='F')
    R[[x for x in range(n)], [x for x in range(n)]] = 1
    R[[x for x in range(n - k)], [x for x in range(k, n)]] = 1
    R[[x for x in range(1, n)], [x for x in range(n - 1)]] = 1
    R[[x for x in range(k, n)], [x for x in range(n - k)]] = 1
    assert bandwidth(R) == (k, k)

def test_issymetric_ishermitian_dtypes():
    if False:
        for i in range(10):
            print('nop')
    n = 5
    for t in np.typecodes['All']:
        A = np.zeros([n, n], dtype=t)
        if t in 'eUVOMm':
            raises(TypeError, issymmetric, A)
            raises(TypeError, ishermitian, A)
        elif t == 'G':
            pass
        else:
            assert issymmetric(A)
            assert ishermitian(A)

def test_issymmetric_ishermitian_invalid_input():
    if False:
        i = 10
        return i + 15
    A = np.array([1, 2, 3])
    raises(ValueError, issymmetric, A)
    raises(ValueError, ishermitian, A)
    A = np.array([[[1, 2, 3], [4, 5, 6]]])
    raises(ValueError, issymmetric, A)
    raises(ValueError, ishermitian, A)
    A = np.array([[1, 2, 3], [4, 5, 6]])
    raises(ValueError, issymmetric, A)
    raises(ValueError, ishermitian, A)

def test_issymetric_complex_decimals():
    if False:
        return 10
    A = np.arange(1, 10).astype(complex).reshape(3, 3)
    A += np.arange(-4, 5).astype(complex).reshape(3, 3) * 1j
    A /= np.pi
    A = A + A.T
    assert issymmetric(A)

def test_ishermitian_complex_decimals():
    if False:
        i = 10
        return i + 15
    A = np.arange(1, 10).astype(complex).reshape(3, 3)
    A += np.arange(-4, 5).astype(complex).reshape(3, 3) * 1j
    A /= np.pi
    A = A + A.T.conj()
    assert ishermitian(A)

def test_issymmetric_approximate_results():
    if False:
        while True:
            i = 10
    n = 20
    rng = np.random.RandomState(123456789)
    x = rng.uniform(high=5.0, size=[n, n])
    y = x @ x.T
    p = rng.standard_normal([n, n])
    z = p @ y @ p.T
    assert issymmetric(z, atol=1e-10)
    assert issymmetric(z, atol=1e-10, rtol=0.0)
    assert issymmetric(z, atol=0.0, rtol=1e-12)
    assert issymmetric(z, atol=1e-13, rtol=1e-12)

def test_ishermitian_approximate_results():
    if False:
        i = 10
        return i + 15
    n = 20
    rng = np.random.RandomState(987654321)
    x = rng.uniform(high=5.0, size=[n, n])
    y = x @ x.T
    p = rng.standard_normal([n, n]) + rng.standard_normal([n, n]) * 1j
    z = p @ y @ p.conj().T
    assert ishermitian(z, atol=1e-10)
    assert ishermitian(z, atol=1e-10, rtol=0.0)
    assert ishermitian(z, atol=0.0, rtol=1e-12)
    assert ishermitian(z, atol=1e-13, rtol=1e-12)