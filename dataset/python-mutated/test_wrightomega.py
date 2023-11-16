import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
import scipy.special as sc
from scipy.special._testutils import assert_func_equal

def test_wrightomega_nan():
    if False:
        for i in range(10):
            print('nop')
    pts = [complex(np.nan, 0), complex(0, np.nan), complex(np.nan, np.nan), complex(np.nan, 1), complex(1, np.nan)]
    for p in pts:
        res = sc.wrightomega(p)
        assert_(np.isnan(res.real))
        assert_(np.isnan(res.imag))

def test_wrightomega_inf_branch():
    if False:
        i = 10
        return i + 15
    pts = [complex(-np.inf, np.pi / 4), complex(-np.inf, -np.pi / 4), complex(-np.inf, 3 * np.pi / 4), complex(-np.inf, -3 * np.pi / 4)]
    expected_results = [complex(0.0, 0.0), complex(0.0, -0.0), complex(-0.0, 0.0), complex(-0.0, -0.0)]
    for (p, expected) in zip(pts, expected_results):
        res = sc.wrightomega(p)
        assert_equal(res.real, expected.real)
        assert_equal(res.imag, expected.imag)

def test_wrightomega_inf():
    if False:
        while True:
            i = 10
    pts = [complex(np.inf, 10), complex(-np.inf, 10), complex(10, np.inf), complex(10, -np.inf)]
    for p in pts:
        assert_equal(sc.wrightomega(p), p)

def test_wrightomega_singular():
    if False:
        for i in range(10):
            print('nop')
    pts = [complex(-1.0, np.pi), complex(-1.0, -np.pi)]
    for p in pts:
        res = sc.wrightomega(p)
        assert_equal(res, -1.0)
        assert_(np.signbit(res.imag) == np.bool_(False))

@pytest.mark.parametrize('x, desired', [(-np.inf, 0), (np.inf, np.inf)])
def test_wrightomega_real_infinities(x, desired):
    if False:
        print('Hello World!')
    assert sc.wrightomega(x) == desired

def test_wrightomega_real_nan():
    if False:
        return 10
    assert np.isnan(sc.wrightomega(np.nan))

def test_wrightomega_real_series_crossover():
    if False:
        for i in range(10):
            print('nop')
    desired_error = 2 * np.finfo(float).eps
    crossover = 1e+20
    x_before_crossover = np.nextafter(crossover, -np.inf)
    x_after_crossover = np.nextafter(crossover, np.inf)
    desired_before_crossover = 9.999999999999998e+19
    desired_after_crossover = 1.0000000000000002e+20
    assert_allclose(sc.wrightomega(x_before_crossover), desired_before_crossover, atol=0, rtol=desired_error)
    assert_allclose(sc.wrightomega(x_after_crossover), desired_after_crossover, atol=0, rtol=desired_error)

def test_wrightomega_exp_approximation_crossover():
    if False:
        print('Hello World!')
    desired_error = 2 * np.finfo(float).eps
    crossover = -50
    x_before_crossover = np.nextafter(crossover, np.inf)
    x_after_crossover = np.nextafter(crossover, -np.inf)
    desired_before_crossover = 1.9287498479639315e-22
    desired_after_crossover = 1.9287498479639042e-22
    assert_allclose(sc.wrightomega(x_before_crossover), desired_before_crossover, atol=0, rtol=desired_error)
    assert_allclose(sc.wrightomega(x_after_crossover), desired_after_crossover, atol=0, rtol=desired_error)

def test_wrightomega_real_versus_complex():
    if False:
        i = 10
        return i + 15
    x = np.linspace(-500, 500, 1001)
    results = sc.wrightomega(x + 0j).real
    assert_func_equal(sc.wrightomega, results, x, atol=0, rtol=1e-14)