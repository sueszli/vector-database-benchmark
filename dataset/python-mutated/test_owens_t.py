import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc

def test_symmetries():
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(1234)
    (a, h) = (np.random.rand(100), np.random.rand(100))
    assert_equal(sc.owens_t(h, a), sc.owens_t(-h, a))
    assert_equal(sc.owens_t(h, a), -sc.owens_t(h, -a))

def test_special_cases():
    if False:
        while True:
            i = 10
    assert_equal(sc.owens_t(5, 0), 0)
    assert_allclose(sc.owens_t(0, 5), 0.5 * np.arctan(5) / np.pi, rtol=5e-14)
    assert_allclose(sc.owens_t(5, 1), 1.4332574485503513e-07, rtol=5e-14)

def test_nans():
    if False:
        while True:
            i = 10
    assert_equal(sc.owens_t(20, np.nan), np.nan)
    assert_equal(sc.owens_t(np.nan, 20), np.nan)
    assert_equal(sc.owens_t(np.nan, np.nan), np.nan)

def test_infs():
    if False:
        while True:
            i = 10
    (h, a) = (0, np.inf)
    res = 1 / (2 * np.pi) * np.arctan(a)
    assert_allclose(sc.owens_t(h, a), res, rtol=5e-14)
    assert_allclose(sc.owens_t(h, -a), -res, rtol=5e-14)
    h = 1
    res = 0.07932762696572854
    assert_allclose(sc.owens_t(h, np.inf), res, rtol=5e-14)
    assert_allclose(sc.owens_t(h, -np.inf), -res, rtol=5e-14)
    assert_equal(sc.owens_t(np.inf, 1), 0)
    assert_equal(sc.owens_t(-np.inf, 1), 0)
    assert_equal(sc.owens_t(np.inf, np.inf), 0)
    assert_equal(sc.owens_t(-np.inf, np.inf), 0)
    assert_equal(sc.owens_t(np.inf, -np.inf), -0.0)
    assert_equal(sc.owens_t(-np.inf, -np.inf), -0.0)