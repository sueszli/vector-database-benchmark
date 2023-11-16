import numpy as np
from numpy.testing import assert_allclose, assert_
from scipy.special._testutils import FuncData
from scipy.special import gamma, gammaln, loggamma

def test_identities1():
    if False:
        for i in range(10):
            print('nop')
    x = np.array([-99.5, -9.5, -0.5, 0.5, 9.5, 99.5])
    y = x.copy()
    (x, y) = np.meshgrid(x, y)
    z = (x + 1j * y).flatten()
    dataset = np.vstack((z, gamma(z))).T

    def f(z):
        if False:
            for i in range(10):
                print('nop')
        return np.exp(loggamma(z))
    FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()

def test_identities2():
    if False:
        for i in range(10):
            print('nop')
    x = np.array([-99.5, -9.5, -0.5, 0.5, 9.5, 99.5])
    y = x.copy()
    (x, y) = np.meshgrid(x, y)
    z = (x + 1j * y).flatten()
    dataset = np.vstack((z, np.log(z) + loggamma(z))).T

    def f(z):
        if False:
            return 10
        return loggamma(z + 1)
    FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()

def test_complex_dispatch_realpart():
    if False:
        print('Hello World!')
    x = np.r_[-np.logspace(10, -10), np.logspace(-10, 10)] + 0.5
    dataset = np.vstack((x, gammaln(x))).T

    def f(z):
        if False:
            while True:
                i = 10
        z = np.array(z, dtype='complex128')
        return loggamma(z).real
    FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()

def test_real_dispatch():
    if False:
        for i in range(10):
            print('nop')
    x = np.logspace(-10, 10) + 0.5
    dataset = np.vstack((x, gammaln(x))).T
    FuncData(loggamma, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()
    assert_(loggamma(0) == np.inf)
    assert_(np.isnan(loggamma(-1)))

def test_gh_6536():
    if False:
        for i in range(10):
            print('nop')
    z = loggamma(complex(-3.4, +0.0))
    zbar = loggamma(complex(-3.4, -0.0))
    assert_allclose(z, zbar.conjugate(), rtol=1e-15, atol=0)

def test_branch_cut():
    if False:
        for i in range(10):
            print('nop')
    x = -np.logspace(300, -30, 100)
    z = np.asarray([complex(x0, 0.0) for x0 in x])
    zbar = np.asarray([complex(x0, -0.0) for x0 in x])
    assert_allclose(z, zbar.conjugate(), rtol=1e-15, atol=0)