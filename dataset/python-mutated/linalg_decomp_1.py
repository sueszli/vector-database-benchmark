"""Recipes for more efficient work with linalg using classes


intended for use for multivariate normal and linear regression
calculations

x  is the data (nobs, nvars)
m  is the moment matrix (x'x) or a covariance matrix Sigma

examples:
x'sigma^{-1}x
z = Px  where P=Sigma^{-1/2}  or P=Sigma^{1/2}

Initially assume positive definite, then add spectral cutoff and
regularization of moment matrix, and extend to PCA

maybe extend to sparse if some examples work out
(transformation matrix P for random effect and for toeplitz)


Author: josef-pktd
Created on 2010-10-20
"""
import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly

class PlainMatrixArray:
    """Class that defines linalg operation on an array

    simplest version as benchmark

    linear algebra recipes for multivariate normal and linear
    regression calculations

    """

    def __init__(self, data=None, sym=None):
        if False:
            print('Hello World!')
        if data is not None:
            if sym is None:
                self.x = np.asarray(data)
                self.m = np.dot(self.x.T, self.x)
            else:
                raise ValueError('data and sym cannot be both given')
        elif sym is not None:
            self.m = np.asarray(sym)
            self.x = np.eye(*self.m.shape)
        else:
            raise ValueError('either data or sym need to be given')

    @cache_readonly
    def minv(self):
        if False:
            i = 10
            return i + 15
        return np.linalg.inv(self.m)

    def m_y(self, y):
        if False:
            while True:
                i = 10
        return np.dot(self.m, y)

    def minv_y(self, y):
        if False:
            return 10
        return np.dot(self.minv, y)

    @cache_readonly
    def mpinv(self):
        if False:
            return 10
        return linalg.pinv(self.m)

    @cache_readonly
    def xpinv(self):
        if False:
            i = 10
            return i + 15
        return linalg.pinv(self.x)

    def yt_m_y(self, y):
        if False:
            for i in range(10):
                print('nop')
        return np.dot(y.T, np.dot(self.m, y))

    def yt_minv_y(self, y):
        if False:
            return 10
        return np.dot(y.T, np.dot(self.minv, y))

    def y_m_yt(self, y):
        if False:
            i = 10
            return i + 15
        return np.dot(y, np.dot(self.m, y.T))

    def y_minv_yt(self, y):
        if False:
            i = 10
            return i + 15
        return np.dot(y, np.dot(self.minv, y.T))

    @cache_readonly
    def mdet(self):
        if False:
            return 10
        return linalg.det(self.m)

    @cache_readonly
    def mlogdet(self):
        if False:
            i = 10
            return i + 15
        return np.log(linalg.det(self.m))

    @cache_readonly
    def meigh(self):
        if False:
            while True:
                i = 10
        (evals, evecs) = linalg.eigh(self.m)
        sortind = np.argsort(evals)[::-1]
        return (evals[sortind], evecs[:, sortind])

    @cache_readonly
    def mhalf(self):
        if False:
            i = 10
            return i + 15
        (evals, evecs) = self.meigh
        return np.dot(np.diag(evals ** 0.5), evecs.T)

    @cache_readonly
    def minvhalf(self):
        if False:
            print('Hello World!')
        (evals, evecs) = self.meigh
        return np.dot(evecs, 1.0 / np.sqrt(evals) * evecs.T)

class SvdArray(PlainMatrixArray):
    """Class that defines linalg operation on an array

    svd version, where svd is taken on original data array, if
    or when it matters

    no spectral cutoff in first version
    """

    def __init__(self, data=None, sym=None):
        if False:
            return 10
        super(SvdArray, self).__init__(data=data, sym=sym)
        (u, s, v) = np.linalg.svd(self.x, full_matrices=1)
        (self.u, self.s, self.v) = (u, s, v)
        self.sdiag = linalg.diagsvd(s, *x.shape)
        self.sinvdiag = linalg.diagsvd(1.0 / s, *x.shape)

    def _sdiagpow(self, p):
        if False:
            while True:
                i = 10
        return linalg.diagsvd(np.power(self.s, p), *x.shape)

    @cache_readonly
    def minv(self):
        if False:
            return 10
        sinvv = np.dot(self.sinvdiag, self.v)
        return np.dot(sinvv.T, sinvv)

    @cache_readonly
    def meigh(self):
        if False:
            print('Hello World!')
        evecs = self.v.T
        evals = self.s ** 2
        return (evals, evecs)

    @cache_readonly
    def mdet(self):
        if False:
            print('Hello World!')
        return self.meigh[0].prod()

    @cache_readonly
    def mlogdet(self):
        if False:
            return 10
        return np.log(self.meigh[0]).sum()

    @cache_readonly
    def mhalf(self):
        if False:
            print('Hello World!')
        return np.dot(np.diag(self.s), self.v)

    @cache_readonly
    def xxthalf(self):
        if False:
            for i in range(10):
                print('nop')
        return np.dot(self.u, self.sdiag)

    @cache_readonly
    def xxtinvhalf(self):
        if False:
            return 10
        return np.dot(self.u, self.sinvdiag)

class CholArray(PlainMatrixArray):
    """Class that defines linalg operation on an array

    cholesky version, where svd is taken on original data array, if
    or when it matters

    plan: use cholesky factor and cholesky solve
    nothing implemented yet
    """

    def __init__(self, data=None, sym=None):
        if False:
            while True:
                i = 10
        super(SvdArray, self).__init__(data=data, sym=sym)

    def yt_minv_y(self, y):
        if False:
            return 10
        'xSigmainvx\n        does not use stored cholesky yet\n        '
        return np.dot(x, linalg.cho_solve(linalg.cho_factor(self.m), x))

def testcompare(m1, m2):
    if False:
        print('Hello World!')
    from numpy.testing import assert_almost_equal, assert_approx_equal
    decimal = 12
    assert_almost_equal(m1.minv, m2.minv, decimal=decimal)
    s1 = np.sign(m1.mhalf.sum(1))[:, None]
    s2 = np.sign(m2.mhalf.sum(1))[:, None]
    scorr = s1 / s2
    assert_almost_equal(m1.mhalf, m2.mhalf * scorr, decimal=decimal)
    assert_almost_equal(m1.minvhalf, m2.minvhalf, decimal=decimal)
    (evals1, evecs1) = m1.meigh
    (evals2, evecs2) = m2.meigh
    assert_almost_equal(evals1, evals2, decimal=decimal)
    s1 = np.sign(evecs1.sum(0))
    s2 = np.sign(evecs2.sum(0))
    scorr = s1 / s2
    assert_almost_equal(evecs1, evecs2 * scorr, decimal=decimal)
    assert_approx_equal(m1.mdet, m2.mdet, significant=13)
    assert_approx_equal(m1.mlogdet, m2.mlogdet, significant=13)

def tiny2zero(x, eps=1e-15):
    if False:
        for i in range(10):
            print('nop')
    'replace abs values smaller than eps by zero, makes copy\n    '
    mask = np.abs(x.copy()) < eps
    x[mask] = 0
    return x

def maxabs(x):
    if False:
        print('Hello World!')
    return np.max(np.abs(x))
if __name__ == '__main__':
    n = 5
    y = np.arange(n)
    x = np.random.randn(100, n)
    autocov = 2 * 0.8 ** np.arange(n) + 0.01 * np.random.randn(n)
    sigma = linalg.toeplitz(autocov)
    mat = PlainMatrixArray(sym=sigma)
    print(tiny2zero(mat.mhalf))
    mih = mat.minvhalf
    print(tiny2zero(mih))
    mat2 = PlainMatrixArray(data=x)
    print(maxabs(mat2.yt_minv_y(np.dot(x.T, x)) - mat2.m))
    print(tiny2zero(mat2.minv_y(mat2.m)))
    mat3 = SvdArray(data=x)
    print(mat3.meigh[0])
    print(mat2.meigh[0])
    testcompare(mat2, mat3)
    '\n    m = np.dot(x.T, x)\n\n    u,s,v = np.linalg.svd(x, full_matrices=1)\n    Sig = linalg.diagsvd(s,*x.shape)\n\n    >>> np.max(np.abs(np.dot(u, np.dot(Sig, v)) - x))\n    3.1086244689504383e-015\n    >>> np.max(np.abs(np.dot(u.T, u) - np.eye(100)))\n    3.3306690738754696e-016\n    >>> np.max(np.abs(np.dot(v.T, v) - np.eye(5)))\n    6.6613381477509392e-016\n    >>> np.max(np.abs(np.dot(Sig.T, Sig) - np.diag(s**2)))\n    5.6843418860808015e-014\n\n    >>> evals,evecs = linalg.eigh(np.dot(x.T, x))\n    >>> evals[::-1]\n    array([ 123.36404464,  112.17036442,  102.04198468,   76.60832278,\n             74.70484487])\n\n    >>> s**2\n    array([ 123.36404464,  112.17036442,  102.04198468,   76.60832278,\n             74.70484487])\n\n    >>> np.max(np.abs(np.dot(v.T, np.dot(np.diag(s**2), v)) - m))\n    1.1368683772161603e-013\n\n    >>> us = np.dot(u, Sig)\n    >>> np.max(np.abs(np.dot(us, us.T) - np.dot(x, x.T)))\n    1.0658141036401503e-014\n\n    >>> sv = np.dot(Sig, v)\n    >>> np.max(np.abs(np.dot(sv.T, sv) - np.dot(x.T, x)))\n    1.1368683772161603e-013\n\n\n    '