import numpy as np
from numpy.testing import assert_, assert_allclose
import pytest
from scipy.special import _ufuncs
import scipy.special._orthogonal as orth
from scipy.special._testutils import FuncData

def test_eval_chebyt():
    if False:
        for i in range(10):
            print('nop')
    n = np.arange(0, 10000, 7)
    x = 2 * np.random.rand() - 1
    v1 = np.cos(n * np.arccos(x))
    v2 = _ufuncs.eval_chebyt(n, x)
    assert_(np.allclose(v1, v2, rtol=1e-15))

def test_eval_genlaguerre_restriction():
    if False:
        while True:
            i = 10
    assert_(np.isnan(_ufuncs.eval_genlaguerre(0, -1, 0)))
    assert_(np.isnan(_ufuncs.eval_genlaguerre(0.1, -1, 0)))

def test_warnings():
    if False:
        i = 10
        return i + 15
    with np.errstate(all='raise'):
        _ufuncs.eval_legendre(1, 0)
        _ufuncs.eval_laguerre(1, 1)
        _ufuncs.eval_gegenbauer(1, 1, 0)

class TestPolys:
    """
    Check that the eval_* functions agree with the constructed polynomials

    """

    def check_poly(self, func, cls, param_ranges=[], x_range=[], nn=10, nparam=10, nx=10, rtol=1e-08):
        if False:
            print('Hello World!')
        np.random.seed(1234)
        dataset = []
        for n in np.arange(nn):
            params = [a + (b - a) * np.random.rand(nparam) for (a, b) in param_ranges]
            params = np.asarray(params).T
            if not param_ranges:
                params = [0]
            for p in params:
                if param_ranges:
                    p = (n,) + tuple(p)
                else:
                    p = (n,)
                x = x_range[0] + (x_range[1] - x_range[0]) * np.random.rand(nx)
                x[0] = x_range[0]
                x[1] = x_range[1]
                poly = np.poly1d(cls(*p).coef)
                z = np.c_[np.tile(p, (nx, 1)), x, poly(x)]
                dataset.append(z)
        dataset = np.concatenate(dataset, axis=0)

        def polyfunc(*p):
            if False:
                return 10
            p = (p[0].astype(int),) + p[1:]
            return func(*p)
        with np.errstate(all='raise'):
            ds = FuncData(polyfunc, dataset, list(range(len(param_ranges) + 2)), -1, rtol=rtol)
            ds.check()

    def test_jacobi(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_poly(_ufuncs.eval_jacobi, orth.jacobi, param_ranges=[(-0.99, 10), (-0.99, 10)], x_range=[-1, 1], rtol=1e-05)

    def test_sh_jacobi(self):
        if False:
            print('Hello World!')
        self.check_poly(_ufuncs.eval_sh_jacobi, orth.sh_jacobi, param_ranges=[(1, 10), (0, 1)], x_range=[0, 1], rtol=1e-05)

    def test_gegenbauer(self):
        if False:
            return 10
        self.check_poly(_ufuncs.eval_gegenbauer, orth.gegenbauer, param_ranges=[(-0.499, 10)], x_range=[-1, 1], rtol=1e-07)

    def test_chebyt(self):
        if False:
            print('Hello World!')
        self.check_poly(_ufuncs.eval_chebyt, orth.chebyt, param_ranges=[], x_range=[-1, 1])

    def test_chebyu(self):
        if False:
            return 10
        self.check_poly(_ufuncs.eval_chebyu, orth.chebyu, param_ranges=[], x_range=[-1, 1])

    def test_chebys(self):
        if False:
            return 10
        self.check_poly(_ufuncs.eval_chebys, orth.chebys, param_ranges=[], x_range=[-2, 2])

    def test_chebyc(self):
        if False:
            print('Hello World!')
        self.check_poly(_ufuncs.eval_chebyc, orth.chebyc, param_ranges=[], x_range=[-2, 2])

    def test_sh_chebyt(self):
        if False:
            return 10
        with np.errstate(all='ignore'):
            self.check_poly(_ufuncs.eval_sh_chebyt, orth.sh_chebyt, param_ranges=[], x_range=[0, 1])

    def test_sh_chebyu(self):
        if False:
            return 10
        self.check_poly(_ufuncs.eval_sh_chebyu, orth.sh_chebyu, param_ranges=[], x_range=[0, 1])

    def test_legendre(self):
        if False:
            print('Hello World!')
        self.check_poly(_ufuncs.eval_legendre, orth.legendre, param_ranges=[], x_range=[-1, 1])

    def test_sh_legendre(self):
        if False:
            print('Hello World!')
        with np.errstate(all='ignore'):
            self.check_poly(_ufuncs.eval_sh_legendre, orth.sh_legendre, param_ranges=[], x_range=[0, 1])

    def test_genlaguerre(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_poly(_ufuncs.eval_genlaguerre, orth.genlaguerre, param_ranges=[(-0.99, 10)], x_range=[0, 100])

    def test_laguerre(self):
        if False:
            return 10
        self.check_poly(_ufuncs.eval_laguerre, orth.laguerre, param_ranges=[], x_range=[0, 100])

    def test_hermite(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_poly(_ufuncs.eval_hermite, orth.hermite, param_ranges=[], x_range=[-100, 100])

    def test_hermitenorm(self):
        if False:
            return 10
        self.check_poly(_ufuncs.eval_hermitenorm, orth.hermitenorm, param_ranges=[], x_range=[-100, 100])

class TestRecurrence:
    """
    Check that the eval_* functions sig='ld->d' and 'dd->d' agree.

    """

    def check_poly(self, func, param_ranges=[], x_range=[], nn=10, nparam=10, nx=10, rtol=1e-08):
        if False:
            print('Hello World!')
        np.random.seed(1234)
        dataset = []
        for n in np.arange(nn):
            params = [a + (b - a) * np.random.rand(nparam) for (a, b) in param_ranges]
            params = np.asarray(params).T
            if not param_ranges:
                params = [0]
            for p in params:
                if param_ranges:
                    p = (n,) + tuple(p)
                else:
                    p = (n,)
                x = x_range[0] + (x_range[1] - x_range[0]) * np.random.rand(nx)
                x[0] = x_range[0]
                x[1] = x_range[1]
                kw = dict(sig=(len(p) + 1) * 'd' + '->d')
                z = np.c_[np.tile(p, (nx, 1)), x, func(*p + (x,), **kw)]
                dataset.append(z)
        dataset = np.concatenate(dataset, axis=0)

        def polyfunc(*p):
            if False:
                for i in range(10):
                    print('nop')
            p = (p[0].astype(int),) + p[1:]
            kw = dict(sig='l' + (len(p) - 1) * 'd' + '->d')
            return func(*p, **kw)
        with np.errstate(all='raise'):
            ds = FuncData(polyfunc, dataset, list(range(len(param_ranges) + 2)), -1, rtol=rtol)
            ds.check()

    def test_jacobi(self):
        if False:
            print('Hello World!')
        self.check_poly(_ufuncs.eval_jacobi, param_ranges=[(-0.99, 10), (-0.99, 10)], x_range=[-1, 1])

    def test_sh_jacobi(self):
        if False:
            print('Hello World!')
        self.check_poly(_ufuncs.eval_sh_jacobi, param_ranges=[(1, 10), (0, 1)], x_range=[0, 1])

    def test_gegenbauer(self):
        if False:
            i = 10
            return i + 15
        self.check_poly(_ufuncs.eval_gegenbauer, param_ranges=[(-0.499, 10)], x_range=[-1, 1])

    def test_chebyt(self):
        if False:
            print('Hello World!')
        self.check_poly(_ufuncs.eval_chebyt, param_ranges=[], x_range=[-1, 1])

    def test_chebyu(self):
        if False:
            i = 10
            return i + 15
        self.check_poly(_ufuncs.eval_chebyu, param_ranges=[], x_range=[-1, 1])

    def test_chebys(self):
        if False:
            i = 10
            return i + 15
        self.check_poly(_ufuncs.eval_chebys, param_ranges=[], x_range=[-2, 2])

    def test_chebyc(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_poly(_ufuncs.eval_chebyc, param_ranges=[], x_range=[-2, 2])

    def test_sh_chebyt(self):
        if False:
            while True:
                i = 10
        self.check_poly(_ufuncs.eval_sh_chebyt, param_ranges=[], x_range=[0, 1])

    def test_sh_chebyu(self):
        if False:
            return 10
        self.check_poly(_ufuncs.eval_sh_chebyu, param_ranges=[], x_range=[0, 1])

    def test_legendre(self):
        if False:
            i = 10
            return i + 15
        self.check_poly(_ufuncs.eval_legendre, param_ranges=[], x_range=[-1, 1])

    def test_sh_legendre(self):
        if False:
            return 10
        self.check_poly(_ufuncs.eval_sh_legendre, param_ranges=[], x_range=[0, 1])

    def test_genlaguerre(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_poly(_ufuncs.eval_genlaguerre, param_ranges=[(-0.99, 10)], x_range=[0, 100])

    def test_laguerre(self):
        if False:
            while True:
                i = 10
        self.check_poly(_ufuncs.eval_laguerre, param_ranges=[], x_range=[0, 100])

    def test_hermite(self):
        if False:
            return 10
        v = _ufuncs.eval_hermite(70, 1.0)
        a = -1.457076485701412e+60
        assert_allclose(v, a)

def test_hermite_domain():
    if False:
        for i in range(10):
            print('nop')
    assert np.isnan(_ufuncs.eval_hermite(-1, 1.0))
    assert np.isnan(_ufuncs.eval_hermitenorm(-1, 1.0))

@pytest.mark.parametrize('n', [0, 1, 2])
@pytest.mark.parametrize('x', [0, 1, np.nan])
def test_hermite_nan(n, x):
    if False:
        i = 10
        return i + 15
    assert np.isnan(_ufuncs.eval_hermite(n, x)) == np.any(np.isnan([n, x]))
    assert np.isnan(_ufuncs.eval_hermitenorm(n, x)) == np.any(np.isnan([n, x]))

@pytest.mark.parametrize('n', [0, 1, 2, 3.2])
@pytest.mark.parametrize('alpha', [1, np.nan])
@pytest.mark.parametrize('x', [2, np.nan])
def test_genlaguerre_nan(n, alpha, x):
    if False:
        i = 10
        return i + 15
    nan_laguerre = np.isnan(_ufuncs.eval_genlaguerre(n, alpha, x))
    nan_arg = np.any(np.isnan([n, alpha, x]))
    assert nan_laguerre == nan_arg

@pytest.mark.parametrize('n', [0, 1, 2, 3.2])
@pytest.mark.parametrize('alpha', [0.0, 1, np.nan])
@pytest.mark.parametrize('x', [1e-06, 2, np.nan])
def test_gegenbauer_nan(n, alpha, x):
    if False:
        i = 10
        return i + 15
    nan_gegenbauer = np.isnan(_ufuncs.eval_gegenbauer(n, alpha, x))
    nan_arg = np.any(np.isnan([n, alpha, x]))
    assert nan_gegenbauer == nan_arg