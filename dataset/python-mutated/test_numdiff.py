"""Testing numerical differentiation

Still some problems, with API (args tuple versus *args)
finite difference Hessian has some problems that I did not look at yet

Should Hessian also work per observation, if fun returns 2d

"""
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import approx_fprime, approx_fprime_cs, approx_hess_cs, _approx_fprime_scalar, _approx_fprime_cs_scalar
DEC3 = 3
DEC4 = 4
DEC5 = 5
DEC6 = 6
DEC8 = 8
DEC13 = 13
DEC14 = 14

def maxabs(x, y):
    if False:
        while True:
            i = 10
    return np.abs(x - y).max()

def fun(beta, x):
    if False:
        for i in range(10):
            print('nop')
    return np.dot(x, beta).sum(0)

def fun1(beta, y, x):
    if False:
        return 10
    xb = np.dot(x, beta)
    return (y - xb) ** 2

def fun2(beta, y, x):
    if False:
        print('Hello World!')
    return fun1(beta, y, x).sum(0)

class CheckGradLoglikeMixin:

    def test_score(self):
        if False:
            while True:
                i = 10
        for test_params in self.params:
            sc = self.mod.score(test_params)
            scfd = numdiff.approx_fprime(test_params.ravel(), self.mod.loglike)
            assert_almost_equal(sc, scfd, decimal=1)
            sccs = numdiff.approx_fprime_cs(test_params.ravel(), self.mod.loglike)
            assert_almost_equal(sc, sccs, decimal=11)

    def test_hess(self):
        if False:
            return 10
        for test_params in self.params:
            he = self.mod.hessian(test_params)
            hefd = numdiff.approx_fprime_cs(test_params, self.mod.score)
            assert_almost_equal(he, hefd, decimal=DEC8)
            assert_almost_equal(he, hefd, decimal=7)
            hefd = numdiff.approx_fprime(test_params, self.mod.score, centered=True)
            assert_allclose(he, hefd, rtol=1e-09)
            hefd = numdiff.approx_fprime(test_params, self.mod.score, centered=False)
            assert_almost_equal(he, hefd, decimal=4)
            hescs = numdiff.approx_fprime_cs(test_params.ravel(), self.mod.score)
            assert_allclose(he, hescs, rtol=1e-13)
            hecs = numdiff.approx_hess_cs(test_params.ravel(), self.mod.loglike)
            assert_allclose(he, hecs, rtol=1e-09)
            grad = self.mod.score(test_params)
            (hecs, gradcs) = numdiff.approx_hess1(test_params, self.mod.loglike, 1e-06, return_grad=True)
            assert_almost_equal(he, hecs, decimal=1)
            assert_almost_equal(grad, gradcs, decimal=1)
            (hecs, gradcs) = numdiff.approx_hess2(test_params, self.mod.loglike, 0.0001, return_grad=True)
            assert_almost_equal(he, hecs, decimal=3)
            assert_almost_equal(grad, gradcs, decimal=1)
            hecs = numdiff.approx_hess3(test_params, self.mod.loglike, 1e-05)
            assert_almost_equal(he, hecs, decimal=4)

class TestGradMNLogit(CheckGradLoglikeMixin):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        data = sm.datasets.anes96.load()
        data.exog = np.asarray(data.exog)
        data.endog = np.asarray(data.endog)
        exog = data.exog
        exog = sm.add_constant(exog, prepend=False)
        cls.mod = sm.MNLogit(data.endog, exog)
        res = cls.mod.fit(disp=0)
        cls.params = [res.params.ravel('F')]

    def test_hess(self):
        if False:
            i = 10
            return i + 15
        for test_params in self.params:
            he = self.mod.hessian(test_params)
            hefd = numdiff.approx_fprime_cs(test_params, self.mod.score)
            assert_almost_equal(he, hefd, decimal=DEC8)
            assert_almost_equal(he, hefd, decimal=7)
            hefd = numdiff.approx_fprime(test_params, self.mod.score, centered=True)
            assert_almost_equal(he, hefd, decimal=4)
            hefd = numdiff.approx_fprime(test_params, self.mod.score, 1e-09, centered=False)
            assert_almost_equal(he, hefd, decimal=2)
            hescs = numdiff.approx_fprime_cs(test_params, self.mod.score)
            assert_almost_equal(he, hescs, decimal=DEC8)
            hecs = numdiff.approx_hess_cs(test_params, self.mod.loglike)
            assert_almost_equal(he, hecs, decimal=5)
            hecs = numdiff.approx_hess3(test_params, self.mod.loglike, 0.0001)
            assert_almost_equal(he, hecs, decimal=0)

class TestGradLogit(CheckGradLoglikeMixin):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=False)
        cls.mod = sm.Logit(data.endog, data.exog)
        cls.params = [np.array([1, 0.25, 1.4, -7])]

class CheckDerivativeMixin:

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        nobs = 200
        np.random.seed(187678)
        x = np.random.randn(nobs, 3)
        xk = np.array([1, 2, 3])
        xk = np.array([1.0, 1.0, 1.0])
        beta = xk
        y = np.dot(x, beta) + 0.1 * np.random.randn(nobs)
        xkols = np.dot(np.linalg.pinv(x), y)
        cls.x = x
        cls.y = y
        cls.params = [np.array([1.0, 1.0, 1.0]), xkols]
        cls.init()

    @classmethod
    def init(cls):
        if False:
            return 10
        pass

    def test_grad_fun1_fd(self):
        if False:
            print('Hello World!')
        for test_params in self.params:
            gtrue = self.gradtrue(test_params)
            fun = self.fun()
            epsilon = 1e-06
            gfd = numdiff.approx_fprime(test_params, fun, epsilon=epsilon, args=self.args)
            gfd += numdiff.approx_fprime(test_params, fun, epsilon=-epsilon, args=self.args)
            gfd /= 2.0
            assert_almost_equal(gtrue, gfd, decimal=DEC6)

    def test_grad_fun1_fdc(self):
        if False:
            while True:
                i = 10
        for test_params in self.params:
            gtrue = self.gradtrue(test_params)
            fun = self.fun()
            gfd = numdiff.approx_fprime(test_params, fun, epsilon=1e-08, args=self.args, centered=True)
            assert_almost_equal(gtrue, gfd, decimal=DEC5)

    def test_grad_fun1_cs(self):
        if False:
            print('Hello World!')
        for test_params in self.params:
            gtrue = self.gradtrue(test_params)
            fun = self.fun()
            gcs = numdiff.approx_fprime_cs(test_params, fun, args=self.args)
            assert_almost_equal(gtrue, gcs, decimal=DEC13)

    def test_hess_fun1_fd(self):
        if False:
            return 10
        for test_params in self.params:
            hetrue = self.hesstrue(test_params)
            if hetrue is not None:
                fun = self.fun()
                hefd = numdiff.approx_hess1(test_params, fun, args=self.args)
                assert_almost_equal(hetrue, hefd, decimal=DEC3)
                hefd = numdiff.approx_hess2(test_params, fun, args=self.args)
                assert_almost_equal(hetrue, hefd, decimal=DEC3)
                hefd = numdiff.approx_hess3(test_params, fun, args=self.args)
                assert_almost_equal(hetrue, hefd, decimal=DEC3)

    def test_hess_fun1_cs(self):
        if False:
            print('Hello World!')
        for test_params in self.params:
            hetrue = self.hesstrue(test_params)
            if hetrue is not None:
                fun = self.fun()
                hecs = numdiff.approx_hess_cs(test_params, fun, args=self.args)
                assert_almost_equal(hetrue, hecs, decimal=DEC6)

class TestDerivativeFun(CheckDerivativeMixin):

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        super(TestDerivativeFun, cls).setup_class()
        xkols = np.dot(np.linalg.pinv(cls.x), cls.y)
        cls.params = [np.array([1.0, 1.0, 1.0]), xkols]
        cls.args = (cls.x,)

    def fun(self):
        if False:
            while True:
                i = 10
        return fun

    def gradtrue(self, params):
        if False:
            while True:
                i = 10
        return self.x.sum(0)

    def hesstrue(self, params):
        if False:
            i = 10
            return i + 15
        return np.zeros((3, 3))

class TestDerivativeFun2(CheckDerivativeMixin):

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        super(TestDerivativeFun2, cls).setup_class()
        xkols = np.dot(np.linalg.pinv(cls.x), cls.y)
        cls.params = [np.array([1.0, 1.0, 1.0]), xkols]
        cls.args = (cls.y, cls.x)

    def fun(self):
        if False:
            for i in range(10):
                print('nop')
        return fun2

    def gradtrue(self, params):
        if False:
            for i in range(10):
                print('nop')
        (y, x) = (self.y, self.x)
        return (-x * 2 * (y - np.dot(x, params))[:, None]).sum(0)

    def hesstrue(self, params):
        if False:
            for i in range(10):
                print('nop')
        x = self.x
        return 2 * np.dot(x.T, x)

class TestDerivativeFun1(CheckDerivativeMixin):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        super(TestDerivativeFun1, cls).setup_class()
        xkols = np.dot(np.linalg.pinv(cls.x), cls.y)
        cls.params = [np.array([1.0, 1.0, 1.0]), xkols]
        cls.args = (cls.y, cls.x)

    def fun(self):
        if False:
            for i in range(10):
                print('nop')
        return fun1

    def gradtrue(self, params):
        if False:
            print('Hello World!')
        (y, x) = (self.y, self.x)
        return -x * 2 * (y - np.dot(x, params))[:, None]

    def hesstrue(self, params):
        if False:
            i = 10
            return i + 15
        return None
        (y, x) = (self.y, self.x)
        return -x * 2 * (y - np.dot(x, params))[:, None]

def test_dtypes():
    if False:
        i = 10
        return i + 15

    def f(x):
        if False:
            return 10
        return 2 * x
    desired = np.array([[2, 0], [0, 2]])
    assert_allclose(approx_fprime(np.array([1, 2]), f), desired)
    assert_allclose(approx_fprime(np.array([1.0, 2.0]), f), desired)
    assert_allclose(approx_fprime(np.array([1.0 + 0j, 2.0 + 0j]), f), desired)

def test_vectorized():
    if False:
        i = 10
        return i + 15

    def f(x):
        if False:
            return 10
        return 2 * x
    desired = np.array([2, 2])
    p = np.array([[1, 2]]).T
    assert_allclose(_approx_fprime_scalar(p, f), desired[:, None], rtol=1e-08)
    assert_allclose(_approx_fprime_scalar(p.squeeze(), f), desired, rtol=1e-08)
    assert_allclose(_approx_fprime_cs_scalar(p, f), desired[:, None], rtol=1e-08)
    assert_allclose(_approx_fprime_cs_scalar(p.squeeze(), f), desired, rtol=1e-08)
    assert_allclose(approx_fprime_cs(p.T, f).squeeze(), desired, rtol=1e-08)
if __name__ == '__main__':
    epsilon = 1e-06
    nobs = 200
    x = np.arange(nobs * 3).reshape(nobs, -1)
    x = np.random.randn(nobs, 3)
    xk = np.array([1, 2, 3])
    xk = np.array([1.0, 1.0, 1.0])
    beta = xk
    y = np.dot(x, beta) + 0.1 * np.random.randn(nobs)
    xkols = np.dot(np.linalg.pinv(x), y)
    print(approx_fprime((1, 2, 3), fun, epsilon, x))
    gradtrue = x.sum(0)
    print(x.sum(0))
    gradcs = approx_fprime_cs((1, 2, 3), fun, (x,), h=1e-20)
    print(gradcs, maxabs(gradcs, gradtrue))
    print(approx_hess_cs((1, 2, 3), fun, (x,), h=1e-20))
    print(approx_hess_cs((1, 2, 3), fun2, (y, x), h=1e-20) - 2 * np.dot(x.T, x))
    print(numdiff.approx_hess(xk, fun2, 0.001, (y, x))[0] - 2 * np.dot(x.T, x))
    gt = -x * 2 * (y - np.dot(x, [1, 2, 3]))[:, None]
    g = approx_fprime_cs((1, 2, 3), fun1, (y, x), h=1e-20)
    gd = numdiff.approx_fprime((1, 2, 3), fun1, epsilon, (y, x))
    print(maxabs(g, gt))
    print(maxabs(gd, gt))
    data = sm.datasets.spector.load()
    data.exog = sm.add_constant(data.exog, prepend=False)
    mod = sm.Logit(data.endog, data.exog)
    test_params = [1, 0.25, 1.4, -7]
    loglike = mod.loglike
    score = mod.score
    hess = mod.hessian
    print('sm', score(test_params))
    print('fd', numdiff.approx_fprime(test_params, loglike, epsilon))
    print('cs', numdiff.approx_fprime_cs(test_params, loglike))
    print('sm', hess(test_params))
    print('fd', numdiff.approx_fprime(test_params, score, epsilon))
    print('cs', numdiff.approx_fprime_cs(test_params, score))
    hesscs = numdiff.approx_hess_cs(test_params, loglike)
    print('cs', hesscs)
    print(maxabs(hess(test_params), hesscs))
    data = sm.datasets.anes96.load()
    exog = data.exog
    exog = sm.add_constant(exog, prepend=False)
    res1 = sm.MNLogit(data.endog, exog).fit(method='newton', disp=0)
    datap = sm.datasets.randhie.load()
    nobs = len(datap.endog)
    exogp = sm.add_constant(datap.exog.view(float).reshape(nobs, -1), prepend=False)
    modp = sm.Poisson(datap.endog, exogp)
    resp = modp.fit(method='newton', disp=0)