"""
Unit tests for the basin hopping global minimization algorithm.
"""
import copy
from numpy.testing import assert_almost_equal, assert_equal, assert_, assert_allclose
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import Storage, RandomDisplacement, Metropolis, AdaptiveStepsize

def func1d(x):
    if False:
        print('Hello World!')
    f = cos(14.5 * x - 0.3) + (x + 0.2) * x
    df = np.array(-14.5 * sin(14.5 * x - 0.3) + 2.0 * x + 0.2)
    return (f, df)

def func2d_nograd(x):
    if False:
        i = 10
        return i + 15
    f = cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
    return f

def func2d(x):
    if False:
        print('Hello World!')
    f = cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
    df = np.zeros(2)
    df[0] = -14.5 * sin(14.5 * x[0] - 0.3) + 2.0 * x[0] + 0.2
    df[1] = 2.0 * x[1] + 0.2
    return (f, df)

def func2d_easyderiv(x):
    if False:
        return 10
    f = 2.0 * x[0] ** 2 + 2.0 * x[0] * x[1] + 2.0 * x[1] ** 2 - 6.0 * x[0]
    df = np.zeros(2)
    df[0] = 4.0 * x[0] + 2.0 * x[1] - 6.0
    df[1] = 2.0 * x[0] + 4.0 * x[1]
    return (f, df)

class MyTakeStep1(RandomDisplacement):
    """use a copy of displace, but have it set a special parameter to
    make sure it's actually being used."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.been_called = False
        super().__init__()

    def __call__(self, x):
        if False:
            print('Hello World!')
        self.been_called = True
        return super().__call__(x)

def myTakeStep2(x):
    if False:
        for i in range(10):
            print('nop')
    'redo RandomDisplacement in function form without the attribute stepsize\n    to make sure everything still works ok\n    '
    s = 0.5
    x += np.random.uniform(-s, s, np.shape(x))
    return x

class MyAcceptTest:
    """pass a custom accept test

    This does nothing but make sure it's being used and ensure all the
    possible return values are accepted
    """

    def __init__(self):
        if False:
            return 10
        self.been_called = False
        self.ncalls = 0
        self.testres = [False, 'force accept', True, np.bool_(True), np.bool_(False), [], {}, 0, 1]

    def __call__(self, **kwargs):
        if False:
            while True:
                i = 10
        self.been_called = True
        self.ncalls += 1
        if self.ncalls - 1 < len(self.testres):
            return self.testres[self.ncalls - 1]
        else:
            return True

class MyCallBack:
    """pass a custom callback function

    This makes sure it's being used. It also returns True after 10
    steps to ensure that it's stopping early.

    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.been_called = False
        self.ncalls = 0

    def __call__(self, x, f, accepted):
        if False:
            for i in range(10):
                print('nop')
        self.been_called = True
        self.ncalls += 1
        if self.ncalls == 10:
            return True

class TestBasinHopping:

    def setup_method(self):
        if False:
            return 10
        ' Tests setup.\n\n        Run tests based on the 1-D and 2-D functions described above.\n        '
        self.x0 = (1.0, [1.0, 1.0])
        self.sol = (-0.195, np.array([-0.195, -0.1]))
        self.tol = 3
        self.niter = 100
        self.disp = False
        np.random.seed(1234)
        self.kwargs = {'method': 'L-BFGS-B', 'jac': True}
        self.kwargs_nograd = {'method': 'L-BFGS-B'}

    def test_TypeError(self):
        if False:
            for i in range(10):
                print('nop')
        i = 1
        assert_raises(TypeError, basinhopping, func2d, self.x0[i], take_step=1)
        assert_raises(TypeError, basinhopping, func2d, self.x0[i], accept_test=1)

    def test_input_validation(self):
        if False:
            print('Hello World!')
        msg = 'target_accept_rate has to be in range \\(0, 1\\)'
        with assert_raises(ValueError, match=msg):
            basinhopping(func1d, self.x0[0], target_accept_rate=0.0)
        with assert_raises(ValueError, match=msg):
            basinhopping(func1d, self.x0[0], target_accept_rate=1.0)
        msg = 'stepwise_factor has to be in range \\(0, 1\\)'
        with assert_raises(ValueError, match=msg):
            basinhopping(func1d, self.x0[0], stepwise_factor=0.0)
        with assert_raises(ValueError, match=msg):
            basinhopping(func1d, self.x0[0], stepwise_factor=1.0)

    def test_1d_grad(self):
        if False:
            while True:
                i = 10
        i = 0
        res = basinhopping(func1d, self.x0[i], minimizer_kwargs=self.kwargs, niter=self.niter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_2d(self):
        if False:
            return 10
        i = 1
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=self.niter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)
        assert_(res.nfev > 0)

    def test_njev(self):
        if False:
            while True:
                i = 10
        i = 1
        minimizer_kwargs = self.kwargs.copy()
        minimizer_kwargs['method'] = 'BFGS'
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=minimizer_kwargs, niter=self.niter, disp=self.disp)
        assert_(res.nfev > 0)
        assert_equal(res.nfev, res.njev)

    def test_jac(self):
        if False:
            i = 10
            return i + 15
        minimizer_kwargs = self.kwargs.copy()
        minimizer_kwargs['method'] = 'BFGS'
        res = basinhopping(func2d_easyderiv, [0.0, 0.0], minimizer_kwargs=minimizer_kwargs, niter=self.niter, disp=self.disp)
        assert_(hasattr(res.lowest_optimization_result, 'jac'))
        (_, jacobian) = func2d_easyderiv(res.x)
        assert_almost_equal(res.lowest_optimization_result.jac, jacobian, self.tol)

    def test_2d_nograd(self):
        if False:
            i = 10
            return i + 15
        i = 1
        res = basinhopping(func2d_nograd, self.x0[i], minimizer_kwargs=self.kwargs_nograd, niter=self.niter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_all_minimizers(self):
        if False:
            i = 10
            return i + 15
        i = 1
        methods = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP']
        minimizer_kwargs = copy.copy(self.kwargs)
        for method in methods:
            minimizer_kwargs['method'] = method
            res = basinhopping(func2d, self.x0[i], minimizer_kwargs=minimizer_kwargs, niter=self.niter, disp=self.disp)
            assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_all_nograd_minimizers(self):
        if False:
            while True:
                i = 10
        i = 1
        methods = ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP', 'Nelder-Mead', 'Powell', 'COBYLA']
        minimizer_kwargs = copy.copy(self.kwargs_nograd)
        for method in methods:
            minimizer_kwargs['method'] = method
            res = basinhopping(func2d_nograd, self.x0[i], minimizer_kwargs=minimizer_kwargs, niter=self.niter, disp=self.disp)
            tol = self.tol
            if method == 'COBYLA':
                tol = 2
            assert_almost_equal(res.x, self.sol[i], decimal=tol)

    def test_pass_takestep(self):
        if False:
            print('Hello World!')
        takestep = MyTakeStep1()
        initial_step_size = takestep.stepsize
        i = 1
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=self.niter, disp=self.disp, take_step=takestep)
        assert_almost_equal(res.x, self.sol[i], self.tol)
        assert_(takestep.been_called)
        assert_(initial_step_size != takestep.stepsize)

    def test_pass_simple_takestep(self):
        if False:
            return 10
        takestep = myTakeStep2
        i = 1
        res = basinhopping(func2d_nograd, self.x0[i], minimizer_kwargs=self.kwargs_nograd, niter=self.niter, disp=self.disp, take_step=takestep)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_pass_accept_test(self):
        if False:
            return 10
        accept_test = MyAcceptTest()
        i = 1
        basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=10, disp=self.disp, accept_test=accept_test)
        assert_(accept_test.been_called)

    def test_pass_callback(self):
        if False:
            i = 10
            return i + 15
        callback = MyCallBack()
        i = 1
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=30, disp=self.disp, callback=callback)
        assert_(callback.been_called)
        assert_('callback' in res.message[0])
        assert_equal(res.nit, 9)

    def test_minimizer_fail(self):
        if False:
            return 10
        i = 1
        self.kwargs['options'] = dict(maxiter=0)
        self.niter = 10
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=self.niter, disp=self.disp)
        assert_equal(res.nit + 1, res.minimization_failures)

    def test_niter_zero(self):
        if False:
            for i in range(10):
                print('nop')
        i = 0
        basinhopping(func1d, self.x0[i], minimizer_kwargs=self.kwargs, niter=0, disp=self.disp)

    def test_seed_reproducibility(self):
        if False:
            for i in range(10):
                print('nop')
        minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': True}
        f_1 = []

        def callback(x, f, accepted):
            if False:
                i = 10
                return i + 15
            f_1.append(f)
        basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs, niter=10, callback=callback, seed=10)
        f_2 = []

        def callback2(x, f, accepted):
            if False:
                for i in range(10):
                    print('nop')
            f_2.append(f)
        basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs, niter=10, callback=callback2, seed=10)
        assert_equal(np.array(f_1), np.array(f_2))

    def test_random_gen(self):
        if False:
            i = 10
            return i + 15
        rng = np.random.default_rng(1)
        minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': True}
        res1 = basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs, niter=10, seed=rng)
        rng = np.random.default_rng(1)
        res2 = basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs, niter=10, seed=rng)
        assert_equal(res1.x, res2.x)

    def test_monotonic_basin_hopping(self):
        if False:
            print('Hello World!')
        i = 0
        res = basinhopping(func1d, self.x0[i], minimizer_kwargs=self.kwargs, niter=self.niter, disp=self.disp, T=0)
        assert_almost_equal(res.x, self.sol[i], self.tol)

class Test_Storage:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.x0 = np.array(1)
        self.f0 = 0
        minres = OptimizeResult(success=True)
        minres.x = self.x0
        minres.fun = self.f0
        self.storage = Storage(minres)

    def test_higher_f_rejected(self):
        if False:
            return 10
        new_minres = OptimizeResult(success=True)
        new_minres.x = self.x0 + 1
        new_minres.fun = self.f0 + 1
        ret = self.storage.update(new_minres)
        minres = self.storage.get_lowest()
        assert_equal(self.x0, minres.x)
        assert_equal(self.f0, minres.fun)
        assert_(not ret)

    @pytest.mark.parametrize('success', [True, False])
    def test_lower_f_accepted(self, success):
        if False:
            return 10
        new_minres = OptimizeResult(success=success)
        new_minres.x = self.x0 + 1
        new_minres.fun = self.f0 - 1
        ret = self.storage.update(new_minres)
        minres = self.storage.get_lowest()
        assert (self.x0 != minres.x) == success
        assert (self.f0 != minres.fun) == success
        assert ret is success

class Test_RandomDisplacement:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.stepsize = 1.0
        self.displace = RandomDisplacement(stepsize=self.stepsize)
        self.N = 300000
        self.x0 = np.zeros([self.N])

    def test_random(self):
        if False:
            while True:
                i = 10
        x = self.displace(self.x0)
        v = (2.0 * self.stepsize) ** 2 / 12
        assert_almost_equal(np.mean(x), 0.0, 1)
        assert_almost_equal(np.var(x), v, 1)

class Test_Metropolis:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.T = 2.0
        self.met = Metropolis(self.T)
        self.res_new = OptimizeResult(success=True, fun=0.0)
        self.res_old = OptimizeResult(success=True, fun=1.0)

    def test_boolean_return(self):
        if False:
            while True:
                i = 10
        ret = self.met(res_new=self.res_new, res_old=self.res_old)
        assert isinstance(ret, bool)

    def test_lower_f_accepted(self):
        if False:
            i = 10
            return i + 15
        assert_(self.met(res_new=self.res_new, res_old=self.res_old))

    def test_accept(self):
        if False:
            for i in range(10):
                print('nop')
        one_accept = False
        one_reject = False
        for i in range(1000):
            if one_accept and one_reject:
                break
            res_new = OptimizeResult(success=True, fun=1.0)
            res_old = OptimizeResult(success=True, fun=0.5)
            ret = self.met(res_new=res_new, res_old=res_old)
            if ret:
                one_accept = True
            else:
                one_reject = True
        assert_(one_accept)
        assert_(one_reject)

    def test_GH7495(self):
        if False:
            print('Hello World!')
        met = Metropolis(2)
        res_new = OptimizeResult(success=True, fun=0.0)
        res_old = OptimizeResult(success=True, fun=2000)
        with np.errstate(over='raise'):
            met.accept_reject(res_new=res_new, res_old=res_old)

    def test_gh7799(self):
        if False:
            i = 10
            return i + 15

        def func(x):
            if False:
                i = 10
                return i + 15
            return (x ** 2 - 8) ** 2 + (x + 2) ** 2
        x0 = -4
        limit = 50
        con = ({'type': 'ineq', 'fun': lambda x: func(x) - limit},)
        res = basinhopping(func, x0, 30, minimizer_kwargs={'constraints': con})
        assert res.success
        assert_allclose(res.fun, limit, rtol=1e-06)

    def test_accept_gh7799(self):
        if False:
            return 10
        met = Metropolis(0)
        res_new = OptimizeResult(success=True, fun=0.0)
        res_old = OptimizeResult(success=True, fun=1.0)
        assert met(res_new=res_new, res_old=res_old)
        res_new.success = False
        assert not met(res_new=res_new, res_old=res_old)
        res_old.success = False
        assert met(res_new=res_new, res_old=res_old)

    def test_reject_all_gh7799(self):
        if False:
            for i in range(10):
                print('nop')

        def fun(x):
            if False:
                for i in range(10):
                    print('nop')
            return x @ x

        def constraint(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + 1
        kwargs = {'constraints': {'type': 'eq', 'fun': constraint}, 'bounds': [(0, 1), (0, 1)], 'method': 'slsqp'}
        res = basinhopping(fun, x0=[2, 3], niter=10, minimizer_kwargs=kwargs)
        assert not res.success

class Test_AdaptiveStepsize:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.stepsize = 1.0
        self.ts = RandomDisplacement(stepsize=self.stepsize)
        self.target_accept_rate = 0.5
        self.takestep = AdaptiveStepsize(takestep=self.ts, verbose=False, accept_rate=self.target_accept_rate)

    def test_adaptive_increase(self):
        if False:
            while True:
                i = 10
        x = 0.0
        self.takestep(x)
        self.takestep.report(False)
        for i in range(self.takestep.interval):
            self.takestep(x)
            self.takestep.report(True)
        assert_(self.ts.stepsize > self.stepsize)

    def test_adaptive_decrease(self):
        if False:
            i = 10
            return i + 15
        x = 0.0
        self.takestep(x)
        self.takestep.report(True)
        for i in range(self.takestep.interval):
            self.takestep(x)
            self.takestep.report(False)
        assert_(self.ts.stepsize < self.stepsize)

    def test_all_accepted(self):
        if False:
            i = 10
            return i + 15
        x = 0.0
        for i in range(self.takestep.interval + 1):
            self.takestep(x)
            self.takestep.report(True)
        assert_(self.ts.stepsize > self.stepsize)

    def test_all_rejected(self):
        if False:
            return 10
        x = 0.0
        for i in range(self.takestep.interval + 1):
            self.takestep(x)
            self.takestep.report(False)
        assert_(self.ts.stepsize < self.stepsize)