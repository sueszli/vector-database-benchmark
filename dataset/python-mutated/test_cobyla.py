import math
import numpy as np
from numpy.testing import assert_allclose, assert_, assert_array_equal
from scipy.optimize import fmin_cobyla, minimize, Bounds

class TestCobyla:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.x0 = [4.95, 0.66]
        self.solution = [math.sqrt(25 - (2.0 / 3) ** 2), 2.0 / 3]
        self.opts = {'disp': False, 'rhobeg': 1, 'tol': 1e-05, 'maxiter': 100}

    def fun(self, x):
        if False:
            return 10
        return x[0] ** 2 + abs(x[1]) ** 3

    def con1(self, x):
        if False:
            return 10
        return x[0] ** 2 + x[1] ** 2 - 25

    def con2(self, x):
        if False:
            i = 10
            return i + 15
        return -self.con1(x)

    def test_simple(self):
        if False:
            return 10
        x = fmin_cobyla(self.fun, self.x0, [self.con1, self.con2], rhobeg=1, rhoend=1e-05, maxfun=100, disp=True)
        assert_allclose(x, self.solution, atol=0.0001)

    def test_minimize_simple(self):
        if False:
            return 10

        class Callback:

            def __init__(self):
                if False:
                    return 10
                self.n_calls = 0
                self.last_x = None

            def __call__(self, x):
                if False:
                    while True:
                        i = 10
                self.n_calls += 1
                self.last_x = x
        callback = Callback()
        cons = ({'type': 'ineq', 'fun': self.con1}, {'type': 'ineq', 'fun': self.con2})
        sol = minimize(self.fun, self.x0, method='cobyla', constraints=cons, callback=callback, options=self.opts)
        assert_allclose(sol.x, self.solution, atol=0.0001)
        assert_(sol.success, sol.message)
        assert_(sol.maxcv < 1e-05, sol)
        assert_(sol.nfev < 70, sol)
        assert_(sol.fun < self.fun(self.solution) + 0.001, sol)
        assert_(sol.nfev == callback.n_calls, 'Callback is not called exactly once for every function eval.')
        assert_array_equal(sol.x, callback.last_x, 'Last design vector sent to the callback is not equal to returned value.')

    def test_minimize_constraint_violation(self):
        if False:
            return 10
        np.random.seed(1234)
        pb = np.random.rand(10, 10)
        spread = np.random.rand(10)

        def p(w):
            if False:
                for i in range(10):
                    print('nop')
            return pb.dot(w)

        def f(w):
            if False:
                i = 10
                return i + 15
            return -(w * spread).sum()

        def c1(w):
            if False:
                for i in range(10):
                    print('nop')
            return 500 - abs(p(w)).sum()

        def c2(w):
            if False:
                return 10
            return 5 - abs(p(w).sum())

        def c3(w):
            if False:
                print('Hello World!')
            return 5 - abs(p(w)).max()
        cons = ({'type': 'ineq', 'fun': c1}, {'type': 'ineq', 'fun': c2}, {'type': 'ineq', 'fun': c3})
        w0 = np.zeros((10,))
        sol = minimize(f, w0, method='cobyla', constraints=cons, options={'catol': 1e-06})
        assert_(sol.maxcv > 1e-06)
        assert_(not sol.success)

def test_vector_constraints():
    if False:
        i = 10
        return i + 15

    def fun(x):
        if False:
            return 10
        return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2

    def fmin(x):
        if False:
            i = 10
            return i + 15
        return fun(x) - 1

    def cons1(x):
        if False:
            return 10
        a = np.array([[1, -2, 2], [-1, -2, 6], [-1, 2, 2]])
        return np.array([a[i, 0] * x[0] + a[i, 1] * x[1] + a[i, 2] for i in range(len(a))])

    def cons2(x):
        if False:
            while True:
                i = 10
        return x
    x0 = np.array([2, 0])
    cons_list = [fun, cons1, cons2]
    xsol = [1.4, 1.7]
    fsol = 0.8
    sol = fmin_cobyla(fun, x0, cons_list, rhoend=1e-05)
    assert_allclose(sol, xsol, atol=0.0001)
    sol = fmin_cobyla(fun, x0, fmin, rhoend=1e-05)
    assert_allclose(fun(sol), 1, atol=0.0001)
    constraints = [{'type': 'ineq', 'fun': cons} for cons in cons_list]
    sol = minimize(fun, x0, constraints=constraints, tol=1e-05)
    assert_allclose(sol.x, xsol, atol=0.0001)
    assert_(sol.success, sol.message)
    assert_allclose(sol.fun, fsol, atol=0.0001)
    constraints = {'type': 'ineq', 'fun': fmin}
    sol = minimize(fun, x0, constraints=constraints, tol=1e-05)
    assert_allclose(sol.fun, 1, atol=0.0001)

class TestBounds:

    def test_basic(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                return 10
            return np.sum(x ** 2)
        lb = [-1, None, 1, None, -0.5]
        ub = [-0.5, -0.5, None, None, -0.5]
        bounds = [(a, b) for (a, b) in zip(lb, ub)]
        res = minimize(f, x0=[1, 2, 3, 4, 5], method='cobyla', bounds=bounds)
        ref = [-0.5, -0.5, 1, 0, -0.5]
        assert res.success
        assert_allclose(res.x, ref, atol=0.001)

    def test_unbounded(self):
        if False:
            return 10

        def f(x):
            if False:
                print('Hello World!')
            return np.sum(x ** 2)
        bounds = Bounds([-np.inf, -np.inf], [np.inf, np.inf])
        res = minimize(f, x0=[1, 2], method='cobyla', bounds=bounds)
        assert res.success
        assert_allclose(res.x, 0, atol=0.001)
        bounds = Bounds([1, -np.inf], [np.inf, np.inf])
        res = minimize(f, x0=[1, 2], method='cobyla', bounds=bounds)
        assert res.success
        assert_allclose(res.x, [1, 0], atol=0.001)