"""
Unit tests for optimization routines from _root.py.
"""
from numpy.testing import assert_, assert_equal
import pytest
from pytest import raises as assert_raises, warns as assert_warns
import numpy as np
from scipy.optimize import root

class TestRoot:

    def test_tol_parameter(self):
        if False:
            i = 10
            return i + 15

        def func(z):
            if False:
                return 10
            (x, y) = z
            return np.array([x ** 3 - 1, y ** 3 - 1])

        def dfunc(z):
            if False:
                i = 10
                return i + 15
            (x, y) = z
            return np.array([[3 * x ** 2, 0], [0, 3 * y ** 2]])
        for method in ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'diagbroyden', 'krylov']:
            if method in ('linearmixing', 'excitingmixing'):
                continue
            if method in ('hybr', 'lm'):
                jac = dfunc
            else:
                jac = None
            sol1 = root(func, [1.1, 1.1], jac=jac, tol=0.0001, method=method)
            sol2 = root(func, [1.1, 1.1], jac=jac, tol=0.5, method=method)
            msg = f'{method}: {func(sol1.x)} vs. {func(sol2.x)}'
            assert_(sol1.success, msg)
            assert_(sol2.success, msg)
            assert_(abs(func(sol1.x)).max() < abs(func(sol2.x)).max(), msg)

    def test_tol_norm(self):
        if False:
            return 10

        def norm(x):
            if False:
                return 10
            return abs(x[0])
        for method in ['excitingmixing', 'diagbroyden', 'linearmixing', 'anderson', 'broyden1', 'broyden2', 'krylov']:
            root(np.zeros_like, np.zeros(2), method=method, options={'tol_norm': norm})

    def test_minimize_scalar_coerce_args_param(self):
        if False:
            for i in range(10):
                print('nop')

        def func(z, f=1):
            if False:
                for i in range(10):
                    print('nop')
            (x, y) = z
            return np.array([x ** 3 - 1, y ** 3 - f])
        root(func, [1.1, 1.1], args=1.5)

    def test_f_size(self):
        if False:
            i = 10
            return i + 15

        class fun:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.count = 0

            def __call__(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                self.count += 1
                if not self.count % 5:
                    ret = x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0
                else:
                    ret = [x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0, 0.5 * (x[1] - x[0]) ** 3 + x[1]]
                return ret
        F = fun()
        with assert_raises(ValueError):
            root(F, [0.1, 0.0], method='lm')

    def test_gh_10370(self):
        if False:
            print('Hello World!')

        def fun(x, ignored):
            if False:
                while True:
                    i = 10
            return [3 * x[0] - 0.25 * x[1] ** 2 + 10, 0.1 * x[0] ** 2 + 5 * x[1] - 2]

        def grad(x, ignored):
            if False:
                while True:
                    i = 10
            return [[3, 0.5 * x[1]], [0.2 * x[0], 5]]

        def fun_grad(x, ignored):
            if False:
                while True:
                    i = 10
            return (fun(x, ignored), grad(x, ignored))
        x0 = np.zeros(2)
        ref = root(fun, x0, args=(1,), method='krylov')
        message = 'Method krylov does not use the jacobian'
        with assert_warns(RuntimeWarning, match=message):
            res1 = root(fun, x0, args=(1,), method='krylov', jac=grad)
        with assert_warns(RuntimeWarning, match=message):
            res2 = root(fun_grad, x0, args=(1,), method='krylov', jac=True)
        assert_equal(res1.x, ref.x)
        assert_equal(res2.x, ref.x)
        assert res1.success is res2.success is ref.success is True

    @pytest.mark.parametrize('method', ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane'])
    def test_method_in_result(self, method):
        if False:
            return 10

        def func(x):
            if False:
                for i in range(10):
                    print('nop')
            return x - 1
        res = root(func, x0=[1], method=method)
        assert res.method == method