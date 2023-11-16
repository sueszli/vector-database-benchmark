import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import shgo, Bounds, minimize_scalar, minimize, rosen, rosen_der, rosen_hess, NonlinearConstraint
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO

class StructTestFunction:

    def __init__(self, bounds, expected_x, expected_fun=None, expected_xl=None, expected_funl=None):
        if False:
            print('Hello World!')
        self.bounds = bounds
        self.expected_x = expected_x
        self.expected_fun = expected_fun
        self.expected_xl = expected_xl
        self.expected_funl = expected_funl

def wrap_constraints(g):
    if False:
        print('Hello World!')
    cons = []
    if g is not None:
        if not isinstance(g, (tuple, list)):
            g = (g,)
        else:
            pass
        for g in g:
            cons.append({'type': 'ineq', 'fun': g})
        cons = tuple(cons)
    else:
        cons = None
    return cons

class StructTest1(StructTestFunction):

    def f(self, x):
        if False:
            return 10
        return x[0] ** 2 + x[1] ** 2

    def g(x):
        if False:
            for i in range(10):
                print('nop')
        return -(numpy.sum(x, axis=0) - 6.0)
    cons = wrap_constraints(g)
test1_1 = StructTest1(bounds=[(-1, 6), (-1, 6)], expected_x=[0, 0])
test1_2 = StructTest1(bounds=[(0, 1), (0, 1)], expected_x=[0, 0])
test1_3 = StructTest1(bounds=[(None, None), (None, None)], expected_x=[0, 0])

class StructTest2(StructTestFunction):
    """
    Scalar function with several minima to test all minimiser retrievals
    """

    def f(self, x):
        if False:
            return 10
        return (x - 30) * numpy.sin(x)

    def g(x):
        if False:
            for i in range(10):
                print('nop')
        return 58 - numpy.sum(x, axis=0)
    cons = wrap_constraints(g)
test2_1 = StructTest2(bounds=[(0, 60)], expected_x=[1.53567906], expected_fun=-28.44677132, expected_xl=numpy.array([[1.53567906], [55.01782167], [7.80894889], [48.74797493], [14.07445705], [42.4913859], [20.31743841], [36.28607535], [26.43039605], [30.76371366]]), expected_funl=numpy.array([-28.44677132, -24.99785984, -22.16855376, -18.72136195, -15.89423937, -12.45154942, -9.63133158, -6.20801301, -3.43727232, -0.46353338]))
test2_2 = StructTest2(bounds=[(0, 4.5)], expected_x=[1.53567906], expected_fun=[-28.44677132], expected_xl=numpy.array([[1.53567906]]), expected_funl=numpy.array([-28.44677132]))

class StructTest3(StructTestFunction):
    """
    Hock and Schittkowski 18 problem (HS18). Hoch and Schittkowski (1981)
    http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
    Minimize: f = 0.01 * (x_1)**2 + (x_2)**2

    Subject to: x_1 * x_2 - 25.0 >= 0,
                (x_1)**2 + (x_2)**2 - 25.0 >= 0,
                2 <= x_1 <= 50,
                0 <= x_2 <= 50.

    Approx. Answer:
        f([(250)**0.5 , (2.5)**0.5]) = 5.0


    """

    def f(self, x):
        if False:
            for i in range(10):
                print('nop')
        return 0.01 * x[0] ** 2 + x[1] ** 2

    def g1(x):
        if False:
            i = 10
            return i + 15
        return x[0] * x[1] - 25.0

    def g2(x):
        if False:
            i = 10
            return i + 15
        return x[0] ** 2 + x[1] ** 2 - 25.0

    def g(x):
        if False:
            print('Hello World!')
        return (x[0] * x[1] - 25.0, x[0] ** 2 + x[1] ** 2 - 25.0)
    __nlc = NonlinearConstraint(g, 0, np.inf)
    cons = (__nlc,)
test3_1 = StructTest3(bounds=[(2, 50), (0, 50)], expected_x=[250 ** 0.5, 2.5 ** 0.5], expected_fun=5.0)

class StructTest4(StructTestFunction):
    """
    Hock and Schittkowski 11 problem (HS11). Hoch and Schittkowski (1981)

    NOTE: Did not find in original reference to HS collection, refer to
          Henderson (2015) problem 7 instead. 02.03.2016
    """

    def f(self, x):
        if False:
            while True:
                i = 10
        return (x[0] - 10) ** 2 + 5 * (x[1] - 12) ** 2 + x[2] ** 4 + 3 * (x[3] - 11) ** 2 + 10 * x[4] ** 6 + 7 * x[5] ** 2 + x[6] ** 4 - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6]

    def g1(x):
        if False:
            print('Hello World!')
        return -(2 * x[0] ** 2 + 3 * x[1] ** 4 + x[2] + 4 * x[3] ** 2 + 5 * x[4] - 127)

    def g2(x):
        if False:
            return 10
        return -(7 * x[0] + 3 * x[1] + 10 * x[2] ** 2 + x[3] - x[4] - 282.0)

    def g3(x):
        if False:
            print('Hello World!')
        return -(23 * x[0] + x[1] ** 2 + 6 * x[5] ** 2 - 8 * x[6] - 196)

    def g4(x):
        if False:
            i = 10
            return i + 15
        return -(4 * x[0] ** 2 + x[1] ** 2 - 3 * x[0] * x[1] + 2 * x[2] ** 2 + 5 * x[5] - 11 * x[6])
    g = (g1, g2, g3, g4)
    cons = wrap_constraints(g)
test4_1 = StructTest4(bounds=[(-10, 10)] * 7, expected_x=[2.330499, 1.951372, -0.4775414, 4.365726, -0.624487, 1.038131, 1.594227], expected_fun=680.6300573)

class StructTest5(StructTestFunction):

    def f(self, x):
        if False:
            return 10
        return -(x[1] + 47.0) * numpy.sin(numpy.sqrt(abs(x[0] / 2.0 + (x[1] + 47.0)))) - x[0] * numpy.sin(numpy.sqrt(abs(x[0] - (x[1] + 47.0))))
    g = None
    cons = wrap_constraints(g)
test5_1 = StructTest5(bounds=[(-512, 512), (-512, 512)], expected_fun=[-959.6406627208505], expected_x=[512.0, 404.23180542])

class StructTestLJ(StructTestFunction):
    """
    LennardJones objective function. Used to test symmetry constraints
    settings.
    """

    def f(self, x, *args):
        if False:
            for i in range(10):
                print('nop')
        print(f'x = {x}')
        self.N = args[0]
        k = int(self.N / 3)
        s = 0.0
        for i in range(k - 1):
            for j in range(i + 1, k):
                a = 3 * i
                b = 3 * j
                xd = x[a] - x[b]
                yd = x[a + 1] - x[b + 1]
                zd = x[a + 2] - x[b + 2]
                ed = xd * xd + yd * yd + zd * zd
                ud = ed * ed * ed
                if ed > 0.0:
                    s += (1.0 / ud - 2.0) / ud
        return s
    g = None
    cons = wrap_constraints(g)
N = 6
boundsLJ = list(zip([-4.0] * 6, [4.0] * 6))
testLJ = StructTestLJ(bounds=boundsLJ, expected_fun=[-1.0], expected_x=None)

class StructTestS(StructTestFunction):

    def f(self, x):
        if False:
            print('Hello World!')
        return (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 + (x[2] - 0.5) ** 2 + (x[3] - 0.5) ** 2
    g = None
    cons = wrap_constraints(g)
test_s = StructTestS(bounds=[(0, 2.0)] * 4, expected_fun=0.0, expected_x=numpy.ones(4) - 0.5)

class StructTestTable(StructTestFunction):

    def f(self, x):
        if False:
            for i in range(10):
                print('nop')
        if x[0] == 3.0 and x[1] == 3.0:
            return 50
        else:
            return 100
    g = None
    cons = wrap_constraints(g)
test_table = StructTestTable(bounds=[(-10, 10), (-10, 10)], expected_fun=[50], expected_x=[3.0, 3.0])

class StructTestInfeasible(StructTestFunction):
    """
    Test function with no feasible domain.
    """

    def f(self, x, *args):
        if False:
            while True:
                i = 10
        return x[0] ** 2 + x[1] ** 2

    def g1(x):
        if False:
            return 10
        return x[0] + x[1] - 1

    def g2(x):
        if False:
            i = 10
            return i + 15
        return -(x[0] + x[1] - 1)

    def g3(x):
        if False:
            print('Hello World!')
        return -x[0] + x[1] - 1

    def g4(x):
        if False:
            i = 10
            return i + 15
        return -(-x[0] + x[1] - 1)
    g = (g1, g2, g3, g4)
    cons = wrap_constraints(g)
test_infeasible = StructTestInfeasible(bounds=[(2, 50), (-1, 1)], expected_fun=None, expected_x=None)

@pytest.mark.skip('Not a test')
def run_test(test, args=(), test_atol=1e-05, n=100, iters=None, callback=None, minimizer_kwargs=None, options=None, sampling_method='sobol', workers=1):
    if False:
        print('Hello World!')
    res = shgo(test.f, test.bounds, args=args, constraints=test.cons, n=n, iters=iters, callback=callback, minimizer_kwargs=minimizer_kwargs, options=options, sampling_method=sampling_method, workers=workers)
    print(f'res = {res}')
    logging.info(f'res = {res}')
    if test.expected_x is not None:
        numpy.testing.assert_allclose(res.x, test.expected_x, rtol=test_atol, atol=test_atol)
    if test.expected_fun is not None:
        numpy.testing.assert_allclose(res.fun, test.expected_fun, atol=test_atol)
    if test.expected_xl is not None:
        numpy.testing.assert_allclose(res.xl, test.expected_xl, atol=test_atol)
    if test.expected_funl is not None:
        numpy.testing.assert_allclose(res.funl, test.expected_funl, atol=test_atol)
    return

class TestShgoSobolTestFunctions:
    """
    Global optimisation tests with Sobol sampling:
    """

    def test_f1_1_sobol(self):
        if False:
            while True:
                i = 10
        'Multivariate test function 1:\n        x[0]**2 + x[1]**2 with bounds=[(-1, 6), (-1, 6)]'
        run_test(test1_1)

    def test_f1_2_sobol(self):
        if False:
            for i in range(10):
                print('nop')
        'Multivariate test function 1:\n         x[0]**2 + x[1]**2 with bounds=[(0, 1), (0, 1)]'
        run_test(test1_2)

    def test_f1_3_sobol(self):
        if False:
            while True:
                i = 10
        'Multivariate test function 1:\n        x[0]**2 + x[1]**2 with bounds=[(None, None),(None, None)]'
        options = {'disp': True}
        run_test(test1_3, options=options)

    def test_f2_1_sobol(self):
        if False:
            return 10
        'Univariate test function on\n        f(x) = (x - 30) * sin(x) with bounds=[(0, 60)]'
        run_test(test2_1)

    def test_f2_2_sobol(self):
        if False:
            for i in range(10):
                print('nop')
        'Univariate test function on\n        f(x) = (x - 30) * sin(x) bounds=[(0, 4.5)]'
        run_test(test2_2)

    def test_f3_sobol(self):
        if False:
            i = 10
            return i + 15
        'NLP: Hock and Schittkowski problem 18'
        run_test(test3_1)

    @pytest.mark.slow
    def test_f4_sobol(self):
        if False:
            return 10
        'NLP: (High dimensional) Hock and Schittkowski 11 problem (HS11)'
        options = {'infty_constraints': False}
        run_test(test4_1, n=990 * 2, options=options)

    def test_f5_1_sobol(self):
        if False:
            return 10
        'NLP: Eggholder, multimodal'
        run_test(test5_1, n=60)

    def test_f5_2_sobol(self):
        if False:
            for i in range(10):
                print('nop')
        'NLP: Eggholder, multimodal'
        run_test(test5_1, n=60, iters=5)

class TestShgoSimplicialTestFunctions:
    """
    Global optimisation tests with Simplicial sampling:
    """

    def test_f1_1_simplicial(self):
        if False:
            i = 10
            return i + 15
        'Multivariate test function 1:\n        x[0]**2 + x[1]**2 with bounds=[(-1, 6), (-1, 6)]'
        run_test(test1_1, n=1, sampling_method='simplicial')

    def test_f1_2_simplicial(self):
        if False:
            return 10
        'Multivariate test function 1:\n        x[0]**2 + x[1]**2 with bounds=[(0, 1), (0, 1)]'
        run_test(test1_2, n=1, sampling_method='simplicial')

    def test_f1_3_simplicial(self):
        if False:
            i = 10
            return i + 15
        'Multivariate test function 1: x[0]**2 + x[1]**2\n        with bounds=[(None, None),(None, None)]'
        run_test(test1_3, n=5, sampling_method='simplicial')

    def test_f2_1_simplicial(self):
        if False:
            for i in range(10):
                print('nop')
        'Univariate test function on\n        f(x) = (x - 30) * sin(x) with bounds=[(0, 60)]'
        options = {'minimize_every_iter': False}
        run_test(test2_1, n=200, iters=7, options=options, sampling_method='simplicial')

    def test_f2_2_simplicial(self):
        if False:
            print('Hello World!')
        'Univariate test function on\n        f(x) = (x - 30) * sin(x) bounds=[(0, 4.5)]'
        run_test(test2_2, n=1, sampling_method='simplicial')

    def test_f3_simplicial(self):
        if False:
            return 10
        'NLP: Hock and Schittkowski problem 18'
        run_test(test3_1, n=1, sampling_method='simplicial')

    @pytest.mark.slow
    def test_f4_simplicial(self):
        if False:
            for i in range(10):
                print('nop')
        'NLP: (High dimensional) Hock and Schittkowski 11 problem (HS11)'
        run_test(test4_1, n=1, sampling_method='simplicial')

    def test_lj_symmetry_old(self):
        if False:
            return 10
        'LJ: Symmetry-constrained test function'
        options = {'symmetry': True, 'disp': True}
        args = (6,)
        run_test(testLJ, args=args, n=300, options=options, iters=1, sampling_method='simplicial')

    def test_f5_1_lj_symmetry(self):
        if False:
            for i in range(10):
                print('nop')
        'LJ: Symmetry constrained test function'
        options = {'symmetry': [0] * 6, 'disp': True}
        args = (6,)
        run_test(testLJ, args=args, n=300, options=options, iters=1, sampling_method='simplicial')

    def test_f5_2_cons_symmetry(self):
        if False:
            i = 10
            return i + 15
        'Symmetry constrained test function'
        options = {'symmetry': [0, 0], 'disp': True}
        run_test(test1_1, n=200, options=options, iters=1, sampling_method='simplicial')

    def test_f5_3_cons_symmetry(self):
        if False:
            for i in range(10):
                print('nop')
        'Assymmetrically constrained test function'
        options = {'symmetry': [0, 0, 0, 3], 'disp': True}
        run_test(test_s, n=10000, options=options, iters=1, sampling_method='simplicial')

    @pytest.mark.skip('Not a test')
    def test_f0_min_variance(self):
        if False:
            print('Hello World!')
        'Return a minimum on a perfectly symmetric problem, based on\n            gh10429'
        avg = 0.5
        cons = {'type': 'eq', 'fun': lambda x: numpy.mean(x) - avg}
        res = shgo(numpy.var, bounds=6 * [(0, 1)], constraints=cons)
        assert res.success
        assert_allclose(res.fun, 0, atol=1e-15)
        assert_allclose(res.x, 0.5)

    @pytest.mark.skip('Not a test')
    def test_f0_min_variance_1D(self):
        if False:
            i = 10
            return i + 15
        'Return a minimum on a perfectly symmetric 1D problem, based on\n            gh10538'

        def fun(x):
            if False:
                return 10
            return x * (x - 1.0) * (x - 0.5)
        bounds = [(0, 1)]
        res = shgo(fun, bounds=bounds)
        ref = minimize_scalar(fun, bounds=bounds[0])
        assert res.success
        assert_allclose(res.fun, ref.fun)
        assert_allclose(res.x, ref.x, rtol=1e-06)

class TestShgoArguments:

    def test_1_1_simpl_iter(self):
        if False:
            for i in range(10):
                print('nop')
        'Iterative simplicial sampling on TestFunction 1 (multivariate)'
        run_test(test1_2, n=None, iters=2, sampling_method='simplicial')

    def test_1_2_simpl_iter(self):
        if False:
            return 10
        'Iterative simplicial on TestFunction 2 (univariate)'
        options = {'minimize_every_iter': False}
        run_test(test2_1, n=None, iters=9, options=options, sampling_method='simplicial')

    def test_2_1_sobol_iter(self):
        if False:
            print('Hello World!')
        'Iterative Sobol sampling on TestFunction 1 (multivariate)'
        run_test(test1_2, n=None, iters=1, sampling_method='sobol')

    def test_2_2_sobol_iter(self):
        if False:
            for i in range(10):
                print('nop')
        'Iterative Sobol sampling on TestFunction 2 (univariate)'
        res = shgo(test2_1.f, test2_1.bounds, constraints=test2_1.cons, n=None, iters=1, sampling_method='sobol')
        numpy.testing.assert_allclose(res.x, test2_1.expected_x, rtol=1e-05, atol=1e-05)
        numpy.testing.assert_allclose(res.fun, test2_1.expected_fun, atol=1e-05)

    def test_3_1_disp_simplicial(self):
        if False:
            for i in range(10):
                print('nop')
        'Iterative sampling on TestFunction 1 and 2  (multi and univariate)\n        '

        def callback_func(x):
            if False:
                return 10
            print('Local minimization callback test')
        for test in [test1_1, test2_1]:
            shgo(test.f, test.bounds, iters=1, sampling_method='simplicial', callback=callback_func, options={'disp': True})
            shgo(test.f, test.bounds, n=1, sampling_method='simplicial', callback=callback_func, options={'disp': True})

    def test_3_2_disp_sobol(self):
        if False:
            i = 10
            return i + 15
        'Iterative sampling on TestFunction 1 and 2 (multi and univariate)'

        def callback_func(x):
            if False:
                while True:
                    i = 10
            print('Local minimization callback test')
        for test in [test1_1, test2_1]:
            shgo(test.f, test.bounds, iters=1, sampling_method='sobol', callback=callback_func, options={'disp': True})
            shgo(test.f, test.bounds, n=1, sampling_method='simplicial', callback=callback_func, options={'disp': True})

    def test_args_gh14589(self):
        if False:
            for i in range(10):
                print('nop')
        'Using `args` used to cause `shgo` to fail; see #14589, #15986,\n        #16506'
        res = shgo(func=lambda x, y, z: x * z + y, bounds=[(0, 3)], args=(1, 2))
        ref = shgo(func=lambda x: 2 * x + 1, bounds=[(0, 3)])
        assert_allclose(res.fun, ref.fun)
        assert_allclose(res.x, ref.x)

    @pytest.mark.slow
    def test_4_1_known_f_min(self):
        if False:
            return 10
        'Test known function minima stopping criteria'
        options = {'f_min': test4_1.expected_fun, 'f_tol': 1e-06, 'minimize_every_iter': True}
        run_test(test4_1, n=None, test_atol=1e-05, options=options, sampling_method='simplicial')

    @pytest.mark.slow
    def test_4_2_known_f_min(self):
        if False:
            return 10
        'Test Global mode limiting local evaluations'
        options = {'f_min': test4_1.expected_fun, 'f_tol': 1e-06, 'minimize_every_iter': True, 'local_iter': 1}
        run_test(test4_1, n=None, test_atol=1e-05, options=options, sampling_method='simplicial')

    def test_4_4_known_f_min(self):
        if False:
            print('Hello World!')
        'Test Global mode limiting local evaluations for 1D funcs'
        options = {'f_min': test2_1.expected_fun, 'f_tol': 1e-06, 'minimize_every_iter': True, 'local_iter': 1, 'infty_constraints': False}
        res = shgo(test2_1.f, test2_1.bounds, constraints=test2_1.cons, n=None, iters=None, options=options, sampling_method='sobol')
        numpy.testing.assert_allclose(res.x, test2_1.expected_x, rtol=1e-05, atol=1e-05)

    def test_5_1_simplicial_argless(self):
        if False:
            return 10
        'Test Default simplicial sampling settings on TestFunction 1'
        res = shgo(test1_1.f, test1_1.bounds, constraints=test1_1.cons)
        numpy.testing.assert_allclose(res.x, test1_1.expected_x, rtol=1e-05, atol=1e-05)

    def test_5_2_sobol_argless(self):
        if False:
            while True:
                i = 10
        'Test Default sobol sampling settings on TestFunction 1'
        res = shgo(test1_1.f, test1_1.bounds, constraints=test1_1.cons, sampling_method='sobol')
        numpy.testing.assert_allclose(res.x, test1_1.expected_x, rtol=1e-05, atol=1e-05)

    def test_6_1_simplicial_max_iter(self):
        if False:
            while True:
                i = 10
        'Test that maximum iteration option works on TestFunction 3'
        options = {'max_iter': 2}
        res = shgo(test3_1.f, test3_1.bounds, constraints=test3_1.cons, options=options, sampling_method='simplicial')
        numpy.testing.assert_allclose(res.x, test3_1.expected_x, rtol=1e-05, atol=1e-05)
        numpy.testing.assert_allclose(res.fun, test3_1.expected_fun, atol=1e-05)

    def test_6_2_simplicial_min_iter(self):
        if False:
            print('Hello World!')
        'Test that maximum iteration option works on TestFunction 3'
        options = {'min_iter': 2}
        res = shgo(test3_1.f, test3_1.bounds, constraints=test3_1.cons, options=options, sampling_method='simplicial')
        numpy.testing.assert_allclose(res.x, test3_1.expected_x, rtol=1e-05, atol=1e-05)
        numpy.testing.assert_allclose(res.fun, test3_1.expected_fun, atol=1e-05)

    def test_7_1_minkwargs(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the minimizer_kwargs arguments for solvers with constraints'
        for solver in ['COBYLA', 'SLSQP']:
            minimizer_kwargs = {'method': solver, 'constraints': test3_1.cons}
            run_test(test3_1, n=100, test_atol=0.001, minimizer_kwargs=minimizer_kwargs, sampling_method='sobol')

    def test_7_2_minkwargs(self):
        if False:
            print('Hello World!')
        'Test the minimizer_kwargs default inits'
        minimizer_kwargs = {'ftol': 1e-05}
        options = {'disp': True}
        SHGO(test3_1.f, test3_1.bounds, constraints=test3_1.cons[0], minimizer_kwargs=minimizer_kwargs, options=options)

    def test_7_3_minkwargs(self):
        if False:
            i = 10
            return i + 15
        'Test minimizer_kwargs arguments for solvers without constraints'
        for solver in ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']:

            def jac(x):
                if False:
                    i = 10
                    return i + 15
                return numpy.array([2 * x[0], 2 * x[1]]).T

            def hess(x):
                if False:
                    return 10
                return numpy.array([[2, 0], [0, 2]])
            minimizer_kwargs = {'method': solver, 'jac': jac, 'hess': hess}
            logging.info(f'Solver = {solver}')
            logging.info('=' * 100)
            run_test(test1_1, n=100, test_atol=0.001, minimizer_kwargs=minimizer_kwargs, sampling_method='sobol')

    def test_8_homology_group_diff(self):
        if False:
            for i in range(10):
                print('nop')
        options = {'minhgrd': 1, 'minimize_every_iter': True}
        run_test(test1_1, n=None, iters=None, options=options, sampling_method='simplicial')

    def test_9_cons_g(self):
        if False:
            i = 10
            return i + 15
        'Test single function constraint passing'
        SHGO(test3_1.f, test3_1.bounds, constraints=test3_1.cons[0])

    @pytest.mark.xfail(IS_PYPY and sys.platform == 'win32', reason='Failing and fix in PyPy not planned (see gh-18632)')
    def test_10_finite_time(self):
        if False:
            i = 10
            return i + 15
        'Test single function constraint passing'
        options = {'maxtime': 1e-15}

        def f(x):
            if False:
                return 10
            time.sleep(1e-14)
            return 0.0
        res = shgo(f, test1_1.bounds, iters=5, options=options)
        assert res.nit == 1

    def test_11_f_min_0(self):
        if False:
            while True:
                i = 10
        'Test to cover the case where f_lowest == 0'
        options = {'f_min': 0.0, 'disp': True}
        res = shgo(test1_2.f, test1_2.bounds, n=10, iters=None, options=options, sampling_method='sobol')
        numpy.testing.assert_equal(0, res.x[0])
        numpy.testing.assert_equal(0, res.x[1])

    @pytest.mark.skip(reason='no way of currently testing this')
    def test_12_sobol_inf_cons(self):
        if False:
            while True:
                i = 10
        'Test to cover the case where f_lowest == 0'
        options = {'maxtime': 1e-15, 'f_min': 0.0}
        res = shgo(test1_2.f, test1_2.bounds, n=1, iters=None, options=options, sampling_method='sobol')
        numpy.testing.assert_equal(0.0, res.fun)

    def test_13_high_sobol(self):
        if False:
            i = 10
            return i + 15
        'Test init of high-dimensional sobol sequences'

        def f(x):
            if False:
                while True:
                    i = 10
            return 0
        bounds = [(None, None)] * 41
        SHGOc = SHGO(f, bounds, sampling_method='sobol')
        SHGOc.sampling_function(2, 50)

    def test_14_local_iter(self):
        if False:
            i = 10
            return i + 15
        'Test limited local iterations for a pseudo-global mode'
        options = {'local_iter': 4}
        run_test(test5_1, n=60, options=options)

    def test_15_min_every_iter(self):
        if False:
            i = 10
            return i + 15
        'Test minimize every iter options and cover function cache'
        options = {'minimize_every_iter': True}
        run_test(test1_1, n=1, iters=7, options=options, sampling_method='sobol')

    def test_16_disp_bounds_minimizer(self):
        if False:
            i = 10
            return i + 15
        'Test disp=True with minimizers that do not support bounds '
        options = {'disp': True}
        minimizer_kwargs = {'method': 'nelder-mead'}
        run_test(test1_2, sampling_method='simplicial', options=options, minimizer_kwargs=minimizer_kwargs)

    def test_17_custom_sampling(self):
        if False:
            while True:
                i = 10
        'Test the functionality to add custom sampling methods to shgo'

        def sample(n, d):
            if False:
                while True:
                    i = 10
            return numpy.random.uniform(size=(n, d))
        run_test(test1_1, n=30, sampling_method=sample)

    def test_18_bounds_class(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                i = 10
                return i + 15
            return numpy.square(x).sum()
        lb = [-6.0, 1.0, -5.0]
        ub = [-1.0, 3.0, 5.0]
        bounds_old = list(zip(lb, ub))
        bounds_new = Bounds(lb, ub)
        res_old_bounds = shgo(f, bounds_old)
        res_new_bounds = shgo(f, bounds_new)
        assert res_new_bounds.nfev == res_old_bounds.nfev
        assert res_new_bounds.message == res_old_bounds.message
        assert res_new_bounds.success == res_old_bounds.success
        x_opt = numpy.array([-1.0, 1.0, 0.0])
        numpy.testing.assert_allclose(res_new_bounds.x, x_opt)
        numpy.testing.assert_allclose(res_new_bounds.x, res_old_bounds.x)

    def test_19_parallelization(self):
        if False:
            i = 10
            return i + 15
        'Test the functionality to add custom sampling methods to shgo'
        with Pool(2) as p:
            run_test(test1_1, n=30, workers=p.map)
        run_test(test1_1, n=30, workers=map)
        with Pool(2) as p:
            run_test(test_s, n=30, workers=p.map)
        run_test(test_s, n=30, workers=map)

    def test_20_constrained_args(self):
        if False:
            return 10
        'Test that constraints can be passed to arguments'

        def eggholder(x):
            if False:
                i = 10
                return i + 15
            return -(x[1] + 47.0) * numpy.sin(numpy.sqrt(abs(x[0] / 2.0 + (x[1] + 47.0)))) - x[0] * numpy.sin(numpy.sqrt(abs(x[0] - (x[1] + 47.0))))

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return 24.55 * x[0] + 26.75 * x[1] + 39 * x[2] + 40.5 * x[3]
        bounds = [(0, 1.0)] * 4

        def g1_modified(x, i):
            if False:
                while True:
                    i = 10
            return i * 2.3 * x[0] + i * 5.6 * x[1] + 11.1 * x[2] + 1.3 * x[3] - 5

        def g2(x):
            if False:
                for i in range(10):
                    print('nop')
            return 12 * x[0] + 11.9 * x[1] + 41.8 * x[2] + 52.1 * x[3] - 21 - 1.645 * numpy.sqrt(0.28 * x[0] ** 2 + 0.19 * x[1] ** 2 + 20.5 * x[2] ** 2 + 0.62 * x[3] ** 2)

        def h1(x):
            if False:
                for i in range(10):
                    print('nop')
            return x[0] + x[1] + x[2] + x[3] - 1
        cons = ({'type': 'ineq', 'fun': g1_modified, 'args': (0,)}, {'type': 'ineq', 'fun': g2}, {'type': 'eq', 'fun': h1})
        shgo(f, bounds, n=300, iters=1, constraints=cons)
        shgo(f, bounds, n=300, iters=1, constraints=cons, sampling_method='sobol')

    def test_21_1_jac_true(self):
        if False:
            i = 10
            return i + 15
        'Test that shgo can handle objective functions that return the\n        gradient alongside the objective value. Fixes gh-13547'

        def func(x):
            if False:
                for i in range(10):
                    print('nop')
            return (numpy.sum(numpy.power(x, 2)), 2 * x)
        shgo(func, bounds=[[-1, 1], [1, 2]], n=100, iters=5, sampling_method='sobol', minimizer_kwargs={'method': 'SLSQP', 'jac': True})

        def func(x):
            if False:
                return 10
            return (numpy.sum(x ** 2), 2 * x)
        bounds = [[-1, 1], [1, 2], [-1, 1], [1, 2], [0, 3]]
        res = shgo(func, bounds=bounds, sampling_method='sobol', minimizer_kwargs={'method': 'SLSQP', 'jac': True})
        ref = minimize(func, x0=[1, 1, 1, 1, 1], bounds=bounds, jac=True)
        assert res.success
        assert_allclose(res.fun, ref.fun)
        assert_allclose(res.x, ref.x, atol=1e-15)

    @pytest.mark.parametrize('derivative', ['jac', 'hess', 'hessp'])
    def test_21_2_derivative_options(self, derivative):
        if False:
            i = 10
            return i + 15
        "shgo used to raise an error when passing `options` with 'jac'\n        # see gh-12963. check that this is resolved\n        "

        def objective(x):
            if False:
                i = 10
                return i + 15
            return 3 * x[0] * x[0] + 2 * x[0] + 5

        def gradient(x):
            if False:
                print('Hello World!')
            return 6 * x[0] + 2

        def hess(x):
            if False:
                for i in range(10):
                    print('nop')
            return 6

        def hessp(x, p):
            if False:
                while True:
                    i = 10
            return 6 * p
        derivative_funcs = {'jac': gradient, 'hess': hess, 'hessp': hessp}
        options = {derivative: derivative_funcs[derivative]}
        minimizer_kwargs = {'method': 'trust-constr'}
        bounds = [(-100, 100)]
        res = shgo(objective, bounds, minimizer_kwargs=minimizer_kwargs, options=options)
        ref = minimize(objective, x0=[0], bounds=bounds, **minimizer_kwargs, **options)
        assert res.success
        numpy.testing.assert_allclose(res.fun, ref.fun)
        numpy.testing.assert_allclose(res.x, ref.x)

    def test_21_3_hess_options_rosen(self):
        if False:
            i = 10
            return i + 15
        'Ensure the Hessian gets passed correctly to the local minimizer\n        routine. Previous report gh-14533.\n        '
        bounds = [(0, 1.6), (0, 1.6), (0, 1.4), (0, 1.4), (0, 1.4)]
        options = {'jac': rosen_der, 'hess': rosen_hess}
        minimizer_kwargs = {'method': 'Newton-CG'}
        res = shgo(rosen, bounds, minimizer_kwargs=minimizer_kwargs, options=options)
        ref = minimize(rosen, numpy.zeros(5), method='Newton-CG', **options)
        assert res.success
        assert_allclose(res.fun, ref.fun)
        assert_allclose(res.x, ref.x, atol=1e-15)

    def test_21_arg_tuple_sobol(self):
        if False:
            for i in range(10):
                print('nop')
        'shgo used to raise an error when passing `args` with Sobol sampling\n        # see gh-12114. check that this is resolved'

        def fun(x, k):
            if False:
                while True:
                    i = 10
            return x[0] ** k
        constraints = {'type': 'ineq', 'fun': lambda x: x[0] - 1}
        bounds = [(0, 10)]
        res = shgo(fun, bounds, args=(1,), constraints=constraints, sampling_method='sobol')
        ref = minimize(fun, numpy.zeros(1), bounds=bounds, args=(1,), constraints=constraints)
        assert res.success
        assert_allclose(res.fun, ref.fun)
        assert_allclose(res.x, ref.x)

class TestShgoFailures:

    def test_1_maxiter(self):
        if False:
            return 10
        'Test failure on insufficient iterations'
        options = {'maxiter': 2}
        res = shgo(test4_1.f, test4_1.bounds, n=2, iters=None, options=options, sampling_method='sobol')
        numpy.testing.assert_equal(False, res.success)
        numpy.testing.assert_equal(4, res.tnev)

    def test_2_sampling(self):
        if False:
            print('Hello World!')
        'Rejection of unknown sampling method'
        assert_raises(ValueError, shgo, test1_1.f, test1_1.bounds, sampling_method='not_Sobol')

    def test_3_1_no_min_pool_sobol(self):
        if False:
            while True:
                i = 10
        'Check that the routine stops when no minimiser is found\n           after maximum specified function evaluations'
        options = {'maxfev': 10, 'disp': True}
        res = shgo(test_table.f, test_table.bounds, n=3, options=options, sampling_method='sobol')
        numpy.testing.assert_equal(False, res.success)
        numpy.testing.assert_equal(12, res.nfev)

    def test_3_2_no_min_pool_simplicial(self):
        if False:
            print('Hello World!')
        'Check that the routine stops when no minimiser is found\n           after maximum specified sampling evaluations'
        options = {'maxev': 10, 'disp': True}
        res = shgo(test_table.f, test_table.bounds, n=3, options=options, sampling_method='simplicial')
        numpy.testing.assert_equal(False, res.success)

    def test_4_1_bound_err(self):
        if False:
            while True:
                i = 10
        'Specified bounds ub > lb'
        bounds = [(6, 3), (3, 5)]
        assert_raises(ValueError, shgo, test1_1.f, bounds)

    def test_4_2_bound_err(self):
        if False:
            for i in range(10):
                print('nop')
        'Specified bounds are of the form (lb, ub)'
        bounds = [(3, 5, 5), (3, 5)]
        assert_raises(ValueError, shgo, test1_1.f, bounds)

    def test_5_1_1_infeasible_sobol(self):
        if False:
            return 10
        'Ensures the algorithm terminates on infeasible problems\n           after maxev is exceeded. Use infty constraints option'
        options = {'maxev': 100, 'disp': True}
        res = shgo(test_infeasible.f, test_infeasible.bounds, constraints=test_infeasible.cons, n=100, options=options, sampling_method='sobol')
        numpy.testing.assert_equal(False, res.success)

    def test_5_1_2_infeasible_sobol(self):
        if False:
            i = 10
            return i + 15
        'Ensures the algorithm terminates on infeasible problems\n           after maxev is exceeded. Do not use infty constraints option'
        options = {'maxev': 100, 'disp': True, 'infty_constraints': False}
        res = shgo(test_infeasible.f, test_infeasible.bounds, constraints=test_infeasible.cons, n=100, options=options, sampling_method='sobol')
        numpy.testing.assert_equal(False, res.success)

    def test_5_2_infeasible_simplicial(self):
        if False:
            i = 10
            return i + 15
        'Ensures the algorithm terminates on infeasible problems\n           after maxev is exceeded.'
        options = {'maxev': 1000, 'disp': False}
        res = shgo(test_infeasible.f, test_infeasible.bounds, constraints=test_infeasible.cons, n=100, options=options, sampling_method='simplicial')
        numpy.testing.assert_equal(False, res.success)

    def test_6_1_lower_known_f_min(self):
        if False:
            while True:
                i = 10
        'Test Global mode limiting local evaluations with f* too high'
        options = {'f_min': test2_1.expected_fun + 2.0, 'f_tol': 1e-06, 'minimize_every_iter': True, 'local_iter': 1, 'infty_constraints': False}
        args = (test2_1.f, test2_1.bounds)
        kwargs = {'constraints': test2_1.cons, 'n': None, 'iters': None, 'options': options, 'sampling_method': 'sobol'}
        warns(UserWarning, shgo, *args, **kwargs)

    def test(self):
        if False:
            print('Hello World!')
        from scipy.optimize import rosen, shgo
        bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]

        def fun(x):
            if False:
                return 10
            fun.nfev += 1
            return rosen(x)
        fun.nfev = 0
        result = shgo(fun, bounds)
        print(result.x, result.fun, fun.nfev)

class TestShgoReturns:

    def test_1_nfev_simplicial(self):
        if False:
            print('Hello World!')
        bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]

        def fun(x):
            if False:
                print('Hello World!')
            fun.nfev += 1
            return rosen(x)
        fun.nfev = 0
        result = shgo(fun, bounds)
        numpy.testing.assert_equal(fun.nfev, result.nfev)

    def test_1_nfev_sobol(self):
        if False:
            while True:
                i = 10
        bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]

        def fun(x):
            if False:
                while True:
                    i = 10
            fun.nfev += 1
            return rosen(x)
        fun.nfev = 0
        result = shgo(fun, bounds, sampling_method='sobol')
        numpy.testing.assert_equal(fun.nfev, result.nfev)

def test_vector_constraint():
    if False:
        return 10

    def quad(x):
        if False:
            return 10
        x = np.asarray(x)
        return [np.sum(x ** 2)]
    nlc = NonlinearConstraint(quad, [2.2], [3])
    oldc = new_constraint_to_old(nlc, np.array([1.0, 1.0]))
    res = shgo(rosen, [(0, 10), (0, 10)], constraints=oldc, sampling_method='sobol')
    assert np.all(np.sum(res.x ** 2) >= 2.2)
    assert np.all(np.sum(res.x ** 2) <= 3.0)
    assert res.success

@pytest.mark.filterwarnings('ignore:delta_grad')
def test_trust_constr():
    if False:
        i = 10
        return i + 15

    def quad(x):
        if False:
            for i in range(10):
                print('nop')
        x = np.asarray(x)
        return [np.sum(x ** 2)]
    nlc = NonlinearConstraint(quad, [2.6], [3])
    minimizer_kwargs = {'method': 'trust-constr'}
    res = shgo(rosen, [(0, 10), (0, 10)], constraints=nlc, sampling_method='sobol', minimizer_kwargs=minimizer_kwargs)
    assert np.all(np.sum(res.x ** 2) >= 2.6)
    assert np.all(np.sum(res.x ** 2) <= 3.0)
    assert res.success

def test_equality_constraints():
    if False:
        return 10
    bounds = [(0.9, 4.0)] * 2

    def faulty(x):
        if False:
            i = 10
            return i + 15
        return x[0] + x[1]
    nlc = NonlinearConstraint(faulty, 3.9, 3.9)
    res = shgo(rosen, bounds=bounds, constraints=nlc)
    assert_allclose(np.sum(res.x), 3.9)

    def faulty(x):
        if False:
            return 10
        return x[0] + x[1] - 3.9
    constraints = {'type': 'eq', 'fun': faulty}
    res = shgo(rosen, bounds=bounds, constraints=constraints)
    assert_allclose(np.sum(res.x), 3.9)
    bounds = [(0, 1.0)] * 4

    def faulty(x):
        if False:
            for i in range(10):
                print('nop')
        return x[0] + x[1] + x[2] + x[3] - 1
    constraints = {'type': 'eq', 'fun': faulty}
    res = shgo(lambda x: -np.prod(x), bounds=bounds, constraints=constraints, sampling_method='sobol')
    assert_allclose(np.sum(res.x), 1.0)

def test_gh16971():
    if False:
        while True:
            i = 10

    def cons(x):
        if False:
            while True:
                i = 10
        return np.sum(x ** 2) - 0
    c = {'fun': cons, 'type': 'ineq'}
    minimizer_kwargs = {'method': 'COBYLA', 'options': {'rhobeg': 5, 'tol': 0.5, 'catol': 0.05}}
    s = SHGO(rosen, [(0, 10)] * 2, constraints=c, minimizer_kwargs=minimizer_kwargs)
    assert s.minimizer_kwargs['method'].lower() == 'cobyla'
    assert s.minimizer_kwargs['options']['catol'] == 0.05