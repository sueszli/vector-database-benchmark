from itertools import product
import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_, assert_allclose, assert_equal, suppress_warnings
from pytest import raises as assert_raises
from scipy.sparse import issparse, lil_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import least_squares, Bounds
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
from scipy.optimize._lsq.common import EPS, make_strictly_feasible, CL_scaling_vector

def fun_trivial(x, a=0):
    if False:
        return 10
    return (x - a) ** 2 + 5.0

def jac_trivial(x, a=0.0):
    if False:
        for i in range(10):
            print('nop')
    return 2 * (x - a)

def fun_2d_trivial(x):
    if False:
        while True:
            i = 10
    return np.array([x[0], x[1]])

def jac_2d_trivial(x):
    if False:
        i = 10
        return i + 15
    return np.identity(2)

def fun_rosenbrock(x):
    if False:
        i = 10
        return i + 15
    return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])

def jac_rosenbrock(x):
    if False:
        return 10
    return np.array([[-20 * x[0], 10], [-1, 0]])

def jac_rosenbrock_bad_dim(x):
    if False:
        while True:
            i = 10
    return np.array([[-20 * x[0], 10], [-1, 0], [0.0, 0.0]])

def fun_rosenbrock_cropped(x):
    if False:
        print('Hello World!')
    return fun_rosenbrock(x)[0]

def jac_rosenbrock_cropped(x):
    if False:
        for i in range(10):
            print('nop')
    return jac_rosenbrock(x)[0]

def fun_wrong_dimensions(x):
    if False:
        print('Hello World!')
    return np.array([x, x ** 2, x ** 3])

def jac_wrong_dimensions(x, a=0.0):
    if False:
        return 10
    return np.atleast_3d(jac_trivial(x, a=a))

def fun_bvp(x):
    if False:
        while True:
            i = 10
    n = int(np.sqrt(x.shape[0]))
    u = np.zeros((n + 2, n + 2))
    x = x.reshape((n, n))
    u[1:-1, 1:-1] = x
    y = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * x + x ** 3
    return y.ravel()

class BroydenTridiagonal:

    def __init__(self, n=100, mode='sparse'):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(0)
        self.n = n
        self.x0 = -np.ones(n)
        self.lb = np.linspace(-2, -1.5, n)
        self.ub = np.linspace(-0.8, 0.0, n)
        self.lb += 0.1 * np.random.randn(n)
        self.ub += 0.1 * np.random.randn(n)
        self.x0 += 0.1 * np.random.randn(n)
        self.x0 = make_strictly_feasible(self.x0, self.lb, self.ub)
        if mode == 'sparse':
            self.sparsity = lil_matrix((n, n), dtype=int)
            i = np.arange(n)
            self.sparsity[i, i] = 1
            i = np.arange(1, n)
            self.sparsity[i, i - 1] = 1
            i = np.arange(n - 1)
            self.sparsity[i, i + 1] = 1
            self.jac = self._jac
        elif mode == 'operator':
            self.jac = lambda x: aslinearoperator(self._jac(x))
        elif mode == 'dense':
            self.sparsity = None
            self.jac = lambda x: self._jac(x).toarray()
        else:
            assert_(False)

    def fun(self, x):
        if False:
            while True:
                i = 10
        f = (3 - x) * x + 1
        f[1:] -= x[:-1]
        f[:-1] -= 2 * x[1:]
        return f

    def _jac(self, x):
        if False:
            print('Hello World!')
        J = lil_matrix((self.n, self.n))
        i = np.arange(self.n)
        J[i, i] = 3 - 2 * x
        i = np.arange(1, self.n)
        J[i, i - 1] = -1
        i = np.arange(self.n - 1)
        J[i, i + 1] = -2
        return J

class ExponentialFittingProblem:
    """Provide data and function for exponential fitting in the form
    y = a + exp(b * x) + noise."""

    def __init__(self, a, b, noise, n_outliers=1, x_range=(-1, 1), n_points=11, random_seed=None):
        if False:
            i = 10
            return i + 15
        np.random.seed(random_seed)
        self.m = n_points
        self.n = 2
        self.p0 = np.zeros(2)
        self.x = np.linspace(x_range[0], x_range[1], n_points)
        self.y = a + np.exp(b * self.x)
        self.y += noise * np.random.randn(self.m)
        outliers = np.random.randint(0, self.m, n_outliers)
        self.y[outliers] += 50 * noise * np.random.rand(n_outliers)
        self.p_opt = np.array([a, b])

    def fun(self, p):
        if False:
            print('Hello World!')
        return p[0] + np.exp(p[1] * self.x) - self.y

    def jac(self, p):
        if False:
            for i in range(10):
                print('nop')
        J = np.empty((self.m, self.n))
        J[:, 0] = 1
        J[:, 1] = self.x * np.exp(p[1] * self.x)
        return J

def cubic_soft_l1(z):
    if False:
        for i in range(10):
            print('nop')
    rho = np.empty((3, z.size))
    t = 1 + z
    rho[0] = 3 * (t ** (1 / 3) - 1)
    rho[1] = t ** (-2 / 3)
    rho[2] = -2 / 3 * t ** (-5 / 3)
    return rho
LOSSES = list(IMPLEMENTED_LOSSES.keys()) + [cubic_soft_l1]

class BaseMixin:

    def test_basic(self):
        if False:
            return 10
        res = least_squares(fun_trivial, 2.0, method=self.method)
        assert_allclose(res.x, 0, atol=0.0001)
        assert_allclose(res.fun, fun_trivial(res.x))

    def test_args_kwargs(self):
        if False:
            return 10
        a = 3.0
        for jac in ['2-point', '3-point', 'cs', jac_trivial]:
            with suppress_warnings() as sup:
                sup.filter(UserWarning, "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'")
                res = least_squares(fun_trivial, 2.0, jac, args=(a,), method=self.method)
                res1 = least_squares(fun_trivial, 2.0, jac, kwargs={'a': a}, method=self.method)
            assert_allclose(res.x, a, rtol=0.0001)
            assert_allclose(res1.x, a, rtol=0.0001)
            assert_raises(TypeError, least_squares, fun_trivial, 2.0, args=(3, 4), method=self.method)
            assert_raises(TypeError, least_squares, fun_trivial, 2.0, kwargs={'kaboom': 3}, method=self.method)

    def test_jac_options(self):
        if False:
            print('Hello World!')
        for jac in ['2-point', '3-point', 'cs', jac_trivial]:
            with suppress_warnings() as sup:
                sup.filter(UserWarning, "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'")
                res = least_squares(fun_trivial, 2.0, jac, method=self.method)
            assert_allclose(res.x, 0, atol=0.0001)
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, jac='oops', method=self.method)

    def test_nfev_options(self):
        if False:
            i = 10
            return i + 15
        for max_nfev in [None, 20]:
            res = least_squares(fun_trivial, 2.0, max_nfev=max_nfev, method=self.method)
            assert_allclose(res.x, 0, atol=0.0001)

    def test_x_scale_options(self):
        if False:
            print('Hello World!')
        for x_scale in [1.0, np.array([0.5]), 'jac']:
            res = least_squares(fun_trivial, 2.0, x_scale=x_scale)
            assert_allclose(res.x, 0)
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, x_scale='auto', method=self.method)
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, x_scale=-1.0, method=self.method)
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, x_scale=None, method=self.method)
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, x_scale=1.0 + 2j, method=self.method)

    def test_diff_step(self):
        if False:
            while True:
                i = 10
        res1 = least_squares(fun_trivial, 2.0, diff_step=0.1, method=self.method)
        res2 = least_squares(fun_trivial, 2.0, diff_step=-0.1, method=self.method)
        res3 = least_squares(fun_trivial, 2.0, diff_step=None, method=self.method)
        assert_allclose(res1.x, 0, atol=0.0001)
        assert_allclose(res2.x, 0, atol=0.0001)
        assert_allclose(res3.x, 0, atol=0.0001)
        assert_equal(res1.x, res2.x)
        assert_equal(res1.nfev, res2.nfev)

    def test_incorrect_options_usage(self):
        if False:
            return 10
        assert_raises(TypeError, least_squares, fun_trivial, 2.0, method=self.method, options={'no_such_option': 100})
        assert_raises(TypeError, least_squares, fun_trivial, 2.0, method=self.method, options={'max_nfev': 100})

    def test_full_result(self):
        if False:
            while True:
                i = 10
        res = least_squares(fun_trivial, 2.0, method=self.method)
        assert_allclose(res.x, 0, atol=0.0001)
        assert_allclose(res.cost, 12.5)
        assert_allclose(res.fun, 5)
        assert_allclose(res.jac, 0, atol=0.0001)
        assert_allclose(res.grad, 0, atol=0.01)
        assert_allclose(res.optimality, 0, atol=0.01)
        assert_equal(res.active_mask, 0)
        if self.method == 'lm':
            assert_(res.nfev < 30)
            assert_(res.njev is None)
        else:
            assert_(res.nfev < 10)
            assert_(res.njev < 10)
        assert_(res.status > 0)
        assert_(res.success)

    def test_full_result_single_fev(self):
        if False:
            for i in range(10):
                print('nop')
        if self.method == 'lm':
            return
        res = least_squares(fun_trivial, 2.0, method=self.method, max_nfev=1)
        assert_equal(res.x, np.array([2]))
        assert_equal(res.cost, 40.5)
        assert_equal(res.fun, np.array([9]))
        assert_equal(res.jac, np.array([[4]]))
        assert_equal(res.grad, np.array([36]))
        assert_equal(res.optimality, 36)
        assert_equal(res.active_mask, np.array([0]))
        assert_equal(res.nfev, 1)
        assert_equal(res.njev, 1)
        assert_equal(res.status, 0)
        assert_equal(res.success, 0)

    def test_rosenbrock(self):
        if False:
            print('Hello World!')
        x0 = [-2, 1]
        x_opt = [1, 1]
        for (jac, x_scale, tr_solver) in product(['2-point', '3-point', 'cs', jac_rosenbrock], [1.0, np.array([1.0, 0.2]), 'jac'], ['exact', 'lsmr']):
            with suppress_warnings() as sup:
                sup.filter(UserWarning, "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'")
                res = least_squares(fun_rosenbrock, x0, jac, x_scale=x_scale, tr_solver=tr_solver, method=self.method)
            assert_allclose(res.x, x_opt)

    def test_rosenbrock_cropped(self):
        if False:
            for i in range(10):
                print('nop')
        x0 = [-2, 1]
        if self.method == 'lm':
            assert_raises(ValueError, least_squares, fun_rosenbrock_cropped, x0, method='lm')
        else:
            for (jac, x_scale, tr_solver) in product(['2-point', '3-point', 'cs', jac_rosenbrock_cropped], [1.0, np.array([1.0, 0.2]), 'jac'], ['exact', 'lsmr']):
                res = least_squares(fun_rosenbrock_cropped, x0, jac, x_scale=x_scale, tr_solver=tr_solver, method=self.method)
                assert_allclose(res.cost, 0, atol=1e-14)

    def test_fun_wrong_dimensions(self):
        if False:
            for i in range(10):
                print('nop')
        assert_raises(ValueError, least_squares, fun_wrong_dimensions, 2.0, method=self.method)

    def test_jac_wrong_dimensions(self):
        if False:
            i = 10
            return i + 15
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, jac_wrong_dimensions, method=self.method)

    def test_fun_and_jac_inconsistent_dimensions(self):
        if False:
            for i in range(10):
                print('nop')
        x0 = [1, 2]
        assert_raises(ValueError, least_squares, fun_rosenbrock, x0, jac_rosenbrock_bad_dim, method=self.method)

    def test_x0_multidimensional(self):
        if False:
            i = 10
            return i + 15
        x0 = np.ones(4).reshape(2, 2)
        assert_raises(ValueError, least_squares, fun_trivial, x0, method=self.method)

    def test_x0_complex_scalar(self):
        if False:
            for i in range(10):
                print('nop')
        x0 = 2.0 + 0.0 * 1j
        assert_raises(ValueError, least_squares, fun_trivial, x0, method=self.method)

    def test_x0_complex_array(self):
        if False:
            i = 10
            return i + 15
        x0 = [1.0, 2.0 + 0.0 * 1j]
        assert_raises(ValueError, least_squares, fun_trivial, x0, method=self.method)

    def test_bvp(self):
        if False:
            return 10
        n = 10
        x0 = np.ones(n ** 2)
        if self.method == 'lm':
            max_nfev = 5000
        else:
            max_nfev = 100
        res = least_squares(fun_bvp, x0, ftol=0.01, method=self.method, max_nfev=max_nfev)
        assert_(res.nfev < max_nfev)
        assert_(res.cost < 0.5)

    def test_error_raised_when_all_tolerances_below_eps(self):
        if False:
            while True:
                i = 10
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, method=self.method, ftol=None, xtol=None, gtol=None)

    def test_convergence_with_only_one_tolerance_enabled(self):
        if False:
            i = 10
            return i + 15
        if self.method == 'lm':
            return
        x0 = [-2, 1]
        x_opt = [1, 1]
        for (ftol, xtol, gtol) in [(1e-08, None, None), (None, 1e-08, None), (None, None, 1e-08)]:
            res = least_squares(fun_rosenbrock, x0, jac=jac_rosenbrock, ftol=ftol, gtol=gtol, xtol=xtol, method=self.method)
            assert_allclose(res.x, x_opt)

class BoundsMixin:

    def test_inconsistent(self):
        if False:
            for i in range(10):
                print('nop')
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, bounds=(10.0, 0.0), method=self.method)

    def test_infeasible(self):
        if False:
            i = 10
            return i + 15
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, bounds=(3.0, 4), method=self.method)

    def test_wrong_number(self):
        if False:
            print('Hello World!')
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, bounds=(1.0, 2, 3), method=self.method)

    def test_inconsistent_shape(self):
        if False:
            for i in range(10):
                print('nop')
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, bounds=(1.0, [2.0, 3.0]), method=self.method)
        assert_raises(ValueError, least_squares, fun_rosenbrock, [1.0, 2.0], bounds=([0.0], [3.0, 4.0]), method=self.method)

    def test_in_bounds(self):
        if False:
            i = 10
            return i + 15
        for jac in ['2-point', '3-point', 'cs', jac_trivial]:
            res = least_squares(fun_trivial, 2.0, jac=jac, bounds=(-1.0, 3.0), method=self.method)
            assert_allclose(res.x, 0.0, atol=0.0001)
            assert_equal(res.active_mask, [0])
            assert_(-1 <= res.x <= 3)
            res = least_squares(fun_trivial, 2.0, jac=jac, bounds=(0.5, 3.0), method=self.method)
            assert_allclose(res.x, 0.5, atol=0.0001)
            assert_equal(res.active_mask, [-1])
            assert_(0.5 <= res.x <= 3)

    def test_bounds_shape(self):
        if False:
            return 10

        def get_bounds_direct(lb, ub):
            if False:
                print('Hello World!')
            return (lb, ub)

        def get_bounds_instances(lb, ub):
            if False:
                for i in range(10):
                    print('nop')
            return Bounds(lb, ub)
        for jac in ['2-point', '3-point', 'cs', jac_2d_trivial]:
            for bounds_func in [get_bounds_direct, get_bounds_instances]:
                x0 = [1.0, 1.0]
                res = least_squares(fun_2d_trivial, x0, jac=jac)
                assert_allclose(res.x, [0.0, 0.0])
                res = least_squares(fun_2d_trivial, x0, jac=jac, bounds=bounds_func(0.5, [2.0, 2.0]), method=self.method)
                assert_allclose(res.x, [0.5, 0.5])
                res = least_squares(fun_2d_trivial, x0, jac=jac, bounds=bounds_func([0.3, 0.2], 3.0), method=self.method)
                assert_allclose(res.x, [0.3, 0.2])
                res = least_squares(fun_2d_trivial, x0, jac=jac, bounds=bounds_func([-1, 0.5], [1.0, 3.0]), method=self.method)
                assert_allclose(res.x, [0.0, 0.5], atol=1e-05)

    def test_bounds_instances(self):
        if False:
            return 10
        res = least_squares(fun_trivial, 0.5, bounds=Bounds())
        assert_allclose(res.x, 0.0, atol=0.0001)
        res = least_squares(fun_trivial, 3.0, bounds=Bounds(lb=1.0))
        assert_allclose(res.x, 1.0, atol=0.0001)
        res = least_squares(fun_trivial, 0.5, bounds=Bounds(lb=-1.0, ub=1.0))
        assert_allclose(res.x, 0.0, atol=0.0001)
        res = least_squares(fun_trivial, -3.0, bounds=Bounds(ub=-1.0))
        assert_allclose(res.x, -1.0, atol=0.0001)
        res = least_squares(fun_2d_trivial, [0.5, 0.5], bounds=Bounds(lb=[-1.0, -1.0], ub=1.0))
        assert_allclose(res.x, [0.0, 0.0], atol=1e-05)
        res = least_squares(fun_2d_trivial, [0.5, 0.5], bounds=Bounds(lb=[0.1, 0.1]))
        assert_allclose(res.x, [0.1, 0.1], atol=1e-05)

    def test_rosenbrock_bounds(self):
        if False:
            print('Hello World!')
        x0_1 = np.array([-2.0, 1.0])
        x0_2 = np.array([2.0, 2.0])
        x0_3 = np.array([-2.0, 2.0])
        x0_4 = np.array([0.0, 2.0])
        x0_5 = np.array([-1.2, 1.0])
        problems = [(x0_1, ([-np.inf, -1.5], np.inf)), (x0_2, ([-np.inf, 1.5], np.inf)), (x0_3, ([-np.inf, 1.5], np.inf)), (x0_4, ([-np.inf, 1.5], [1.0, np.inf])), (x0_2, ([1.0, 1.5], [3.0, 3.0])), (x0_5, ([-50.0, 0.0], [0.5, 100]))]
        for (x0, bounds) in problems:
            for (jac, x_scale, tr_solver) in product(['2-point', '3-point', 'cs', jac_rosenbrock], [1.0, [1.0, 0.5], 'jac'], ['exact', 'lsmr']):
                res = least_squares(fun_rosenbrock, x0, jac, bounds, x_scale=x_scale, tr_solver=tr_solver, method=self.method)
                assert_allclose(res.optimality, 0.0, atol=1e-05)

class SparseMixin:

    def test_exact_tr_solver(self):
        if False:
            while True:
                i = 10
        p = BroydenTridiagonal()
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac, tr_solver='exact', method=self.method)
        assert_raises(ValueError, least_squares, p.fun, p.x0, tr_solver='exact', jac_sparsity=p.sparsity, method=self.method)

    def test_equivalence(self):
        if False:
            for i in range(10):
                print('nop')
        sparse = BroydenTridiagonal(mode='sparse')
        dense = BroydenTridiagonal(mode='dense')
        res_sparse = least_squares(sparse.fun, sparse.x0, jac=sparse.jac, method=self.method)
        res_dense = least_squares(dense.fun, dense.x0, jac=sparse.jac, method=self.method)
        assert_equal(res_sparse.nfev, res_dense.nfev)
        assert_allclose(res_sparse.x, res_dense.x, atol=1e-20)
        assert_allclose(res_sparse.cost, 0, atol=1e-20)
        assert_allclose(res_dense.cost, 0, atol=1e-20)

    def test_tr_options(self):
        if False:
            print('Hello World!')
        p = BroydenTridiagonal()
        res = least_squares(p.fun, p.x0, p.jac, method=self.method, tr_options={'btol': 1e-10})
        assert_allclose(res.cost, 0, atol=1e-20)

    def test_wrong_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        p = BroydenTridiagonal()
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac, tr_solver='best', method=self.method)
        assert_raises(TypeError, least_squares, p.fun, p.x0, p.jac, tr_solver='lsmr', tr_options={'tol': 1e-10})

    def test_solver_selection(self):
        if False:
            i = 10
            return i + 15
        sparse = BroydenTridiagonal(mode='sparse')
        dense = BroydenTridiagonal(mode='dense')
        res_sparse = least_squares(sparse.fun, sparse.x0, jac=sparse.jac, method=self.method)
        res_dense = least_squares(dense.fun, dense.x0, jac=dense.jac, method=self.method)
        assert_allclose(res_sparse.cost, 0, atol=1e-20)
        assert_allclose(res_dense.cost, 0, atol=1e-20)
        assert_(issparse(res_sparse.jac))
        assert_(isinstance(res_dense.jac, np.ndarray))

    def test_numerical_jac(self):
        if False:
            for i in range(10):
                print('nop')
        p = BroydenTridiagonal()
        for jac in ['2-point', '3-point', 'cs']:
            res_dense = least_squares(p.fun, p.x0, jac, method=self.method)
            res_sparse = least_squares(p.fun, p.x0, jac, method=self.method, jac_sparsity=p.sparsity)
            assert_equal(res_dense.nfev, res_sparse.nfev)
            assert_allclose(res_dense.x, res_sparse.x, atol=1e-20)
            assert_allclose(res_dense.cost, 0, atol=1e-20)
            assert_allclose(res_sparse.cost, 0, atol=1e-20)

    def test_with_bounds(self):
        if False:
            while True:
                i = 10
        p = BroydenTridiagonal()
        for (jac, jac_sparsity) in product([p.jac, '2-point', '3-point', 'cs'], [None, p.sparsity]):
            res_1 = least_squares(p.fun, p.x0, jac, bounds=(p.lb, np.inf), method=self.method, jac_sparsity=jac_sparsity)
            res_2 = least_squares(p.fun, p.x0, jac, bounds=(-np.inf, p.ub), method=self.method, jac_sparsity=jac_sparsity)
            res_3 = least_squares(p.fun, p.x0, jac, bounds=(p.lb, p.ub), method=self.method, jac_sparsity=jac_sparsity)
            assert_allclose(res_1.optimality, 0, atol=1e-10)
            assert_allclose(res_2.optimality, 0, atol=1e-10)
            assert_allclose(res_3.optimality, 0, atol=1e-10)

    def test_wrong_jac_sparsity(self):
        if False:
            while True:
                i = 10
        p = BroydenTridiagonal()
        sparsity = p.sparsity[:-1]
        assert_raises(ValueError, least_squares, p.fun, p.x0, jac_sparsity=sparsity, method=self.method)

    def test_linear_operator(self):
        if False:
            i = 10
            return i + 15
        p = BroydenTridiagonal(mode='operator')
        res = least_squares(p.fun, p.x0, p.jac, method=self.method)
        assert_allclose(res.cost, 0.0, atol=1e-20)
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac, method=self.method, tr_solver='exact')

    def test_x_scale_jac_scale(self):
        if False:
            print('Hello World!')
        p = BroydenTridiagonal()
        res = least_squares(p.fun, p.x0, p.jac, method=self.method, x_scale='jac')
        assert_allclose(res.cost, 0.0, atol=1e-20)
        p = BroydenTridiagonal(mode='operator')
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac, method=self.method, x_scale='jac')

class LossFunctionMixin:

    def test_options(self):
        if False:
            i = 10
            return i + 15
        for loss in LOSSES:
            res = least_squares(fun_trivial, 2.0, loss=loss, method=self.method)
            assert_allclose(res.x, 0, atol=1e-15)
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, loss='hinge', method=self.method)

    def test_fun(self):
        if False:
            i = 10
            return i + 15
        for loss in LOSSES:
            res = least_squares(fun_trivial, 2.0, loss=loss, method=self.method)
            assert_equal(res.fun, fun_trivial(res.x))

    def test_grad(self):
        if False:
            while True:
                i = 10
        x = np.array([2.0])
        res = least_squares(fun_trivial, x, jac_trivial, loss='linear', max_nfev=1, method=self.method)
        assert_equal(res.grad, 2 * x * (x ** 2 + 5))
        res = least_squares(fun_trivial, x, jac_trivial, loss='huber', max_nfev=1, method=self.method)
        assert_equal(res.grad, 2 * x)
        res = least_squares(fun_trivial, x, jac_trivial, loss='soft_l1', max_nfev=1, method=self.method)
        assert_allclose(res.grad, 2 * x * (x ** 2 + 5) / (1 + (x ** 2 + 5) ** 2) ** 0.5)
        res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy', max_nfev=1, method=self.method)
        assert_allclose(res.grad, 2 * x * (x ** 2 + 5) / (1 + (x ** 2 + 5) ** 2))
        res = least_squares(fun_trivial, x, jac_trivial, loss='arctan', max_nfev=1, method=self.method)
        assert_allclose(res.grad, 2 * x * (x ** 2 + 5) / (1 + (x ** 2 + 5) ** 4))
        res = least_squares(fun_trivial, x, jac_trivial, loss=cubic_soft_l1, max_nfev=1, method=self.method)
        assert_allclose(res.grad, 2 * x * (x ** 2 + 5) / (1 + (x ** 2 + 5) ** 2) ** (2 / 3))

    def test_jac(self):
        if False:
            i = 10
            return i + 15
        x = 2.0
        f = x ** 2 + 5
        res = least_squares(fun_trivial, x, jac_trivial, loss='linear', max_nfev=1, method=self.method)
        assert_equal(res.jac, 2 * x)
        res = least_squares(fun_trivial, x, jac_trivial, loss='huber', max_nfev=1, method=self.method)
        assert_equal(res.jac, 2 * x * EPS ** 0.5)
        res = least_squares(fun_trivial, x, jac_trivial, loss='huber', f_scale=10, max_nfev=1)
        assert_equal(res.jac, 2 * x)
        res = least_squares(fun_trivial, x, jac_trivial, loss='soft_l1', max_nfev=1, method=self.method)
        assert_allclose(res.jac, 2 * x * (1 + f ** 2) ** (-0.75))
        res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy', max_nfev=1, method=self.method)
        assert_allclose(res.jac, 2 * x * EPS ** 0.5)
        res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy', f_scale=10, max_nfev=1, method=self.method)
        fs = f / 10
        assert_allclose(res.jac, 2 * x * (1 - fs ** 2) ** 0.5 / (1 + fs ** 2))
        res = least_squares(fun_trivial, x, jac_trivial, loss='arctan', max_nfev=1, method=self.method)
        assert_allclose(res.jac, 2 * x * EPS ** 0.5)
        res = least_squares(fun_trivial, x, jac_trivial, loss='arctan', f_scale=20.0, max_nfev=1, method=self.method)
        fs = f / 20
        assert_allclose(res.jac, 2 * x * (1 - 3 * fs ** 4) ** 0.5 / (1 + fs ** 4))
        res = least_squares(fun_trivial, x, jac_trivial, loss=cubic_soft_l1, max_nfev=1)
        assert_allclose(res.jac, 2 * x * EPS ** 0.5)
        res = least_squares(fun_trivial, x, jac_trivial, loss=cubic_soft_l1, f_scale=6, max_nfev=1)
        fs = f / 6
        assert_allclose(res.jac, 2 * x * (1 - fs ** 2 / 3) ** 0.5 * (1 + fs ** 2) ** (-5 / 6))

    def test_robustness(self):
        if False:
            return 10
        for noise in [0.1, 1.0]:
            p = ExponentialFittingProblem(1, 0.1, noise, random_seed=0)
            for jac in ['2-point', '3-point', 'cs', p.jac]:
                res_lsq = least_squares(p.fun, p.p0, jac=jac, method=self.method)
                assert_allclose(res_lsq.optimality, 0, atol=0.01)
                for loss in LOSSES:
                    if loss == 'linear':
                        continue
                    res_robust = least_squares(p.fun, p.p0, jac=jac, loss=loss, f_scale=noise, method=self.method)
                    assert_allclose(res_robust.optimality, 0, atol=0.01)
                    assert_(norm(res_robust.x - p.p_opt) < norm(res_lsq.x - p.p_opt))

class TestDogbox(BaseMixin, BoundsMixin, SparseMixin, LossFunctionMixin):
    method = 'dogbox'

class TestTRF(BaseMixin, BoundsMixin, SparseMixin, LossFunctionMixin):
    method = 'trf'

    def test_lsmr_regularization(self):
        if False:
            while True:
                i = 10
        p = BroydenTridiagonal()
        for regularize in [True, False]:
            res = least_squares(p.fun, p.x0, p.jac, method='trf', tr_options={'regularize': regularize})
            assert_allclose(res.cost, 0, atol=1e-20)

class TestLM(BaseMixin):
    method = 'lm'

    def test_bounds_not_supported(self):
        if False:
            i = 10
            return i + 15
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, bounds=(-3.0, 3.0), method='lm')

    def test_m_less_n_not_supported(self):
        if False:
            i = 10
            return i + 15
        x0 = [-2, 1]
        assert_raises(ValueError, least_squares, fun_rosenbrock_cropped, x0, method='lm')

    def test_sparse_not_supported(self):
        if False:
            print('Hello World!')
        p = BroydenTridiagonal()
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac, method='lm')

    def test_jac_sparsity_not_supported(self):
        if False:
            while True:
                i = 10
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, jac_sparsity=[1], method='lm')

    def test_LinearOperator_not_supported(self):
        if False:
            for i in range(10):
                print('nop')
        p = BroydenTridiagonal(mode='operator')
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac, method='lm')

    def test_loss(self):
        if False:
            while True:
                i = 10
        res = least_squares(fun_trivial, 2.0, loss='linear', method='lm')
        assert_allclose(res.x, 0.0, atol=0.0001)
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, method='lm', loss='huber')

def test_basic():
    if False:
        print('Hello World!')
    res = least_squares(fun_trivial, 2.0)
    assert_allclose(res.x, 0, atol=1e-10)

def test_small_tolerances_for_lm():
    if False:
        for i in range(10):
            print('nop')
    for (ftol, xtol, gtol) in [(None, 1e-13, 1e-13), (1e-13, None, 1e-13), (1e-13, 1e-13, None)]:
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, xtol=xtol, ftol=ftol, gtol=gtol, method='lm')

def test_fp32_gh12991():
    if False:
        return 10
    np.random.seed(1)
    x = np.linspace(0, 1, 100).astype('float32')
    y = np.random.random(100).astype('float32')

    def func(p, x):
        if False:
            i = 10
            return i + 15
        return p[0] + p[1] * x

    def err(p, x, y):
        if False:
            while True:
                i = 10
        return func(p, x) - y
    res = least_squares(err, [-1.0, -1.0], args=(x, y))
    assert res.nfev > 2
    assert_allclose(res.x, np.array([0.4082241, 0.15530563]), atol=5e-05)

def test_gh_18793_and_19351():
    if False:
        return 10
    answer = 1e-12
    initial_guess = 1.1e-12

    def chi2(x):
        if False:
            i = 10
            return i + 15
        return (x - answer) ** 2
    gtol = 1e-15
    res = least_squares(chi2, x0=initial_guess, gtol=1e-15, bounds=(0, np.inf))
    (scaling, _) = CL_scaling_vector(res.x, res.grad, np.atleast_1d(0), np.atleast_1d(np.inf))
    assert res.status == 1
    assert np.linalg.norm(res.grad * scaling, ord=np.inf) < gtol

def test_gh_19103():
    if False:
        return 10
    ydata = np.array([0.0] * 66 + [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0, 1.0, 6.0, 5.0, 0.0, 0.0, 2.0, 8.0, 4.0, 4.0, 6.0, 9.0, 7.0, 2.0, 7.0, 8.0, 2.0, 13.0, 9.0, 8.0, 11.0, 10.0, 13.0, 14.0, 19.0, 11.0, 15.0, 18.0, 26.0, 19.0, 32.0, 29.0, 28.0, 36.0, 32.0, 35.0, 36.0, 43.0, 52.0, 32.0, 58.0, 56.0, 52.0, 67.0, 53.0, 72.0, 88.0, 77.0, 95.0, 94.0, 84.0, 86.0, 101.0, 107.0, 108.0, 118.0, 96.0, 115.0, 138.0, 137.0])
    xdata = np.arange(0, ydata.size) * 0.1

    def exponential_wrapped(params):
        if False:
            for i in range(10):
                print('nop')
        (A, B, x0) = params
        return A * np.exp(B * (xdata - x0)) - ydata
    x0 = [0.01, 1.0, 5.0]
    bounds = ((0.01, 0, 0), (np.inf, 10, 20.9))
    res = least_squares(exponential_wrapped, x0, method='trf', bounds=bounds)
    assert res.success