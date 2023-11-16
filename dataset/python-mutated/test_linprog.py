"""
Unit test for Linear Programming
"""
import sys
import platform
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal, assert_array_less, assert_warns, suppress_warnings
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
has_umfpack = True
try:
    from scikits.umfpack import UmfpackWarning
except ImportError:
    has_umfpack = False
has_cholmod = True
try:
    import sksparse
    from sksparse.cholmod import cholesky as cholmod
except ImportError:
    has_cholmod = False

def _assert_iteration_limit_reached(res, maxiter):
    if False:
        while True:
            i = 10
    assert_(not res.success, 'Incorrectly reported success')
    assert_(res.success < maxiter, 'Incorrectly reported number of iterations')
    assert_equal(res.status, 1, 'Failed to report iteration limit reached')

def _assert_infeasible(res):
    if False:
        i = 10
        return i + 15
    assert_(not res.success, 'incorrectly reported success')
    assert_equal(res.status, 2, 'failed to report infeasible status')

def _assert_unbounded(res):
    if False:
        i = 10
        return i + 15
    assert_(not res.success, 'incorrectly reported success')
    assert_equal(res.status, 3, 'failed to report unbounded status')

def _assert_unable_to_find_basic_feasible_sol(res):
    if False:
        i = 10
        return i + 15
    assert_(not res.success, 'incorrectly reported success')
    assert_(res.status in (2, 4), 'failed to report optimization failure')

def _assert_success(res, desired_fun=None, desired_x=None, rtol=1e-08, atol=1e-08):
    if False:
        i = 10
        return i + 15
    if not res.success:
        msg = 'linprog status {}, message: {}'.format(res.status, res.message)
        raise AssertionError(msg)
    assert_equal(res.status, 0)
    if desired_fun is not None:
        assert_allclose(res.fun, desired_fun, err_msg='converged to an unexpected objective value', rtol=rtol, atol=atol)
    if desired_x is not None:
        assert_allclose(res.x, desired_x, err_msg='converged to an unexpected solution', rtol=rtol, atol=atol)

def magic_square(n):
    if False:
        i = 10
        return i + 15
    '\n    Generates a linear program for which integer solutions represent an\n    n x n magic square; binary decision variables represent the presence\n    (or absence) of an integer 1 to n^2 in each position of the square.\n    '
    np.random.seed(0)
    M = n * (n ** 2 + 1) / 2
    numbers = np.arange(n ** 4) // n ** 2 + 1
    numbers = numbers.reshape(n ** 2, n, n)
    zeros = np.zeros((n ** 2, n, n))
    A_list = []
    b_list = []
    for i in range(n ** 2):
        A_row = zeros.copy()
        A_row[i, :, :] = 1
        A_list.append(A_row.flatten())
        b_list.append(1)
    for i in range(n):
        for j in range(n):
            A_row = zeros.copy()
            A_row[:, i, j] = 1
            A_list.append(A_row.flatten())
            b_list.append(1)
    for i in range(n):
        A_row = zeros.copy()
        A_row[:, i, :] = numbers[:, i, :]
        A_list.append(A_row.flatten())
        b_list.append(M)
    for i in range(n):
        A_row = zeros.copy()
        A_row[:, :, i] = numbers[:, :, i]
        A_list.append(A_row.flatten())
        b_list.append(M)
    A_row = zeros.copy()
    A_row[:, range(n), range(n)] = numbers[:, range(n), range(n)]
    A_list.append(A_row.flatten())
    b_list.append(M)
    A_row = zeros.copy()
    A_row[:, range(n), range(-1, -n - 1, -1)] = numbers[:, range(n), range(-1, -n - 1, -1)]
    A_list.append(A_row.flatten())
    b_list.append(M)
    A = np.array(np.vstack(A_list), dtype=float)
    b = np.array(b_list, dtype=float)
    c = np.random.rand(A.shape[1])
    return (A, b, c, numbers, M)

def lpgen_2d(m, n):
    if False:
        i = 10
        return i + 15
    ' -> A b c LP test: m*n vars, m+n constraints\n        row sums == n/m, col sums == 1\n        https://gist.github.com/denis-bz/8647461\n    '
    np.random.seed(0)
    c = -np.random.exponential(size=(m, n))
    Arow = np.zeros((m, m * n))
    brow = np.zeros(m)
    for j in range(m):
        j1 = j + 1
        Arow[j, j * n:j1 * n] = 1
        brow[j] = n / m
    Acol = np.zeros((n, m * n))
    bcol = np.zeros(n)
    for j in range(n):
        j1 = j + 1
        Acol[j, j::n] = 1
        bcol[j] = 1
    A = np.vstack((Arow, Acol))
    b = np.hstack((brow, bcol))
    return (A, b, c.ravel())

def very_random_gen(seed=0):
    if False:
        print('Hello World!')
    np.random.seed(seed)
    (m_eq, m_ub, n) = (10, 20, 50)
    c = np.random.rand(n) - 0.5
    A_ub = np.random.rand(m_ub, n) - 0.5
    b_ub = np.random.rand(m_ub) - 0.5
    A_eq = np.random.rand(m_eq, n) - 0.5
    b_eq = np.random.rand(m_eq) - 0.5
    lb = -np.random.rand(n)
    ub = np.random.rand(n)
    lb[lb < -np.random.rand()] = -np.inf
    ub[ub > np.random.rand()] = np.inf
    bounds = np.vstack((lb, ub)).T
    return (c, A_ub, b_ub, A_eq, b_eq, bounds)

def nontrivial_problem():
    if False:
        i = 10
        return i + 15
    c = [-1, 8, 4, -6]
    A_ub = [[-7, -7, 6, 9], [1, -1, -3, 0], [10, -10, -7, 7], [6, -1, 3, 4]]
    b_ub = [-3, 6, -6, 6]
    A_eq = [[-10, 1, 1, -8]]
    b_eq = [-4]
    x_star = [101 / 1391, 1462 / 1391, 0, 752 / 1391]
    f_star = 7083 / 1391
    return (c, A_ub, b_ub, A_eq, b_eq, x_star, f_star)

def l1_regression_prob(seed=0, m=8, d=9, n=100):
    if False:
        for i in range(10):
            print('nop')
    '\n    Training data is {(x0, y0), (x1, y2), ..., (xn-1, yn-1)}\n        x in R^d\n        y in R\n    n: number of training samples\n    d: dimension of x, i.e. x in R^d\n    phi: feature map R^d -> R^m\n    m: dimension of feature space\n    '
    np.random.seed(seed)
    phi = np.random.normal(0, 1, size=(m, d))
    w_true = np.random.randn(m)
    x = np.random.normal(0, 1, size=(d, n))
    y = w_true @ (phi @ x) + np.random.normal(0, 1e-05, size=n)
    c = np.ones(m + n)
    c[:m] = 0
    A_ub = scipy.sparse.lil_matrix((2 * n, n + m))
    idx = 0
    for ii in range(n):
        A_ub[idx, :m] = phi @ x[:, ii]
        A_ub[idx, m + ii] = -1
        A_ub[idx + 1, :m] = -1 * phi @ x[:, ii]
        A_ub[idx + 1, m + ii] = -1
        idx += 2
    A_ub = A_ub.tocsc()
    b_ub = np.zeros(2 * n)
    b_ub[0::2] = y
    b_ub[1::2] = -y
    bnds = [(None, None)] * m + [(0, None)] * n
    return (c, A_ub, b_ub, bnds)

def generic_callback_test(self):
    if False:
        print('Hello World!')
    last_cb = {}

    def cb(res):
        if False:
            for i in range(10):
                print('nop')
        message = res.pop('message')
        complete = res.pop('complete')
        assert_(res.pop('phase') in (1, 2))
        assert_(res.pop('status') in range(4))
        assert_(isinstance(res.pop('nit'), int))
        assert_(isinstance(complete, bool))
        assert_(isinstance(message, str))
        last_cb['x'] = res['x']
        last_cb['fun'] = res['fun']
        last_cb['slack'] = res['slack']
        last_cb['con'] = res['con']
    c = np.array([-3, -2])
    A_ub = [[2, 1], [1, 1], [1, 0]]
    b_ub = [10, 8, 4]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, callback=cb, method=self.method)
    _assert_success(res, desired_fun=-18.0, desired_x=[2, 6])
    assert_allclose(last_cb['fun'], res['fun'])
    assert_allclose(last_cb['x'], res['x'])
    assert_allclose(last_cb['con'], res['con'])
    assert_allclose(last_cb['slack'], res['slack'])

def test_unknown_solvers_and_options():
    if False:
        i = 10
        return i + 15
    c = np.array([-3, -2])
    A_ub = [[2, 1], [1, 1], [1, 0]]
    b_ub = [10, 8, 4]
    assert_raises(ValueError, linprog, c, A_ub=A_ub, b_ub=b_ub, method='ekki-ekki-ekki')
    assert_raises(ValueError, linprog, c, A_ub=A_ub, b_ub=b_ub, method='highs-ekki')
    message = "Unrecognized options detected: {'rr_method': 'ekki-ekki-ekki'}"
    with pytest.warns(OptimizeWarning, match=message):
        linprog(c, A_ub=A_ub, b_ub=b_ub, options={'rr_method': 'ekki-ekki-ekki'})

def test_choose_solver():
    if False:
        print('Hello World!')
    c = np.array([-3, -2])
    A_ub = [[2, 1], [1, 1], [1, 0]]
    b_ub = [10, 8, 4]
    res = linprog(c, A_ub, b_ub, method='highs')
    _assert_success(res, desired_fun=-18.0, desired_x=[2, 6])

def test_deprecation():
    if False:
        while True:
            i = 10
    with pytest.warns(DeprecationWarning):
        linprog(1, method='interior-point')
    with pytest.warns(DeprecationWarning):
        linprog(1, method='revised simplex')
    with pytest.warns(DeprecationWarning):
        linprog(1, method='simplex')

def test_highs_status_message():
    if False:
        return 10
    res = linprog(1, method='highs')
    msg = 'Optimization terminated successfully. (HiGHS Status 7:'
    assert res.status == 0
    assert res.message.startswith(msg)
    (A, b, c, numbers, M) = magic_square(6)
    bounds = [(0, 1)] * len(c)
    integrality = [1] * len(c)
    options = {'time_limit': 0.1}
    res = linprog(c=c, A_eq=A, b_eq=b, bounds=bounds, method='highs', options=options, integrality=integrality)
    msg = 'Time limit reached. (HiGHS Status 13:'
    assert res.status == 1
    assert res.message.startswith(msg)
    options = {'maxiter': 10}
    res = linprog(c=c, A_eq=A, b_eq=b, bounds=bounds, method='highs-ds', options=options)
    msg = 'Iteration limit reached. (HiGHS Status 14:'
    assert res.status == 1
    assert res.message.startswith(msg)
    res = linprog(1, bounds=(1, -1), method='highs')
    msg = 'The problem is infeasible. (HiGHS Status 8:'
    assert res.status == 2
    assert res.message.startswith(msg)
    res = linprog(-1, method='highs')
    msg = 'The problem is unbounded. (HiGHS Status 10:'
    assert res.status == 3
    assert res.message.startswith(msg)
    from scipy.optimize._linprog_highs import _highs_to_scipy_status_message
    (status, message) = _highs_to_scipy_status_message(58, 'Hello!')
    msg = 'The HiGHS status code was not recognized. (HiGHS Status 58:'
    assert status == 4
    assert message.startswith(msg)
    (status, message) = _highs_to_scipy_status_message(None, None)
    msg = 'HiGHS did not provide a status code. (HiGHS Status None: None)'
    assert status == 4
    assert message.startswith(msg)

def test_bug_17380():
    if False:
        return 10
    linprog([1, 1], A_ub=[[-1, 0]], b_ub=[-2.5], integrality=[1, 1])
A_ub = None
b_ub = None
A_eq = None
b_eq = None
bounds = None

class LinprogCommonTests:
    """
    Base class for `linprog` tests. Generally, each test will be performed
    once for every derived class of LinprogCommonTests, each of which will
    typically change self.options and/or self.method. Effectively, these tests
    are run for many combination of method (simplex, revised simplex, and
    interior point) and options (such as pivoting rule or sparse treatment).
    """

    def test_callback(self):
        if False:
            i = 10
            return i + 15
        generic_callback_test(self)

    def test_disp(self):
        if False:
            for i in range(10):
                print('nop')
        (A, b, c) = lpgen_2d(20, 20)
        res = linprog(c, A_ub=A, b_ub=b, method=self.method, options={'disp': True})
        _assert_success(res, desired_fun=-64.049494229)

    def test_docstring_example(self):
        if False:
            i = 10
            return i + 15
        c = [-1, 4]
        A = [[-3, 1], [1, 2]]
        b = [6, 4]
        x0_bounds = (None, None)
        x1_bounds = (-3, None)
        res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds), options=self.options, method=self.method)
        _assert_success(res, desired_fun=-22)

    def test_type_error(self):
        if False:
            for i in range(10):
                print('nop')
        c = [1]
        A_eq = [[1]]
        b_eq = 'hello'
        assert_raises(TypeError, linprog, c, A_eq=A_eq, b_eq=b_eq, method=self.method, options=self.options)

    def test_aliasing_b_ub(self):
        if False:
            return 10
        c = np.array([1.0])
        A_ub = np.array([[1.0]])
        b_ub_orig = np.array([3.0])
        b_ub = b_ub_orig.copy()
        bounds = (-4.0, np.inf)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=-4, desired_x=[-4])
        assert_allclose(b_ub_orig, b_ub)

    def test_aliasing_b_eq(self):
        if False:
            return 10
        c = np.array([1.0])
        A_eq = np.array([[1.0]])
        b_eq_orig = np.array([3.0])
        b_eq = b_eq_orig.copy()
        bounds = (-4.0, np.inf)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=3, desired_x=[3])
        assert_allclose(b_eq_orig, b_eq)

    def test_non_ndarray_args(self):
        if False:
            i = 10
            return i + 15
        c = [1.0]
        A_ub = [[1.0]]
        b_ub = [3.0]
        A_eq = [[1.0]]
        b_eq = [2.0]
        bounds = (-1.0, 10.0)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=2, desired_x=[2])

    def test_unknown_options(self):
        if False:
            i = 10
            return i + 15
        c = np.array([-3, -2])
        A_ub = [[2, 1], [1, 1], [1, 0]]
        b_ub = [10, 8, 4]

        def f(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, options={}):
            if False:
                i = 10
                return i + 15
            linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=options)
        o = {key: self.options[key] for key in self.options}
        o['spam'] = 42
        assert_warns(OptimizeWarning, f, c, A_ub=A_ub, b_ub=b_ub, options=o)

    def test_integrality_without_highs(self):
        if False:
            print('Hello World!')
        A_ub = np.array([[-1, 1], [3, 2], [2, 3]])
        b_ub = np.array([1, 12, 12])
        c = -np.array([0, 1])
        bounds = [(0, np.inf)] * len(c)
        integrality = [1] * len(c)
        with np.testing.assert_warns(OptimizeWarning):
            res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=self.method, integrality=integrality)
        np.testing.assert_allclose(res.x, [1.8, 2.8])
        np.testing.assert_allclose(res.fun, -2.8)

    def test_invalid_inputs(self):
        if False:
            i = 10
            return i + 15

        def f(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
            if False:
                i = 10
                return i + 15
            linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        assert_raises(ValueError, f, [1, 2, 3], bounds=[(1, 2), (3, 4)])
        with np.testing.suppress_warnings() as sup:
            sup.filter(VisibleDeprecationWarning, 'Creating an ndarray from ragged')
            assert_raises(ValueError, f, [1, 2, 3], bounds=[(1, 2), (3, 4), (3, 4, 5)])
        assert_raises(ValueError, f, [1, 2, 3], bounds=[(1, -2), (1, 2)])
        assert_raises(ValueError, f, [1, 2], A_ub=[[1, 2]], b_ub=[1, 2])
        assert_raises(ValueError, f, [1, 2], A_ub=[[1]], b_ub=[1])
        assert_raises(ValueError, f, [1, 2], A_eq=[[1, 2]], b_eq=[1, 2])
        assert_raises(ValueError, f, [1, 2], A_eq=[[1]], b_eq=[1])
        assert_raises(ValueError, f, [1, 2], A_eq=[1], b_eq=1)
        if '_sparse_presolve' in self.options and self.options['_sparse_presolve']:
            return
        assert_raises(ValueError, f, [1, 2], A_ub=np.zeros((1, 1, 3)), b_eq=1)

    def test_sparse_constraints(self):
        if False:
            return 10

        def f(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
            if False:
                i = 10
                return i + 15
            linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        np.random.seed(0)
        m = 100
        n = 150
        A_eq = scipy.sparse.rand(m, n, 0.5)
        x_valid = np.random.randn(n)
        c = np.random.randn(n)
        ub = x_valid + np.random.rand(n)
        lb = x_valid - np.random.rand(n)
        bounds = np.column_stack((lb, ub))
        b_eq = A_eq * x_valid
        if self.method in {'simplex', 'revised simplex'}:
            with assert_raises(ValueError, match=f"Method '{self.method}' does not support sparse constraint matrices."):
                linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
        else:
            options = {**self.options}
            if self.method in {'interior-point'}:
                options['sparse'] = True
            res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=options)
            assert res.success

    def test_maxiter(self):
        if False:
            return 10
        c = [4, 8, 3, 0, 0, 0]
        A = [[2, 5, 3, -1, 0, 0], [3, 2.5, 8, 0, -1, 0], [8, 10, 4, 0, 0, -1]]
        b = [185, 155, 600]
        np.random.seed(0)
        maxiter = 3
        res = linprog(c, A_eq=A, b_eq=b, method=self.method, options={'maxiter': maxiter})
        _assert_iteration_limit_reached(res, maxiter)
        assert_equal(res.nit, maxiter)

    def test_bounds_fixed(self):
        if False:
            i = 10
            return i + 15
        do_presolve = self.options.get('presolve', True)
        res = linprog([1], bounds=(1, 1), method=self.method, options=self.options)
        _assert_success(res, 1, 1)
        if do_presolve:
            assert_equal(res.nit, 0)
        res = linprog([1, 2, 3], bounds=[(5, 5), (-1, -1), (3, 3)], method=self.method, options=self.options)
        _assert_success(res, 12, [5, -1, 3])
        if do_presolve:
            assert_equal(res.nit, 0)
        res = linprog([1, 1], bounds=[(1, 1), (1, 3)], method=self.method, options=self.options)
        _assert_success(res, 2, [1, 1])
        if do_presolve:
            assert_equal(res.nit, 0)
        res = linprog([1, 1, 2], A_eq=[[1, 0, 0], [0, 1, 0]], b_eq=[1, 7], bounds=[(-5, 5), (0, 10), (3.5, 3.5)], method=self.method, options=self.options)
        _assert_success(res, 15, [1, 7, 3.5])
        if do_presolve:
            assert_equal(res.nit, 0)

    def test_bounds_infeasible(self):
        if False:
            return 10
        do_presolve = self.options.get('presolve', True)
        res = linprog([1], bounds=(1, -2), method=self.method, options=self.options)
        _assert_infeasible(res)
        if do_presolve:
            assert_equal(res.nit, 0)
        res = linprog([1], bounds=[(1, -2)], method=self.method, options=self.options)
        _assert_infeasible(res)
        if do_presolve:
            assert_equal(res.nit, 0)
        res = linprog([1, 2, 3], bounds=[(5, 0), (1, 2), (3, 4)], method=self.method, options=self.options)
        _assert_infeasible(res)
        if do_presolve:
            assert_equal(res.nit, 0)

    def test_bounds_infeasible_2(self):
        if False:
            print('Hello World!')
        do_presolve = self.options.get('presolve', True)
        simplex_without_presolve = not do_presolve and self.method == 'simplex'
        c = [1, 2, 3]
        bounds_1 = [(1, 2), (np.inf, np.inf), (3, 4)]
        bounds_2 = [(1, 2), (-np.inf, -np.inf), (3, 4)]
        if simplex_without_presolve:

            def g(c, bounds):
                if False:
                    for i in range(10):
                        print('nop')
                res = linprog(c, bounds=bounds, method=self.method, options=self.options)
                return res
            with pytest.warns(RuntimeWarning):
                with pytest.raises(IndexError):
                    g(c, bounds=bounds_1)
            with pytest.warns(RuntimeWarning):
                with pytest.raises(IndexError):
                    g(c, bounds=bounds_2)
        else:
            res = linprog(c=c, bounds=bounds_1, method=self.method, options=self.options)
            _assert_infeasible(res)
            if do_presolve:
                assert_equal(res.nit, 0)
            res = linprog(c=c, bounds=bounds_2, method=self.method, options=self.options)
            _assert_infeasible(res)
            if do_presolve:
                assert_equal(res.nit, 0)

    def test_empty_constraint_1(self):
        if False:
            return 10
        c = [-1, -2]
        res = linprog(c, method=self.method, options=self.options)
        _assert_unbounded(res)

    def test_empty_constraint_2(self):
        if False:
            for i in range(10):
                print('nop')
        c = [-1, 1, -1, 1]
        bounds = [(0, np.inf), (-np.inf, 0), (-1, 1), (-1, 1)]
        res = linprog(c, bounds=bounds, method=self.method, options=self.options)
        _assert_unbounded(res)
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_empty_constraint_3(self):
        if False:
            i = 10
            return i + 15
        c = [1, -1, 1, -1]
        bounds = [(0, np.inf), (-np.inf, 0), (-1, 1), (-1, 1)]
        res = linprog(c, bounds=bounds, method=self.method, options=self.options)
        _assert_success(res, desired_x=[0, 0, -1, 1], desired_fun=-2)

    def test_inequality_constraints(self):
        if False:
            for i in range(10):
                print('nop')
        c = np.array([3, 2]) * -1
        A_ub = [[2, 1], [1, 1], [1, 0]]
        b_ub = [10, 8, 4]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=-18, desired_x=[2, 6])

    def test_inequality_constraints2(self):
        if False:
            i = 10
            return i + 15
        c = [6, 3]
        A_ub = [[0, 3], [-1, -1], [-2, 1]]
        b_ub = [2, -1, -1]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=5, desired_x=[2 / 3, 1 / 3])

    def test_bounds_simple(self):
        if False:
            for i in range(10):
                print('nop')
        c = [1, 2]
        bounds = (1, 2)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_x=[1, 1])
        bounds = [(1, 2), (1, 2)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_x=[1, 1])

    def test_bounded_below_only_1(self):
        if False:
            print('Hello World!')
        c = np.array([1.0])
        A_eq = np.array([[1.0]])
        b_eq = np.array([3.0])
        bounds = (1.0, None)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=3, desired_x=[3])

    def test_bounded_below_only_2(self):
        if False:
            for i in range(10):
                print('nop')
        c = np.ones(3)
        A_eq = np.eye(3)
        b_eq = np.array([1, 2, 3])
        bounds = (0.5, np.inf)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_x=b_eq, desired_fun=np.sum(b_eq))

    def test_bounded_above_only_1(self):
        if False:
            return 10
        c = np.array([1.0])
        A_eq = np.array([[1.0]])
        b_eq = np.array([3.0])
        bounds = (None, 10.0)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=3, desired_x=[3])

    def test_bounded_above_only_2(self):
        if False:
            for i in range(10):
                print('nop')
        c = np.ones(3)
        A_eq = np.eye(3)
        b_eq = np.array([1, 2, 3])
        bounds = (-np.inf, 4)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_x=b_eq, desired_fun=np.sum(b_eq))

    def test_bounds_infinity(self):
        if False:
            i = 10
            return i + 15
        c = np.ones(3)
        A_eq = np.eye(3)
        b_eq = np.array([1, 2, 3])
        bounds = (-np.inf, np.inf)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_x=b_eq, desired_fun=np.sum(b_eq))

    def test_bounds_mixed(self):
        if False:
            return 10
        c = np.array([-1, 4]) * -1
        A_ub = np.array([[-3, 1], [1, 2]], dtype=np.float64)
        b_ub = [6, 4]
        x0_bounds = (-np.inf, np.inf)
        x1_bounds = (-3, np.inf)
        bounds = (x0_bounds, x1_bounds)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=-80 / 7, desired_x=[-8 / 7, 18 / 7])

    def test_bounds_equal_but_infeasible(self):
        if False:
            for i in range(10):
                print('nop')
        c = [-4, 1]
        A_ub = [[7, -2], [0, 1], [2, -2]]
        b_ub = [14, 0, 3]
        bounds = [(2, 2), (0, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_infeasible(res)

    def test_bounds_equal_but_infeasible2(self):
        if False:
            print('Hello World!')
        c = [-4, 1]
        A_eq = [[7, -2], [0, 1], [2, -2]]
        b_eq = [14, 0, 3]
        bounds = [(2, 2), (0, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_infeasible(res)

    def test_bounds_equal_no_presolve(self):
        if False:
            return 10
        c = [1, 2]
        A_ub = [[1, 2], [1.1, 2.2]]
        b_ub = [4, 8]
        bounds = [(1, 2), (2, 2)]
        o = {key: self.options[key] for key in self.options}
        o['presolve'] = False
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=o)
        _assert_infeasible(res)

    def test_zero_column_1(self):
        if False:
            print('Hello World!')
        (m, n) = (3, 4)
        np.random.seed(0)
        c = np.random.rand(n)
        c[1] = 1
        A_eq = np.random.rand(m, n)
        A_eq[:, 1] = 0
        b_eq = np.random.rand(m)
        A_ub = [[1, 0, 1, 1]]
        b_ub = 3
        bounds = [(-10, 10), (-10, 10), (-10, None), (None, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=-9.70878367304134)

    def test_zero_column_2(self):
        if False:
            while True:
                i = 10
        if self.method in {'highs-ds', 'highs-ipm'}:
            pytest.xfail()
        np.random.seed(0)
        (m, n) = (2, 4)
        c = np.random.rand(n)
        c[1] = -1
        A_eq = np.random.rand(m, n)
        A_eq[:, 1] = 0
        b_eq = np.random.rand(m)
        A_ub = np.random.rand(m, n)
        A_ub[:, 1] = 0
        b_ub = np.random.rand(m)
        bounds = (None, None)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_unbounded(res)
        if self.options.get('presolve', True) and 'highs' not in self.method:
            assert_equal(res.nit, 0)

    def test_zero_row_1(self):
        if False:
            while True:
                i = 10
        c = [1, 2, 3]
        A_eq = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        b_eq = [0, 3, 0]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=3)

    def test_zero_row_2(self):
        if False:
            i = 10
            return i + 15
        A_ub = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        b_ub = [0, 3, 0]
        c = [1, 2, 3]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=0)

    def test_zero_row_3(self):
        if False:
            while True:
                i = 10
        (m, n) = (2, 4)
        c = np.random.rand(n)
        A_eq = np.random.rand(m, n)
        A_eq[0, :] = 0
        b_eq = np.random.rand(m)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_infeasible(res)
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_zero_row_4(self):
        if False:
            return 10
        (m, n) = (2, 4)
        c = np.random.rand(n)
        A_ub = np.random.rand(m, n)
        A_ub[0, :] = 0
        b_ub = -np.random.rand(m)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_infeasible(res)
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_singleton_row_eq_1(self):
        if False:
            return 10
        c = [1, 1, 1, 2]
        A_eq = [[1, 0, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]]
        b_eq = [1, 2, 2, 4]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_infeasible(res)
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_singleton_row_eq_2(self):
        if False:
            return 10
        c = [1, 1, 1, 2]
        A_eq = [[1, 0, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]]
        b_eq = [1, 2, 1, 4]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=4)

    def test_singleton_row_ub_1(self):
        if False:
            while True:
                i = 10
        c = [1, 1, 1, 2]
        A_ub = [[1, 0, 0, 0], [0, 2, 0, 0], [-1, 0, 0, 0], [1, 1, 1, 1]]
        b_ub = [1, 2, -2, 4]
        bounds = [(None, None), (0, None), (0, None), (0, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_infeasible(res)
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_singleton_row_ub_2(self):
        if False:
            print('Hello World!')
        c = [1, 1, 1, 2]
        A_ub = [[1, 0, 0, 0], [0, 2, 0, 0], [-1, 0, 0, 0], [1, 1, 1, 1]]
        b_ub = [1, 2, -0.5, 4]
        bounds = [(None, None), (0, None), (0, None), (0, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=0.5)

    def test_infeasible(self):
        if False:
            for i in range(10):
                print('nop')
        c = [-1, -1]
        A_ub = [[1, 0], [0, 1], [-1, -1]]
        b_ub = [2, 2, -5]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_infeasible(res)

    def test_infeasible_inequality_bounds(self):
        if False:
            print('Hello World!')
        c = [1]
        A_ub = [[2]]
        b_ub = 4
        bounds = (5, 6)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_infeasible(res)
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_unbounded(self):
        if False:
            i = 10
            return i + 15
        c = np.array([1, 1]) * -1
        A_ub = [[-1, 1], [-1, -1]]
        b_ub = [-1, -2]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_unbounded(res)

    def test_unbounded_below_no_presolve_corrected(self):
        if False:
            while True:
                i = 10
        c = [1]
        bounds = [(None, 1)]
        o = {key: self.options[key] for key in self.options}
        o['presolve'] = False
        res = linprog(c=c, bounds=bounds, method=self.method, options=o)
        if self.method == 'revised simplex':
            assert_equal(res.status, 5)
        else:
            _assert_unbounded(res)

    def test_unbounded_no_nontrivial_constraints_1(self):
        if False:
            while True:
                i = 10
        '\n        Test whether presolve pathway for detecting unboundedness after\n        constraint elimination is working.\n        '
        c = np.array([0, 0, 0, 1, -1, -1])
        A_ub = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1]])
        b_ub = np.array([2, -2, 0])
        bounds = [(None, None), (None, None), (None, None), (-1, 1), (-1, 1), (0, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_unbounded(res)
        if not self.method.lower().startswith('highs'):
            assert_equal(res.x[-1], np.inf)
            assert_equal(res.message[:36], 'The problem is (trivially) unbounded')

    def test_unbounded_no_nontrivial_constraints_2(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test whether presolve pathway for detecting unboundedness after\n        constraint elimination is working.\n        '
        c = np.array([0, 0, 0, 1, -1, 1])
        A_ub = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]])
        b_ub = np.array([2, -2, 0])
        bounds = [(None, None), (None, None), (None, None), (-1, 1), (-1, 1), (None, 0)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_unbounded(res)
        if not self.method.lower().startswith('highs'):
            assert_equal(res.x[-1], -np.inf)
            assert_equal(res.message[:36], 'The problem is (trivially) unbounded')

    def test_cyclic_recovery(self):
        if False:
            i = 10
            return i + 15
        c = np.array([100, 10, 1]) * -1
        A_ub = [[1, 0, 0], [20, 1, 0], [200, 20, 1]]
        b_ub = [1, 100, 10000]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_x=[0, 0, 10000], atol=5e-06, rtol=1e-07)

    def test_cyclic_bland(self):
        if False:
            return 10
        c = np.array([-10, 57, 9, 24.0])
        A_ub = np.array([[0.5, -5.5, -2.5, 9], [0.5, -1.5, -0.5, 1], [1, 0, 0, 0]])
        b_ub = [0, 0, 1]
        maxiter = 100
        o = {key: val for (key, val) in self.options.items()}
        o['maxiter'] = maxiter
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=o)
        if self.method == 'simplex' and (not self.options.get('bland')):
            _assert_iteration_limit_reached(res, o['maxiter'])
        else:
            _assert_success(res, desired_x=[1, 0, 1, 0])

    def test_remove_redundancy_infeasibility(self):
        if False:
            while True:
                i = 10
        (m, n) = (10, 10)
        c = np.random.rand(n)
        A_eq = np.random.rand(m, n)
        b_eq = np.random.rand(m)
        A_eq[-1, :] = 2 * A_eq[-2, :]
        b_eq[-1] *= -1
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, 'A_eq does not appear...')
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_infeasible(res)

    def test_nontrivial_problem(self):
        if False:
            print('Hello World!')
        (c, A_ub, b_ub, A_eq, b_eq, x_star, f_star) = nontrivial_problem()
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)

    def test_lpgen_problem(self):
        if False:
            return 10
        (A_ub, b_ub, c) = lpgen_2d(20, 20)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "Solving system with option 'sym_pos'")
            sup.filter(RuntimeWarning, 'invalid value encountered')
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=-64.049494229)

    def test_network_flow(self):
        if False:
            for i in range(10):
                print('nop')
        c = [2, 4, 9, 11, 4, 3, 8, 7, 0, 15, 16, 18]
        (n, p) = (-1, 1)
        A_eq = [[n, n, p, 0, p, 0, 0, 0, 0, p, 0, 0], [p, 0, 0, p, 0, p, 0, 0, 0, 0, 0, 0], [0, 0, n, n, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, p, p, 0, 0, p, 0], [0, 0, 0, 0, n, n, n, 0, p, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, n, n, 0, 0, p], [0, 0, 0, 0, 0, 0, 0, 0, 0, n, n, n]]
        b_eq = [0, 19, -16, 33, 0, 0, -36]
        with suppress_warnings() as sup:
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=755, atol=1e-06, rtol=1e-07)

    def test_network_flow_limited_capacity(self):
        if False:
            print('Hello World!')
        c = [2, 2, 1, 3, 1]
        bounds = [[0, 4], [0, 2], [0, 2], [0, 3], [0, 5]]
        (n, p) = (-1, 1)
        A_eq = [[n, n, 0, 0, 0], [p, 0, n, n, 0], [0, p, p, 0, n], [0, 0, 0, p, p]]
        b_eq = [-4, 0, 0, 4]
        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(RuntimeWarning, 'scipy.linalg.solve\nIll...')
            sup.filter(OptimizeWarning, 'A_eq does not appear...')
            sup.filter(OptimizeWarning, 'Solving system with option...')
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=14)

    def test_simplex_algorithm_wikipedia_example(self):
        if False:
            for i in range(10):
                print('nop')
        c = [-2, -3, -4]
        A_ub = [[3, 2, 1], [2, 5, 3]]
        b_ub = [10, 15]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=-20)

    def test_enzo_example(self):
        if False:
            return 10
        c = [4, 8, 3, 0, 0, 0]
        A_eq = [[2, 5, 3, -1, 0, 0], [3, 2.5, 8, 0, -1, 0], [8, 10, 4, 0, 0, -1]]
        b_eq = [185, 155, 600]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=317.5, desired_x=[66.25, 0, 17.5, 0, 183.75, 0], atol=6e-06, rtol=1e-07)

    def test_enzo_example_b(self):
        if False:
            i = 10
            return i + 15
        c = [2.8, 6.3, 10.8, -2.8, -6.3, -10.8]
        A_eq = [[-1, -1, -1, 0, 0, 0], [0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1]]
        b_eq = [-0.5, 0.4, 0.3, 0.3, 0.3]
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, 'A_eq does not appear...')
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=-1.77, desired_x=[0.3, 0.2, 0.0, 0.0, 0.1, 0.3])

    def test_enzo_example_c_with_degeneracy(self):
        if False:
            while True:
                i = 10
        m = 20
        c = -np.ones(m)
        tmp = 2 * np.pi * np.arange(1, m + 1) / (m + 1)
        A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))
        b_eq = [0, 0]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=0, desired_x=np.zeros(m))

    def test_enzo_example_c_with_unboundedness(self):
        if False:
            return 10
        m = 50
        c = -np.ones(m)
        tmp = 2 * np.pi * np.arange(m) / (m + 1)
        row0 = np.cos(tmp) - 1
        row0[0] = 0.0
        row1 = np.sin(tmp)
        row1[0] = 0.0
        A_eq = np.vstack((row0, row1))
        b_eq = [0, 0]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_unbounded(res)

    def test_enzo_example_c_with_infeasibility(self):
        if False:
            i = 10
            return i + 15
        m = 50
        c = -np.ones(m)
        tmp = 2 * np.pi * np.arange(m) / (m + 1)
        A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))
        b_eq = [1, 1]
        o = {key: self.options[key] for key in self.options}
        o['presolve'] = False
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=o)
        _assert_infeasible(res)

    def test_basic_artificial_vars(self):
        if False:
            print('Hello World!')
        c = np.array([-0.1, -0.07, 0.004, 0.004, 0.004, 0.004])
        A_ub = np.array([[1.0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0], [0, -1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0], [1.0, 1.0, 0, 0, 0, 0]])
        b_ub = np.array([3.0, 3.0, 3.0, 3.0, 20.0])
        A_eq = np.array([[1.0, 0, -1, 1, -1, 1], [0, -1.0, -1, 1, -1, 1]])
        b_eq = np.array([0, 0])
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=0, desired_x=np.zeros_like(c), atol=2e-06)

    def test_optimize_result(self):
        if False:
            print('Hello World!')
        (c, A_ub, b_ub, A_eq, b_eq, bounds) = very_random_gen(0)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
        assert_(res.success)
        assert_(res.nit)
        assert_(not res.status)
        if 'highs' not in self.method:
            assert_(res.message == 'Optimization terminated successfully.')
        assert_allclose(c @ res.x, res.fun)
        assert_allclose(b_eq - A_eq @ res.x, res.con, atol=1e-11)
        assert_allclose(b_ub - A_ub @ res.x, res.slack, atol=1e-11)
        for key in ['eqlin', 'ineqlin', 'lower', 'upper']:
            if key in res.keys():
                assert isinstance(res[key]['marginals'], np.ndarray)
                assert isinstance(res[key]['residual'], np.ndarray)

    def test_bug_5400(self):
        if False:
            print('Hello World!')
        bounds = [(0, None), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 900), (0, 900), (0, 900), (0, 900), (0, 900), (0, 900), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]
        f = 1 / 9
        g = -10000.0
        h = -3.1
        A_ub = np.array([[1, -2.99, 0, 0, -3, 0, 0, 0, -1, -1, 0, -1, -1, 1, 1, 0, 0, 0, 0], [1, 0, -2.9, h, 0, -3, 0, -1, 0, 0, -1, 0, -1, 0, 0, 1, 1, 0, 0], [1, 0, 0, h, 0, 0, -3, -1, -1, 0, -1, -1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1], [0, 1.99, -1, -1, 0, 0, 0, -1, f, f, 0, 0, 0, g, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, -1, -1, 0, 0, 0, -1, f, f, 0, g, 0, 0, 0, 0], [0, -1, 1.9, 2.1, 0, 0, 0, f, -1, -1, 0, 0, 0, 0, 0, g, 0, 0, 0], [0, 0, 0, 0, -1, 2, -1, 0, 0, 0, f, -1, f, 0, 0, 0, g, 0, 0], [0, -1, -1, 2.1, 0, 0, 0, f, f, -1, 0, 0, 0, 0, 0, 0, 0, g, 0], [0, 0, 0, 0, -1, -1, 2, 0, 0, 0, f, f, -1, 0, 0, 0, 0, 0, g]])
        b_ub = np.array([0.0, 0, 0, 100, 100, 100, 100, 100, 100, 900, 900, 900, 900, 900, 900, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        c = np.array([-1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "Solving system with option 'sym_pos'")
            sup.filter(RuntimeWarning, 'invalid value encountered')
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=-106.63507541835018)

    def test_bug_6139(self):
        if False:
            i = 10
            return i + 15
        c = np.array([1, 1, 1])
        A_eq = np.array([[1.0, 0.0, 0.0], [-1000.0, 0.0, -1000.0]])
        b_eq = np.array([5.0, -10000.0])
        A_ub = -np.array([[0.0, 1000000.0, 1010000.0]])
        b_ub = -np.array([10000000.0])
        bounds = (None, None)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=14.95, desired_x=np.array([5, 4.95, 5]))

    def test_bug_6690(self):
        if False:
            return 10
        A_eq = np.array([[0, 0, 0, 0.93, 0, 0.65, 0, 0, 0.83, 0]])
        b_eq = np.array([0.9626])
        A_ub = np.array([[0, 0, 0, 1.18, 0, 0, 0, -0.2, 0, -0.22], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0.43, 0, 0, 0, 0, 0, 0], [0, -1.22, -0.25, 0, 0, 0, -2.06, 0, 0, 1.37], [0, 0, 0, 0, 0, 0, 0, -0.25, 0, 0]])
        b_ub = np.array([0.615, 0, 0.172, -0.869, -0.022])
        bounds = np.array([[-0.84, -0.97, 0.34, 0.4, -0.33, -0.74, 0.47, 0.09, -1.45, -0.73], [0.37, 0.02, 2.86, 0.86, 1.18, 0.5, 1.76, 0.17, 0.32, -0.15]]).T
        c = np.array([-1.64, 0.7, 1.8, -1.06, -1.16, 0.26, 2.13, 1.53, 0.66, 0.28])
        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(OptimizeWarning, "Solving system with option 'cholesky'")
            sup.filter(OptimizeWarning, "Solving system with option 'sym_pos'")
            sup.filter(RuntimeWarning, 'invalid value encountered')
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        desired_fun = -1.19099999999
        desired_x = np.array([0.37, -0.97, 0.34, 0.4, 1.18, 0.5, 0.47, 0.09, 0.32, -0.73])
        _assert_success(res, desired_fun=desired_fun, desired_x=desired_x)
        atol = 1e-06
        assert_array_less(bounds[:, 0] - atol, res.x)
        assert_array_less(res.x, bounds[:, 1] + atol)

    def test_bug_7044(self):
        if False:
            print('Hello World!')
        (A_eq, b_eq, c, _, _) = magic_square(3)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, 'A_eq does not appear...')
            sup.filter(RuntimeWarning, 'invalid value encountered')
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        desired_fun = 1.730550597
        _assert_success(res, desired_fun=desired_fun)
        assert_allclose(A_eq.dot(res.x), b_eq)
        assert_array_less(np.zeros(res.x.size) - 1e-05, res.x)

    def test_bug_7237(self):
        if False:
            while True:
                i = 10
        c = np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0])
        A_ub = np.array([[1.0, -724.0, 911.0, -551.0, -555.0, -896.0, 478.0, -80.0, -293.0], [1.0, 566.0, 42.0, 937.0, 233.0, 883.0, 392.0, -909.0, 57.0], [1.0, -208.0, -894.0, 539.0, 321.0, 532.0, -924.0, 942.0, 55.0], [1.0, 857.0, -859.0, 83.0, 462.0, -265.0, -971.0, 826.0, 482.0], [1.0, 314.0, -424.0, 245.0, -424.0, 194.0, -443.0, -104.0, -429.0], [1.0, 540.0, 679.0, 361.0, 149.0, -827.0, 876.0, 633.0, 302.0], [0.0, -1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0], [0.0, -0.0, -1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0], [0.0, -0.0, -0.0, -1.0, -0.0, -0.0, -0.0, -0.0, -0.0], [0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, -0.0, -0.0], [0.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, -0.0], [0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0], [0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0], [0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        b_ub = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        A_eq = np.array([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        b_eq = np.array([[1.0]])
        bounds = [(None, None)] * 9
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=108.568535, atol=1e-06)

    def test_bug_8174(self):
        if False:
            return 10
        A_ub = np.array([[22714, 1008, 13380, -2713.5, -1116], [-4986, -1092, -31220, 17386.5, 684], [-4986, 0, 0, -2713.5, 0], [22714, 0, 0, 17386.5, 0]])
        b_ub = np.zeros(A_ub.shape[0])
        c = -np.ones(A_ub.shape[1])
        bounds = [(0, 1)] * A_ub.shape[1]
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered')
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        if self.options.get('tol', 1e-09) < 1e-10 and self.method == 'simplex':
            _assert_unable_to_find_basic_feasible_sol(res)
        else:
            _assert_success(res, desired_fun=-2.0080717488789235, atol=1e-06)

    def test_bug_8174_2(self):
        if False:
            print('Hello World!')
        c = np.array([1, 0, 0, 0, 0, 0, 0])
        A_ub = -np.identity(7)
        b_ub = np.array([[-2], [-2], [-2], [-2], [-2], [-2], [-2]])
        A_eq = np.array([[1, 1, 1, 1, 1, 1, 0], [0.3, 1.3, 0.9, 0, 0, 0, -1], [0.3, 0, 0, 0, 0, 0, -2 / 3], [0, 0.65, 0, 0, 0, 0, -1 / 15], [0, 0, 0.3, 0, 0, 0, -1 / 15]])
        b_eq = np.array([[100], [0], [0], [0], [0]])
        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(OptimizeWarning, 'A_eq does not appear...')
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=43.3333333331385)

    def test_bug_8561(self):
        if False:
            i = 10
            return i + 15
        c = np.array([7, 0, -4, 1.5, 1.5])
        A_ub = np.array([[4, 5.5, 1.5, 1.0, -3.5], [1, -2.5, -2, 2.5, 0.5], [3, -0.5, 4, -12.5, -7], [-1, 4.5, 2, -3.5, -2], [5.5, 2, -4.5, -1, 9.5]])
        b_ub = np.array([0, 0, 0, 0, 1])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, options=self.options, method=self.method)
        _assert_success(res, desired_x=[0, 0, 19, 16 / 3, 29 / 3])

    def test_bug_8662(self):
        if False:
            print('Hello World!')
        c = [-10, 10, 6, 3]
        A_ub = [[8, -8, -4, 6], [-8, 8, 4, -6], [-4, 4, 8, -4], [3, -3, -3, -10]]
        b_ub = [9, -9, -9, -4]
        bounds = [(0, None), (0, None), (0, None), (0, None)]
        desired_fun = 36.0
        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(RuntimeWarning, 'invalid value encountered')
            sup.filter(LinAlgWarning)
            res1 = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        A_ub.append([0, 0, -1, 0])
        b_ub.append(0)
        bounds[2] = (None, None)
        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(RuntimeWarning, 'invalid value encountered')
            sup.filter(LinAlgWarning)
            res2 = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        rtol = 1e-05
        _assert_success(res1, desired_fun=desired_fun, rtol=rtol)
        _assert_success(res2, desired_fun=desired_fun, rtol=rtol)

    def test_bug_8663(self):
        if False:
            while True:
                i = 10
        c = [1, 5]
        A_eq = [[0, -7]]
        b_eq = [-6]
        bounds = [(0, None), (None, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_x=[0, 6.0 / 7], desired_fun=5 * 6.0 / 7)

    def test_bug_8664(self):
        if False:
            print('Hello World!')
        c = [4]
        A_ub = [[2], [5]]
        b_ub = [4, 4]
        A_eq = [[0], [-8], [9]]
        b_eq = [3, 2, 10]
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            sup.filter(OptimizeWarning, 'Solving system with option...')
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_infeasible(res)

    def test_bug_8973(self):
        if False:
            print('Hello World!')
        '\n        Test whether bug described at:\n        https://github.com/scipy/scipy/issues/8973\n        was fixed.\n        '
        c = np.array([0, 0, 0, 1, -1])
        A_ub = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        b_ub = np.array([2, -2])
        bounds = [(None, None), (None, None), (None, None), (-1, 1), (-1, 1)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=-2)
        assert_equal(c @ res.x, res.fun)

    def test_bug_8973_2(self):
        if False:
            return 10
        '\n        Additional test for:\n        https://github.com/scipy/scipy/issues/8973\n        suggested in\n        https://github.com/scipy/scipy/pull/8985\n        review by @antonior92\n        '
        c = np.zeros(1)
        A_ub = np.array([[1]])
        b_ub = np.array([-2])
        bounds = (None, None)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_x=[-2], desired_fun=0)

    def test_bug_10124(self):
        if False:
            print('Hello World!')
        "\n        Test for linprog docstring problem\n        'disp'=True caused revised simplex failure\n        "
        c = np.zeros(1)
        A_ub = np.array([[1]])
        b_ub = np.array([-2])
        bounds = (None, None)
        c = [-1, 4]
        A_ub = [[-3, 1], [1, 2]]
        b_ub = [6, 4]
        bounds = [(None, None), (-3, None)]
        o = {'disp': True}
        o.update(self.options)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=o)
        _assert_success(res, desired_x=[10, -3], desired_fun=-22)

    def test_bug_10349(self):
        if False:
            print('Hello World!')
        '\n        Test for redundancy removal tolerance issue\n        https://github.com/scipy/scipy/issues/10349\n        '
        A_eq = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1], [1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 0, 1]])
        b_eq = np.array([221, 210, 10, 141, 198, 102])
        c = np.concatenate((0, 1, np.zeros(4)), axis=None)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, 'A_eq does not appear...')
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
        _assert_success(res, desired_x=[129, 92, 12, 198, 0, 10], desired_fun=92)

    @pytest.mark.skipif(sys.platform == 'darwin', reason='Failing on some local macOS builds, see gh-13846')
    def test_bug_10466(self):
        if False:
            print('Hello World!')
        '\n        Test that autoscale fixes poorly-scaled problem\n        '
        c = [-8.0, -0.0, -8.0, -0.0, -8.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]
        A_eq = [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        b_eq = [314572800.0, 419430400.0, 524288000.0, 1006632960.0, 1073741820.0, 1073741820.0, 1073741820.0, 1073741820.0, 1073741820.0, 1073741820.0]
        o = {}
        if not self.method.startswith('highs'):
            o = {'autoscale': True}
        o.update(self.options)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, 'Solving system with option...')
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(RuntimeWarning, 'scipy.linalg.solve\nIll...')
            sup.filter(RuntimeWarning, 'divide by zero encountered...')
            sup.filter(RuntimeWarning, 'overflow encountered...')
            sup.filter(RuntimeWarning, 'invalid value encountered...')
            sup.filter(LinAlgWarning, 'Ill-conditioned matrix...')
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=o)
        assert_allclose(res.fun, -8589934560)

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class LinprogSimplexTests(LinprogCommonTests):
    method = 'simplex'

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class LinprogIPTests(LinprogCommonTests):
    method = 'interior-point'

    def test_bug_10466(self):
        if False:
            for i in range(10):
                print('nop')
        pytest.skip('Test is failing, but solver is deprecated.')

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class LinprogRSTests(LinprogCommonTests):
    method = 'revised simplex'

    def test_bug_5400(self):
        if False:
            i = 10
            return i + 15
        pytest.skip('Intermittent failure acceptable.')

    def test_bug_8662(self):
        if False:
            return 10
        pytest.skip('Intermittent failure acceptable.')

    def test_network_flow(self):
        if False:
            for i in range(10):
                print('nop')
        pytest.skip('Intermittent failure acceptable.')

class LinprogHiGHSTests(LinprogCommonTests):

    def test_callback(self):
        if False:
            return 10

        def cb(res):
            if False:
                return 10
            return None
        c = np.array([-3, -2])
        A_ub = [[2, 1], [1, 1], [1, 0]]
        b_ub = [10, 8, 4]
        assert_raises(NotImplementedError, linprog, c, A_ub=A_ub, b_ub=b_ub, callback=cb, method=self.method)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, method=self.method)
        _assert_success(res, desired_fun=-18.0, desired_x=[2, 6])

    @pytest.mark.parametrize('options', [{'maxiter': -1}, {'disp': -1}, {'presolve': -1}, {'time_limit': -1}, {'dual_feasibility_tolerance': -1}, {'primal_feasibility_tolerance': -1}, {'ipm_optimality_tolerance': -1}, {'simplex_dual_edge_weight_strategy': 'ekki'}])
    def test_invalid_option_values(self, options):
        if False:
            print('Hello World!')

        def f(options):
            if False:
                while True:
                    i = 10
            linprog(1, method=self.method, options=options)
        options.update(self.options)
        assert_warns(OptimizeWarning, f, options=options)

    def test_crossover(self):
        if False:
            print('Hello World!')
        (A_eq, b_eq, c, _, _) = magic_square(4)
        bounds = (0, 1)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
        assert_equal(res.crossover_nit == 0, self.method != 'highs-ipm')

    def test_marginals(self):
        if False:
            print('Hello World!')
        (c, A_ub, b_ub, A_eq, b_eq, bounds) = very_random_gen(seed=0)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
        (lb, ub) = bounds.T

        def f_bub(x):
            if False:
                while True:
                    i = 10
            return linprog(c, A_ub, x, A_eq, b_eq, bounds, method=self.method).fun
        dfdbub = approx_derivative(f_bub, b_ub, method='3-point', f0=res.fun)
        assert_allclose(res.ineqlin.marginals, dfdbub)

        def f_beq(x):
            if False:
                print('Hello World!')
            return linprog(c, A_ub, b_ub, A_eq, x, bounds, method=self.method).fun
        dfdbeq = approx_derivative(f_beq, b_eq, method='3-point', f0=res.fun)
        assert_allclose(res.eqlin.marginals, dfdbeq)

        def f_lb(x):
            if False:
                i = 10
                return i + 15
            bounds = np.array([x, ub]).T
            return linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method).fun
        with np.errstate(invalid='ignore'):
            dfdlb = approx_derivative(f_lb, lb, method='3-point', f0=res.fun)
            dfdlb[~np.isfinite(lb)] = 0
        assert_allclose(res.lower.marginals, dfdlb)

        def f_ub(x):
            if False:
                print('Hello World!')
            bounds = np.array([lb, x]).T
            return linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method).fun
        with np.errstate(invalid='ignore'):
            dfdub = approx_derivative(f_ub, ub, method='3-point', f0=res.fun)
            dfdub[~np.isfinite(ub)] = 0
        assert_allclose(res.upper.marginals, dfdub)

    def test_dual_feasibility(self):
        if False:
            return 10
        (c, A_ub, b_ub, A_eq, b_eq, bounds) = very_random_gen(seed=42)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
        resid = -c + A_ub.T @ res.ineqlin.marginals + A_eq.T @ res.eqlin.marginals + res.upper.marginals + res.lower.marginals
        assert_allclose(resid, 0, atol=1e-12)

    def test_complementary_slackness(self):
        if False:
            print('Hello World!')
        (c, A_ub, b_ub, A_eq, b_eq, bounds) = very_random_gen(seed=42)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
        assert np.allclose(res.ineqlin.marginals @ (b_ub - A_ub @ res.x), 0)

class TestLinprogSimplexDefault(LinprogSimplexTests):

    def setup_method(self):
        if False:
            return 10
        self.options = {}

    def test_bug_5400(self):
        if False:
            for i in range(10):
                print('nop')
        pytest.skip('Simplex fails on this problem.')

    def test_bug_7237_low_tol(self):
        if False:
            while True:
                i = 10
        pytest.skip('Simplex fails on this problem.')

    def test_bug_8174_low_tol(self):
        if False:
            print('Hello World!')
        self.options.update({'tol': 1e-12})
        with pytest.warns(OptimizeWarning):
            super().test_bug_8174()

class TestLinprogSimplexBland(LinprogSimplexTests):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.options = {'bland': True}

    def test_bug_5400(self):
        if False:
            for i in range(10):
                print('nop')
        pytest.skip('Simplex fails on this problem.')

    def test_bug_8174_low_tol(self):
        if False:
            for i in range(10):
                print('nop')
        self.options.update({'tol': 1e-12})
        with pytest.raises(AssertionError):
            with pytest.warns(OptimizeWarning):
                super().test_bug_8174()

class TestLinprogSimplexNoPresolve(LinprogSimplexTests):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.options = {'presolve': False}
    is_32_bit = np.intp(0).itemsize < 8
    is_linux = sys.platform.startswith('linux')

    @pytest.mark.xfail(condition=is_32_bit and is_linux, reason='Fails with warning on 32-bit linux')
    def test_bug_5400(self):
        if False:
            i = 10
            return i + 15
        super().test_bug_5400()

    def test_bug_6139_low_tol(self):
        if False:
            i = 10
            return i + 15
        self.options.update({'tol': 1e-12})
        with pytest.raises(AssertionError, match='linprog status 4'):
            return super().test_bug_6139()

    def test_bug_7237_low_tol(self):
        if False:
            return 10
        pytest.skip('Simplex fails on this problem.')

    def test_bug_8174_low_tol(self):
        if False:
            print('Hello World!')
        self.options.update({'tol': 1e-12})
        with pytest.warns(OptimizeWarning):
            super().test_bug_8174()

    def test_unbounded_no_nontrivial_constraints_1(self):
        if False:
            print('Hello World!')
        pytest.skip('Tests behavior specific to presolve')

    def test_unbounded_no_nontrivial_constraints_2(self):
        if False:
            for i in range(10):
                print('nop')
        pytest.skip('Tests behavior specific to presolve')

class TestLinprogIPDense(LinprogIPTests):
    options = {'sparse': False}
if has_cholmod:

    class TestLinprogIPSparseCholmod(LinprogIPTests):
        options = {'sparse': True, 'cholesky': True}
if has_umfpack:

    class TestLinprogIPSparseUmfpack(LinprogIPTests):
        options = {'sparse': True, 'cholesky': False}

        def test_network_flow_limited_capacity(self):
            if False:
                i = 10
                return i + 15
            pytest.skip('Failing due to numerical issues on some platforms.')

class TestLinprogIPSparse(LinprogIPTests):
    options = {'sparse': True, 'cholesky': False, 'sym_pos': False}

    @pytest.mark.xfail_on_32bit('This test is sensitive to machine epsilon level perturbations in linear system solution in _linprog_ip._sym_solve.')
    def test_bug_6139(self):
        if False:
            i = 10
            return i + 15
        super().test_bug_6139()

    @pytest.mark.xfail(reason='Fails with ATLAS, see gh-7877')
    def test_bug_6690(self):
        if False:
            return 10
        super().test_bug_6690()

    def test_magic_square_sparse_no_presolve(self):
        if False:
            print('Hello World!')
        (A_eq, b_eq, c, _, _) = magic_square(3)
        bounds = (0, 1)
        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(MatrixRankWarning, 'Matrix is exactly singular')
            sup.filter(OptimizeWarning, 'Solving system with option...')
            o = {key: self.options[key] for key in self.options}
            o['presolve'] = False
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=o)
        _assert_success(res, desired_fun=1.730550597)

    def test_sparse_solve_options(self):
        if False:
            for i in range(10):
                print('nop')
        (A_eq, b_eq, c, _, _) = magic_square(3)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, 'A_eq does not appear...')
            sup.filter(OptimizeWarning, 'Invalid permc_spec option')
            o = {key: self.options[key] for key in self.options}
            permc_specs = ('NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', 'COLAMD', 'ekki-ekki-ekki')
            for permc_spec in permc_specs:
                o['permc_spec'] = permc_spec
                res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=o)
                _assert_success(res, desired_fun=1.730550597)

class TestLinprogIPSparsePresolve(LinprogIPTests):
    options = {'sparse': True, '_sparse_presolve': True}

    @pytest.mark.xfail_on_32bit('This test is sensitive to machine epsilon level perturbations in linear system solution in _linprog_ip._sym_solve.')
    def test_bug_6139(self):
        if False:
            i = 10
            return i + 15
        super().test_bug_6139()

    def test_enzo_example_c_with_infeasibility(self):
        if False:
            print('Hello World!')
        pytest.skip('_sparse_presolve=True incompatible with presolve=False')

    @pytest.mark.xfail(reason='Fails with ATLAS, see gh-7877')
    def test_bug_6690(self):
        if False:
            i = 10
            return i + 15
        super().test_bug_6690()

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class TestLinprogIPSpecific:
    method = 'interior-point'

    def test_solver_select(self):
        if False:
            i = 10
            return i + 15
        if has_cholmod:
            options = {'sparse': True, 'cholesky': True}
        elif has_umfpack:
            options = {'sparse': True, 'cholesky': False}
        else:
            options = {'sparse': True, 'cholesky': False, 'sym_pos': False}
        (A, b, c) = lpgen_2d(20, 20)
        res1 = linprog(c, A_ub=A, b_ub=b, method=self.method, options=options)
        res2 = linprog(c, A_ub=A, b_ub=b, method=self.method)
        assert_allclose(res1.fun, res2.fun, err_msg='linprog default solver unexpected result', rtol=2e-15, atol=1e-15)

    def test_unbounded_below_no_presolve_original(self):
        if False:
            return 10
        c = [-1]
        bounds = [(None, 1)]
        res = linprog(c=c, bounds=bounds, method=self.method, options={'presolve': False, 'cholesky': True})
        _assert_success(res, desired_fun=-1)

    def test_cholesky(self):
        if False:
            while True:
                i = 10
        (A, b, c) = lpgen_2d(20, 20)
        res = linprog(c, A_ub=A, b_ub=b, method=self.method, options={'cholesky': True})
        _assert_success(res, desired_fun=-64.049494229)

    def test_alternate_initial_point(self):
        if False:
            while True:
                i = 10
        (A, b, c) = lpgen_2d(20, 20)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'scipy.linalg.solve\nIll...')
            sup.filter(OptimizeWarning, 'Solving system with option...')
            sup.filter(LinAlgWarning, 'Ill-conditioned matrix...')
            res = linprog(c, A_ub=A, b_ub=b, method=self.method, options={'ip': True, 'disp': True})
        _assert_success(res, desired_fun=-64.049494229)

    def test_bug_8664(self):
        if False:
            for i in range(10):
                print('nop')
        c = [4]
        A_ub = [[2], [5]]
        b_ub = [4, 4]
        A_eq = [[0], [-8], [9]]
        b_eq = [3, 2, 10]
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            sup.filter(OptimizeWarning, 'Solving system with option...')
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options={'presolve': False})
        assert_(not res.success, 'Incorrectly reported success')

class TestLinprogRSCommon(LinprogRSTests):
    options = {}

    def test_cyclic_bland(self):
        if False:
            print('Hello World!')
        pytest.skip('Intermittent failure acceptable.')

    def test_nontrivial_problem_with_guess(self):
        if False:
            while True:
                i = 10
        (c, A_ub, b_ub, A_eq, b_eq, x_star, f_star) = nontrivial_problem()
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_unbounded_variables(self):
        if False:
            while True:
                i = 10
        (c, A_ub, b_ub, A_eq, b_eq, x_star, f_star) = nontrivial_problem()
        bounds = [(None, None), (None, None), (0, None), (None, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_bounded_variables(self):
        if False:
            print('Hello World!')
        (c, A_ub, b_ub, A_eq, b_eq, x_star, f_star) = nontrivial_problem()
        bounds = [(None, 1), (1, None), (0, None), (0.4, 0.6)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_negative_unbounded_variable(self):
        if False:
            i = 10
            return i + 15
        (c, A_ub, b_ub, A_eq, b_eq, x_star, f_star) = nontrivial_problem()
        b_eq = [4]
        x_star = np.array([-219 / 385, 582 / 385, 0, 4 / 10])
        f_star = 3951 / 385
        bounds = [(None, None), (1, None), (0, None), (0.4, 0.6)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_bad_guess(self):
        if False:
            i = 10
            return i + 15
        (c, A_ub, b_ub, A_eq, b_eq, x_star, f_star) = nontrivial_problem()
        bad_guess = [1, 2, 3, 0.5]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=bad_guess)
        assert_equal(res.status, 6)

    def test_redundant_constraints_with_guess(self):
        if False:
            for i in range(10):
                print('nop')
        (A, b, c, _, _) = magic_square(3)
        p = np.random.rand(*c.shape)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, 'A_eq does not appear...')
            sup.filter(RuntimeWarning, 'invalid value encountered')
            sup.filter(LinAlgWarning)
            res = linprog(c, A_eq=A, b_eq=b, method=self.method)
            res2 = linprog(c, A_eq=A, b_eq=b, method=self.method, x0=res.x)
            res3 = linprog(c + p, A_eq=A, b_eq=b, method=self.method, x0=res.x)
        _assert_success(res2, desired_fun=1.730550597)
        assert_equal(res2.nit, 0)
        _assert_success(res3)
        assert_(res3.nit < res.nit)

class TestLinprogRSBland(LinprogRSTests):
    options = {'pivot': 'bland'}

class TestLinprogHiGHSSimplexDual(LinprogHiGHSTests):
    method = 'highs-ds'
    options = {}

    def test_lad_regression(self):
        if False:
            return 10
        '\n        The scaled model should be optimal, i.e. not produce unscaled model\n        infeasible.  See https://github.com/ERGO-Code/HiGHS/issues/494.\n        '
        (c, A_ub, b_ub, bnds) = l1_regression_prob()
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bnds, method=self.method, options=self.options)
        assert_equal(res.status, 0)
        assert_(res.x is not None)
        assert_(np.all(res.slack > -1e-06))
        assert_(np.all(res.x <= [np.inf if ub is None else ub for (lb, ub) in bnds]))
        assert_(np.all(res.x >= [-np.inf if lb is None else lb - 1e-07 for (lb, ub) in bnds]))

class TestLinprogHiGHSIPM(LinprogHiGHSTests):
    method = 'highs-ipm'
    options = {}

class TestLinprogHiGHSMIP:
    method = 'highs'
    options = {}

    @pytest.mark.xfail(condition=sys.maxsize < 2 ** 32 and platform.system() == 'Linux', run=False, reason='gh-16347')
    def test_mip1(self):
        if False:
            while True:
                i = 10
        n = 4
        (A, b, c, numbers, M) = magic_square(n)
        bounds = [(0, 1)] * len(c)
        integrality = [1] * len(c)
        res = linprog(c=c * 0, A_eq=A, b_eq=b, bounds=bounds, method=self.method, integrality=integrality)
        s = (numbers.flatten() * res.x).reshape(n ** 2, n, n)
        square = np.sum(s, axis=0)
        np.testing.assert_allclose(square.sum(axis=0), M)
        np.testing.assert_allclose(square.sum(axis=1), M)
        np.testing.assert_allclose(np.diag(square).sum(), M)
        np.testing.assert_allclose(np.diag(square[:, ::-1]).sum(), M)
        np.testing.assert_allclose(res.x, np.round(res.x), atol=1e-12)

    def test_mip2(self):
        if False:
            print('Hello World!')
        A_ub = np.array([[2, -2], [-8, 10]])
        b_ub = np.array([-1, 13])
        c = -np.array([1, 1])
        bounds = np.array([(0, np.inf)] * len(c))
        integrality = np.ones_like(c)
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=self.method, integrality=integrality)
        np.testing.assert_allclose(res.x, [1, 2])
        np.testing.assert_allclose(res.fun, -3)

    def test_mip3(self):
        if False:
            i = 10
            return i + 15
        A_ub = np.array([[-1, 1], [3, 2], [2, 3]])
        b_ub = np.array([1, 12, 12])
        c = -np.array([0, 1])
        bounds = [(0, np.inf)] * len(c)
        integrality = [1] * len(c)
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=self.method, integrality=integrality)
        np.testing.assert_allclose(res.fun, -2)
        assert np.allclose(res.x, [1, 2]) or np.allclose(res.x, [2, 2])

    def test_mip4(self):
        if False:
            return 10
        A_ub = np.array([[-1, -2], [-4, -1], [2, 1]])
        b_ub = np.array([14, -33, 20])
        c = np.array([8, 1])
        bounds = [(0, np.inf)] * len(c)
        integrality = [0, 1]
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=self.method, integrality=integrality)
        np.testing.assert_allclose(res.x, [6.5, 7])
        np.testing.assert_allclose(res.fun, 59)

    def test_mip5(self):
        if False:
            return 10
        A_ub = np.array([[1, 1, 1]])
        b_ub = np.array([7])
        A_eq = np.array([[4, 2, 1]])
        b_eq = np.array([12])
        c = np.array([-3, -2, -1])
        bounds = [(0, np.inf), (0, np.inf), (0, 1)]
        integrality = [0, 1, 0]
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, integrality=integrality)
        np.testing.assert_allclose(res.x, [0, 6, 0])
        np.testing.assert_allclose(res.fun, -12)
        assert res.get('mip_node_count', None) is not None
        assert res.get('mip_dual_bound', None) is not None
        assert res.get('mip_gap', None) is not None

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_mip6(self):
        if False:
            return 10
        A_eq = np.array([[22, 13, 26, 33, 21, 3, 14, 26], [39, 16, 22, 28, 26, 30, 23, 24], [18, 14, 29, 27, 30, 38, 26, 26], [41, 26, 28, 36, 18, 38, 16, 26]])
        b_eq = np.array([7872, 10466, 11322, 12058])
        c = np.array([2, 10, 13, 17, 7, 5, 7, 3])
        bounds = [(0, np.inf)] * 8
        integrality = [1] * 8
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, integrality=integrality)
        np.testing.assert_allclose(res.fun, 1854)

    @pytest.mark.xslow
    def test_mip_rel_gap_passdown(self):
        if False:
            print('Hello World!')
        A_eq = np.array([[22, 13, 26, 33, 21, 3, 14, 26], [39, 16, 22, 28, 26, 30, 23, 24], [18, 14, 29, 27, 30, 38, 26, 26], [41, 26, 28, 36, 18, 38, 16, 26]])
        b_eq = np.array([7872, 10466, 11322, 12058])
        c = np.array([2, 10, 13, 17, 7, 5, 7, 3])
        bounds = [(0, np.inf)] * 8
        integrality = [1] * 8
        mip_rel_gaps = [0.5, 0.25, 0.01, 0.001]
        sol_mip_gaps = []
        for mip_rel_gap in mip_rel_gaps:
            res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, integrality=integrality, options={'mip_rel_gap': mip_rel_gap})
            final_mip_gap = res['mip_gap']
            assert final_mip_gap <= mip_rel_gap
            sol_mip_gaps.append(final_mip_gap)
        gap_diffs = np.diff(np.flip(sol_mip_gaps))
        assert np.all(gap_diffs >= 0)
        assert not np.all(gap_diffs == 0)

    def test_semi_continuous(self):
        if False:
            return 10
        c = np.array([1.0, 1.0, -1, -1])
        bounds = np.array([[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.5, 1.5]])
        integrality = np.array([2, 3, 2, 3])
        res = linprog(c, bounds=bounds, integrality=integrality, method='highs')
        np.testing.assert_allclose(res.x, [0, 0, 1.5, 1])
        assert res.status == 0

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class AutoscaleTests:
    options = {'autoscale': True}
    test_bug_6139 = LinprogCommonTests.test_bug_6139
    test_bug_6690 = LinprogCommonTests.test_bug_6690
    test_bug_7237 = LinprogCommonTests.test_bug_7237

class TestAutoscaleIP(AutoscaleTests):
    method = 'interior-point'

    def test_bug_6139(self):
        if False:
            return 10
        self.options['tol'] = 1e-10
        return AutoscaleTests.test_bug_6139(self)

class TestAutoscaleSimplex(AutoscaleTests):
    method = 'simplex'

class TestAutoscaleRS(AutoscaleTests):
    method = 'revised simplex'

    def test_nontrivial_problem_with_guess(self):
        if False:
            for i in range(10):
                print('nop')
        (c, A_ub, b_ub, A_eq, b_eq, x_star, f_star) = nontrivial_problem()
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_bad_guess(self):
        if False:
            return 10
        (c, A_ub, b_ub, A_eq, b_eq, x_star, f_star) = nontrivial_problem()
        bad_guess = [1, 2, 3, 0.5]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=bad_guess)
        assert_equal(res.status, 6)

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class RRTests:
    method = 'interior-point'
    LCT = LinprogCommonTests
    test_RR_infeasibility = LCT.test_remove_redundancy_infeasibility
    test_bug_10349 = LCT.test_bug_10349
    test_bug_7044 = LCT.test_bug_7044
    test_NFLC = LCT.test_network_flow_limited_capacity
    test_enzo_example_b = LCT.test_enzo_example_b

class TestRRSVD(RRTests):
    options = {'rr_method': 'SVD'}

class TestRRPivot(RRTests):
    options = {'rr_method': 'pivot'}

class TestRRID(RRTests):
    options = {'rr_method': 'ID'}