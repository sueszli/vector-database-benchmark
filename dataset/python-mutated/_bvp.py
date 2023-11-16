"""Boundary value problem solver."""
from warnings import warn
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult
EPS = np.finfo(float).eps

def estimate_fun_jac(fun, x, y, p, f0=None):
    if False:
        while True:
            i = 10
    'Estimate derivatives of an ODE system rhs with forward differences.\n\n    Returns\n    -------\n    df_dy : ndarray, shape (n, n, m)\n        Derivatives with respect to y. An element (i, j, q) corresponds to\n        d f_i(x_q, y_q) / d (y_q)_j.\n    df_dp : ndarray with shape (n, k, m) or None\n        Derivatives with respect to p. An element (i, j, q) corresponds to\n        d f_i(x_q, y_q, p) / d p_j. If `p` is empty, None is returned.\n    '
    (n, m) = y.shape
    if f0 is None:
        f0 = fun(x, y, p)
    dtype = y.dtype
    df_dy = np.empty((n, n, m), dtype=dtype)
    h = EPS ** 0.5 * (1 + np.abs(y))
    for i in range(n):
        y_new = y.copy()
        y_new[i] += h[i]
        hi = y_new[i] - y[i]
        f_new = fun(x, y_new, p)
        df_dy[:, i, :] = (f_new - f0) / hi
    k = p.shape[0]
    if k == 0:
        df_dp = None
    else:
        df_dp = np.empty((n, k, m), dtype=dtype)
        h = EPS ** 0.5 * (1 + np.abs(p))
        for i in range(k):
            p_new = p.copy()
            p_new[i] += h[i]
            hi = p_new[i] - p[i]
            f_new = fun(x, y, p_new)
            df_dp[:, i, :] = (f_new - f0) / hi
    return (df_dy, df_dp)

def estimate_bc_jac(bc, ya, yb, p, bc0=None):
    if False:
        return 10
    'Estimate derivatives of boundary conditions with forward differences.\n\n    Returns\n    -------\n    dbc_dya : ndarray, shape (n + k, n)\n        Derivatives with respect to ya. An element (i, j) corresponds to\n        d bc_i / d ya_j.\n    dbc_dyb : ndarray, shape (n + k, n)\n        Derivatives with respect to yb. An element (i, j) corresponds to\n        d bc_i / d ya_j.\n    dbc_dp : ndarray with shape (n + k, k) or None\n        Derivatives with respect to p. An element (i, j) corresponds to\n        d bc_i / d p_j. If `p` is empty, None is returned.\n    '
    n = ya.shape[0]
    k = p.shape[0]
    if bc0 is None:
        bc0 = bc(ya, yb, p)
    dtype = ya.dtype
    dbc_dya = np.empty((n, n + k), dtype=dtype)
    h = EPS ** 0.5 * (1 + np.abs(ya))
    for i in range(n):
        ya_new = ya.copy()
        ya_new[i] += h[i]
        hi = ya_new[i] - ya[i]
        bc_new = bc(ya_new, yb, p)
        dbc_dya[i] = (bc_new - bc0) / hi
    dbc_dya = dbc_dya.T
    h = EPS ** 0.5 * (1 + np.abs(yb))
    dbc_dyb = np.empty((n, n + k), dtype=dtype)
    for i in range(n):
        yb_new = yb.copy()
        yb_new[i] += h[i]
        hi = yb_new[i] - yb[i]
        bc_new = bc(ya, yb_new, p)
        dbc_dyb[i] = (bc_new - bc0) / hi
    dbc_dyb = dbc_dyb.T
    if k == 0:
        dbc_dp = None
    else:
        h = EPS ** 0.5 * (1 + np.abs(p))
        dbc_dp = np.empty((k, n + k), dtype=dtype)
        for i in range(k):
            p_new = p.copy()
            p_new[i] += h[i]
            hi = p_new[i] - p[i]
            bc_new = bc(ya, yb, p_new)
            dbc_dp[i] = (bc_new - bc0) / hi
        dbc_dp = dbc_dp.T
    return (dbc_dya, dbc_dyb, dbc_dp)

def compute_jac_indices(n, m, k):
    if False:
        while True:
            i = 10
    'Compute indices for the collocation system Jacobian construction.\n\n    See `construct_global_jac` for the explanation.\n    '
    i_col = np.repeat(np.arange((m - 1) * n), n)
    j_col = np.tile(np.arange(n), n * (m - 1)) + np.repeat(np.arange(m - 1) * n, n ** 2)
    i_bc = np.repeat(np.arange((m - 1) * n, m * n + k), n)
    j_bc = np.tile(np.arange(n), n + k)
    i_p_col = np.repeat(np.arange((m - 1) * n), k)
    j_p_col = np.tile(np.arange(m * n, m * n + k), (m - 1) * n)
    i_p_bc = np.repeat(np.arange((m - 1) * n, m * n + k), k)
    j_p_bc = np.tile(np.arange(m * n, m * n + k), n + k)
    i = np.hstack((i_col, i_col, i_bc, i_bc, i_p_col, i_p_bc))
    j = np.hstack((j_col, j_col + n, j_bc, j_bc + (m - 1) * n, j_p_col, j_p_bc))
    return (i, j)

def stacked_matmul(a, b):
    if False:
        i = 10
        return i + 15
    'Stacked matrix multiply: out[i,:,:] = np.dot(a[i,:,:], b[i,:,:]).\n\n    Empirical optimization. Use outer Python loop and BLAS for large\n    matrices, otherwise use a single einsum call.\n    '
    if a.shape[1] > 50:
        out = np.empty((a.shape[0], a.shape[1], b.shape[2]))
        for i in range(a.shape[0]):
            out[i] = np.dot(a[i], b[i])
        return out
    else:
        return np.einsum('...ij,...jk->...ik', a, b)

def construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle, df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp):
    if False:
        while True:
            i = 10
    'Construct the Jacobian of the collocation system.\n\n    There are n * m + k functions: m - 1 collocations residuals, each\n    containing n components, followed by n + k boundary condition residuals.\n\n    There are n * m + k variables: m vectors of y, each containing n\n    components, followed by k values of vector p.\n\n    For example, let m = 4, n = 2 and k = 1, then the Jacobian will have\n    the following sparsity structure:\n\n        1 1 2 2 0 0 0 0  5\n        1 1 2 2 0 0 0 0  5\n        0 0 1 1 2 2 0 0  5\n        0 0 1 1 2 2 0 0  5\n        0 0 0 0 1 1 2 2  5\n        0 0 0 0 1 1 2 2  5\n\n        3 3 0 0 0 0 4 4  6\n        3 3 0 0 0 0 4 4  6\n        3 3 0 0 0 0 4 4  6\n\n    Zeros denote identically zero values, other values denote different kinds\n    of blocks in the matrix (see below). The blank row indicates the separation\n    of collocation residuals from boundary conditions. And the blank column\n    indicates the separation of y values from p values.\n\n    Refer to [1]_  (p. 306) for the formula of n x n blocks for derivatives\n    of collocation residuals with respect to y.\n\n    Parameters\n    ----------\n    n : int\n        Number of equations in the ODE system.\n    m : int\n        Number of nodes in the mesh.\n    k : int\n        Number of the unknown parameters.\n    i_jac, j_jac : ndarray\n        Row and column indices returned by `compute_jac_indices`. They\n        represent different blocks in the Jacobian matrix in the following\n        order (see the scheme above):\n\n            * 1: m - 1 diagonal n x n blocks for the collocation residuals.\n            * 2: m - 1 off-diagonal n x n blocks for the collocation residuals.\n            * 3 : (n + k) x n block for the dependency of the boundary\n              conditions on ya.\n            * 4: (n + k) x n block for the dependency of the boundary\n              conditions on yb.\n            * 5: (m - 1) * n x k block for the dependency of the collocation\n              residuals on p.\n            * 6: (n + k) x k block for the dependency of the boundary\n              conditions on p.\n\n    df_dy : ndarray, shape (n, n, m)\n        Jacobian of f with respect to y computed at the mesh nodes.\n    df_dy_middle : ndarray, shape (n, n, m - 1)\n        Jacobian of f with respect to y computed at the middle between the\n        mesh nodes.\n    df_dp : ndarray with shape (n, k, m) or None\n        Jacobian of f with respect to p computed at the mesh nodes.\n    df_dp_middle : ndarray with shape (n, k, m - 1) or None\n        Jacobian of f with respect to p computed at the middle between the\n        mesh nodes.\n    dbc_dya, dbc_dyb : ndarray, shape (n, n)\n        Jacobian of bc with respect to ya and yb.\n    dbc_dp : ndarray with shape (n, k) or None\n        Jacobian of bc with respect to p.\n\n    Returns\n    -------\n    J : csc_matrix, shape (n * m + k, n * m + k)\n        Jacobian of the collocation system in a sparse form.\n\n    References\n    ----------\n    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual\n       Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,\n       Number 3, pp. 299-316, 2001.\n    '
    df_dy = np.transpose(df_dy, (2, 0, 1))
    df_dy_middle = np.transpose(df_dy_middle, (2, 0, 1))
    h = h[:, np.newaxis, np.newaxis]
    dtype = df_dy.dtype
    dPhi_dy_0 = np.empty((m - 1, n, n), dtype=dtype)
    dPhi_dy_0[:] = -np.identity(n)
    dPhi_dy_0 -= h / 6 * (df_dy[:-1] + 2 * df_dy_middle)
    T = stacked_matmul(df_dy_middle, df_dy[:-1])
    dPhi_dy_0 -= h ** 2 / 12 * T
    dPhi_dy_1 = np.empty((m - 1, n, n), dtype=dtype)
    dPhi_dy_1[:] = np.identity(n)
    dPhi_dy_1 -= h / 6 * (df_dy[1:] + 2 * df_dy_middle)
    T = stacked_matmul(df_dy_middle, df_dy[1:])
    dPhi_dy_1 += h ** 2 / 12 * T
    values = np.hstack((dPhi_dy_0.ravel(), dPhi_dy_1.ravel(), dbc_dya.ravel(), dbc_dyb.ravel()))
    if k > 0:
        df_dp = np.transpose(df_dp, (2, 0, 1))
        df_dp_middle = np.transpose(df_dp_middle, (2, 0, 1))
        T = stacked_matmul(df_dy_middle, df_dp[:-1] - df_dp[1:])
        df_dp_middle += 0.125 * h * T
        dPhi_dp = -h / 6 * (df_dp[:-1] + df_dp[1:] + 4 * df_dp_middle)
        values = np.hstack((values, dPhi_dp.ravel(), dbc_dp.ravel()))
    J = coo_matrix((values, (i_jac, j_jac)))
    return csc_matrix(J)

def collocation_fun(fun, y, p, x, h):
    if False:
        while True:
            i = 10
    'Evaluate collocation residuals.\n\n    This function lies in the core of the method. The solution is sought\n    as a cubic C1 continuous spline with derivatives matching the ODE rhs\n    at given nodes `x`. Collocation conditions are formed from the equality\n    of the spline derivatives and rhs of the ODE system in the middle points\n    between nodes.\n\n    Such method is classified to Lobbato IIIA family in ODE literature.\n    Refer to [1]_ for the formula and some discussion.\n\n    Returns\n    -------\n    col_res : ndarray, shape (n, m - 1)\n        Collocation residuals at the middle points of the mesh intervals.\n    y_middle : ndarray, shape (n, m - 1)\n        Values of the cubic spline evaluated at the middle points of the mesh\n        intervals.\n    f : ndarray, shape (n, m)\n        RHS of the ODE system evaluated at the mesh nodes.\n    f_middle : ndarray, shape (n, m - 1)\n        RHS of the ODE system evaluated at the middle points of the mesh\n        intervals (and using `y_middle`).\n\n    References\n    ----------\n    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual\n           Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,\n           Number 3, pp. 299-316, 2001.\n    '
    f = fun(x, y, p)
    y_middle = 0.5 * (y[:, 1:] + y[:, :-1]) - 0.125 * h * (f[:, 1:] - f[:, :-1])
    f_middle = fun(x[:-1] + 0.5 * h, y_middle, p)
    col_res = y[:, 1:] - y[:, :-1] - h / 6 * (f[:, :-1] + f[:, 1:] + 4 * f_middle)
    return (col_res, y_middle, f, f_middle)

def prepare_sys(n, m, k, fun, bc, fun_jac, bc_jac, x, h):
    if False:
        for i in range(10):
            print('nop')
    'Create the function and the Jacobian for the collocation system.'
    x_middle = x[:-1] + 0.5 * h
    (i_jac, j_jac) = compute_jac_indices(n, m, k)

    def col_fun(y, p):
        if False:
            while True:
                i = 10
        return collocation_fun(fun, y, p, x, h)

    def sys_jac(y, p, y_middle, f, f_middle, bc0):
        if False:
            for i in range(10):
                print('nop')
        if fun_jac is None:
            (df_dy, df_dp) = estimate_fun_jac(fun, x, y, p, f)
            (df_dy_middle, df_dp_middle) = estimate_fun_jac(fun, x_middle, y_middle, p, f_middle)
        else:
            (df_dy, df_dp) = fun_jac(x, y, p)
            (df_dy_middle, df_dp_middle) = fun_jac(x_middle, y_middle, p)
        if bc_jac is None:
            (dbc_dya, dbc_dyb, dbc_dp) = estimate_bc_jac(bc, y[:, 0], y[:, -1], p, bc0)
        else:
            (dbc_dya, dbc_dyb, dbc_dp) = bc_jac(y[:, 0], y[:, -1], p)
        return construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle, df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp)
    return (col_fun, sys_jac)

def solve_newton(n, m, h, col_fun, bc, jac, y, p, B, bvp_tol, bc_tol):
    if False:
        while True:
            i = 10
    'Solve the nonlinear collocation system by a Newton method.\n\n    This is a simple Newton method with a backtracking line search. As\n    advised in [1]_, an affine-invariant criterion function F = ||J^-1 r||^2\n    is used, where J is the Jacobian matrix at the current iteration and r is\n    the vector or collocation residuals (values of the system lhs).\n\n    The method alters between full Newton iterations and the fixed-Jacobian\n    iterations based\n\n    There are other tricks proposed in [1]_, but they are not used as they\n    don\'t seem to improve anything significantly, and even break the\n    convergence on some test problems I tried.\n\n    All important parameters of the algorithm are defined inside the function.\n\n    Parameters\n    ----------\n    n : int\n        Number of equations in the ODE system.\n    m : int\n        Number of nodes in the mesh.\n    h : ndarray, shape (m-1,)\n        Mesh intervals.\n    col_fun : callable\n        Function computing collocation residuals.\n    bc : callable\n        Function computing boundary condition residuals.\n    jac : callable\n        Function computing the Jacobian of the whole system (including\n        collocation and boundary condition residuals). It is supposed to\n        return csc_matrix.\n    y : ndarray, shape (n, m)\n        Initial guess for the function values at the mesh nodes.\n    p : ndarray, shape (k,)\n        Initial guess for the unknown parameters.\n    B : ndarray with shape (n, n) or None\n        Matrix to force the S y(a) = 0 condition for a problems with the\n        singular term. If None, the singular term is assumed to be absent.\n    bvp_tol : float\n        Tolerance to which we want to solve a BVP.\n    bc_tol : float\n        Tolerance to which we want to satisfy the boundary conditions.\n\n    Returns\n    -------\n    y : ndarray, shape (n, m)\n        Final iterate for the function values at the mesh nodes.\n    p : ndarray, shape (k,)\n        Final iterate for the unknown parameters.\n    singular : bool\n        True, if the LU decomposition failed because Jacobian turned out\n        to be singular.\n\n    References\n    ----------\n    .. [1]  U. Ascher, R. Mattheij and R. Russell "Numerical Solution of\n       Boundary Value Problems for Ordinary Differential Equations"\n    '
    tol_r = 2 / 3 * h * 0.05 * bvp_tol
    max_njev = 4
    max_iter = 8
    sigma = 0.2
    tau = 0.5
    n_trial = 4
    (col_res, y_middle, f, f_middle) = col_fun(y, p)
    bc_res = bc(y[:, 0], y[:, -1], p)
    res = np.hstack((col_res.ravel(order='F'), bc_res))
    njev = 0
    singular = False
    recompute_jac = True
    for iteration in range(max_iter):
        if recompute_jac:
            J = jac(y, p, y_middle, f, f_middle, bc_res)
            njev += 1
            try:
                LU = splu(J)
            except RuntimeError:
                singular = True
                break
            step = LU.solve(res)
            cost = np.dot(step, step)
        y_step = step[:m * n].reshape((n, m), order='F')
        p_step = step[m * n:]
        alpha = 1
        for trial in range(n_trial + 1):
            y_new = y - alpha * y_step
            if B is not None:
                y_new[:, 0] = np.dot(B, y_new[:, 0])
            p_new = p - alpha * p_step
            (col_res, y_middle, f, f_middle) = col_fun(y_new, p_new)
            bc_res = bc(y_new[:, 0], y_new[:, -1], p_new)
            res = np.hstack((col_res.ravel(order='F'), bc_res))
            step_new = LU.solve(res)
            cost_new = np.dot(step_new, step_new)
            if cost_new < (1 - 2 * alpha * sigma) * cost:
                break
            if trial < n_trial:
                alpha *= tau
        y = y_new
        p = p_new
        if njev == max_njev:
            break
        if np.all(np.abs(col_res) < tol_r * (1 + np.abs(f_middle))) and np.all(np.abs(bc_res) < bc_tol):
            break
        if alpha == 1:
            step = step_new
            cost = cost_new
            recompute_jac = False
        else:
            recompute_jac = True
    return (y, p, singular)

def print_iteration_header():
    if False:
        while True:
            i = 10
    print('{:^15}{:^15}{:^15}{:^15}{:^15}'.format('Iteration', 'Max residual', 'Max BC residual', 'Total nodes', 'Nodes added'))

def print_iteration_progress(iteration, residual, bc_residual, total_nodes, nodes_added):
    if False:
        i = 10
        return i + 15
    print('{:^15}{:^15.2e}{:^15.2e}{:^15}{:^15}'.format(iteration, residual, bc_residual, total_nodes, nodes_added))

class BVPResult(OptimizeResult):
    pass
TERMINATION_MESSAGES = {0: 'The algorithm converged to the desired accuracy.', 1: 'The maximum number of mesh nodes is exceeded.', 2: 'A singular Jacobian encountered when solving the collocation system.', 3: 'The solver was unable to satisfy boundary conditions tolerance on iteration 10.'}

def estimate_rms_residuals(fun, sol, x, h, p, r_middle, f_middle):
    if False:
        while True:
            i = 10
    'Estimate rms values of collocation residuals using Lobatto quadrature.\n\n    The residuals are defined as the difference between the derivatives of\n    our solution and rhs of the ODE system. We use relative residuals, i.e.,\n    normalized by 1 + np.abs(f). RMS values are computed as sqrt from the\n    normalized integrals of the squared relative residuals over each interval.\n    Integrals are estimated using 5-point Lobatto quadrature [1]_, we use the\n    fact that residuals at the mesh nodes are identically zero.\n\n    In [2] they don\'t normalize integrals by interval lengths, which gives\n    a higher rate of convergence of the residuals by the factor of h**0.5.\n    I chose to do such normalization for an ease of interpretation of return\n    values as RMS estimates.\n\n    Returns\n    -------\n    rms_res : ndarray, shape (m - 1,)\n        Estimated rms values of the relative residuals over each interval.\n\n    References\n    ----------\n    .. [1] http://mathworld.wolfram.com/LobattoQuadrature.html\n    .. [2] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual\n       Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,\n       Number 3, pp. 299-316, 2001.\n    '
    x_middle = x[:-1] + 0.5 * h
    s = 0.5 * h * (3 / 7) ** 0.5
    x1 = x_middle + s
    x2 = x_middle - s
    y1 = sol(x1)
    y2 = sol(x2)
    y1_prime = sol(x1, 1)
    y2_prime = sol(x2, 1)
    f1 = fun(x1, y1, p)
    f2 = fun(x2, y2, p)
    r1 = y1_prime - f1
    r2 = y2_prime - f2
    r_middle /= 1 + np.abs(f_middle)
    r1 /= 1 + np.abs(f1)
    r2 /= 1 + np.abs(f2)
    r1 = np.sum(np.real(r1 * np.conj(r1)), axis=0)
    r2 = np.sum(np.real(r2 * np.conj(r2)), axis=0)
    r_middle = np.sum(np.real(r_middle * np.conj(r_middle)), axis=0)
    return (0.5 * (32 / 45 * r_middle + 49 / 90 * (r1 + r2))) ** 0.5

def create_spline(y, yp, x, h):
    if False:
        for i in range(10):
            print('nop')
    'Create a cubic spline given values and derivatives.\n\n    Formulas for the coefficients are taken from interpolate.CubicSpline.\n\n    Returns\n    -------\n    sol : PPoly\n        Constructed spline as a PPoly instance.\n    '
    from scipy.interpolate import PPoly
    (n, m) = y.shape
    c = np.empty((4, n, m - 1), dtype=y.dtype)
    slope = (y[:, 1:] - y[:, :-1]) / h
    t = (yp[:, :-1] + yp[:, 1:] - 2 * slope) / h
    c[0] = t / h
    c[1] = (slope - yp[:, :-1]) / h - t
    c[2] = yp[:, :-1]
    c[3] = y[:, :-1]
    c = np.moveaxis(c, 1, 0)
    return PPoly(c, x, extrapolate=True, axis=1)

def modify_mesh(x, insert_1, insert_2):
    if False:
        print('Hello World!')
    'Insert nodes into a mesh.\n\n    Nodes removal logic is not established, its impact on the solver is\n    presumably negligible. So, only insertion is done in this function.\n\n    Parameters\n    ----------\n    x : ndarray, shape (m,)\n        Mesh nodes.\n    insert_1 : ndarray\n        Intervals to each insert 1 new node in the middle.\n    insert_2 : ndarray\n        Intervals to each insert 2 new nodes, such that divide an interval\n        into 3 equal parts.\n\n    Returns\n    -------\n    x_new : ndarray\n        New mesh nodes.\n\n    Notes\n    -----\n    `insert_1` and `insert_2` should not have common values.\n    '
    return np.sort(np.hstack((x, 0.5 * (x[insert_1] + x[insert_1 + 1]), (2 * x[insert_2] + x[insert_2 + 1]) / 3, (x[insert_2] + 2 * x[insert_2 + 1]) / 3)))

def wrap_functions(fun, bc, fun_jac, bc_jac, k, a, S, D, dtype):
    if False:
        while True:
            i = 10
    'Wrap functions for unified usage in the solver.'
    if fun_jac is None:
        fun_jac_wrapped = None
    if bc_jac is None:
        bc_jac_wrapped = None
    if k == 0:

        def fun_p(x, y, _):
            if False:
                i = 10
                return i + 15
            return np.asarray(fun(x, y), dtype)

        def bc_wrapped(ya, yb, _):
            if False:
                for i in range(10):
                    print('nop')
            return np.asarray(bc(ya, yb), dtype)
        if fun_jac is not None:

            def fun_jac_p(x, y, _):
                if False:
                    for i in range(10):
                        print('nop')
                return (np.asarray(fun_jac(x, y), dtype), None)
        if bc_jac is not None:

            def bc_jac_wrapped(ya, yb, _):
                if False:
                    print('Hello World!')
                (dbc_dya, dbc_dyb) = bc_jac(ya, yb)
                return (np.asarray(dbc_dya, dtype), np.asarray(dbc_dyb, dtype), None)
    else:

        def fun_p(x, y, p):
            if False:
                return 10
            return np.asarray(fun(x, y, p), dtype)

        def bc_wrapped(x, y, p):
            if False:
                for i in range(10):
                    print('nop')
            return np.asarray(bc(x, y, p), dtype)
        if fun_jac is not None:

            def fun_jac_p(x, y, p):
                if False:
                    for i in range(10):
                        print('nop')
                (df_dy, df_dp) = fun_jac(x, y, p)
                return (np.asarray(df_dy, dtype), np.asarray(df_dp, dtype))
        if bc_jac is not None:

            def bc_jac_wrapped(ya, yb, p):
                if False:
                    for i in range(10):
                        print('nop')
                (dbc_dya, dbc_dyb, dbc_dp) = bc_jac(ya, yb, p)
                return (np.asarray(dbc_dya, dtype), np.asarray(dbc_dyb, dtype), np.asarray(dbc_dp, dtype))
    if S is None:
        fun_wrapped = fun_p
    else:

        def fun_wrapped(x, y, p):
            if False:
                print('Hello World!')
            f = fun_p(x, y, p)
            if x[0] == a:
                f[:, 0] = np.dot(D, f[:, 0])
                f[:, 1:] += np.dot(S, y[:, 1:]) / (x[1:] - a)
            else:
                f += np.dot(S, y) / (x - a)
            return f
    if fun_jac is not None:
        if S is None:
            fun_jac_wrapped = fun_jac_p
        else:
            Sr = S[:, :, np.newaxis]

            def fun_jac_wrapped(x, y, p):
                if False:
                    i = 10
                    return i + 15
                (df_dy, df_dp) = fun_jac_p(x, y, p)
                if x[0] == a:
                    df_dy[:, :, 0] = np.dot(D, df_dy[:, :, 0])
                    df_dy[:, :, 1:] += Sr / (x[1:] - a)
                else:
                    df_dy += Sr / (x - a)
                return (df_dy, df_dp)
    return (fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped)

def solve_bvp(fun, bc, x, y, p=None, S=None, fun_jac=None, bc_jac=None, tol=0.001, max_nodes=1000, verbose=0, bc_tol=None):
    if False:
        return 10
    'Solve a boundary value problem for a system of ODEs.\n\n    This function numerically solves a first order system of ODEs subject to\n    two-point boundary conditions::\n\n        dy / dx = f(x, y, p) + S * y / (x - a), a <= x <= b\n        bc(y(a), y(b), p) = 0\n\n    Here x is a 1-D independent variable, y(x) is an N-D\n    vector-valued function and p is a k-D vector of unknown\n    parameters which is to be found along with y(x). For the problem to be\n    determined, there must be n + k boundary conditions, i.e., bc must be an\n    (n + k)-D function.\n\n    The last singular term on the right-hand side of the system is optional.\n    It is defined by an n-by-n matrix S, such that the solution must satisfy\n    S y(a) = 0. This condition will be forced during iterations, so it must not\n    contradict boundary conditions. See [2]_ for the explanation how this term\n    is handled when solving BVPs numerically.\n\n    Problems in a complex domain can be solved as well. In this case, y and p\n    are considered to be complex, and f and bc are assumed to be complex-valued\n    functions, but x stays real. Note that f and bc must be complex\n    differentiable (satisfy Cauchy-Riemann equations [4]_), otherwise you\n    should rewrite your problem for real and imaginary parts separately. To\n    solve a problem in a complex domain, pass an initial guess for y with a\n    complex data type (see below).\n\n    Parameters\n    ----------\n    fun : callable\n        Right-hand side of the system. The calling signature is ``fun(x, y)``,\n        or ``fun(x, y, p)`` if parameters are present. All arguments are\n        ndarray: ``x`` with shape (m,), ``y`` with shape (n, m), meaning that\n        ``y[:, i]`` corresponds to ``x[i]``, and ``p`` with shape (k,). The\n        return value must be an array with shape (n, m) and with the same\n        layout as ``y``.\n    bc : callable\n        Function evaluating residuals of the boundary conditions. The calling\n        signature is ``bc(ya, yb)``, or ``bc(ya, yb, p)`` if parameters are\n        present. All arguments are ndarray: ``ya`` and ``yb`` with shape (n,),\n        and ``p`` with shape (k,). The return value must be an array with\n        shape (n + k,).\n    x : array_like, shape (m,)\n        Initial mesh. Must be a strictly increasing sequence of real numbers\n        with ``x[0]=a`` and ``x[-1]=b``.\n    y : array_like, shape (n, m)\n        Initial guess for the function values at the mesh nodes, ith column\n        corresponds to ``x[i]``. For problems in a complex domain pass `y`\n        with a complex data type (even if the initial guess is purely real).\n    p : array_like with shape (k,) or None, optional\n        Initial guess for the unknown parameters. If None (default), it is\n        assumed that the problem doesn\'t depend on any parameters.\n    S : array_like with shape (n, n) or None\n        Matrix defining the singular term. If None (default), the problem is\n        solved without the singular term.\n    fun_jac : callable or None, optional\n        Function computing derivatives of f with respect to y and p. The\n        calling signature is ``fun_jac(x, y)``, or ``fun_jac(x, y, p)`` if\n        parameters are present. The return must contain 1 or 2 elements in the\n        following order:\n\n            * df_dy : array_like with shape (n, n, m), where an element\n              (i, j, q) equals to d f_i(x_q, y_q, p) / d (y_q)_j.\n            * df_dp : array_like with shape (n, k, m), where an element\n              (i, j, q) equals to d f_i(x_q, y_q, p) / d p_j.\n\n        Here q numbers nodes at which x and y are defined, whereas i and j\n        number vector components. If the problem is solved without unknown\n        parameters, df_dp should not be returned.\n\n        If `fun_jac` is None (default), the derivatives will be estimated\n        by the forward finite differences.\n    bc_jac : callable or None, optional\n        Function computing derivatives of bc with respect to ya, yb, and p.\n        The calling signature is ``bc_jac(ya, yb)``, or ``bc_jac(ya, yb, p)``\n        if parameters are present. The return must contain 2 or 3 elements in\n        the following order:\n\n            * dbc_dya : array_like with shape (n, n), where an element (i, j)\n              equals to d bc_i(ya, yb, p) / d ya_j.\n            * dbc_dyb : array_like with shape (n, n), where an element (i, j)\n              equals to d bc_i(ya, yb, p) / d yb_j.\n            * dbc_dp : array_like with shape (n, k), where an element (i, j)\n              equals to d bc_i(ya, yb, p) / d p_j.\n\n        If the problem is solved without unknown parameters, dbc_dp should not\n        be returned.\n\n        If `bc_jac` is None (default), the derivatives will be estimated by\n        the forward finite differences.\n    tol : float, optional\n        Desired tolerance of the solution. If we define ``r = y\' - f(x, y)``,\n        where y is the found solution, then the solver tries to achieve on each\n        mesh interval ``norm(r / (1 + abs(f)) < tol``, where ``norm`` is\n        estimated in a root mean squared sense (using a numerical quadrature\n        formula). Default is 1e-3.\n    max_nodes : int, optional\n        Maximum allowed number of the mesh nodes. If exceeded, the algorithm\n        terminates. Default is 1000.\n    verbose : {0, 1, 2}, optional\n        Level of algorithm\'s verbosity:\n\n            * 0 (default) : work silently.\n            * 1 : display a termination report.\n            * 2 : display progress during iterations.\n    bc_tol : float, optional\n        Desired absolute tolerance for the boundary condition residuals: `bc`\n        value should satisfy ``abs(bc) < bc_tol`` component-wise.\n        Equals to `tol` by default. Up to 10 iterations are allowed to achieve this\n        tolerance.\n\n    Returns\n    -------\n    Bunch object with the following fields defined:\n    sol : PPoly\n        Found solution for y as `scipy.interpolate.PPoly` instance, a C1\n        continuous cubic spline.\n    p : ndarray or None, shape (k,)\n        Found parameters. None, if the parameters were not present in the\n        problem.\n    x : ndarray, shape (m,)\n        Nodes of the final mesh.\n    y : ndarray, shape (n, m)\n        Solution values at the mesh nodes.\n    yp : ndarray, shape (n, m)\n        Solution derivatives at the mesh nodes.\n    rms_residuals : ndarray, shape (m - 1,)\n        RMS values of the relative residuals over each mesh interval (see the\n        description of `tol` parameter).\n    niter : int\n        Number of completed iterations.\n    status : int\n        Reason for algorithm termination:\n\n            * 0: The algorithm converged to the desired accuracy.\n            * 1: The maximum number of mesh nodes is exceeded.\n            * 2: A singular Jacobian encountered when solving the collocation\n              system.\n\n    message : string\n        Verbal description of the termination reason.\n    success : bool\n        True if the algorithm converged to the desired accuracy (``status=0``).\n\n    Notes\n    -----\n    This function implements a 4th order collocation algorithm with the\n    control of residuals similar to [1]_. A collocation system is solved\n    by a damped Newton method with an affine-invariant criterion function as\n    described in [3]_.\n\n    Note that in [1]_  integral residuals are defined without normalization\n    by interval lengths. So, their definition is different by a multiplier of\n    h**0.5 (h is an interval length) from the definition used here.\n\n    .. versionadded:: 0.18.0\n\n    References\n    ----------\n    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual\n           Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,\n           Number 3, pp. 299-316, 2001.\n    .. [2] L.F. Shampine, P. H. Muir and H. Xu, "A User-Friendly Fortran BVP\n           Solver".\n    .. [3] U. Ascher, R. Mattheij and R. Russell "Numerical Solution of\n           Boundary Value Problems for Ordinary Differential Equations".\n    .. [4] `Cauchy-Riemann equations\n            <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ on\n            Wikipedia.\n\n    Examples\n    --------\n    In the first example, we solve Bratu\'s problem::\n\n        y\'\' + k * exp(y) = 0\n        y(0) = y(1) = 0\n\n    for k = 1.\n\n    We rewrite the equation as a first-order system and implement its\n    right-hand side evaluation::\n\n        y1\' = y2\n        y2\' = -exp(y1)\n\n    >>> import numpy as np\n    >>> def fun(x, y):\n    ...     return np.vstack((y[1], -np.exp(y[0])))\n\n    Implement evaluation of the boundary condition residuals:\n\n    >>> def bc(ya, yb):\n    ...     return np.array([ya[0], yb[0]])\n\n    Define the initial mesh with 5 nodes:\n\n    >>> x = np.linspace(0, 1, 5)\n\n    This problem is known to have two solutions. To obtain both of them, we\n    use two different initial guesses for y. We denote them by subscripts\n    a and b.\n\n    >>> y_a = np.zeros((2, x.size))\n    >>> y_b = np.zeros((2, x.size))\n    >>> y_b[0] = 3\n\n    Now we are ready to run the solver.\n\n    >>> from scipy.integrate import solve_bvp\n    >>> res_a = solve_bvp(fun, bc, x, y_a)\n    >>> res_b = solve_bvp(fun, bc, x, y_b)\n\n    Let\'s plot the two found solutions. We take an advantage of having the\n    solution in a spline form to produce a smooth plot.\n\n    >>> x_plot = np.linspace(0, 1, 100)\n    >>> y_plot_a = res_a.sol(x_plot)[0]\n    >>> y_plot_b = res_b.sol(x_plot)[0]\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(x_plot, y_plot_a, label=\'y_a\')\n    >>> plt.plot(x_plot, y_plot_b, label=\'y_b\')\n    >>> plt.legend()\n    >>> plt.xlabel("x")\n    >>> plt.ylabel("y")\n    >>> plt.show()\n\n    We see that the two solutions have similar shape, but differ in scale\n    significantly.\n\n    In the second example, we solve a simple Sturm-Liouville problem::\n\n        y\'\' + k**2 * y = 0\n        y(0) = y(1) = 0\n\n    It is known that a non-trivial solution y = A * sin(k * x) is possible for\n    k = pi * n, where n is an integer. To establish the normalization constant\n    A = 1 we add a boundary condition::\n\n        y\'(0) = k\n\n    Again, we rewrite our equation as a first-order system and implement its\n    right-hand side evaluation::\n\n        y1\' = y2\n        y2\' = -k**2 * y1\n\n    >>> def fun(x, y, p):\n    ...     k = p[0]\n    ...     return np.vstack((y[1], -k**2 * y[0]))\n\n    Note that parameters p are passed as a vector (with one element in our\n    case).\n\n    Implement the boundary conditions:\n\n    >>> def bc(ya, yb, p):\n    ...     k = p[0]\n    ...     return np.array([ya[0], yb[0], ya[1] - k])\n\n    Set up the initial mesh and guess for y. We aim to find the solution for\n    k = 2 * pi, to achieve that we set values of y to approximately follow\n    sin(2 * pi * x):\n\n    >>> x = np.linspace(0, 1, 5)\n    >>> y = np.zeros((2, x.size))\n    >>> y[0, 1] = 1\n    >>> y[0, 3] = -1\n\n    Run the solver with 6 as an initial guess for k.\n\n    >>> sol = solve_bvp(fun, bc, x, y, p=[6])\n\n    We see that the found k is approximately correct:\n\n    >>> sol.p[0]\n    6.28329460046\n\n    And, finally, plot the solution to see the anticipated sinusoid:\n\n    >>> x_plot = np.linspace(0, 1, 100)\n    >>> y_plot = sol.sol(x_plot)[0]\n    >>> plt.plot(x_plot, y_plot)\n    >>> plt.xlabel("x")\n    >>> plt.ylabel("y")\n    >>> plt.show()\n    '
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError('`x` must be 1 dimensional.')
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError('`x` must be strictly increasing.')
    a = x[0]
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.complexfloating):
        dtype = complex
    else:
        dtype = float
    y = y.astype(dtype, copy=False)
    if y.ndim != 2:
        raise ValueError('`y` must be 2 dimensional.')
    if y.shape[1] != x.shape[0]:
        raise ValueError('`y` is expected to have {} columns, but actually has {}.'.format(x.shape[0], y.shape[1]))
    if p is None:
        p = np.array([])
    else:
        p = np.asarray(p, dtype=dtype)
    if p.ndim != 1:
        raise ValueError('`p` must be 1 dimensional.')
    if tol < 100 * EPS:
        warn(f'`tol` is too low, setting to {100 * EPS:.2e}')
        tol = 100 * EPS
    if verbose not in [0, 1, 2]:
        raise ValueError('`verbose` must be in [0, 1, 2].')
    n = y.shape[0]
    k = p.shape[0]
    if S is not None:
        S = np.asarray(S, dtype=dtype)
        if S.shape != (n, n):
            raise ValueError('`S` is expected to have shape {}, but actually has {}'.format((n, n), S.shape))
        B = np.identity(n) - np.dot(pinv(S), S)
        y[:, 0] = np.dot(B, y[:, 0])
        D = pinv(np.identity(n) - S)
    else:
        B = None
        D = None
    if bc_tol is None:
        bc_tol = tol
    max_iteration = 10
    (fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped) = wrap_functions(fun, bc, fun_jac, bc_jac, k, a, S, D, dtype)
    f = fun_wrapped(x, y, p)
    if f.shape != y.shape:
        raise ValueError('`fun` return is expected to have shape {}, but actually has {}.'.format(y.shape, f.shape))
    bc_res = bc_wrapped(y[:, 0], y[:, -1], p)
    if bc_res.shape != (n + k,):
        raise ValueError('`bc` return is expected to have shape {}, but actually has {}.'.format((n + k,), bc_res.shape))
    status = 0
    iteration = 0
    if verbose == 2:
        print_iteration_header()
    while True:
        m = x.shape[0]
        (col_fun, jac_sys) = prepare_sys(n, m, k, fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped, x, h)
        (y, p, singular) = solve_newton(n, m, h, col_fun, bc_wrapped, jac_sys, y, p, B, tol, bc_tol)
        iteration += 1
        (col_res, y_middle, f, f_middle) = collocation_fun(fun_wrapped, y, p, x, h)
        bc_res = bc_wrapped(y[:, 0], y[:, -1], p)
        max_bc_res = np.max(abs(bc_res))
        r_middle = 1.5 * col_res / h
        sol = create_spline(y, f, x, h)
        rms_res = estimate_rms_residuals(fun_wrapped, sol, x, h, p, r_middle, f_middle)
        max_rms_res = np.max(rms_res)
        if singular:
            status = 2
            break
        (insert_1,) = np.nonzero((rms_res > tol) & (rms_res < 100 * tol))
        (insert_2,) = np.nonzero(rms_res >= 100 * tol)
        nodes_added = insert_1.shape[0] + 2 * insert_2.shape[0]
        if m + nodes_added > max_nodes:
            status = 1
            if verbose == 2:
                nodes_added = f'({nodes_added})'
                print_iteration_progress(iteration, max_rms_res, max_bc_res, m, nodes_added)
            break
        if verbose == 2:
            print_iteration_progress(iteration, max_rms_res, max_bc_res, m, nodes_added)
        if nodes_added > 0:
            x = modify_mesh(x, insert_1, insert_2)
            h = np.diff(x)
            y = sol(x)
        elif max_bc_res <= bc_tol:
            status = 0
            break
        elif iteration >= max_iteration:
            status = 3
            break
    if verbose > 0:
        if status == 0:
            print('Solved in {} iterations, number of nodes {}. \nMaximum relative residual: {:.2e} \nMaximum boundary residual: {:.2e}'.format(iteration, x.shape[0], max_rms_res, max_bc_res))
        elif status == 1:
            print('Number of nodes is exceeded after iteration {}. \nMaximum relative residual: {:.2e} \nMaximum boundary residual: {:.2e}'.format(iteration, max_rms_res, max_bc_res))
        elif status == 2:
            print('Singular Jacobian encountered when solving the collocation system on iteration {}. \nMaximum relative residual: {:.2e} \nMaximum boundary residual: {:.2e}'.format(iteration, max_rms_res, max_bc_res))
        elif status == 3:
            print('The solver was unable to satisfy boundary conditions tolerance on iteration {}. \nMaximum relative residual: {:.2e} \nMaximum boundary residual: {:.2e}'.format(iteration, max_rms_res, max_bc_res))
    if p.size == 0:
        p = None
    return BVPResult(sol=sol, p=p, x=x, y=y, yp=f, rms_residuals=rms_res, niter=iteration, status=status, message=TERMINATION_MESSAGES[status], success=status == 0)