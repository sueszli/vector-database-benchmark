"""Linear least squares with bound constraints on independent variables."""
import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.optimize import OptimizeResult
from scipy.optimize._minimize import Bounds
from .common import in_bounds, compute_grad
from .trf_linear import trf_linear
from .bvls import bvls

def prepare_bounds(bounds, n):
    if False:
        i = 10
        return i + 15
    if len(bounds) != 2:
        raise ValueError('`bounds` must contain 2 elements.')
    (lb, ub) = (np.asarray(b, dtype=float) for b in bounds)
    if lb.ndim == 0:
        lb = np.resize(lb, n)
    if ub.ndim == 0:
        ub = np.resize(ub, n)
    return (lb, ub)
TERMINATION_MESSAGES = {-1: 'The algorithm was not able to make progress on the last iteration.', 0: 'The maximum number of iterations is exceeded.', 1: 'The first-order optimality measure is less than `tol`.', 2: 'The relative change of the cost function is less than `tol`.', 3: 'The unconstrained solution is optimal.'}

def lsq_linear(A, b, bounds=(-np.inf, np.inf), method='trf', tol=1e-10, lsq_solver=None, lsmr_tol=None, max_iter=None, verbose=0, *, lsmr_maxiter=None):
    if False:
        for i in range(10):
            print('nop')
    'Solve a linear least-squares problem with bounds on the variables.\n\n    Given a m-by-n design matrix A and a target vector b with m elements,\n    `lsq_linear` solves the following optimization problem::\n\n        minimize 0.5 * ||A x - b||**2\n        subject to lb <= x <= ub\n\n    This optimization problem is convex, hence a found minimum (if iterations\n    have converged) is guaranteed to be global.\n\n    Parameters\n    ----------\n    A : array_like, sparse matrix of LinearOperator, shape (m, n)\n        Design matrix. Can be `scipy.sparse.linalg.LinearOperator`.\n    b : array_like, shape (m,)\n        Target vector.\n    bounds : 2-tuple of array_like or `Bounds`, optional\n        Lower and upper bounds on parameters. Defaults to no bounds.\n        There are two ways to specify the bounds:\n\n            - Instance of `Bounds` class.\n\n            - 2-tuple of array_like: Each element of the tuple must be either\n              an array with the length equal to the number of parameters, or a\n              scalar (in which case the bound is taken to be the same for all\n              parameters). Use ``np.inf`` with an appropriate sign to disable\n              bounds on all or some parameters.\n\n    method : \'trf\' or \'bvls\', optional\n        Method to perform minimization.\n\n            * \'trf\' : Trust Region Reflective algorithm adapted for a linear\n              least-squares problem. This is an interior-point-like method\n              and the required number of iterations is weakly correlated with\n              the number of variables.\n            * \'bvls\' : Bounded-variable least-squares algorithm. This is\n              an active set method, which requires the number of iterations\n              comparable to the number of variables. Can\'t be used when `A` is\n              sparse or LinearOperator.\n\n        Default is \'trf\'.\n    tol : float, optional\n        Tolerance parameter. The algorithm terminates if a relative change\n        of the cost function is less than `tol` on the last iteration.\n        Additionally, the first-order optimality measure is considered:\n\n            * ``method=\'trf\'`` terminates if the uniform norm of the gradient,\n              scaled to account for the presence of the bounds, is less than\n              `tol`.\n            * ``method=\'bvls\'`` terminates if Karush-Kuhn-Tucker conditions\n              are satisfied within `tol` tolerance.\n\n    lsq_solver : {None, \'exact\', \'lsmr\'}, optional\n        Method of solving unbounded least-squares problems throughout\n        iterations:\n\n            * \'exact\' : Use dense QR or SVD decomposition approach. Can\'t be\n              used when `A` is sparse or LinearOperator.\n            * \'lsmr\' : Use `scipy.sparse.linalg.lsmr` iterative procedure\n              which requires only matrix-vector product evaluations. Can\'t\n              be used with ``method=\'bvls\'``.\n\n        If None (default), the solver is chosen based on type of `A`.\n    lsmr_tol : None, float or \'auto\', optional\n        Tolerance parameters \'atol\' and \'btol\' for `scipy.sparse.linalg.lsmr`\n        If None (default), it is set to ``1e-2 * tol``. If \'auto\', the\n        tolerance will be adjusted based on the optimality of the current\n        iterate, which can speed up the optimization process, but is not always\n        reliable.\n    max_iter : None or int, optional\n        Maximum number of iterations before termination. If None (default), it\n        is set to 100 for ``method=\'trf\'`` or to the number of variables for\n        ``method=\'bvls\'`` (not counting iterations for \'bvls\' initialization).\n    verbose : {0, 1, 2}, optional\n        Level of algorithm\'s verbosity:\n\n            * 0 : work silently (default).\n            * 1 : display a termination report.\n            * 2 : display progress during iterations.\n    lsmr_maxiter : None or int, optional\n        Maximum number of iterations for the lsmr least squares solver,\n        if it is used (by setting ``lsq_solver=\'lsmr\'``). If None (default), it\n        uses lsmr\'s default of ``min(m, n)`` where ``m`` and ``n`` are the\n        number of rows and columns of `A`, respectively. Has no effect if\n        ``lsq_solver=\'exact\'``.\n\n    Returns\n    -------\n    OptimizeResult with the following fields defined:\n    x : ndarray, shape (n,)\n        Solution found.\n    cost : float\n        Value of the cost function at the solution.\n    fun : ndarray, shape (m,)\n        Vector of residuals at the solution.\n    optimality : float\n        First-order optimality measure. The exact meaning depends on `method`,\n        refer to the description of `tol` parameter.\n    active_mask : ndarray of int, shape (n,)\n        Each component shows whether a corresponding constraint is active\n        (that is, whether a variable is at the bound):\n\n            *  0 : a constraint is not active.\n            * -1 : a lower bound is active.\n            *  1 : an upper bound is active.\n\n        Might be somewhat arbitrary for the `trf` method as it generates a\n        sequence of strictly feasible iterates and active_mask is determined\n        within a tolerance threshold.\n    unbounded_sol : tuple\n        Unbounded least squares solution tuple returned by the least squares\n        solver (set with `lsq_solver` option). If `lsq_solver` is not set or is\n        set to ``\'exact\'``, the tuple contains an ndarray of shape (n,) with\n        the unbounded solution, an ndarray with the sum of squared residuals,\n        an int with the rank of `A`, and an ndarray with the singular values\n        of `A` (see NumPy\'s ``linalg.lstsq`` for more information). If\n        `lsq_solver` is set to ``\'lsmr\'``, the tuple contains an ndarray of\n        shape (n,) with the unbounded solution, an int with the exit code,\n        an int with the number of iterations, and five floats with\n        various norms and the condition number of `A` (see SciPy\'s\n        ``sparse.linalg.lsmr`` for more information). This output can be\n        useful for determining the convergence of the least squares solver,\n        particularly the iterative ``\'lsmr\'`` solver. The unbounded least\n        squares problem is to minimize ``0.5 * ||A x - b||**2``.\n    nit : int\n        Number of iterations. Zero if the unconstrained solution is optimal.\n    status : int\n        Reason for algorithm termination:\n\n            * -1 : the algorithm was not able to make progress on the last\n              iteration.\n            *  0 : the maximum number of iterations is exceeded.\n            *  1 : the first-order optimality measure is less than `tol`.\n            *  2 : the relative change of the cost function is less than `tol`.\n            *  3 : the unconstrained solution is optimal.\n\n    message : str\n        Verbal description of the termination reason.\n    success : bool\n        True if one of the convergence criteria is satisfied (`status` > 0).\n\n    See Also\n    --------\n    nnls : Linear least squares with non-negativity constraint.\n    least_squares : Nonlinear least squares with bounds on the variables.\n\n    Notes\n    -----\n    The algorithm first computes the unconstrained least-squares solution by\n    `numpy.linalg.lstsq` or `scipy.sparse.linalg.lsmr` depending on\n    `lsq_solver`. This solution is returned as optimal if it lies within the\n    bounds.\n\n    Method \'trf\' runs the adaptation of the algorithm described in [STIR]_ for\n    a linear least-squares problem. The iterations are essentially the same as\n    in the nonlinear least-squares algorithm, but as the quadratic function\n    model is always accurate, we don\'t need to track or modify the radius of\n    a trust region. The line search (backtracking) is used as a safety net\n    when a selected step does not decrease the cost function. Read more\n    detailed description of the algorithm in `scipy.optimize.least_squares`.\n\n    Method \'bvls\' runs a Python implementation of the algorithm described in\n    [BVLS]_. The algorithm maintains active and free sets of variables, on\n    each iteration chooses a new variable to move from the active set to the\n    free set and then solves the unconstrained least-squares problem on free\n    variables. This algorithm is guaranteed to give an accurate solution\n    eventually, but may require up to n iterations for a problem with n\n    variables. Additionally, an ad-hoc initialization procedure is\n    implemented, that determines which variables to set free or active\n    initially. It takes some number of iterations before actual BVLS starts,\n    but can significantly reduce the number of further iterations.\n\n    References\n    ----------\n    .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,\n              and Conjugate Gradient Method for Large-Scale Bound-Constrained\n              Minimization Problems," SIAM Journal on Scientific Computing,\n              Vol. 21, Number 1, pp 1-23, 1999.\n    .. [BVLS] P. B. Start and R. L. Parker, "Bounded-Variable Least-Squares:\n              an Algorithm and Applications", Computational Statistics, 10,\n              129-141, 1995.\n\n    Examples\n    --------\n    In this example, a problem with a large sparse matrix and bounds on the\n    variables is solved.\n\n    >>> import numpy as np\n    >>> from scipy.sparse import rand\n    >>> from scipy.optimize import lsq_linear\n    >>> rng = np.random.default_rng()\n    ...\n    >>> m = 20000\n    >>> n = 10000\n    ...\n    >>> A = rand(m, n, density=1e-4, random_state=rng)\n    >>> b = rng.standard_normal(m)\n    ...\n    >>> lb = rng.standard_normal(n)\n    >>> ub = lb + 1\n    ...\n    >>> res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol=\'auto\', verbose=1)\n    # may vary\n    The relative change of the cost function is less than `tol`.\n    Number of iterations 16, initial cost 1.5039e+04, final cost 1.1112e+04,\n    first-order optimality 4.66e-08.\n    '
    if method not in ['trf', 'bvls']:
        raise ValueError("`method` must be 'trf' or 'bvls'")
    if lsq_solver not in [None, 'exact', 'lsmr']:
        raise ValueError("`solver` must be None, 'exact' or 'lsmr'.")
    if verbose not in [0, 1, 2]:
        raise ValueError('`verbose` must be in [0, 1, 2].')
    if issparse(A):
        A = csr_matrix(A)
    elif not isinstance(A, LinearOperator):
        A = np.atleast_2d(np.asarray(A))
    if method == 'bvls':
        if lsq_solver == 'lsmr':
            raise ValueError("method='bvls' can't be used with lsq_solver='lsmr'")
        if not isinstance(A, np.ndarray):
            raise ValueError("method='bvls' can't be used with `A` being sparse or LinearOperator.")
    if lsq_solver is None:
        if isinstance(A, np.ndarray):
            lsq_solver = 'exact'
        else:
            lsq_solver = 'lsmr'
    elif lsq_solver == 'exact' and (not isinstance(A, np.ndarray)):
        raise ValueError("`exact` solver can't be used when `A` is sparse or LinearOperator.")
    if len(A.shape) != 2:
        raise ValueError('`A` must have at most 2 dimensions.')
    if max_iter is not None and max_iter <= 0:
        raise ValueError('`max_iter` must be None or positive integer.')
    (m, n) = A.shape
    b = np.atleast_1d(b)
    if b.ndim != 1:
        raise ValueError('`b` must have at most 1 dimension.')
    if b.size != m:
        raise ValueError('Inconsistent shapes between `A` and `b`.')
    if isinstance(bounds, Bounds):
        lb = bounds.lb
        ub = bounds.ub
    else:
        (lb, ub) = prepare_bounds(bounds, n)
    if lb.shape != (n,) and ub.shape != (n,):
        raise ValueError('Bounds have wrong shape.')
    if np.any(lb >= ub):
        raise ValueError('Each lower bound must be strictly less than each upper bound.')
    if lsmr_maxiter is not None and lsmr_maxiter < 1:
        raise ValueError('`lsmr_maxiter` must be None or positive integer.')
    if not (isinstance(lsmr_tol, float) and lsmr_tol > 0 or lsmr_tol in ('auto', None)):
        raise ValueError("`lsmr_tol` must be None, 'auto', or positive float.")
    if lsq_solver == 'exact':
        unbd_lsq = np.linalg.lstsq(A, b, rcond=-1)
    elif lsq_solver == 'lsmr':
        first_lsmr_tol = lsmr_tol
        if lsmr_tol is None or lsmr_tol == 'auto':
            first_lsmr_tol = 0.01 * tol
        unbd_lsq = lsmr(A, b, maxiter=lsmr_maxiter, atol=first_lsmr_tol, btol=first_lsmr_tol)
    x_lsq = unbd_lsq[0]
    if in_bounds(x_lsq, lb, ub):
        r = A @ x_lsq - b
        cost = 0.5 * np.dot(r, r)
        termination_status = 3
        termination_message = TERMINATION_MESSAGES[termination_status]
        g = compute_grad(A, r)
        g_norm = norm(g, ord=np.inf)
        if verbose > 0:
            print(termination_message)
            print('Final cost {:.4e}, first-order optimality {:.2e}'.format(cost, g_norm))
        return OptimizeResult(x=x_lsq, fun=r, cost=cost, optimality=g_norm, active_mask=np.zeros(n), unbounded_sol=unbd_lsq, nit=0, status=termination_status, message=termination_message, success=True)
    if method == 'trf':
        res = trf_linear(A, b, x_lsq, lb, ub, tol, lsq_solver, lsmr_tol, max_iter, verbose, lsmr_maxiter=lsmr_maxiter)
    elif method == 'bvls':
        res = bvls(A, b, x_lsq, lb, ub, tol, max_iter, verbose)
    res.unbounded_sol = unbd_lsq
    res.message = TERMINATION_MESSAGES[res.status]
    res.success = res.status > 0
    if verbose > 0:
        print(res.message)
        print('Number of iterations {}, initial cost {:.4e}, final cost {:.4e}, first-order optimality {:.2e}.'.format(res.nit, res.initial_cost, res.cost, res.optimality))
    del res.initial_cost
    return res