"""Generic interface for least-squares minimization."""
from warnings import warn
import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import _minpack, OptimizeResult
from scipy.optimize._numdiff import approx_derivative, group_columns
from scipy.optimize._minimize import Bounds
from .trf import trf
from .dogbox import dogbox
from .common import EPS, in_bounds, make_strictly_feasible
TERMINATION_MESSAGES = {-1: 'Improper input parameters status returned from `leastsq`', 0: 'The maximum number of function evaluations is exceeded.', 1: '`gtol` termination condition is satisfied.', 2: '`ftol` termination condition is satisfied.', 3: '`xtol` termination condition is satisfied.', 4: 'Both `ftol` and `xtol` termination conditions are satisfied.'}
FROM_MINPACK_TO_COMMON = {0: -1, 1: 2, 2: 3, 3: 4, 4: 1, 5: 0}

def call_minpack(fun, x0, jac, ftol, xtol, gtol, max_nfev, x_scale, diff_step):
    if False:
        return 10
    n = x0.size
    if diff_step is None:
        epsfcn = EPS
    else:
        epsfcn = diff_step ** 2
    if isinstance(x_scale, str) and x_scale == 'jac':
        diag = None
    else:
        diag = 1 / x_scale
    full_output = True
    col_deriv = False
    factor = 100.0
    if jac is None:
        if max_nfev is None:
            max_nfev = 100 * n * (n + 1)
        (x, info, status) = _minpack._lmdif(fun, x0, (), full_output, ftol, xtol, gtol, max_nfev, epsfcn, factor, diag)
    else:
        if max_nfev is None:
            max_nfev = 100 * n
        (x, info, status) = _minpack._lmder(fun, jac, x0, (), full_output, col_deriv, ftol, xtol, gtol, max_nfev, factor, diag)
    f = info['fvec']
    if callable(jac):
        J = jac(x)
    else:
        J = np.atleast_2d(approx_derivative(fun, x))
    cost = 0.5 * np.dot(f, f)
    g = J.T.dot(f)
    g_norm = norm(g, ord=np.inf)
    nfev = info['nfev']
    njev = info.get('njev', None)
    status = FROM_MINPACK_TO_COMMON[status]
    active_mask = np.zeros_like(x0, dtype=int)
    return OptimizeResult(x=x, cost=cost, fun=f, jac=J, grad=g, optimality=g_norm, active_mask=active_mask, nfev=nfev, njev=njev, status=status)

def prepare_bounds(bounds, n):
    if False:
        i = 10
        return i + 15
    (lb, ub) = (np.asarray(b, dtype=float) for b in bounds)
    if lb.ndim == 0:
        lb = np.resize(lb, n)
    if ub.ndim == 0:
        ub = np.resize(ub, n)
    return (lb, ub)

def check_tolerance(ftol, xtol, gtol, method):
    if False:
        while True:
            i = 10

    def check(tol, name):
        if False:
            return 10
        if tol is None:
            tol = 0
        elif tol < EPS:
            warn('Setting `{}` below the machine epsilon ({:.2e}) effectively disables the corresponding termination condition.'.format(name, EPS))
        return tol
    ftol = check(ftol, 'ftol')
    xtol = check(xtol, 'xtol')
    gtol = check(gtol, 'gtol')
    if method == 'lm' and (ftol < EPS or xtol < EPS or gtol < EPS):
        raise ValueError("All tolerances must be higher than machine epsilon ({:.2e}) for method 'lm'.".format(EPS))
    elif ftol < EPS and xtol < EPS and (gtol < EPS):
        raise ValueError('At least one of the tolerances must be higher than machine epsilon ({:.2e}).'.format(EPS))
    return (ftol, xtol, gtol)

def check_x_scale(x_scale, x0):
    if False:
        i = 10
        return i + 15
    if isinstance(x_scale, str) and x_scale == 'jac':
        return x_scale
    try:
        x_scale = np.asarray(x_scale, dtype=float)
        valid = np.all(np.isfinite(x_scale)) and np.all(x_scale > 0)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError("`x_scale` must be 'jac' or array_like with positive numbers.")
    if x_scale.ndim == 0:
        x_scale = np.resize(x_scale, x0.shape)
    if x_scale.shape != x0.shape:
        raise ValueError('Inconsistent shapes between `x_scale` and `x0`.')
    return x_scale

def check_jac_sparsity(jac_sparsity, m, n):
    if False:
        for i in range(10):
            print('nop')
    if jac_sparsity is None:
        return None
    if not issparse(jac_sparsity):
        jac_sparsity = np.atleast_2d(jac_sparsity)
    if jac_sparsity.shape != (m, n):
        raise ValueError('`jac_sparsity` has wrong shape.')
    return (jac_sparsity, group_columns(jac_sparsity))

def huber(z, rho, cost_only):
    if False:
        while True:
            i = 10
    mask = z <= 1
    rho[0, mask] = z[mask]
    rho[0, ~mask] = 2 * z[~mask] ** 0.5 - 1
    if cost_only:
        return
    rho[1, mask] = 1
    rho[1, ~mask] = z[~mask] ** (-0.5)
    rho[2, mask] = 0
    rho[2, ~mask] = -0.5 * z[~mask] ** (-1.5)

def soft_l1(z, rho, cost_only):
    if False:
        for i in range(10):
            print('nop')
    t = 1 + z
    rho[0] = 2 * (t ** 0.5 - 1)
    if cost_only:
        return
    rho[1] = t ** (-0.5)
    rho[2] = -0.5 * t ** (-1.5)

def cauchy(z, rho, cost_only):
    if False:
        print('Hello World!')
    rho[0] = np.log1p(z)
    if cost_only:
        return
    t = 1 + z
    rho[1] = 1 / t
    rho[2] = -1 / t ** 2

def arctan(z, rho, cost_only):
    if False:
        i = 10
        return i + 15
    rho[0] = np.arctan(z)
    if cost_only:
        return
    t = 1 + z ** 2
    rho[1] = 1 / t
    rho[2] = -2 * z / t ** 2
IMPLEMENTED_LOSSES = dict(linear=None, huber=huber, soft_l1=soft_l1, cauchy=cauchy, arctan=arctan)

def construct_loss_function(m, loss, f_scale):
    if False:
        i = 10
        return i + 15
    if loss == 'linear':
        return None
    if not callable(loss):
        loss = IMPLEMENTED_LOSSES[loss]
        rho = np.empty((3, m))

        def loss_function(f, cost_only=False):
            if False:
                for i in range(10):
                    print('nop')
            z = (f / f_scale) ** 2
            loss(z, rho, cost_only=cost_only)
            if cost_only:
                return 0.5 * f_scale ** 2 * np.sum(rho[0])
            rho[0] *= f_scale ** 2
            rho[2] /= f_scale ** 2
            return rho
    else:

        def loss_function(f, cost_only=False):
            if False:
                return 10
            z = (f / f_scale) ** 2
            rho = loss(z)
            if cost_only:
                return 0.5 * f_scale ** 2 * np.sum(rho[0])
            rho[0] *= f_scale ** 2
            rho[2] /= f_scale ** 2
            return rho
    return loss_function

def least_squares(fun, x0, jac='2-point', bounds=(-np.inf, np.inf), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs={}):
    if False:
        for i in range(10):
            print('nop')
    'Solve a nonlinear least-squares problem with bounds on the variables.\n\n    Given the residuals f(x) (an m-D real function of n real\n    variables) and the loss function rho(s) (a scalar function), `least_squares`\n    finds a local minimum of the cost function F(x)::\n\n        minimize F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m - 1)\n        subject to lb <= x <= ub\n\n    The purpose of the loss function rho(s) is to reduce the influence of\n    outliers on the solution.\n\n    Parameters\n    ----------\n    fun : callable\n        Function which computes the vector of residuals, with the signature\n        ``fun(x, *args, **kwargs)``, i.e., the minimization proceeds with\n        respect to its first argument. The argument ``x`` passed to this\n        function is an ndarray of shape (n,) (never a scalar, even for n=1).\n        It must allocate and return a 1-D array_like of shape (m,) or a scalar.\n        If the argument ``x`` is complex or the function ``fun`` returns\n        complex residuals, it must be wrapped in a real function of real\n        arguments, as shown at the end of the Examples section.\n    x0 : array_like with shape (n,) or float\n        Initial guess on independent variables. If float, it will be treated\n        as a 1-D array with one element. When `method` is \'trf\', the initial\n        guess might be slightly adjusted to lie sufficiently within the given\n        `bounds`.\n    jac : {\'2-point\', \'3-point\', \'cs\', callable}, optional\n        Method of computing the Jacobian matrix (an m-by-n matrix, where\n        element (i, j) is the partial derivative of f[i] with respect to\n        x[j]). The keywords select a finite difference scheme for numerical\n        estimation. The scheme \'3-point\' is more accurate, but requires\n        twice as many operations as \'2-point\' (default). The scheme \'cs\'\n        uses complex steps, and while potentially the most accurate, it is\n        applicable only when `fun` correctly handles complex inputs and\n        can be analytically continued to the complex plane. Method \'lm\'\n        always uses the \'2-point\' scheme. If callable, it is used as\n        ``jac(x, *args, **kwargs)`` and should return a good approximation\n        (or the exact value) for the Jacobian as an array_like (np.atleast_2d\n        is applied), a sparse matrix (csr_matrix preferred for performance) or\n        a `scipy.sparse.linalg.LinearOperator`.\n    bounds : 2-tuple of array_like or `Bounds`, optional\n        There are two ways to specify bounds:\n\n            1. Instance of `Bounds` class\n            2. Lower and upper bounds on independent variables. Defaults to no\n               bounds. Each array must match the size of `x0` or be a scalar,\n               in the latter case a bound will be the same for all variables.\n               Use ``np.inf`` with an appropriate sign to disable bounds on all\n               or some variables.\n    method : {\'trf\', \'dogbox\', \'lm\'}, optional\n        Algorithm to perform minimization.\n\n            * \'trf\' : Trust Region Reflective algorithm, particularly suitable\n              for large sparse problems with bounds. Generally robust method.\n            * \'dogbox\' : dogleg algorithm with rectangular trust regions,\n              typical use case is small problems with bounds. Not recommended\n              for problems with rank-deficient Jacobian.\n            * \'lm\' : Levenberg-Marquardt algorithm as implemented in MINPACK.\n              Doesn\'t handle bounds and sparse Jacobians. Usually the most\n              efficient method for small unconstrained problems.\n\n        Default is \'trf\'. See Notes for more information.\n    ftol : float or None, optional\n        Tolerance for termination by the change of the cost function. Default\n        is 1e-8. The optimization process is stopped when ``dF < ftol * F``,\n        and there was an adequate agreement between a local quadratic model and\n        the true model in the last step.\n\n        If None and \'method\' is not \'lm\', the termination by this condition is\n        disabled. If \'method\' is \'lm\', this tolerance must be higher than\n        machine epsilon.\n    xtol : float or None, optional\n        Tolerance for termination by the change of the independent variables.\n        Default is 1e-8. The exact condition depends on the `method` used:\n\n            * For \'trf\' and \'dogbox\' : ``norm(dx) < xtol * (xtol + norm(x))``.\n            * For \'lm\' : ``Delta < xtol * norm(xs)``, where ``Delta`` is\n              a trust-region radius and ``xs`` is the value of ``x``\n              scaled according to `x_scale` parameter (see below).\n\n        If None and \'method\' is not \'lm\', the termination by this condition is\n        disabled. If \'method\' is \'lm\', this tolerance must be higher than\n        machine epsilon.\n    gtol : float or None, optional\n        Tolerance for termination by the norm of the gradient. Default is 1e-8.\n        The exact condition depends on a `method` used:\n\n            * For \'trf\' : ``norm(g_scaled, ord=np.inf) < gtol``, where\n              ``g_scaled`` is the value of the gradient scaled to account for\n              the presence of the bounds [STIR]_.\n            * For \'dogbox\' : ``norm(g_free, ord=np.inf) < gtol``, where\n              ``g_free`` is the gradient with respect to the variables which\n              are not in the optimal state on the boundary.\n            * For \'lm\' : the maximum absolute value of the cosine of angles\n              between columns of the Jacobian and the residual vector is less\n              than `gtol`, or the residual vector is zero.\n\n        If None and \'method\' is not \'lm\', the termination by this condition is\n        disabled. If \'method\' is \'lm\', this tolerance must be higher than\n        machine epsilon.\n    x_scale : array_like or \'jac\', optional\n        Characteristic scale of each variable. Setting `x_scale` is equivalent\n        to reformulating the problem in scaled variables ``xs = x / x_scale``.\n        An alternative view is that the size of a trust region along jth\n        dimension is proportional to ``x_scale[j]``. Improved convergence may\n        be achieved by setting `x_scale` such that a step of a given size\n        along any of the scaled variables has a similar effect on the cost\n        function. If set to \'jac\', the scale is iteratively updated using the\n        inverse norms of the columns of the Jacobian matrix (as described in\n        [JJMore]_).\n    loss : str or callable, optional\n        Determines the loss function. The following keyword values are allowed:\n\n            * \'linear\' (default) : ``rho(z) = z``. Gives a standard\n              least-squares problem.\n            * \'soft_l1\' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth\n              approximation of l1 (absolute value) loss. Usually a good\n              choice for robust least squares.\n            * \'huber\' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works\n              similarly to \'soft_l1\'.\n            * \'cauchy\' : ``rho(z) = ln(1 + z)``. Severely weakens outliers\n              influence, but may cause difficulties in optimization process.\n            * \'arctan\' : ``rho(z) = arctan(z)``. Limits a maximum loss on\n              a single residual, has properties similar to \'cauchy\'.\n\n        If callable, it must take a 1-D ndarray ``z=f**2`` and return an\n        array_like with shape (3, m) where row 0 contains function values,\n        row 1 contains first derivatives and row 2 contains second\n        derivatives. Method \'lm\' supports only \'linear\' loss.\n    f_scale : float, optional\n        Value of soft margin between inlier and outlier residuals, default\n        is 1.0. The loss function is evaluated as follows\n        ``rho_(f**2) = C**2 * rho(f**2 / C**2)``, where ``C`` is `f_scale`,\n        and ``rho`` is determined by `loss` parameter. This parameter has\n        no effect with ``loss=\'linear\'``, but for other `loss` values it is\n        of crucial importance.\n    max_nfev : None or int, optional\n        Maximum number of function evaluations before the termination.\n        If None (default), the value is chosen automatically:\n\n            * For \'trf\' and \'dogbox\' : 100 * n.\n            * For \'lm\' :  100 * n if `jac` is callable and 100 * n * (n + 1)\n              otherwise (because \'lm\' counts function calls in Jacobian\n              estimation).\n\n    diff_step : None or array_like, optional\n        Determines the relative step size for the finite difference\n        approximation of the Jacobian. The actual step is computed as\n        ``x * diff_step``. If None (default), then `diff_step` is taken to be\n        a conventional "optimal" power of machine epsilon for the finite\n        difference scheme used [NR]_.\n    tr_solver : {None, \'exact\', \'lsmr\'}, optional\n        Method for solving trust-region subproblems, relevant only for \'trf\'\n        and \'dogbox\' methods.\n\n            * \'exact\' is suitable for not very large problems with dense\n              Jacobian matrices. The computational complexity per iteration is\n              comparable to a singular value decomposition of the Jacobian\n              matrix.\n            * \'lsmr\' is suitable for problems with sparse and large Jacobian\n              matrices. It uses the iterative procedure\n              `scipy.sparse.linalg.lsmr` for finding a solution of a linear\n              least-squares problem and only requires matrix-vector product\n              evaluations.\n\n        If None (default), the solver is chosen based on the type of Jacobian\n        returned on the first iteration.\n    tr_options : dict, optional\n        Keyword options passed to trust-region solver.\n\n            * ``tr_solver=\'exact\'``: `tr_options` are ignored.\n            * ``tr_solver=\'lsmr\'``: options for `scipy.sparse.linalg.lsmr`.\n              Additionally,  ``method=\'trf\'`` supports  \'regularize\' option\n              (bool, default is True), which adds a regularization term to the\n              normal equation, which improves convergence if the Jacobian is\n              rank-deficient [Byrd]_ (eq. 3.4).\n\n    jac_sparsity : {None, array_like, sparse matrix}, optional\n        Defines the sparsity structure of the Jacobian matrix for finite\n        difference estimation, its shape must be (m, n). If the Jacobian has\n        only few non-zero elements in *each* row, providing the sparsity\n        structure will greatly speed up the computations [Curtis]_. A zero\n        entry means that a corresponding element in the Jacobian is identically\n        zero. If provided, forces the use of \'lsmr\' trust-region solver.\n        If None (default), then dense differencing will be used. Has no effect\n        for \'lm\' method.\n    verbose : {0, 1, 2}, optional\n        Level of algorithm\'s verbosity:\n\n            * 0 (default) : work silently.\n            * 1 : display a termination report.\n            * 2 : display progress during iterations (not supported by \'lm\'\n              method).\n\n    args, kwargs : tuple and dict, optional\n        Additional arguments passed to `fun` and `jac`. Both empty by default.\n        The calling signature is ``fun(x, *args, **kwargs)`` and the same for\n        `jac`.\n\n    Returns\n    -------\n    result : OptimizeResult\n        `OptimizeResult` with the following fields defined:\n\n            x : ndarray, shape (n,)\n                Solution found.\n            cost : float\n                Value of the cost function at the solution.\n            fun : ndarray, shape (m,)\n                Vector of residuals at the solution.\n            jac : ndarray, sparse matrix or LinearOperator, shape (m, n)\n                Modified Jacobian matrix at the solution, in the sense that J^T J\n                is a Gauss-Newton approximation of the Hessian of the cost function.\n                The type is the same as the one used by the algorithm.\n            grad : ndarray, shape (m,)\n                Gradient of the cost function at the solution.\n            optimality : float\n                First-order optimality measure. In unconstrained problems, it is\n                always the uniform norm of the gradient. In constrained problems,\n                it is the quantity which was compared with `gtol` during iterations.\n            active_mask : ndarray of int, shape (n,)\n                Each component shows whether a corresponding constraint is active\n                (that is, whether a variable is at the bound):\n\n                    *  0 : a constraint is not active.\n                    * -1 : a lower bound is active.\n                    *  1 : an upper bound is active.\n\n                Might be somewhat arbitrary for \'trf\' method as it generates a\n                sequence of strictly feasible iterates and `active_mask` is\n                determined within a tolerance threshold.\n            nfev : int\n                Number of function evaluations done. Methods \'trf\' and \'dogbox\' do\n                not count function calls for numerical Jacobian approximation, as\n                opposed to \'lm\' method.\n            njev : int or None\n                Number of Jacobian evaluations done. If numerical Jacobian\n                approximation is used in \'lm\' method, it is set to None.\n            status : int\n                The reason for algorithm termination:\n\n                    * -1 : improper input parameters status returned from MINPACK.\n                    *  0 : the maximum number of function evaluations is exceeded.\n                    *  1 : `gtol` termination condition is satisfied.\n                    *  2 : `ftol` termination condition is satisfied.\n                    *  3 : `xtol` termination condition is satisfied.\n                    *  4 : Both `ftol` and `xtol` termination conditions are satisfied.\n\n            message : str\n                Verbal description of the termination reason.\n            success : bool\n                True if one of the convergence criteria is satisfied (`status` > 0).\n\n    See Also\n    --------\n    leastsq : A legacy wrapper for the MINPACK implementation of the\n              Levenberg-Marquadt algorithm.\n    curve_fit : Least-squares minimization applied to a curve-fitting problem.\n\n    Notes\n    -----\n    Method \'lm\' (Levenberg-Marquardt) calls a wrapper over least-squares\n    algorithms implemented in MINPACK (lmder, lmdif). It runs the\n    Levenberg-Marquardt algorithm formulated as a trust-region type algorithm.\n    The implementation is based on paper [JJMore]_, it is very robust and\n    efficient with a lot of smart tricks. It should be your first choice\n    for unconstrained problems. Note that it doesn\'t support bounds. Also,\n    it doesn\'t work when m < n.\n\n    Method \'trf\' (Trust Region Reflective) is motivated by the process of\n    solving a system of equations, which constitute the first-order optimality\n    condition for a bound-constrained minimization problem as formulated in\n    [STIR]_. The algorithm iteratively solves trust-region subproblems\n    augmented by a special diagonal quadratic term and with trust-region shape\n    determined by the distance from the bounds and the direction of the\n    gradient. This enhancements help to avoid making steps directly into bounds\n    and efficiently explore the whole space of variables. To further improve\n    convergence, the algorithm considers search directions reflected from the\n    bounds. To obey theoretical requirements, the algorithm keeps iterates\n    strictly feasible. With dense Jacobians trust-region subproblems are\n    solved by an exact method very similar to the one described in [JJMore]_\n    (and implemented in MINPACK). The difference from the MINPACK\n    implementation is that a singular value decomposition of a Jacobian\n    matrix is done once per iteration, instead of a QR decomposition and series\n    of Givens rotation eliminations. For large sparse Jacobians a 2-D subspace\n    approach of solving trust-region subproblems is used [STIR]_, [Byrd]_.\n    The subspace is spanned by a scaled gradient and an approximate\n    Gauss-Newton solution delivered by `scipy.sparse.linalg.lsmr`. When no\n    constraints are imposed the algorithm is very similar to MINPACK and has\n    generally comparable performance. The algorithm works quite robust in\n    unbounded and bounded problems, thus it is chosen as a default algorithm.\n\n    Method \'dogbox\' operates in a trust-region framework, but considers\n    rectangular trust regions as opposed to conventional ellipsoids [Voglis]_.\n    The intersection of a current trust region and initial bounds is again\n    rectangular, so on each iteration a quadratic minimization problem subject\n    to bound constraints is solved approximately by Powell\'s dogleg method\n    [NumOpt]_. The required Gauss-Newton step can be computed exactly for\n    dense Jacobians or approximately by `scipy.sparse.linalg.lsmr` for large\n    sparse Jacobians. The algorithm is likely to exhibit slow convergence when\n    the rank of Jacobian is less than the number of variables. The algorithm\n    often outperforms \'trf\' in bounded problems with a small number of\n    variables.\n\n    Robust loss functions are implemented as described in [BA]_. The idea\n    is to modify a residual vector and a Jacobian matrix on each iteration\n    such that computed gradient and Gauss-Newton Hessian approximation match\n    the true gradient and Hessian approximation of the cost function. Then\n    the algorithm proceeds in a normal way, i.e., robust loss functions are\n    implemented as a simple wrapper over standard least-squares algorithms.\n\n    .. versionadded:: 0.17.0\n\n    References\n    ----------\n    .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,\n              and Conjugate Gradient Method for Large-Scale Bound-Constrained\n              Minimization Problems," SIAM Journal on Scientific Computing,\n              Vol. 21, Number 1, pp 1-23, 1999.\n    .. [NR] William H. Press et. al., "Numerical Recipes. The Art of Scientific\n            Computing. 3rd edition", Sec. 5.7.\n    .. [Byrd] R. H. Byrd, R. B. Schnabel and G. A. Shultz, "Approximate\n              solution of the trust region problem by minimization over\n              two-dimensional subspaces", Math. Programming, 40, pp. 247-263,\n              1988.\n    .. [Curtis] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of\n                sparse Jacobian matrices", Journal of the Institute of\n                Mathematics and its Applications, 13, pp. 117-120, 1974.\n    .. [JJMore] J. J. More, "The Levenberg-Marquardt Algorithm: Implementation\n                and Theory," Numerical Analysis, ed. G. A. Watson, Lecture\n                Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.\n    .. [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region\n                Dogleg Approach for Unconstrained and Bound Constrained\n                Nonlinear Optimization", WSEAS International Conference on\n                Applied Mathematics, Corfu, Greece, 2004.\n    .. [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization,\n                2nd edition", Chapter 4.\n    .. [BA] B. Triggs et. al., "Bundle Adjustment - A Modern Synthesis",\n            Proceedings of the International Workshop on Vision Algorithms:\n            Theory and Practice, pp. 298-372, 1999.\n\n    Examples\n    --------\n    In this example we find a minimum of the Rosenbrock function without bounds\n    on independent variables.\n\n    >>> import numpy as np\n    >>> def fun_rosenbrock(x):\n    ...     return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])\n\n    Notice that we only provide the vector of the residuals. The algorithm\n    constructs the cost function as a sum of squares of the residuals, which\n    gives the Rosenbrock function. The exact minimum is at ``x = [1.0, 1.0]``.\n\n    >>> from scipy.optimize import least_squares\n    >>> x0_rosenbrock = np.array([2, 2])\n    >>> res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)\n    >>> res_1.x\n    array([ 1.,  1.])\n    >>> res_1.cost\n    9.8669242910846867e-30\n    >>> res_1.optimality\n    8.8928864934219529e-14\n\n    We now constrain the variables, in such a way that the previous solution\n    becomes infeasible. Specifically, we require that ``x[1] >= 1.5``, and\n    ``x[0]`` left unconstrained. To this end, we specify the `bounds` parameter\n    to `least_squares` in the form ``bounds=([-np.inf, 1.5], np.inf)``.\n\n    We also provide the analytic Jacobian:\n\n    >>> def jac_rosenbrock(x):\n    ...     return np.array([\n    ...         [-20 * x[0], 10],\n    ...         [-1, 0]])\n\n    Putting this all together, we see that the new solution lies on the bound:\n\n    >>> res_2 = least_squares(fun_rosenbrock, x0_rosenbrock, jac_rosenbrock,\n    ...                       bounds=([-np.inf, 1.5], np.inf))\n    >>> res_2.x\n    array([ 1.22437075,  1.5       ])\n    >>> res_2.cost\n    0.025213093946805685\n    >>> res_2.optimality\n    1.5885401433157753e-07\n\n    Now we solve a system of equations (i.e., the cost function should be zero\n    at a minimum) for a Broyden tridiagonal vector-valued function of 100000\n    variables:\n\n    >>> def fun_broyden(x):\n    ...     f = (3 - x) * x + 1\n    ...     f[1:] -= x[:-1]\n    ...     f[:-1] -= 2 * x[1:]\n    ...     return f\n\n    The corresponding Jacobian matrix is sparse. We tell the algorithm to\n    estimate it by finite differences and provide the sparsity structure of\n    Jacobian to significantly speed up this process.\n\n    >>> from scipy.sparse import lil_matrix\n    >>> def sparsity_broyden(n):\n    ...     sparsity = lil_matrix((n, n), dtype=int)\n    ...     i = np.arange(n)\n    ...     sparsity[i, i] = 1\n    ...     i = np.arange(1, n)\n    ...     sparsity[i, i - 1] = 1\n    ...     i = np.arange(n - 1)\n    ...     sparsity[i, i + 1] = 1\n    ...     return sparsity\n    ...\n    >>> n = 100000\n    >>> x0_broyden = -np.ones(n)\n    ...\n    >>> res_3 = least_squares(fun_broyden, x0_broyden,\n    ...                       jac_sparsity=sparsity_broyden(n))\n    >>> res_3.cost\n    4.5687069299604613e-23\n    >>> res_3.optimality\n    1.1650454296851518e-11\n\n    Let\'s also solve a curve fitting problem using robust loss function to\n    take care of outliers in the data. Define the model function as\n    ``y = a + b * exp(c * t)``, where t is a predictor variable, y is an\n    observation and a, b, c are parameters to estimate.\n\n    First, define the function which generates the data with noise and\n    outliers, define the model parameters, and generate data:\n\n    >>> from numpy.random import default_rng\n    >>> rng = default_rng()\n    >>> def gen_data(t, a, b, c, noise=0., n_outliers=0, seed=None):\n    ...     rng = default_rng(seed)\n    ...\n    ...     y = a + b * np.exp(t * c)\n    ...\n    ...     error = noise * rng.standard_normal(t.size)\n    ...     outliers = rng.integers(0, t.size, n_outliers)\n    ...     error[outliers] *= 10\n    ...\n    ...     return y + error\n    ...\n    >>> a = 0.5\n    >>> b = 2.0\n    >>> c = -1\n    >>> t_min = 0\n    >>> t_max = 10\n    >>> n_points = 15\n    ...\n    >>> t_train = np.linspace(t_min, t_max, n_points)\n    >>> y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)\n\n    Define function for computing residuals and initial estimate of\n    parameters.\n\n    >>> def fun(x, t, y):\n    ...     return x[0] + x[1] * np.exp(x[2] * t) - y\n    ...\n    >>> x0 = np.array([1.0, 1.0, 0.0])\n\n    Compute a standard least-squares solution:\n\n    >>> res_lsq = least_squares(fun, x0, args=(t_train, y_train))\n\n    Now compute two solutions with two different robust loss functions. The\n    parameter `f_scale` is set to 0.1, meaning that inlier residuals should\n    not significantly exceed 0.1 (the noise level used).\n\n    >>> res_soft_l1 = least_squares(fun, x0, loss=\'soft_l1\', f_scale=0.1,\n    ...                             args=(t_train, y_train))\n    >>> res_log = least_squares(fun, x0, loss=\'cauchy\', f_scale=0.1,\n    ...                         args=(t_train, y_train))\n\n    And, finally, plot all the curves. We see that by selecting an appropriate\n    `loss`  we can get estimates close to optimal even in the presence of\n    strong outliers. But keep in mind that generally it is recommended to try\n    \'soft_l1\' or \'huber\' losses first (if at all necessary) as the other two\n    options may cause difficulties in optimization process.\n\n    >>> t_test = np.linspace(t_min, t_max, n_points * 10)\n    >>> y_true = gen_data(t_test, a, b, c)\n    >>> y_lsq = gen_data(t_test, *res_lsq.x)\n    >>> y_soft_l1 = gen_data(t_test, *res_soft_l1.x)\n    >>> y_log = gen_data(t_test, *res_log.x)\n    ...\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(t_train, y_train, \'o\')\n    >>> plt.plot(t_test, y_true, \'k\', linewidth=2, label=\'true\')\n    >>> plt.plot(t_test, y_lsq, label=\'linear loss\')\n    >>> plt.plot(t_test, y_soft_l1, label=\'soft_l1 loss\')\n    >>> plt.plot(t_test, y_log, label=\'cauchy loss\')\n    >>> plt.xlabel("t")\n    >>> plt.ylabel("y")\n    >>> plt.legend()\n    >>> plt.show()\n\n    In the next example, we show how complex-valued residual functions of\n    complex variables can be optimized with ``least_squares()``. Consider the\n    following function:\n\n    >>> def f(z):\n    ...     return z - (0.5 + 0.5j)\n\n    We wrap it into a function of real variables that returns real residuals\n    by simply handling the real and imaginary parts as independent variables:\n\n    >>> def f_wrap(x):\n    ...     fx = f(x[0] + 1j*x[1])\n    ...     return np.array([fx.real, fx.imag])\n\n    Thus, instead of the original m-D complex function of n complex\n    variables we optimize a 2m-D real function of 2n real variables:\n\n    >>> from scipy.optimize import least_squares\n    >>> res_wrapped = least_squares(f_wrap, (0.1, 0.1), bounds=([0, 0], [1, 1]))\n    >>> z = res_wrapped.x[0] + res_wrapped.x[1]*1j\n    >>> z\n    (0.49999999999925893+0.49999999999925893j)\n\n    '
    if method not in ['trf', 'dogbox', 'lm']:
        raise ValueError("`method` must be 'trf', 'dogbox' or 'lm'.")
    if jac not in ['2-point', '3-point', 'cs'] and (not callable(jac)):
        raise ValueError("`jac` must be '2-point', '3-point', 'cs' or callable.")
    if tr_solver not in [None, 'exact', 'lsmr']:
        raise ValueError("`tr_solver` must be None, 'exact' or 'lsmr'.")
    if loss not in IMPLEMENTED_LOSSES and (not callable(loss)):
        raise ValueError('`loss` must be one of {} or a callable.'.format(IMPLEMENTED_LOSSES.keys()))
    if method == 'lm' and loss != 'linear':
        raise ValueError("method='lm' supports only 'linear' loss function.")
    if verbose not in [0, 1, 2]:
        raise ValueError('`verbose` must be in [0, 1, 2].')
    if max_nfev is not None and max_nfev <= 0:
        raise ValueError('`max_nfev` must be None or positive integer.')
    if np.iscomplexobj(x0):
        raise ValueError('`x0` must be real.')
    x0 = np.atleast_1d(x0).astype(float)
    if x0.ndim > 1:
        raise ValueError('`x0` must have at most 1 dimension.')
    if isinstance(bounds, Bounds):
        (lb, ub) = (bounds.lb, bounds.ub)
        bounds = (lb, ub)
    elif len(bounds) == 2:
        (lb, ub) = prepare_bounds(bounds, x0.shape[0])
    else:
        raise ValueError('`bounds` must contain 2 elements.')
    if method == 'lm' and (not np.all((lb == -np.inf) & (ub == np.inf))):
        raise ValueError("Method 'lm' doesn't support bounds.")
    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError('Inconsistent shapes between bounds and `x0`.')
    if np.any(lb >= ub):
        raise ValueError('Each lower bound must be strictly less than each upper bound.')
    if not in_bounds(x0, lb, ub):
        raise ValueError('`x0` is infeasible.')
    x_scale = check_x_scale(x_scale, x0)
    (ftol, xtol, gtol) = check_tolerance(ftol, xtol, gtol, method)
    if method == 'trf':
        x0 = make_strictly_feasible(x0, lb, ub)

    def fun_wrapped(x):
        if False:
            return 10
        return np.atleast_1d(fun(x, *args, **kwargs))
    f0 = fun_wrapped(x0)
    if f0.ndim != 1:
        raise ValueError('`fun` must return at most 1-d array_like. f0.shape: {}'.format(f0.shape))
    if not np.all(np.isfinite(f0)):
        raise ValueError('Residuals are not finite in the initial point.')
    n = x0.size
    m = f0.size
    if method == 'lm' and m < n:
        raise ValueError("Method 'lm' doesn't work when the number of residuals is less than the number of variables.")
    loss_function = construct_loss_function(m, loss, f_scale)
    if callable(loss):
        rho = loss_function(f0)
        if rho.shape != (3, m):
            raise ValueError('The return value of `loss` callable has wrong shape.')
        initial_cost = 0.5 * np.sum(rho[0])
    elif loss_function is not None:
        initial_cost = loss_function(f0, cost_only=True)
    else:
        initial_cost = 0.5 * np.dot(f0, f0)
    if callable(jac):
        J0 = jac(x0, *args, **kwargs)
        if issparse(J0):
            J0 = J0.tocsr()

            def jac_wrapped(x, _=None):
                if False:
                    return 10
                return jac(x, *args, **kwargs).tocsr()
        elif isinstance(J0, LinearOperator):

            def jac_wrapped(x, _=None):
                if False:
                    print('Hello World!')
                return jac(x, *args, **kwargs)
        else:
            J0 = np.atleast_2d(J0)

            def jac_wrapped(x, _=None):
                if False:
                    i = 10
                    return i + 15
                return np.atleast_2d(jac(x, *args, **kwargs))
    elif method == 'lm':
        if jac_sparsity is not None:
            raise ValueError("method='lm' does not support `jac_sparsity`.")
        if jac != '2-point':
            warn("jac='{}' works equivalently to '2-point' for method='lm'.".format(jac))
        J0 = jac_wrapped = None
    else:
        if jac_sparsity is not None and tr_solver == 'exact':
            raise ValueError("tr_solver='exact' is incompatible with `jac_sparsity`.")
        jac_sparsity = check_jac_sparsity(jac_sparsity, m, n)

        def jac_wrapped(x, f):
            if False:
                for i in range(10):
                    print('nop')
            J = approx_derivative(fun, x, rel_step=diff_step, method=jac, f0=f, bounds=bounds, args=args, kwargs=kwargs, sparsity=jac_sparsity)
            if J.ndim != 2:
                J = np.atleast_2d(J)
            return J
        J0 = jac_wrapped(x0, f0)
    if J0 is not None:
        if J0.shape != (m, n):
            raise ValueError('The return value of `jac` has wrong shape: expected {}, actual {}.'.format((m, n), J0.shape))
        if not isinstance(J0, np.ndarray):
            if method == 'lm':
                raise ValueError("method='lm' works only with dense Jacobian matrices.")
            if tr_solver == 'exact':
                raise ValueError("tr_solver='exact' works only with dense Jacobian matrices.")
        jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
        if isinstance(J0, LinearOperator) and jac_scale:
            raise ValueError("x_scale='jac' can't be used when `jac` returns LinearOperator.")
        if tr_solver is None:
            if isinstance(J0, np.ndarray):
                tr_solver = 'exact'
            else:
                tr_solver = 'lsmr'
    if method == 'lm':
        result = call_minpack(fun_wrapped, x0, jac_wrapped, ftol, xtol, gtol, max_nfev, x_scale, diff_step)
    elif method == 'trf':
        result = trf(fun_wrapped, jac_wrapped, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale, loss_function, tr_solver, tr_options.copy(), verbose)
    elif method == 'dogbox':
        if tr_solver == 'lsmr' and 'regularize' in tr_options:
            warn("The keyword 'regularize' in `tr_options` is not relevant for 'dogbox' method.")
            tr_options = tr_options.copy()
            del tr_options['regularize']
        result = dogbox(fun_wrapped, jac_wrapped, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale, loss_function, tr_solver, tr_options, verbose)
    result.message = TERMINATION_MESSAGES[result.status]
    result.success = result.status > 0
    if verbose >= 1:
        print(result.message)
        print('Function evaluations {}, initial cost {:.4e}, final cost {:.4e}, first-order optimality {:.2e}.'.format(result.nfev, initial_cost, result.cost, result.optimality))
    return result