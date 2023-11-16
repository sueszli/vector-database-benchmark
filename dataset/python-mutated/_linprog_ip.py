"""Interior-point method for linear programming

The *interior-point* method uses the primal-dual path following algorithm
outlined in [1]_. This algorithm supports sparse constraint matrices and
is typically faster than the simplex methods, especially for large, sparse
problems. Note, however, that the solution returned may be slightly less
accurate than those of the simplex methods and will not, in general,
correspond with a vertex of the polytope defined by the constraints.

    .. versionadded:: 1.0.0

References
----------
.. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
       optimizer for linear programming: an implementation of the
       homogeneous algorithm." High performance optimization. Springer US,
       2000. 197-232.
"""
import numpy as np
import scipy as sp
import scipy.sparse as sps
from warnings import warn
from scipy.linalg import LinAlgError
from ._optimize import OptimizeWarning, OptimizeResult, _check_unknown_options
from ._linprog_util import _postsolve
has_umfpack = True
has_cholmod = True
try:
    import sksparse
    from sksparse.cholmod import cholesky as cholmod
    from sksparse.cholmod import analyze as cholmod_analyze
except ImportError:
    has_cholmod = False
try:
    import scikits.umfpack
except ImportError:
    has_umfpack = False

def _get_solver(M, sparse=False, lstsq=False, sym_pos=True, cholesky=True, permc_spec='MMD_AT_PLUS_A'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Given solver options, return a handle to the appropriate linear system\n    solver.\n\n    Parameters\n    ----------\n    M : 2-D array\n        As defined in [4] Equation 8.31\n    sparse : bool (default = False)\n        True if the system to be solved is sparse. This is typically set\n        True when the original ``A_ub`` and ``A_eq`` arrays are sparse.\n    lstsq : bool (default = False)\n        True if the system is ill-conditioned and/or (nearly) singular and\n        thus a more robust least-squares solver is desired. This is sometimes\n        needed as the solution is approached.\n    sym_pos : bool (default = True)\n        True if the system matrix is symmetric positive definite\n        Sometimes this needs to be set false as the solution is approached,\n        even when the system should be symmetric positive definite, due to\n        numerical difficulties.\n    cholesky : bool (default = True)\n        True if the system is to be solved by Cholesky, rather than LU,\n        decomposition. This is typically faster unless the problem is very\n        small or prone to numerical difficulties.\n    permc_spec : str (default = 'MMD_AT_PLUS_A')\n        Sparsity preservation strategy used by SuperLU. Acceptable values are:\n\n        - ``NATURAL``: natural ordering.\n        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.\n        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.\n        - ``COLAMD``: approximate minimum degree column ordering.\n\n        See SuperLU documentation.\n\n    Returns\n    -------\n    solve : function\n        Handle to the appropriate solver function\n\n    "
    try:
        if sparse:
            if lstsq:

                def solve(r, sym_pos=False):
                    if False:
                        i = 10
                        return i + 15
                    return sps.linalg.lsqr(M, r)[0]
            elif cholesky:
                try:
                    _get_solver.cholmod_factor.cholesky_inplace(M)
                except Exception:
                    _get_solver.cholmod_factor = cholmod_analyze(M)
                    _get_solver.cholmod_factor.cholesky_inplace(M)
                solve = _get_solver.cholmod_factor
            elif has_umfpack and sym_pos:
                solve = sps.linalg.factorized(M)
            else:
                solve = sps.linalg.splu(M, permc_spec=permc_spec).solve
        elif lstsq:

            def solve(r):
                if False:
                    return 10
                return sp.linalg.lstsq(M, r)[0]
        elif cholesky:
            L = sp.linalg.cho_factor(M)

            def solve(r):
                if False:
                    i = 10
                    return i + 15
                return sp.linalg.cho_solve(L, r)
        else:

            def solve(r, sym_pos=sym_pos):
                if False:
                    return 10
                if sym_pos:
                    return sp.linalg.solve(M, r, assume_a='pos')
                else:
                    return sp.linalg.solve(M, r)
    except KeyboardInterrupt:
        raise
    except Exception:
        return None
    return solve

def _get_delta(A, b, c, x, y, z, tau, kappa, gamma, eta, sparse=False, lstsq=False, sym_pos=True, cholesky=True, pc=True, ip=False, permc_spec='MMD_AT_PLUS_A'):
    if False:
        print('Hello World!')
    '\n    Given standard form problem defined by ``A``, ``b``, and ``c``;\n    current variable estimates ``x``, ``y``, ``z``, ``tau``, and ``kappa``;\n    algorithmic parameters ``gamma and ``eta;\n    and options ``sparse``, ``lstsq``, ``sym_pos``, ``cholesky``, ``pc``\n    (predictor-corrector), and ``ip`` (initial point improvement),\n    get the search direction for increments to the variable estimates.\n\n    Parameters\n    ----------\n    As defined in [4], except:\n    sparse : bool\n        True if the system to be solved is sparse. This is typically set\n        True when the original ``A_ub`` and ``A_eq`` arrays are sparse.\n    lstsq : bool\n        True if the system is ill-conditioned and/or (nearly) singular and\n        thus a more robust least-squares solver is desired. This is sometimes\n        needed as the solution is approached.\n    sym_pos : bool\n        True if the system matrix is symmetric positive definite\n        Sometimes this needs to be set false as the solution is approached,\n        even when the system should be symmetric positive definite, due to\n        numerical difficulties.\n    cholesky : bool\n        True if the system is to be solved by Cholesky, rather than LU,\n        decomposition. This is typically faster unless the problem is very\n        small or prone to numerical difficulties.\n    pc : bool\n        True if the predictor-corrector method of Mehrota is to be used. This\n        is almost always (if not always) beneficial. Even though it requires\n        the solution of an additional linear system, the factorization\n        is typically (implicitly) reused so solution is efficient, and the\n        number of algorithm iterations is typically reduced.\n    ip : bool\n        True if the improved initial point suggestion due to [4] section 4.3\n        is desired. It\'s unclear whether this is beneficial.\n    permc_spec : str (default = \'MMD_AT_PLUS_A\')\n        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =\n        True``.) A matrix is factorized in each iteration of the algorithm.\n        This option specifies how to permute the columns of the matrix for\n        sparsity preservation. Acceptable values are:\n\n        - ``NATURAL``: natural ordering.\n        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.\n        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.\n        - ``COLAMD``: approximate minimum degree column ordering.\n\n        This option can impact the convergence of the\n        interior point algorithm; test different values to determine which\n        performs best for your problem. For more information, refer to\n        ``scipy.sparse.linalg.splu``.\n\n    Returns\n    -------\n    Search directions as defined in [4]\n\n    References\n    ----------\n    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point\n           optimizer for linear programming: an implementation of the\n           homogeneous algorithm." High performance optimization. Springer US,\n           2000. 197-232.\n\n    '
    if A.shape[0] == 0:
        (sparse, lstsq, sym_pos, cholesky) = (False, False, True, False)
    n_x = len(x)
    r_P = b * tau - A.dot(x)
    r_D = c * tau - A.T.dot(y) - z
    r_G = c.dot(x) - b.transpose().dot(y) + kappa
    mu = (x.dot(z) + tau * kappa) / (n_x + 1)
    Dinv = x / z
    if sparse:
        M = A.dot(sps.diags(Dinv, 0, format='csc').dot(A.T))
    else:
        M = A.dot(Dinv.reshape(-1, 1) * A.T)
    solve = _get_solver(M, sparse, lstsq, sym_pos, cholesky, permc_spec)
    n_corrections = 1 if pc else 0
    i = 0
    (alpha, d_x, d_z, d_tau, d_kappa) = (0, 0, 0, 0, 0)
    while i <= n_corrections:
        rhatp = eta(gamma) * r_P
        rhatd = eta(gamma) * r_D
        rhatg = eta(gamma) * r_G
        rhatxs = gamma * mu - x * z
        rhattk = gamma * mu - tau * kappa
        if i == 1:
            if ip:
                rhatxs = (1 - alpha) * gamma * mu - x * z - alpha ** 2 * d_x * d_z
                rhattk = (1 - alpha) * gamma * mu - tau * kappa - alpha ** 2 * d_tau * d_kappa
            else:
                rhatxs -= d_x * d_z
                rhattk -= d_tau * d_kappa
        solved = False
        while not solved:
            try:
                (p, q) = _sym_solve(Dinv, A, c, b, solve)
                (u, v) = _sym_solve(Dinv, A, rhatd - 1 / x * rhatxs, rhatp, solve)
                if np.any(np.isnan(p)) or np.any(np.isnan(q)):
                    raise LinAlgError
                solved = True
            except (LinAlgError, ValueError, TypeError) as e:
                if cholesky:
                    cholesky = False
                    warn("Solving system with option 'cholesky':True failed. It is normal for this to happen occasionally, especially as the solution is approached. However, if you see this frequently, consider setting option 'cholesky' to False.", OptimizeWarning, stacklevel=5)
                elif sym_pos:
                    sym_pos = False
                    warn("Solving system with option 'sym_pos':True failed. It is normal for this to happen occasionally, especially as the solution is approached. However, if you see this frequently, consider setting option 'sym_pos' to False.", OptimizeWarning, stacklevel=5)
                elif not lstsq:
                    lstsq = True
                    warn("Solving system with option 'sym_pos':False failed. This may happen occasionally, especially as the solution is approached. However, if you see this frequently, your problem may be numerically challenging. If you cannot improve the formulation, consider setting 'lstsq' to True. Consider also setting `presolve` to True, if it is not already.", OptimizeWarning, stacklevel=5)
                else:
                    raise e
                solve = _get_solver(M, sparse, lstsq, sym_pos, cholesky, permc_spec)
        d_tau = (rhatg + 1 / tau * rhattk - (-c.dot(u) + b.dot(v))) / (1 / tau * kappa + (-c.dot(p) + b.dot(q)))
        d_x = u + p * d_tau
        d_y = v + q * d_tau
        d_z = 1 / x * (rhatxs - z * d_x)
        d_kappa = 1 / tau * (rhattk - kappa * d_tau)
        alpha = _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, 1)
        if ip:
            gamma = 10
        else:
            beta1 = 0.1
            gamma = (1 - alpha) ** 2 * min(beta1, 1 - alpha)
        i += 1
    return (d_x, d_y, d_z, d_tau, d_kappa)

def _sym_solve(Dinv, A, r1, r2, solve):
    if False:
        for i in range(10):
            print('nop')
    '\n    An implementation of [4] equation 8.31 and 8.32\n\n    References\n    ----------\n    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point\n           optimizer for linear programming: an implementation of the\n           homogeneous algorithm." High performance optimization. Springer US,\n           2000. 197-232.\n\n    '
    r = r2 + A.dot(Dinv * r1)
    v = solve(r)
    u = Dinv * (A.T.dot(v) - r1)
    return (u, v)

def _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0):
    if False:
        print('Hello World!')
    '\n    An implementation of [4] equation 8.21\n\n    References\n    ----------\n    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point\n           optimizer for linear programming: an implementation of the\n           homogeneous algorithm." High performance optimization. Springer US,\n           2000. 197-232.\n\n    '
    i_x = d_x < 0
    i_z = d_z < 0
    alpha_x = alpha0 * np.min(x[i_x] / -d_x[i_x]) if np.any(i_x) else 1
    alpha_tau = alpha0 * tau / -d_tau if d_tau < 0 else 1
    alpha_z = alpha0 * np.min(z[i_z] / -d_z[i_z]) if np.any(i_z) else 1
    alpha_kappa = alpha0 * kappa / -d_kappa if d_kappa < 0 else 1
    alpha = np.min([1, alpha_x, alpha_tau, alpha_z, alpha_kappa])
    return alpha

def _get_message(status):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given problem status code, return a more detailed message.\n\n    Parameters\n    ----------\n    status : int\n        An integer representing the exit status of the optimization::\n\n         0 : Optimization terminated successfully\n         1 : Iteration limit reached\n         2 : Problem appears to be infeasible\n         3 : Problem appears to be unbounded\n         4 : Serious numerical difficulties encountered\n\n    Returns\n    -------\n    message : str\n        A string descriptor of the exit status of the optimization.\n\n    '
    messages = ['Optimization terminated successfully.', 'The iteration limit was reached before the algorithm converged.', 'The algorithm terminated successfully and determined that the problem is infeasible.', 'The algorithm terminated successfully and determined that the problem is unbounded.', 'Numerical difficulties were encountered before the problem converged. Please check your problem formulation for errors, independence of linear equality constraints, and reasonable scaling and matrix condition numbers. If you continue to encounter this error, please submit a bug report.']
    return messages[status]

def _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha):
    if False:
        return 10
    '\n    An implementation of [4] Equation 8.9\n\n    References\n    ----------\n    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point\n           optimizer for linear programming: an implementation of the\n           homogeneous algorithm." High performance optimization. Springer US,\n           2000. 197-232.\n\n    '
    x = x + alpha * d_x
    tau = tau + alpha * d_tau
    z = z + alpha * d_z
    kappa = kappa + alpha * d_kappa
    y = y + alpha * d_y
    return (x, y, z, tau, kappa)

def _get_blind_start(shape):
    if False:
        return 10
    '\n    Return the starting point from [4] 4.4\n\n    References\n    ----------\n    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point\n           optimizer for linear programming: an implementation of the\n           homogeneous algorithm." High performance optimization. Springer US,\n           2000. 197-232.\n\n    '
    (m, n) = shape
    x0 = np.ones(n)
    y0 = np.zeros(m)
    z0 = np.ones(n)
    tau0 = 1
    kappa0 = 1
    return (x0, y0, z0, tau0, kappa0)

def _indicators(A, b, c, c0, x, y, z, tau, kappa):
    if False:
        while True:
            i = 10
    '\n    Implementation of several equations from [4] used as indicators of\n    the status of optimization.\n\n    References\n    ----------\n    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point\n           optimizer for linear programming: an implementation of the\n           homogeneous algorithm." High performance optimization. Springer US,\n           2000. 197-232.\n\n    '
    (x0, y0, z0, tau0, kappa0) = _get_blind_start(A.shape)

    def r_p(x, tau):
        if False:
            return 10
        return b * tau - A.dot(x)

    def r_d(y, z, tau):
        if False:
            return 10
        return c * tau - A.T.dot(y) - z

    def r_g(x, y, kappa):
        if False:
            return 10
        return kappa + c.dot(x) - b.dot(y)

    def mu(x, tau, z, kappa):
        if False:
            return 10
        return (x.dot(z) + np.dot(tau, kappa)) / (len(x) + 1)
    obj = c.dot(x / tau) + c0

    def norm(a):
        if False:
            while True:
                i = 10
        return np.linalg.norm(a)
    r_p0 = r_p(x0, tau0)
    r_d0 = r_d(y0, z0, tau0)
    r_g0 = r_g(x0, y0, kappa0)
    mu_0 = mu(x0, tau0, z0, kappa0)
    rho_A = norm(c.T.dot(x) - b.T.dot(y)) / (tau + norm(b.T.dot(y)))
    rho_p = norm(r_p(x, tau)) / max(1, norm(r_p0))
    rho_d = norm(r_d(y, z, tau)) / max(1, norm(r_d0))
    rho_g = norm(r_g(x, y, kappa)) / max(1, norm(r_g0))
    rho_mu = mu(x, tau, z, kappa) / mu_0
    return (rho_p, rho_d, rho_A, rho_g, rho_mu, obj)

def _display_iter(rho_p, rho_d, rho_g, alpha, rho_mu, obj, header=False):
    if False:
        return 10
    '\n    Print indicators of optimization status to the console.\n\n    Parameters\n    ----------\n    rho_p : float\n        The (normalized) primal feasibility, see [4] 4.5\n    rho_d : float\n        The (normalized) dual feasibility, see [4] 4.5\n    rho_g : float\n        The (normalized) duality gap, see [4] 4.5\n    alpha : float\n        The step size, see [4] 4.3\n    rho_mu : float\n        The (normalized) path parameter, see [4] 4.5\n    obj : float\n        The objective function value of the current iterate\n    header : bool\n        True if a header is to be printed\n\n    References\n    ----------\n    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point\n           optimizer for linear programming: an implementation of the\n           homogeneous algorithm." High performance optimization. Springer US,\n           2000. 197-232.\n\n    '
    if header:
        print('Primal Feasibility ', 'Dual Feasibility   ', 'Duality Gap        ', 'Step            ', 'Path Parameter     ', 'Objective          ')
    fmt = '{0:<20.13}{1:<20.13}{2:<20.13}{3:<17.13}{4:<20.13}{5:<20.13}'
    print(fmt.format(float(rho_p), float(rho_d), float(rho_g), alpha if isinstance(alpha, str) else float(alpha), float(rho_mu), float(obj)))

def _ip_hsd(A, b, c, c0, alpha0, beta, maxiter, disp, tol, sparse, lstsq, sym_pos, cholesky, pc, ip, permc_spec, callback, postsolve_args):
    if False:
        return 10
    '\n    Solve a linear programming problem in standard form:\n\n    Minimize::\n\n        c @ x\n\n    Subject to::\n\n        A @ x == b\n            x >= 0\n\n    using the interior point method of [4].\n\n    Parameters\n    ----------\n    A : 2-D array\n        2-D array such that ``A @ x``, gives the values of the equality\n        constraints at ``x``.\n    b : 1-D array\n        1-D array of values representing the RHS of each equality constraint\n        (row) in ``A`` (for standard form problem).\n    c : 1-D array\n        Coefficients of the linear objective function to be minimized (for\n        standard form problem).\n    c0 : float\n        Constant term in objective function due to fixed (and eliminated)\n        variables. (Purely for display.)\n    alpha0 : float\n        The maximal step size for Mehrota\'s predictor-corrector search\n        direction; see :math:`\\beta_3`of [4] Table 8.1\n    beta : float\n        The desired reduction of the path parameter :math:`\\mu` (see  [6]_)\n    maxiter : int\n        The maximum number of iterations of the algorithm.\n    disp : bool\n        Set to ``True`` if indicators of optimization status are to be printed\n        to the console each iteration.\n    tol : float\n        Termination tolerance; see [4]_ Section 4.5.\n    sparse : bool\n        Set to ``True`` if the problem is to be treated as sparse. However,\n        the inputs ``A_eq`` and ``A_ub`` should nonetheless be provided as\n        (dense) arrays rather than sparse matrices.\n    lstsq : bool\n        Set to ``True`` if the problem is expected to be very poorly\n        conditioned. This should always be left as ``False`` unless severe\n        numerical difficulties are frequently encountered, and a better option\n        would be to improve the formulation of the problem.\n    sym_pos : bool\n        Leave ``True`` if the problem is expected to yield a well conditioned\n        symmetric positive definite normal equation matrix (almost always).\n    cholesky : bool\n        Set to ``True`` if the normal equations are to be solved by explicit\n        Cholesky decomposition followed by explicit forward/backward\n        substitution. This is typically faster for moderate, dense problems\n        that are numerically well-behaved.\n    pc : bool\n        Leave ``True`` if the predictor-corrector method of Mehrota is to be\n        used. This is almost always (if not always) beneficial.\n    ip : bool\n        Set to ``True`` if the improved initial point suggestion due to [4]_\n        Section 4.3 is desired. It\'s unclear whether this is beneficial.\n    permc_spec : str (default = \'MMD_AT_PLUS_A\')\n        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =\n        True``.) A matrix is factorized in each iteration of the algorithm.\n        This option specifies how to permute the columns of the matrix for\n        sparsity preservation. Acceptable values are:\n\n        - ``NATURAL``: natural ordering.\n        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.\n        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.\n        - ``COLAMD``: approximate minimum degree column ordering.\n\n        This option can impact the convergence of the\n        interior point algorithm; test different values to determine which\n        performs best for your problem. For more information, refer to\n        ``scipy.sparse.linalg.splu``.\n    callback : callable, optional\n        If a callback function is provided, it will be called within each\n        iteration of the algorithm. The callback function must accept a single\n        `scipy.optimize.OptimizeResult` consisting of the following fields:\n\n            x : 1-D array\n                Current solution vector\n            fun : float\n                Current value of the objective function\n            success : bool\n                True only when an algorithm has completed successfully,\n                so this is always False as the callback function is called\n                only while the algorithm is still iterating.\n            slack : 1-D array\n                The values of the slack variables. Each slack variable\n                corresponds to an inequality constraint. If the slack is zero,\n                the corresponding constraint is active.\n            con : 1-D array\n                The (nominally zero) residuals of the equality constraints,\n                that is, ``b - A_eq @ x``\n            phase : int\n                The phase of the algorithm being executed. This is always\n                1 for the interior-point method because it has only one phase.\n            status : int\n                For revised simplex, this is always 0 because if a different\n                status is detected, the algorithm terminates.\n            nit : int\n                The number of iterations performed.\n            message : str\n                A string descriptor of the exit status of the optimization.\n    postsolve_args : tuple\n        Data needed by _postsolve to convert the solution to the standard-form\n        problem into the solution to the original problem.\n\n    Returns\n    -------\n    x_hat : float\n        Solution vector (for standard form problem).\n    status : int\n        An integer representing the exit status of the optimization::\n\n         0 : Optimization terminated successfully\n         1 : Iteration limit reached\n         2 : Problem appears to be infeasible\n         3 : Problem appears to be unbounded\n         4 : Serious numerical difficulties encountered\n\n    message : str\n        A string descriptor of the exit status of the optimization.\n    iteration : int\n        The number of iterations taken to solve the problem\n\n    References\n    ----------\n    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point\n           optimizer for linear programming: an implementation of the\n           homogeneous algorithm." High performance optimization. Springer US,\n           2000. 197-232.\n    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear\n           Programming based on Newton\'s Method." Unpublished Course Notes,\n           March 2004. Available 2/25/2017 at:\n           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf\n\n    '
    iteration = 0
    (x, y, z, tau, kappa) = _get_blind_start(A.shape)
    ip = ip if pc else False
    (rho_p, rho_d, rho_A, rho_g, rho_mu, obj) = _indicators(A, b, c, c0, x, y, z, tau, kappa)
    go = rho_p > tol or rho_d > tol or rho_A > tol
    if disp:
        _display_iter(rho_p, rho_d, rho_g, '-', rho_mu, obj, header=True)
    if callback is not None:
        (x_o, fun, slack, con) = _postsolve(x / tau, postsolve_args)
        res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack, 'con': con, 'nit': iteration, 'phase': 1, 'complete': False, 'status': 0, 'message': '', 'success': False})
        callback(res)
    status = 0
    message = 'Optimization terminated successfully.'
    if sparse:
        A = sps.csc_matrix(A)
    while go:
        iteration += 1
        if ip:
            gamma = 1

            def eta(g):
                if False:
                    for i in range(10):
                        print('nop')
                return 1
        else:
            gamma = 0 if pc else beta * np.mean(z * x)

            def eta(g=gamma):
                if False:
                    while True:
                        i = 10
                return 1 - g
        try:
            (d_x, d_y, d_z, d_tau, d_kappa) = _get_delta(A, b, c, x, y, z, tau, kappa, gamma, eta, sparse, lstsq, sym_pos, cholesky, pc, ip, permc_spec)
            if ip:
                alpha = 1.0
                (x, y, z, tau, kappa) = _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha)
                x[x < 1] = 1
                z[z < 1] = 1
                tau = max(1, tau)
                kappa = max(1, kappa)
                ip = False
            else:
                alpha = _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0)
                (x, y, z, tau, kappa) = _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha)
        except (LinAlgError, FloatingPointError, ValueError, ZeroDivisionError):
            status = 4
            message = _get_message(status)
            break
        (rho_p, rho_d, rho_A, rho_g, rho_mu, obj) = _indicators(A, b, c, c0, x, y, z, tau, kappa)
        go = rho_p > tol or rho_d > tol or rho_A > tol
        if disp:
            _display_iter(rho_p, rho_d, rho_g, alpha, rho_mu, obj)
        if callback is not None:
            (x_o, fun, slack, con) = _postsolve(x / tau, postsolve_args)
            res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack, 'con': con, 'nit': iteration, 'phase': 1, 'complete': False, 'status': 0, 'message': '', 'success': False})
            callback(res)
        inf1 = rho_p < tol and rho_d < tol and (rho_g < tol) and (tau < tol * max(1, kappa))
        inf2 = rho_mu < tol and tau < tol * min(1, kappa)
        if inf1 or inf2:
            if b.transpose().dot(y) > tol:
                status = 2
            else:
                status = 3
            message = _get_message(status)
            break
        elif iteration >= maxiter:
            status = 1
            message = _get_message(status)
            break
    x_hat = x / tau
    return (x_hat, status, message, iteration)

def _linprog_ip(c, c0, A, b, callback, postsolve_args, maxiter=1000, tol=1e-08, disp=False, alpha0=0.99995, beta=0.1, sparse=False, lstsq=False, sym_pos=True, cholesky=None, pc=True, ip=False, permc_spec='MMD_AT_PLUS_A', **unknown_options):
    if False:
        while True:
            i = 10
    '\n    Minimize a linear objective function subject to linear\n    equality and non-negativity constraints using the interior point method\n    of [4]_. Linear programming is intended to solve problems\n    of the following form:\n\n    Minimize::\n\n        c @ x\n\n    Subject to::\n\n        A @ x == b\n            x >= 0\n\n    User-facing documentation is in _linprog_doc.py.\n\n    Parameters\n    ----------\n    c : 1-D array\n        Coefficients of the linear objective function to be minimized.\n    c0 : float\n        Constant term in objective function due to fixed (and eliminated)\n        variables. (Purely for display.)\n    A : 2-D array\n        2-D array such that ``A @ x``, gives the values of the equality\n        constraints at ``x``.\n    b : 1-D array\n        1-D array of values representing the right hand side of each equality\n        constraint (row) in ``A``.\n    callback : callable, optional\n        Callback function to be executed once per iteration.\n    postsolve_args : tuple\n        Data needed by _postsolve to convert the solution to the standard-form\n        problem into the solution to the original problem.\n\n    Options\n    -------\n    maxiter : int (default = 1000)\n        The maximum number of iterations of the algorithm.\n    tol : float (default = 1e-8)\n        Termination tolerance to be used for all termination criteria;\n        see [4]_ Section 4.5.\n    disp : bool (default = False)\n        Set to ``True`` if indicators of optimization status are to be printed\n        to the console each iteration.\n    alpha0 : float (default = 0.99995)\n        The maximal step size for Mehrota\'s predictor-corrector search\n        direction; see :math:`\\beta_{3}` of [4]_ Table 8.1.\n    beta : float (default = 0.1)\n        The desired reduction of the path parameter :math:`\\mu` (see [6]_)\n        when Mehrota\'s predictor-corrector is not in use (uncommon).\n    sparse : bool (default = False)\n        Set to ``True`` if the problem is to be treated as sparse after\n        presolve. If either ``A_eq`` or ``A_ub`` is a sparse matrix,\n        this option will automatically be set ``True``, and the problem\n        will be treated as sparse even during presolve. If your constraint\n        matrices contain mostly zeros and the problem is not very small (less\n        than about 100 constraints or variables), consider setting ``True``\n        or providing ``A_eq`` and ``A_ub`` as sparse matrices.\n    lstsq : bool (default = False)\n        Set to ``True`` if the problem is expected to be very poorly\n        conditioned. This should always be left ``False`` unless severe\n        numerical difficulties are encountered. Leave this at the default\n        unless you receive a warning message suggesting otherwise.\n    sym_pos : bool (default = True)\n        Leave ``True`` if the problem is expected to yield a well conditioned\n        symmetric positive definite normal equation matrix\n        (almost always). Leave this at the default unless you receive\n        a warning message suggesting otherwise.\n    cholesky : bool (default = True)\n        Set to ``True`` if the normal equations are to be solved by explicit\n        Cholesky decomposition followed by explicit forward/backward\n        substitution. This is typically faster for problems\n        that are numerically well-behaved.\n    pc : bool (default = True)\n        Leave ``True`` if the predictor-corrector method of Mehrota is to be\n        used. This is almost always (if not always) beneficial.\n    ip : bool (default = False)\n        Set to ``True`` if the improved initial point suggestion due to [4]_\n        Section 4.3 is desired. Whether this is beneficial or not\n        depends on the problem.\n    permc_spec : str (default = \'MMD_AT_PLUS_A\')\n        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =\n        True``, and no SuiteSparse.)\n        A matrix is factorized in each iteration of the algorithm.\n        This option specifies how to permute the columns of the matrix for\n        sparsity preservation. Acceptable values are:\n\n        - ``NATURAL``: natural ordering.\n        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.\n        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.\n        - ``COLAMD``: approximate minimum degree column ordering.\n\n        This option can impact the convergence of the\n        interior point algorithm; test different values to determine which\n        performs best for your problem. For more information, refer to\n        ``scipy.sparse.linalg.splu``.\n    unknown_options : dict\n        Optional arguments not used by this particular solver. If\n        `unknown_options` is non-empty a warning is issued listing all\n        unused options.\n\n    Returns\n    -------\n    x : 1-D array\n        Solution vector.\n    status : int\n        An integer representing the exit status of the optimization::\n\n         0 : Optimization terminated successfully\n         1 : Iteration limit reached\n         2 : Problem appears to be infeasible\n         3 : Problem appears to be unbounded\n         4 : Serious numerical difficulties encountered\n\n    message : str\n        A string descriptor of the exit status of the optimization.\n    iteration : int\n        The number of iterations taken to solve the problem.\n\n    Notes\n    -----\n    This method implements the algorithm outlined in [4]_ with ideas from [8]_\n    and a structure inspired by the simpler methods of [6]_.\n\n    The primal-dual path following method begins with initial \'guesses\' of\n    the primal and dual variables of the standard form problem and iteratively\n    attempts to solve the (nonlinear) Karush-Kuhn-Tucker conditions for the\n    problem with a gradually reduced logarithmic barrier term added to the\n    objective. This particular implementation uses a homogeneous self-dual\n    formulation, which provides certificates of infeasibility or unboundedness\n    where applicable.\n\n    The default initial point for the primal and dual variables is that\n    defined in [4]_ Section 4.4 Equation 8.22. Optionally (by setting initial\n    point option ``ip=True``), an alternate (potentially improved) starting\n    point can be calculated according to the additional recommendations of\n    [4]_ Section 4.4.\n\n    A search direction is calculated using the predictor-corrector method\n    (single correction) proposed by Mehrota and detailed in [4]_ Section 4.1.\n    (A potential improvement would be to implement the method of multiple\n    corrections described in [4]_ Section 4.2.) In practice, this is\n    accomplished by solving the normal equations, [4]_ Section 5.1 Equations\n    8.31 and 8.32, derived from the Newton equations [4]_ Section 5 Equations\n    8.25 (compare to [4]_ Section 4 Equations 8.6-8.8). The advantage of\n    solving the normal equations rather than 8.25 directly is that the\n    matrices involved are symmetric positive definite, so Cholesky\n    decomposition can be used rather than the more expensive LU factorization.\n\n    With default options, the solver used to perform the factorization depends\n    on third-party software availability and the conditioning of the problem.\n\n    For dense problems, solvers are tried in the following order:\n\n    1. ``scipy.linalg.cho_factor``\n\n    2. ``scipy.linalg.solve`` with option ``sym_pos=True``\n\n    3. ``scipy.linalg.solve`` with option ``sym_pos=False``\n\n    4. ``scipy.linalg.lstsq``\n\n    For sparse problems:\n\n    1. ``sksparse.cholmod.cholesky`` (if scikit-sparse and SuiteSparse are installed)\n\n    2. ``scipy.sparse.linalg.factorized`` (if scikit-umfpack and SuiteSparse are installed)\n\n    3. ``scipy.sparse.linalg.splu`` (which uses SuperLU distributed with SciPy)\n\n    4. ``scipy.sparse.linalg.lsqr``\n\n    If the solver fails for any reason, successively more robust (but slower)\n    solvers are attempted in the order indicated. Attempting, failing, and\n    re-starting factorization can be time consuming, so if the problem is\n    numerically challenging, options can be set to  bypass solvers that are\n    failing. Setting ``cholesky=False`` skips to solver 2,\n    ``sym_pos=False`` skips to solver 3, and ``lstsq=True`` skips\n    to solver 4 for both sparse and dense problems.\n\n    Potential improvements for combatting issues associated with dense\n    columns in otherwise sparse problems are outlined in [4]_ Section 5.3 and\n    [10]_ Section 4.1-4.2; the latter also discusses the alleviation of\n    accuracy issues associated with the substitution approach to free\n    variables.\n\n    After calculating the search direction, the maximum possible step size\n    that does not activate the non-negativity constraints is calculated, and\n    the smaller of this step size and unity is applied (as in [4]_ Section\n    4.1.) [4]_ Section 4.3 suggests improvements for choosing the step size.\n\n    The new point is tested according to the termination conditions of [4]_\n    Section 4.5. The same tolerance, which can be set using the ``tol`` option,\n    is used for all checks. (A potential improvement would be to expose\n    the different tolerances to be set independently.) If optimality,\n    unboundedness, or infeasibility is detected, the solve procedure\n    terminates; otherwise it repeats.\n\n    The expected problem formulation differs between the top level ``linprog``\n    module and the method specific solvers. The method specific solvers expect a\n    problem in standard form:\n\n    Minimize::\n\n        c @ x\n\n    Subject to::\n\n        A @ x == b\n            x >= 0\n\n    Whereas the top level ``linprog`` module expects a problem of form:\n\n    Minimize::\n\n        c @ x\n\n    Subject to::\n\n        A_ub @ x <= b_ub\n        A_eq @ x == b_eq\n         lb <= x <= ub\n\n    where ``lb = 0`` and ``ub = None`` unless set in ``bounds``.\n\n    The original problem contains equality, upper-bound and variable constraints\n    whereas the method specific solver requires equality constraints and\n    variable non-negativity.\n\n    ``linprog`` module converts the original problem to standard form by\n    converting the simple bounds to upper bound constraints, introducing\n    non-negative slack variables for inequality constraints, and expressing\n    unbounded variables as the difference between two non-negative variables.\n\n\n    References\n    ----------\n    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point\n           optimizer for linear programming: an implementation of the\n           homogeneous algorithm." High performance optimization. Springer US,\n           2000. 197-232.\n    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear\n           Programming based on Newton\'s Method." Unpublished Course Notes,\n           March 2004. Available 2/25/2017 at\n           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf\n    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear\n           programming." Mathematical Programming 71.2 (1995): 221-245.\n    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear\n           programming." Athena Scientific 1 (1997): 997.\n    .. [10] Andersen, Erling D., et al. Implementation of interior point methods\n            for large scale linear programming. HEC/Universite de Geneve, 1996.\n\n    '
    _check_unknown_options(unknown_options)
    if (cholesky or cholesky is None) and sparse and (not has_cholmod):
        if cholesky:
            warn('Sparse cholesky is only available with scikit-sparse. Setting `cholesky = False`', OptimizeWarning, stacklevel=3)
        cholesky = False
    if sparse and lstsq:
        warn("Option combination 'sparse':True and 'lstsq':True is not recommended.", OptimizeWarning, stacklevel=3)
    if lstsq and cholesky:
        warn("Invalid option combination 'lstsq':True and 'cholesky':True; option 'cholesky' has no effect when 'lstsq' is set True.", OptimizeWarning, stacklevel=3)
    valid_permc_spec = ('NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', 'COLAMD')
    if permc_spec.upper() not in valid_permc_spec:
        warn("Invalid permc_spec option: '" + str(permc_spec) + "'. Acceptable values are 'NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', and 'COLAMD'. Reverting to default.", OptimizeWarning, stacklevel=3)
        permc_spec = 'MMD_AT_PLUS_A'
    if not sym_pos and cholesky:
        raise ValueError("Invalid option combination 'sym_pos':False and 'cholesky':True: Cholesky decomposition is only possible for symmetric positive definite matrices.")
    cholesky = cholesky or (cholesky is None and sym_pos and (not lstsq))
    (x, status, message, iteration) = _ip_hsd(A, b, c, c0, alpha0, beta, maxiter, disp, tol, sparse, lstsq, sym_pos, cholesky, pc, ip, permc_spec, callback, postsolve_args)
    return (x, status, message, iteration)