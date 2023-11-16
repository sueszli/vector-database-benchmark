"""Simplex method for  linear programming

The *simplex* method uses a traditional, full-tableau implementation of
Dantzig's simplex algorithm [1]_, [2]_ (*not* the Nelder-Mead simplex).
This algorithm is included for backwards compatibility and educational
purposes.

    .. versionadded:: 0.15.0

Warnings
--------

The simplex method may encounter numerical difficulties when pivot
values are close to the specified tolerance. If encountered try
remove any redundant constraints, change the pivot strategy to Bland's
rule or increase the tolerance value.

Alternatively, more robust methods maybe be used. See
:ref:`'interior-point' <optimize.linprog-interior-point>` and
:ref:`'revised simplex' <optimize.linprog-revised_simplex>`.

References
----------
.. [1] Dantzig, George B., Linear programming and extensions. Rand
       Corporation Research Study Princeton Univ. Press, Princeton, NJ,
       1963
.. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
       Mathematical Programming", McGraw-Hill, Chapter 4.
"""
import numpy as np
from warnings import warn
from ._optimize import OptimizeResult, OptimizeWarning, _check_unknown_options
from ._linprog_util import _postsolve

def _pivot_col(T, tol=1e-09, bland=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Given a linear programming simplex tableau, determine the column\n    of the variable to enter the basis.\n\n    Parameters\n    ----------\n    T : 2-D array\n        A 2-D array representing the simplex tableau, T, corresponding to the\n        linear programming problem. It should have the form:\n\n        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],\n         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],\n         .\n         .\n         .\n         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],\n         [c[0],   c[1], ...,   c[n_total],    0]]\n\n        for a Phase 2 problem, or the form:\n\n        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],\n         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],\n         .\n         .\n         .\n         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],\n         [c[0],   c[1], ...,   c[n_total],   0],\n         [c'[0],  c'[1], ...,  c'[n_total],  0]]\n\n         for a Phase 1 problem (a problem in which a basic feasible solution is\n         sought prior to maximizing the actual objective. ``T`` is modified in\n         place by ``_solve_simplex``.\n    tol : float\n        Elements in the objective row larger than -tol will not be considered\n        for pivoting. Nominally this value is zero, but numerical issues\n        cause a tolerance about zero to be necessary.\n    bland : bool\n        If True, use Bland's rule for selection of the column (select the\n        first column with a negative coefficient in the objective row,\n        regardless of magnitude).\n\n    Returns\n    -------\n    status: bool\n        True if a suitable pivot column was found, otherwise False.\n        A return of False indicates that the linear programming simplex\n        algorithm is complete.\n    col: int\n        The index of the column of the pivot element.\n        If status is False, col will be returned as nan.\n    "
    ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
    if ma.count() == 0:
        return (False, np.nan)
    if bland:
        return (True, np.nonzero(np.logical_not(np.atleast_1d(ma.mask)))[0][0])
    return (True, np.ma.nonzero(ma == ma.min())[0][0])

def _pivot_row(T, basis, pivcol, phase, tol=1e-09, bland=False):
    if False:
        print('Hello World!')
    "\n    Given a linear programming simplex tableau, determine the row for the\n    pivot operation.\n\n    Parameters\n    ----------\n    T : 2-D array\n        A 2-D array representing the simplex tableau, T, corresponding to the\n        linear programming problem. It should have the form:\n\n        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],\n         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],\n         .\n         .\n         .\n         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],\n         [c[0],   c[1], ...,   c[n_total],    0]]\n\n        for a Phase 2 problem, or the form:\n\n        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],\n         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],\n         .\n         .\n         .\n         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],\n         [c[0],   c[1], ...,   c[n_total],   0],\n         [c'[0],  c'[1], ...,  c'[n_total],  0]]\n\n         for a Phase 1 problem (a Problem in which a basic feasible solution is\n         sought prior to maximizing the actual objective. ``T`` is modified in\n         place by ``_solve_simplex``.\n    basis : array\n        A list of the current basic variables.\n    pivcol : int\n        The index of the pivot column.\n    phase : int\n        The phase of the simplex algorithm (1 or 2).\n    tol : float\n        Elements in the pivot column smaller than tol will not be considered\n        for pivoting. Nominally this value is zero, but numerical issues\n        cause a tolerance about zero to be necessary.\n    bland : bool\n        If True, use Bland's rule for selection of the row (if more than one\n        row can be used, choose the one with the lowest variable index).\n\n    Returns\n    -------\n    status: bool\n        True if a suitable pivot row was found, otherwise False. A return\n        of False indicates that the linear programming problem is unbounded.\n    row: int\n        The index of the row of the pivot element. If status is False, row\n        will be returned as nan.\n    "
    if phase == 1:
        k = 2
    else:
        k = 1
    ma = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, pivcol], copy=False)
    if ma.count() == 0:
        return (False, np.nan)
    mb = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, -1], copy=False)
    q = mb / ma
    min_rows = np.ma.nonzero(q == q.min())[0]
    if bland:
        return (True, min_rows[np.argmin(np.take(basis, min_rows))])
    return (True, min_rows[0])

def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-09):
    if False:
        print('Hello World!')
    "\n    Pivot the simplex tableau inplace on the element given by (pivrow, pivol).\n    The entering variable corresponds to the column given by pivcol forcing\n    the variable basis[pivrow] to leave the basis.\n\n    Parameters\n    ----------\n    T : 2-D array\n        A 2-D array representing the simplex tableau, T, corresponding to the\n        linear programming problem. It should have the form:\n\n        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],\n         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],\n         .\n         .\n         .\n         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],\n         [c[0],   c[1], ...,   c[n_total],    0]]\n\n        for a Phase 2 problem, or the form:\n\n        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],\n         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],\n         .\n         .\n         .\n         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],\n         [c[0],   c[1], ...,   c[n_total],   0],\n         [c'[0],  c'[1], ...,  c'[n_total],  0]]\n\n         for a Phase 1 problem (a problem in which a basic feasible solution is\n         sought prior to maximizing the actual objective. ``T`` is modified in\n         place by ``_solve_simplex``.\n    basis : 1-D array\n        An array of the indices of the basic variables, such that basis[i]\n        contains the column corresponding to the basic variable for row i.\n        Basis is modified in place by _apply_pivot.\n    pivrow : int\n        Row index of the pivot.\n    pivcol : int\n        Column index of the pivot.\n    "
    basis[pivrow] = pivcol
    pivval = T[pivrow, pivcol]
    T[pivrow] = T[pivrow] / pivval
    for irow in range(T.shape[0]):
        if irow != pivrow:
            T[irow] = T[irow] - T[pivrow] * T[irow, pivcol]
    if np.isclose(pivval, tol, atol=0, rtol=10000.0):
        message = "The pivot operation produces a pivot value of:{: .1e}, which is only slightly greater than the specified tolerance{: .1e}. This may lead to issues regarding the numerical stability of the simplex method. Removing redundant constraints, changing the pivot strategy via Bland's rule or increasing the tolerance may help reduce the issue.".format(pivval, tol)
        warn(message, OptimizeWarning, stacklevel=5)

def _solve_simplex(T, n, basis, callback, postsolve_args, maxiter=1000, tol=1e-09, phase=2, bland=False, nit0=0):
    if False:
        while True:
            i = 10
    '\n    Solve a linear programming problem in "standard form" using the Simplex\n    Method. Linear Programming is intended to solve the following problem form:\n\n    Minimize::\n\n        c @ x\n\n    Subject to::\n\n        A @ x == b\n            x >= 0\n\n    Parameters\n    ----------\n    T : 2-D array\n        A 2-D array representing the simplex tableau, T, corresponding to the\n        linear programming problem. It should have the form:\n\n        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],\n         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],\n         .\n         .\n         .\n         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],\n         [c[0],   c[1], ...,   c[n_total],    0]]\n\n        for a Phase 2 problem, or the form:\n\n        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],\n         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],\n         .\n         .\n         .\n         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],\n         [c[0],   c[1], ...,   c[n_total],   0],\n         [c\'[0],  c\'[1], ...,  c\'[n_total],  0]]\n\n         for a Phase 1 problem (a problem in which a basic feasible solution is\n         sought prior to maximizing the actual objective. ``T`` is modified in\n         place by ``_solve_simplex``.\n    n : int\n        The number of true variables in the problem.\n    basis : 1-D array\n        An array of the indices of the basic variables, such that basis[i]\n        contains the column corresponding to the basic variable for row i.\n        Basis is modified in place by _solve_simplex\n    callback : callable, optional\n        If a callback function is provided, it will be called within each\n        iteration of the algorithm. The callback must accept a\n        `scipy.optimize.OptimizeResult` consisting of the following fields:\n\n            x : 1-D array\n                Current solution vector\n            fun : float\n                Current value of the objective function\n            success : bool\n                True only when a phase has completed successfully. This\n                will be False for most iterations.\n            slack : 1-D array\n                The values of the slack variables. Each slack variable\n                corresponds to an inequality constraint. If the slack is zero,\n                the corresponding constraint is active.\n            con : 1-D array\n                The (nominally zero) residuals of the equality constraints,\n                that is, ``b - A_eq @ x``\n            phase : int\n                The phase of the optimization being executed. In phase 1 a basic\n                feasible solution is sought and the T has an additional row\n                representing an alternate objective function.\n            status : int\n                An integer representing the exit status of the optimization::\n\n                     0 : Optimization terminated successfully\n                     1 : Iteration limit reached\n                     2 : Problem appears to be infeasible\n                     3 : Problem appears to be unbounded\n                     4 : Serious numerical difficulties encountered\n\n            nit : int\n                The number of iterations performed.\n            message : str\n                A string descriptor of the exit status of the optimization.\n    postsolve_args : tuple\n        Data needed by _postsolve to convert the solution to the standard-form\n        problem into the solution to the original problem.\n    maxiter : int\n        The maximum number of iterations to perform before aborting the\n        optimization.\n    tol : float\n        The tolerance which determines when a solution is "close enough" to\n        zero in Phase 1 to be considered a basic feasible solution or close\n        enough to positive to serve as an optimal solution.\n    phase : int\n        The phase of the optimization being executed. In phase 1 a basic\n        feasible solution is sought and the T has an additional row\n        representing an alternate objective function.\n    bland : bool\n        If True, choose pivots using Bland\'s rule [3]_. In problems which\n        fail to converge due to cycling, using Bland\'s rule can provide\n        convergence at the expense of a less optimal path about the simplex.\n    nit0 : int\n        The initial iteration number used to keep an accurate iteration total\n        in a two-phase problem.\n\n    Returns\n    -------\n    nit : int\n        The number of iterations. Used to keep an accurate iteration total\n        in the two-phase problem.\n    status : int\n        An integer representing the exit status of the optimization::\n\n         0 : Optimization terminated successfully\n         1 : Iteration limit reached\n         2 : Problem appears to be infeasible\n         3 : Problem appears to be unbounded\n         4 : Serious numerical difficulties encountered\n\n    '
    nit = nit0
    status = 0
    message = ''
    complete = False
    if phase == 1:
        m = T.shape[1] - 2
    elif phase == 2:
        m = T.shape[1] - 1
    else:
        raise ValueError("Argument 'phase' to _solve_simplex must be 1 or 2")
    if phase == 2:
        for pivrow in [row for row in range(basis.size) if basis[row] > T.shape[1] - 2]:
            non_zero_row = [col for col in range(T.shape[1] - 1) if abs(T[pivrow, col]) > tol]
            if len(non_zero_row) > 0:
                pivcol = non_zero_row[0]
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1
    if len(basis[:m]) == 0:
        solution = np.empty(T.shape[1] - 1, dtype=np.float64)
    else:
        solution = np.empty(max(T.shape[1] - 1, max(basis[:m]) + 1), dtype=np.float64)
    while not complete:
        (pivcol_found, pivcol) = _pivot_col(T, tol, bland)
        if not pivcol_found:
            pivcol = np.nan
            pivrow = np.nan
            status = 0
            complete = True
        else:
            (pivrow_found, pivrow) = _pivot_row(T, basis, pivcol, phase, tol, bland)
            if not pivrow_found:
                status = 3
                complete = True
        if callback is not None:
            solution[:] = 0
            solution[basis[:n]] = T[:n, -1]
            x = solution[:m]
            (x, fun, slack, con) = _postsolve(x, postsolve_args)
            res = OptimizeResult({'x': x, 'fun': fun, 'slack': slack, 'con': con, 'status': status, 'message': message, 'nit': nit, 'success': status == 0 and complete, 'phase': phase, 'complete': complete})
            callback(res)
        if not complete:
            if nit >= maxiter:
                status = 1
                complete = True
            else:
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1
    return (nit, status)

def _linprog_simplex(c, c0, A, b, callback, postsolve_args, maxiter=1000, tol=1e-09, disp=False, bland=False, **unknown_options):
    if False:
        i = 10
        return i + 15
    '\n    Minimize a linear objective function subject to linear equality and\n    non-negativity constraints using the two phase simplex method.\n    Linear programming is intended to solve problems of the following form:\n\n    Minimize::\n\n        c @ x\n\n    Subject to::\n\n        A @ x == b\n            x >= 0\n\n    User-facing documentation is in _linprog_doc.py.\n\n    Parameters\n    ----------\n    c : 1-D array\n        Coefficients of the linear objective function to be minimized.\n    c0 : float\n        Constant term in objective function due to fixed (and eliminated)\n        variables. (Purely for display.)\n    A : 2-D array\n        2-D array such that ``A @ x``, gives the values of the equality\n        constraints at ``x``.\n    b : 1-D array\n        1-D array of values representing the right hand side of each equality\n        constraint (row) in ``A``.\n    callback : callable, optional\n        If a callback function is provided, it will be called within each\n        iteration of the algorithm. The callback function must accept a single\n        `scipy.optimize.OptimizeResult` consisting of the following fields:\n\n            x : 1-D array\n                Current solution vector\n            fun : float\n                Current value of the objective function\n            success : bool\n                True when an algorithm has completed successfully.\n            slack : 1-D array\n                The values of the slack variables. Each slack variable\n                corresponds to an inequality constraint. If the slack is zero,\n                the corresponding constraint is active.\n            con : 1-D array\n                The (nominally zero) residuals of the equality constraints,\n                that is, ``b - A_eq @ x``\n            phase : int\n                The phase of the algorithm being executed.\n            status : int\n                An integer representing the status of the optimization::\n\n                     0 : Algorithm proceeding nominally\n                     1 : Iteration limit reached\n                     2 : Problem appears to be infeasible\n                     3 : Problem appears to be unbounded\n                     4 : Serious numerical difficulties encountered\n            nit : int\n                The number of iterations performed.\n            message : str\n                A string descriptor of the exit status of the optimization.\n    postsolve_args : tuple\n        Data needed by _postsolve to convert the solution to the standard-form\n        problem into the solution to the original problem.\n\n    Options\n    -------\n    maxiter : int\n       The maximum number of iterations to perform.\n    disp : bool\n        If True, print exit status message to sys.stdout\n    tol : float\n        The tolerance which determines when a solution is "close enough" to\n        zero in Phase 1 to be considered a basic feasible solution or close\n        enough to positive to serve as an optimal solution.\n    bland : bool\n        If True, use Bland\'s anti-cycling rule [3]_ to choose pivots to\n        prevent cycling. If False, choose pivots which should lead to a\n        converged solution more quickly. The latter method is subject to\n        cycling (non-convergence) in rare instances.\n    unknown_options : dict\n        Optional arguments not used by this particular solver. If\n        `unknown_options` is non-empty a warning is issued listing all\n        unused options.\n\n    Returns\n    -------\n    x : 1-D array\n        Solution vector.\n    status : int\n        An integer representing the exit status of the optimization::\n\n         0 : Optimization terminated successfully\n         1 : Iteration limit reached\n         2 : Problem appears to be infeasible\n         3 : Problem appears to be unbounded\n         4 : Serious numerical difficulties encountered\n\n    message : str\n        A string descriptor of the exit status of the optimization.\n    iteration : int\n        The number of iterations taken to solve the problem.\n\n    References\n    ----------\n    .. [1] Dantzig, George B., Linear programming and extensions. Rand\n           Corporation Research Study Princeton Univ. Press, Princeton, NJ,\n           1963\n    .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to\n           Mathematical Programming", McGraw-Hill, Chapter 4.\n    .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.\n           Mathematics of Operations Research (2), 1977: pp. 103-107.\n\n\n    Notes\n    -----\n    The expected problem formulation differs between the top level ``linprog``\n    module and the method specific solvers. The method specific solvers expect a\n    problem in standard form:\n\n    Minimize::\n\n        c @ x\n\n    Subject to::\n\n        A @ x == b\n            x >= 0\n\n    Whereas the top level ``linprog`` module expects a problem of form:\n\n    Minimize::\n\n        c @ x\n\n    Subject to::\n\n        A_ub @ x <= b_ub\n        A_eq @ x == b_eq\n         lb <= x <= ub\n\n    where ``lb = 0`` and ``ub = None`` unless set in ``bounds``.\n\n    The original problem contains equality, upper-bound and variable constraints\n    whereas the method specific solver requires equality constraints and\n    variable non-negativity.\n\n    ``linprog`` module converts the original problem to standard form by\n    converting the simple bounds to upper bound constraints, introducing\n    non-negative slack variables for inequality constraints, and expressing\n    unbounded variables as the difference between two non-negative variables.\n    '
    _check_unknown_options(unknown_options)
    status = 0
    messages = {0: 'Optimization terminated successfully.', 1: 'Iteration limit reached.', 2: 'Optimization failed. Unable to find a feasible starting point.', 3: 'Optimization failed. The problem appears to be unbounded.', 4: 'Optimization failed. Singular matrix encountered.'}
    (n, m) = A.shape
    is_negative_constraint = np.less(b, 0)
    A[is_negative_constraint] *= -1
    b[is_negative_constraint] *= -1
    av = np.arange(n) + m
    basis = av.copy()
    row_constraints = np.hstack((A, np.eye(n), b[:, np.newaxis]))
    row_objective = np.hstack((c, np.zeros(n), c0))
    row_pseudo_objective = -row_constraints.sum(axis=0)
    row_pseudo_objective[av] = 0
    T = np.vstack((row_constraints, row_objective, row_pseudo_objective))
    (nit1, status) = _solve_simplex(T, n, basis, callback=callback, postsolve_args=postsolve_args, maxiter=maxiter, tol=tol, phase=1, bland=bland)
    nit2 = nit1
    if abs(T[-1, -1]) < tol:
        T = T[:-1, :]
        T = np.delete(T, av, 1)
    else:
        status = 2
        messages[status] = "Phase 1 of the simplex method failed to find a feasible solution. The pseudo-objective function evaluates to {0:.1e} which exceeds the required tolerance of {1} for a solution to be considered 'close enough' to zero to be a basic solution. Consider increasing the tolerance to be greater than {0:.1e}. If this tolerance is unacceptably  large the problem may be infeasible.".format(abs(T[-1, -1]), tol)
    if status == 0:
        (nit2, status) = _solve_simplex(T, n, basis, callback=callback, postsolve_args=postsolve_args, maxiter=maxiter, tol=tol, phase=2, bland=bland, nit0=nit1)
    solution = np.zeros(n + m)
    solution[basis[:n]] = T[:n, -1]
    x = solution[:m]
    return (x, status, messages[status], int(nit2))