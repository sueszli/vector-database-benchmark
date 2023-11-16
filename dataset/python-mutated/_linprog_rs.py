"""Revised simplex method for linear programming

The *revised simplex* method uses the method described in [1]_, except
that a factorization [2]_ of the basis matrix, rather than its inverse,
is efficiently maintained and used to solve the linear systems at each
iteration of the algorithm.

.. versionadded:: 1.3.0

References
----------
.. [1] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
.. [2] Bartels, Richard H. "A stabilization of the simplex method."
            Journal in  Numerische Mathematik 16.5 (1971): 414-434.

"""
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve
from ._optimize import _check_unknown_options
from ._bglu_dense import LU
from ._bglu_dense import BGLU as BGLU
from ._linprog_util import _postsolve
from ._optimize import OptimizeResult

def _phase_one(A, b, x0, callback, postsolve_args, maxiter, tol, disp, maxupdate, mast, pivot):
    if False:
        while True:
            i = 10
    '\n    The purpose of phase one is to find an initial basic feasible solution\n    (BFS) to the original problem.\n\n    Generates an auxiliary problem with a trivial BFS and an objective that\n    minimizes infeasibility of the original problem. Solves the auxiliary\n    problem using the main simplex routine (phase two). This either yields\n    a BFS to the original problem or determines that the original problem is\n    infeasible. If feasible, phase one detects redundant rows in the original\n    constraint matrix and removes them, then chooses additional indices as\n    necessary to complete a basis/BFS for the original problem.\n    '
    (m, n) = A.shape
    status = 0
    (A, b, c, basis, x, status) = _generate_auxiliary_problem(A, b, x0, tol)
    if status == 6:
        residual = c.dot(x)
        iter_k = 0
        return (x, basis, A, b, residual, status, iter_k)
    phase_one_n = n
    iter_k = 0
    (x, basis, status, iter_k) = _phase_two(c, A, x, basis, callback, postsolve_args, maxiter, tol, disp, maxupdate, mast, pivot, iter_k, phase_one_n)
    residual = c.dot(x)
    if status == 0 and residual > tol:
        status = 2
    keep_rows = np.ones(m, dtype=bool)
    for basis_column in basis[basis >= n]:
        B = A[:, basis]
        try:
            basis_finder = np.abs(solve(B, A))
            pertinent_row = np.argmax(basis_finder[:, basis_column])
            eligible_columns = np.ones(n, dtype=bool)
            eligible_columns[basis[basis < n]] = 0
            eligible_column_indices = np.where(eligible_columns)[0]
            index = np.argmax(basis_finder[:, :n][pertinent_row, eligible_columns])
            new_basis_column = eligible_column_indices[index]
            if basis_finder[pertinent_row, new_basis_column] < tol:
                keep_rows[pertinent_row] = False
            else:
                basis[basis == basis_column] = new_basis_column
        except LinAlgError:
            status = 4
    A = A[keep_rows, :n]
    basis = basis[keep_rows]
    x = x[:n]
    m = A.shape[0]
    return (x, basis, A, b, residual, status, iter_k)

def _get_more_basis_columns(A, basis):
    if False:
        return 10
    '\n    Called when the auxiliary problem terminates with artificial columns in\n    the basis, which must be removed and replaced with non-artificial\n    columns. Finds additional columns that do not make the matrix singular.\n    '
    (m, n) = A.shape
    a = np.arange(m + n)
    bl = np.zeros(len(a), dtype=bool)
    bl[basis] = 1
    options = a[~bl]
    options = options[options < n]
    B = np.zeros((m, m))
    B[:, 0:len(basis)] = A[:, basis]
    if basis.size > 0 and np.linalg.matrix_rank(B[:, :len(basis)]) < len(basis):
        raise Exception('Basis has dependent columns')
    rank = 0
    for i in range(n):
        new_basis = np.random.permutation(options)[:m - len(basis)]
        B[:, len(basis):] = A[:, new_basis]
        rank = np.linalg.matrix_rank(B)
        if rank == m:
            break
    return np.concatenate((basis, new_basis))

def _generate_auxiliary_problem(A, b, x0, tol):
    if False:
        i = 10
        return i + 15
    "\n    Modifies original problem to create an auxiliary problem with a trivial\n    initial basic feasible solution and an objective that minimizes\n    infeasibility in the original problem.\n\n    Conceptually, this is done by stacking an identity matrix on the right of\n    the original constraint matrix, adding artificial variables to correspond\n    with each of these new columns, and generating a cost vector that is all\n    zeros except for ones corresponding with each of the new variables.\n\n    A initial basic feasible solution is trivial: all variables are zero\n    except for the artificial variables, which are set equal to the\n    corresponding element of the right hand side `b`.\n\n    Running the simplex method on this auxiliary problem drives all of the\n    artificial variables - and thus the cost - to zero if the original problem\n    is feasible. The original problem is declared infeasible otherwise.\n\n    Much of the complexity below is to improve efficiency by using singleton\n    columns in the original problem where possible, thus generating artificial\n    variables only as necessary, and using an initial 'guess' basic feasible\n    solution.\n    "
    status = 0
    (m, n) = A.shape
    if x0 is not None:
        x = x0
    else:
        x = np.zeros(n)
    r = b - A @ x
    A[r < 0] = -A[r < 0]
    b[r < 0] = -b[r < 0]
    r[r < 0] *= -1
    if x0 is None:
        nonzero_constraints = np.arange(m)
    else:
        nonzero_constraints = np.where(r > tol)[0]
    basis = np.where(np.abs(x) > tol)[0]
    if len(nonzero_constraints) == 0 and len(basis) <= m:
        c = np.zeros(n)
        basis = _get_more_basis_columns(A, basis)
        return (A, b, c, basis, x, status)
    elif len(nonzero_constraints) > m - len(basis) or np.any(x < 0):
        c = np.zeros(n)
        status = 6
        return (A, b, c, basis, x, status)
    (cols, rows) = _select_singleton_columns(A, r)
    i_tofix = np.isin(rows, nonzero_constraints)
    i_notinbasis = np.logical_not(np.isin(cols, basis))
    i_fix_without_aux = np.logical_and(i_tofix, i_notinbasis)
    rows = rows[i_fix_without_aux]
    cols = cols[i_fix_without_aux]
    arows = nonzero_constraints[np.logical_not(np.isin(nonzero_constraints, rows))]
    n_aux = len(arows)
    acols = n + np.arange(n_aux)
    basis_ng = np.concatenate((cols, acols))
    basis_ng_rows = np.concatenate((rows, arows))
    A = np.hstack((A, np.zeros((m, n_aux))))
    A[arows, acols] = 1
    x = np.concatenate((x, np.zeros(n_aux)))
    x[basis_ng] = r[basis_ng_rows] / A[basis_ng_rows, basis_ng]
    c = np.zeros(n_aux + n)
    c[acols] = 1
    basis = np.concatenate((basis, basis_ng))
    basis = _get_more_basis_columns(A, basis)
    return (A, b, c, basis, x, status)

def _select_singleton_columns(A, b):
    if False:
        while True:
            i = 10
    '\n    Finds singleton columns for which the singleton entry is of the same sign\n    as the right-hand side; these columns are eligible for inclusion in an\n    initial basis. Determines the rows in which the singleton entries are\n    located. For each of these rows, returns the indices of the one singleton\n    column and its corresponding row.\n    '
    column_indices = np.nonzero(np.sum(np.abs(A) != 0, axis=0) == 1)[0]
    columns = A[:, column_indices]
    row_indices = np.zeros(len(column_indices), dtype=int)
    (nonzero_rows, nonzero_columns) = np.nonzero(columns)
    row_indices[nonzero_columns] = nonzero_rows
    same_sign = A[row_indices, column_indices] * b[row_indices] >= 0
    column_indices = column_indices[same_sign][::-1]
    row_indices = row_indices[same_sign][::-1]
    (unique_row_indices, first_columns) = np.unique(row_indices, return_index=True)
    return (column_indices[first_columns], unique_row_indices)

def _find_nonzero_rows(A, tol):
    if False:
        i = 10
        return i + 15
    '\n    Returns logical array indicating the locations of rows with at least\n    one nonzero element.\n    '
    return np.any(np.abs(A) > tol, axis=1)

def _select_enter_pivot(c_hat, bl, a, rule='bland', tol=1e-12):
    if False:
        while True:
            i = 10
    "\n    Selects a pivot to enter the basis. Currently Bland's rule - the smallest\n    index that has a negative reduced cost - is the default.\n    "
    if rule.lower() == 'mrc':
        return a[~bl][np.argmin(c_hat)]
    else:
        return a[~bl][c_hat < -tol][0]

def _display_iter(phase, iteration, slack, con, fun):
    if False:
        for i in range(10):
            print('nop')
    '\n    Print indicators of optimization status to the console.\n    '
    header = True if not iteration % 20 else False
    if header:
        print('Phase', 'Iteration', 'Minimum Slack      ', 'Constraint Residual', 'Objective          ')
    fmt = '{0:<6}{1:<10}{2:<20.13}{3:<20.13}{4:<20.13}'
    try:
        slack = np.min(slack)
    except ValueError:
        slack = 'NA'
    print(fmt.format(phase, iteration, slack, np.linalg.norm(con), fun))

def _display_and_callback(phase_one_n, x, postsolve_args, status, iteration, disp, callback):
    if False:
        i = 10
        return i + 15
    if phase_one_n is not None:
        phase = 1
        x_postsolve = x[:phase_one_n]
    else:
        phase = 2
        x_postsolve = x
    (x_o, fun, slack, con) = _postsolve(x_postsolve, postsolve_args)
    if callback is not None:
        res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack, 'con': con, 'nit': iteration, 'phase': phase, 'complete': False, 'status': status, 'message': '', 'success': False})
        callback(res)
    if disp:
        _display_iter(phase, iteration, slack, con, fun)

def _phase_two(c, A, x, b, callback, postsolve_args, maxiter, tol, disp, maxupdate, mast, pivot, iteration=0, phase_one_n=None):
    if False:
        i = 10
        return i + 15
    '\n    The heart of the simplex method. Beginning with a basic feasible solution,\n    moves to adjacent basic feasible solutions successively lower reduced cost.\n    Terminates when there are no basic feasible solutions with lower reduced\n    cost or if the problem is determined to be unbounded.\n\n    This implementation follows the revised simplex method based on LU\n    decomposition. Rather than maintaining a tableau or an inverse of the\n    basis matrix, we keep a factorization of the basis matrix that allows\n    efficient solution of linear systems while avoiding stability issues\n    associated with inverted matrices.\n    '
    (m, n) = A.shape
    status = 0
    a = np.arange(n)
    ab = np.arange(m)
    if maxupdate:
        B = BGLU(A, b, maxupdate, mast)
    else:
        B = LU(A, b)
    for iteration in range(iteration, maxiter):
        if disp or callback is not None:
            _display_and_callback(phase_one_n, x, postsolve_args, status, iteration, disp, callback)
        bl = np.zeros(len(a), dtype=bool)
        bl[b] = 1
        xb = x[b]
        cb = c[b]
        try:
            v = B.solve(cb, transposed=True)
        except LinAlgError:
            status = 4
            break
        c_hat = c - v.dot(A)
        c_hat = c_hat[~bl]
        if np.all(c_hat >= -tol):
            break
        j = _select_enter_pivot(c_hat, bl, a, rule=pivot, tol=tol)
        u = B.solve(A[:, j])
        i = u > tol
        if not np.any(i):
            status = 3
            break
        th = xb[i] / u[i]
        l = np.argmin(th)
        th_star = th[l]
        x[b] = x[b] - th_star * u
        x[j] = th_star
        B.update(ab[i][l], j)
        b = B.b
    else:
        iteration += 1
        status = 1
        if disp or callback is not None:
            _display_and_callback(phase_one_n, x, postsolve_args, status, iteration, disp, callback)
    return (x, b, status, iteration)

def _linprog_rs(c, c0, A, b, x0, callback, postsolve_args, maxiter=5000, tol=1e-12, disp=False, maxupdate=10, mast=False, pivot='mrc', **unknown_options):
    if False:
        return 10
    '\n    Solve the following linear programming problem via a two-phase\n    revised simplex algorithm.::\n\n        minimize:     c @ x\n\n        subject to:  A @ x == b\n                     0 <= x < oo\n\n    User-facing documentation is in _linprog_doc.py.\n\n    Parameters\n    ----------\n    c : 1-D array\n        Coefficients of the linear objective function to be minimized.\n    c0 : float\n        Constant term in objective function due to fixed (and eliminated)\n        variables. (Currently unused.)\n    A : 2-D array\n        2-D array which, when matrix-multiplied by ``x``, gives the values of\n        the equality constraints at ``x``.\n    b : 1-D array\n        1-D array of values representing the RHS of each equality constraint\n        (row) in ``A_eq``.\n    x0 : 1-D array, optional\n        Starting values of the independent variables, which will be refined by\n        the optimization algorithm. For the revised simplex method, these must\n        correspond with a basic feasible solution.\n    callback : callable, optional\n        If a callback function is provided, it will be called within each\n        iteration of the algorithm. The callback function must accept a single\n        `scipy.optimize.OptimizeResult` consisting of the following fields:\n\n            x : 1-D array\n                Current solution vector.\n            fun : float\n                Current value of the objective function ``c @ x``.\n            success : bool\n                True only when an algorithm has completed successfully,\n                so this is always False as the callback function is called\n                only while the algorithm is still iterating.\n            slack : 1-D array\n                The values of the slack variables. Each slack variable\n                corresponds to an inequality constraint. If the slack is zero,\n                the corresponding constraint is active.\n            con : 1-D array\n                The (nominally zero) residuals of the equality constraints,\n                that is, ``b - A_eq @ x``.\n            phase : int\n                The phase of the algorithm being executed.\n            status : int\n                For revised simplex, this is always 0 because if a different\n                status is detected, the algorithm terminates.\n            nit : int\n                The number of iterations performed.\n            message : str\n                A string descriptor of the exit status of the optimization.\n    postsolve_args : tuple\n        Data needed by _postsolve to convert the solution to the standard-form\n        problem into the solution to the original problem.\n\n    Options\n    -------\n    maxiter : int\n       The maximum number of iterations to perform in either phase.\n    tol : float\n        The tolerance which determines when a solution is "close enough" to\n        zero in Phase 1 to be considered a basic feasible solution or close\n        enough to positive to serve as an optimal solution.\n    disp : bool\n        Set to ``True`` if indicators of optimization status are to be printed\n        to the console each iteration.\n    maxupdate : int\n        The maximum number of updates performed on the LU factorization.\n        After this many updates is reached, the basis matrix is factorized\n        from scratch.\n    mast : bool\n        Minimize Amortized Solve Time. If enabled, the average time to solve\n        a linear system using the basis factorization is measured. Typically,\n        the average solve time will decrease with each successive solve after\n        initial factorization, as factorization takes much more time than the\n        solve operation (and updates). Eventually, however, the updated\n        factorization becomes sufficiently complex that the average solve time\n        begins to increase. When this is detected, the basis is refactorized\n        from scratch. Enable this option to maximize speed at the risk of\n        nondeterministic behavior. Ignored if ``maxupdate`` is 0.\n    pivot : "mrc" or "bland"\n        Pivot rule: Minimum Reduced Cost (default) or Bland\'s rule. Choose\n        Bland\'s rule if iteration limit is reached and cycling is suspected.\n    unknown_options : dict\n        Optional arguments not used by this particular solver. If\n        `unknown_options` is non-empty a warning is issued listing all\n        unused options.\n\n    Returns\n    -------\n    x : 1-D array\n        Solution vector.\n    status : int\n        An integer representing the exit status of the optimization::\n\n         0 : Optimization terminated successfully\n         1 : Iteration limit reached\n         2 : Problem appears to be infeasible\n         3 : Problem appears to be unbounded\n         4 : Numerical difficulties encountered\n         5 : No constraints; turn presolve on\n         6 : Guess x0 cannot be converted to a basic feasible solution\n\n    message : str\n        A string descriptor of the exit status of the optimization.\n    iteration : int\n        The number of iterations taken to solve the problem.\n    '
    _check_unknown_options(unknown_options)
    messages = ['Optimization terminated successfully.', 'Iteration limit reached.', 'The problem appears infeasible, as the phase one auxiliary problem terminated successfully with a residual of {0:.1e}, greater than the tolerance {1} required for the solution to be considered feasible. Consider increasing the tolerance to be greater than {0:.1e}. If this tolerance is unnaceptably large, the problem is likely infeasible.', 'The problem is unbounded, as the simplex algorithm found a basic feasible solution from which there is a direction with negative reduced cost in which all decision variables increase.', "Numerical difficulties encountered; consider trying method='interior-point'.", 'Problems with no constraints are trivially solved; please turn presolve on.', 'The guess x0 cannot be converted to a basic feasible solution. ']
    if A.size == 0:
        return (np.zeros(c.shape), 5, messages[5], 0)
    (x, basis, A, b, residual, status, iteration) = _phase_one(A, b, x0, callback, postsolve_args, maxiter, tol, disp, maxupdate, mast, pivot)
    if status == 0:
        (x, basis, status, iteration) = _phase_two(c, A, x, basis, callback, postsolve_args, maxiter, tol, disp, maxupdate, mast, pivot, iteration)
    return (x, status, messages[status].format(residual, tol), iteration)