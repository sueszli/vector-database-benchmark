"""
Method agnostic utility functions for linear progamming
"""
import numpy as np
import scipy.sparse as sps
from warnings import warn
from ._optimize import OptimizeWarning
from scipy.optimize._remove_redundancy import _remove_redundancy_svd, _remove_redundancy_pivot_sparse, _remove_redundancy_pivot_dense, _remove_redundancy_id
from collections import namedtuple
_LPProblem = namedtuple('_LPProblem', 'c A_ub b_ub A_eq b_eq bounds x0 integrality')
_LPProblem.__new__.__defaults__ = (None,) * 7
_LPProblem.__doc__ = " Represents a linear-programming problem.\n\n    Attributes\n    ----------\n    c : 1D array\n        The coefficients of the linear objective function to be minimized.\n    A_ub : 2D array, optional\n        The inequality constraint matrix. Each row of ``A_ub`` specifies the\n        coefficients of a linear inequality constraint on ``x``.\n    b_ub : 1D array, optional\n        The inequality constraint vector. Each element represents an\n        upper bound on the corresponding value of ``A_ub @ x``.\n    A_eq : 2D array, optional\n        The equality constraint matrix. Each row of ``A_eq`` specifies the\n        coefficients of a linear equality constraint on ``x``.\n    b_eq : 1D array, optional\n        The equality constraint vector. Each element of ``A_eq @ x`` must equal\n        the corresponding element of ``b_eq``.\n    bounds : various valid formats, optional\n        The bounds of ``x``, as ``min`` and ``max`` pairs.\n        If bounds are specified for all N variables separately, valid formats\n        are:\n        * a 2D array (N x 2);\n        * a sequence of N sequences, each with 2 values.\n        If all variables have the same bounds, the bounds can be specified as\n        a 1-D or 2-D array or sequence with 2 scalar values.\n        If all variables have a lower bound of 0 and no upper bound, the bounds\n        parameter can be omitted (or given as None).\n        Absent lower and/or upper bounds can be specified as -numpy.inf (no\n        lower bound), numpy.inf (no upper bound) or None (both).\n    x0 : 1D array, optional\n        Guess values of the decision variables, which will be refined by\n        the optimization algorithm. This argument is currently used only by the\n        'revised simplex' method, and can only be used if `x0` represents a\n        basic feasible solution.\n    integrality : 1-D array or int, optional\n        Indicates the type of integrality constraint on each decision variable.\n\n        ``0`` : Continuous variable; no integrality constraint.\n\n        ``1`` : Integer variable; decision variable must be an integer\n        within `bounds`.\n\n        ``2`` : Semi-continuous variable; decision variable must be within\n        `bounds` or take value ``0``.\n\n        ``3`` : Semi-integer variable; decision variable must be an integer\n        within `bounds` or take value ``0``.\n\n        By default, all variables are continuous.\n\n        For mixed integrality constraints, supply an array of shape `c.shape`.\n        To infer a constraint on each decision variable from shorter inputs,\n        the argument will be broadcasted to `c.shape` using `np.broadcast_to`.\n\n        This argument is currently used only by the ``'highs'`` method and\n        ignored otherwise.\n\n    Notes\n    -----\n    This namedtuple supports 2 ways of initialization:\n    >>> lp1 = _LPProblem(c=[-1, 4], A_ub=[[-3, 1], [1, 2]], b_ub=[6, 4])\n    >>> lp2 = _LPProblem([-1, 4], [[-3, 1], [1, 2]], [6, 4])\n\n    Note that only ``c`` is a required argument here, whereas all other arguments\n    ``A_ub``, ``b_ub``, ``A_eq``, ``b_eq``, ``bounds``, ``x0`` are optional with\n    default values of None.\n    For example, ``A_eq`` and ``b_eq`` can be set without ``A_ub`` or ``b_ub``:\n    >>> lp3 = _LPProblem(c=[-1, 4], A_eq=[[2, 1]], b_eq=[10])\n    "

def _check_sparse_inputs(options, meth, A_ub, A_eq):
    if False:
        while True:
            i = 10
    "\n    Check the provided ``A_ub`` and ``A_eq`` matrices conform to the specified\n    optional sparsity variables.\n\n    Parameters\n    ----------\n    A_ub : 2-D array, optional\n        2-D array such that ``A_ub @ x`` gives the values of the upper-bound\n        inequality constraints at ``x``.\n    A_eq : 2-D array, optional\n        2-D array such that ``A_eq @ x`` gives the values of the equality\n        constraints at ``x``.\n    options : dict\n        A dictionary of solver options. All methods accept the following\n        generic options:\n\n            maxiter : int\n                Maximum number of iterations to perform.\n            disp : bool\n                Set to True to print convergence messages.\n\n        For method-specific options, see :func:`show_options('linprog')`.\n    method : str, optional\n        The algorithm used to solve the standard form problem.\n\n    Returns\n    -------\n    A_ub : 2-D array, optional\n        2-D array such that ``A_ub @ x`` gives the values of the upper-bound\n        inequality constraints at ``x``.\n    A_eq : 2-D array, optional\n        2-D array such that ``A_eq @ x`` gives the values of the equality\n        constraints at ``x``.\n    options : dict\n        A dictionary of solver options. All methods accept the following\n        generic options:\n\n            maxiter : int\n                Maximum number of iterations to perform.\n            disp : bool\n                Set to True to print convergence messages.\n\n        For method-specific options, see :func:`show_options('linprog')`.\n    "
    _sparse_presolve = options.pop('_sparse_presolve', False)
    if _sparse_presolve and A_eq is not None:
        A_eq = sps.coo_matrix(A_eq)
    if _sparse_presolve and A_ub is not None:
        A_ub = sps.coo_matrix(A_ub)
    sparse_constraint = sps.issparse(A_eq) or sps.issparse(A_ub)
    preferred_methods = {'highs', 'highs-ds', 'highs-ipm'}
    dense_methods = {'simplex', 'revised simplex'}
    if meth in dense_methods and sparse_constraint:
        raise ValueError(f"Method '{meth}' does not support sparse constraint matrices. Please consider using one of {preferred_methods}.")
    sparse = options.get('sparse', False)
    if not sparse and sparse_constraint and (meth == 'interior-point'):
        options['sparse'] = True
        warn("Sparse constraint matrix detected; setting 'sparse':True.", OptimizeWarning, stacklevel=4)
    return (options, A_ub, A_eq)

def _format_A_constraints(A, n_x, sparse_lhs=False):
    if False:
        for i in range(10):
            print('nop')
    'Format the left hand side of the constraints to a 2-D array\n\n    Parameters\n    ----------\n    A : 2-D array\n        2-D array such that ``A @ x`` gives the values of the upper-bound\n        (in)equality constraints at ``x``.\n    n_x : int\n        The number of variables in the linear programming problem.\n    sparse_lhs : bool\n        Whether either of `A_ub` or `A_eq` are sparse. If true return a\n        coo_matrix instead of a numpy array.\n\n    Returns\n    -------\n    np.ndarray or sparse.coo_matrix\n        2-D array such that ``A @ x`` gives the values of the upper-bound\n        (in)equality constraints at ``x``.\n\n    '
    if sparse_lhs:
        return sps.coo_matrix((0, n_x) if A is None else A, dtype=float, copy=True)
    elif A is None:
        return np.zeros((0, n_x), dtype=float)
    else:
        return np.array(A, dtype=float, copy=True)

def _format_b_constraints(b):
    if False:
        print('Hello World!')
    'Format the upper bounds of the constraints to a 1-D array\n\n    Parameters\n    ----------\n    b : 1-D array\n        1-D array of values representing the upper-bound of each (in)equality\n        constraint (row) in ``A``.\n\n    Returns\n    -------\n    1-D np.array\n        1-D array of values representing the upper-bound of each (in)equality\n        constraint (row) in ``A``.\n\n    '
    if b is None:
        return np.array([], dtype=float)
    b = np.array(b, dtype=float, copy=True).squeeze()
    return b if b.size != 1 else b.reshape(-1)

def _clean_inputs(lp):
    if False:
        return 10
    "\n    Given user inputs for a linear programming problem, return the\n    objective vector, upper bound constraints, equality constraints,\n    and simple bounds in a preferred format.\n\n    Parameters\n    ----------\n    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:\n\n        c : 1D array\n            The coefficients of the linear objective function to be minimized.\n        A_ub : 2D array, optional\n            The inequality constraint matrix. Each row of ``A_ub`` specifies the\n            coefficients of a linear inequality constraint on ``x``.\n        b_ub : 1D array, optional\n            The inequality constraint vector. Each element represents an\n            upper bound on the corresponding value of ``A_ub @ x``.\n        A_eq : 2D array, optional\n            The equality constraint matrix. Each row of ``A_eq`` specifies the\n            coefficients of a linear equality constraint on ``x``.\n        b_eq : 1D array, optional\n            The equality constraint vector. Each element of ``A_eq @ x`` must equal\n            the corresponding element of ``b_eq``.\n        bounds : various valid formats, optional\n            The bounds of ``x``, as ``min`` and ``max`` pairs.\n            If bounds are specified for all N variables separately, valid formats are:\n            * a 2D array (2 x N or N x 2);\n            * a sequence of N sequences, each with 2 values.\n            If all variables have the same bounds, a single pair of values can\n            be specified. Valid formats are:\n            * a sequence with 2 scalar values;\n            * a sequence with a single element containing 2 scalar values.\n            If all variables have a lower bound of 0 and no upper bound, the bounds\n            parameter can be omitted (or given as None).\n        x0 : 1D array, optional\n            Guess values of the decision variables, which will be refined by\n            the optimization algorithm. This argument is currently used only by the\n            'revised simplex' method, and can only be used if `x0` represents a\n            basic feasible solution.\n\n    Returns\n    -------\n    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:\n\n        c : 1D array\n            The coefficients of the linear objective function to be minimized.\n        A_ub : 2D array, optional\n            The inequality constraint matrix. Each row of ``A_ub`` specifies the\n            coefficients of a linear inequality constraint on ``x``.\n        b_ub : 1D array, optional\n            The inequality constraint vector. Each element represents an\n            upper bound on the corresponding value of ``A_ub @ x``.\n        A_eq : 2D array, optional\n            The equality constraint matrix. Each row of ``A_eq`` specifies the\n            coefficients of a linear equality constraint on ``x``.\n        b_eq : 1D array, optional\n            The equality constraint vector. Each element of ``A_eq @ x`` must equal\n            the corresponding element of ``b_eq``.\n        bounds : 2D array\n            The bounds of ``x``, as ``min`` and ``max`` pairs, one for each of the N\n            elements of ``x``. The N x 2 array contains lower bounds in the first\n            column and upper bounds in the 2nd. Unbounded variables have lower\n            bound -np.inf and/or upper bound np.inf.\n        x0 : 1D array, optional\n            Guess values of the decision variables, which will be refined by\n            the optimization algorithm. This argument is currently used only by the\n            'revised simplex' method, and can only be used if `x0` represents a\n            basic feasible solution.\n\n    "
    (c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality) = lp
    if c is None:
        raise TypeError
    try:
        c = np.array(c, dtype=np.float64, copy=True).squeeze()
    except ValueError as e:
        raise TypeError('Invalid input for linprog: c must be a 1-D array of numerical coefficients') from e
    else:
        if c.size == 1:
            c = c.reshape(-1)
        n_x = len(c)
        if n_x == 0 or len(c.shape) != 1:
            raise ValueError('Invalid input for linprog: c must be a 1-D array and must not have more than one non-singleton dimension')
        if not np.isfinite(c).all():
            raise ValueError('Invalid input for linprog: c must not contain values inf, nan, or None')
    sparse_lhs = sps.issparse(A_eq) or sps.issparse(A_ub)
    try:
        A_ub = _format_A_constraints(A_ub, n_x, sparse_lhs=sparse_lhs)
    except ValueError as e:
        raise TypeError('Invalid input for linprog: A_ub must be a 2-D array of numerical values') from e
    else:
        n_ub = A_ub.shape[0]
        if len(A_ub.shape) != 2 or A_ub.shape[1] != n_x:
            raise ValueError('Invalid input for linprog: A_ub must have exactly two dimensions, and the number of columns in A_ub must be equal to the size of c')
        if sps.issparse(A_ub) and (not np.isfinite(A_ub.data).all()) or (not sps.issparse(A_ub) and (not np.isfinite(A_ub).all())):
            raise ValueError('Invalid input for linprog: A_ub must not contain values inf, nan, or None')
    try:
        b_ub = _format_b_constraints(b_ub)
    except ValueError as e:
        raise TypeError('Invalid input for linprog: b_ub must be a 1-D array of numerical values, each representing the upper bound of an inequality constraint (row) in A_ub') from e
    else:
        if b_ub.shape != (n_ub,):
            raise ValueError('Invalid input for linprog: b_ub must be a 1-D array; b_ub must not have more than one non-singleton dimension and the number of rows in A_ub must equal the number of values in b_ub')
        if not np.isfinite(b_ub).all():
            raise ValueError('Invalid input for linprog: b_ub must not contain values inf, nan, or None')
    try:
        A_eq = _format_A_constraints(A_eq, n_x, sparse_lhs=sparse_lhs)
    except ValueError as e:
        raise TypeError('Invalid input for linprog: A_eq must be a 2-D array of numerical values') from e
    else:
        n_eq = A_eq.shape[0]
        if len(A_eq.shape) != 2 or A_eq.shape[1] != n_x:
            raise ValueError('Invalid input for linprog: A_eq must have exactly two dimensions, and the number of columns in A_eq must be equal to the size of c')
        if sps.issparse(A_eq) and (not np.isfinite(A_eq.data).all()) or (not sps.issparse(A_eq) and (not np.isfinite(A_eq).all())):
            raise ValueError('Invalid input for linprog: A_eq must not contain values inf, nan, or None')
    try:
        b_eq = _format_b_constraints(b_eq)
    except ValueError as e:
        raise TypeError('Invalid input for linprog: b_eq must be a dense, 1-D array of numerical values, each representing the right hand side of an equality constraint (row) in A_eq') from e
    else:
        if b_eq.shape != (n_eq,):
            raise ValueError('Invalid input for linprog: b_eq must be a 1-D array; b_eq must not have more than one non-singleton dimension and the number of rows in A_eq must equal the number of values in b_eq')
        if not np.isfinite(b_eq).all():
            raise ValueError('Invalid input for linprog: b_eq must not contain values inf, nan, or None')
    if x0 is not None:
        try:
            x0 = np.array(x0, dtype=float, copy=True).squeeze()
        except ValueError as e:
            raise TypeError('Invalid input for linprog: x0 must be a 1-D array of numerical coefficients') from e
        if x0.ndim == 0:
            x0 = x0.reshape(-1)
        if len(x0) == 0 or x0.ndim != 1:
            raise ValueError('Invalid input for linprog: x0 should be a 1-D array; it must not have more than one non-singleton dimension')
        if not x0.size == c.size:
            raise ValueError('Invalid input for linprog: x0 and c should contain the same number of elements')
        if not np.isfinite(x0).all():
            raise ValueError('Invalid input for linprog: x0 must not contain values inf, nan, or None')
    bounds_clean = np.zeros((n_x, 2), dtype=float)
    if bounds is None or np.array_equal(bounds, []) or np.array_equal(bounds, [[]]):
        bounds = (0, np.inf)
    try:
        bounds_conv = np.atleast_2d(np.array(bounds, dtype=float))
    except ValueError as e:
        raise ValueError('Invalid input for linprog: unable to interpret bounds, check values and dimensions: ' + e.args[0]) from e
    except TypeError as e:
        raise TypeError('Invalid input for linprog: unable to interpret bounds, check values and dimensions: ' + e.args[0]) from e
    bsh = bounds_conv.shape
    if len(bsh) > 2:
        raise ValueError('Invalid input for linprog: provide a 2-D array for bounds, not a {:d}-D array.'.format(len(bsh)))
    elif np.all(bsh == (n_x, 2)):
        bounds_clean = bounds_conv
    elif np.all(bsh == (2, 1)) or np.all(bsh == (1, 2)):
        bounds_flat = bounds_conv.flatten()
        bounds_clean[:, 0] = bounds_flat[0]
        bounds_clean[:, 1] = bounds_flat[1]
    elif np.all(bsh == (2, n_x)):
        raise ValueError('Invalid input for linprog: provide a {:d} x 2 array for bounds, not a 2 x {:d} array.'.format(n_x, n_x))
    else:
        raise ValueError('Invalid input for linprog: unable to interpret bounds with this dimension tuple: {}.'.format(bsh))
    i_none = np.isnan(bounds_clean[:, 0])
    bounds_clean[i_none, 0] = -np.inf
    i_none = np.isnan(bounds_clean[:, 1])
    bounds_clean[i_none, 1] = np.inf
    return _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds_clean, x0, integrality)

def _presolve(lp, rr, rr_method, tol=1e-09):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given inputs for a linear programming problem in preferred format,\n    presolve the problem: identify trivial infeasibilities, redundancies,\n    and unboundedness, tighten bounds where possible, and eliminate fixed\n    variables.\n\n    Parameters\n    ----------\n    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:\n\n        c : 1D array\n            The coefficients of the linear objective function to be minimized.\n        A_ub : 2D array, optional\n            The inequality constraint matrix. Each row of ``A_ub`` specifies the\n            coefficients of a linear inequality constraint on ``x``.\n        b_ub : 1D array, optional\n            The inequality constraint vector. Each element represents an\n            upper bound on the corresponding value of ``A_ub @ x``.\n        A_eq : 2D array, optional\n            The equality constraint matrix. Each row of ``A_eq`` specifies the\n            coefficients of a linear equality constraint on ``x``.\n        b_eq : 1D array, optional\n            The equality constraint vector. Each element of ``A_eq @ x`` must equal\n            the corresponding element of ``b_eq``.\n        bounds : 2D array\n            The bounds of ``x``, as ``min`` and ``max`` pairs, one for each of the N\n            elements of ``x``. The N x 2 array contains lower bounds in the first\n            column and upper bounds in the 2nd. Unbounded variables have lower\n            bound -np.inf and/or upper bound np.inf.\n        x0 : 1D array, optional\n            Guess values of the decision variables, which will be refined by\n            the optimization algorithm. This argument is currently used only by the\n            \'revised simplex\' method, and can only be used if `x0` represents a\n            basic feasible solution.\n\n    rr : bool\n        If ``True`` attempts to eliminate any redundant rows in ``A_eq``.\n        Set False if ``A_eq`` is known to be of full row rank, or if you are\n        looking for a potential speedup (at the expense of reliability).\n    rr_method : string\n        Method used to identify and remove redundant rows from the\n        equality constraint matrix after presolve.\n    tol : float\n        The tolerance which determines when a solution is "close enough" to\n        zero in Phase 1 to be considered a basic feasible solution or close\n        enough to positive to serve as an optimal solution.\n\n    Returns\n    -------\n    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:\n\n        c : 1D array\n            The coefficients of the linear objective function to be minimized.\n        A_ub : 2D array, optional\n            The inequality constraint matrix. Each row of ``A_ub`` specifies the\n            coefficients of a linear inequality constraint on ``x``.\n        b_ub : 1D array, optional\n            The inequality constraint vector. Each element represents an\n            upper bound on the corresponding value of ``A_ub @ x``.\n        A_eq : 2D array, optional\n            The equality constraint matrix. Each row of ``A_eq`` specifies the\n            coefficients of a linear equality constraint on ``x``.\n        b_eq : 1D array, optional\n            The equality constraint vector. Each element of ``A_eq @ x`` must equal\n            the corresponding element of ``b_eq``.\n        bounds : 2D array\n            The bounds of ``x``, as ``min`` and ``max`` pairs, possibly tightened.\n        x0 : 1D array, optional\n            Guess values of the decision variables, which will be refined by\n            the optimization algorithm. This argument is currently used only by the\n            \'revised simplex\' method, and can only be used if `x0` represents a\n            basic feasible solution.\n\n    c0 : 1D array\n        Constant term in objective function due to fixed (and eliminated)\n        variables.\n    x : 1D array\n        Solution vector (when the solution is trivial and can be determined\n        in presolve)\n    revstack: list of functions\n        the functions in the list reverse the operations of _presolve()\n        the function signature is x_org = f(x_mod), where x_mod is the result\n        of a presolve step and x_org the value at the start of the step\n        (currently, the revstack contains only one function)\n    complete: bool\n        Whether the solution is complete (solved or determined to be infeasible\n        or unbounded in presolve)\n    status : int\n        An integer representing the exit status of the optimization::\n\n         0 : Optimization terminated successfully\n         1 : Iteration limit reached\n         2 : Problem appears to be infeasible\n         3 : Problem appears to be unbounded\n         4 : Serious numerical difficulties encountered\n\n    message : str\n        A string descriptor of the exit status of the optimization.\n\n    References\n    ----------\n    .. [5] Andersen, Erling D. "Finding all linearly dependent rows in\n           large-scale linear programming." Optimization Methods and Software\n           6.3 (1995): 219-227.\n    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear\n           programming." Mathematical Programming 71.2 (1995): 221-245.\n\n    '
    (c, A_ub, b_ub, A_eq, b_eq, bounds, x0, _) = lp
    revstack = []
    c0 = 0
    complete = False
    x = np.zeros(c.shape)
    status = 0
    message = ''
    lb = bounds[:, 0].copy()
    ub = bounds[:, 1].copy()
    (m_eq, n) = A_eq.shape
    (m_ub, n) = A_ub.shape
    if rr_method is not None and rr_method.lower() not in {'svd', 'pivot', 'id'}:
        message = "'" + str(rr_method) + "' is not a valid option for redundancy removal. Valid options are 'SVD', 'pivot', and 'ID'."
        raise ValueError(message)
    if sps.issparse(A_eq):
        A_eq = A_eq.tocsr()
        A_ub = A_ub.tocsr()

        def where(A):
            if False:
                while True:
                    i = 10
            return A.nonzero()
        vstack = sps.vstack
    else:
        where = np.where
        vstack = np.vstack
    if np.any(ub < lb) or np.any(lb == np.inf) or np.any(ub == -np.inf):
        status = 2
        message = 'The problem is (trivially) infeasible since one or more upper bounds are smaller than the corresponding lower bounds, a lower bound is np.inf or an upper bound is -np.inf.'
        complete = True
        return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
    zero_row = np.array(np.sum(A_eq != 0, axis=1) == 0).flatten()
    if np.any(zero_row):
        if np.any(np.logical_and(zero_row, np.abs(b_eq) > tol)):
            status = 2
            message = 'The problem is (trivially) infeasible due to a row of zeros in the equality constraint matrix with a nonzero corresponding constraint value.'
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
        else:
            A_eq = A_eq[np.logical_not(zero_row), :]
            b_eq = b_eq[np.logical_not(zero_row)]
    zero_row = np.array(np.sum(A_ub != 0, axis=1) == 0).flatten()
    if np.any(zero_row):
        if np.any(np.logical_and(zero_row, b_ub < -tol)):
            status = 2
            message = 'The problem is (trivially) infeasible due to a row of zeros in the equality constraint matrix with a nonzero corresponding  constraint value.'
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
        else:
            A_ub = A_ub[np.logical_not(zero_row), :]
            b_ub = b_ub[np.logical_not(zero_row)]
    A = vstack((A_eq, A_ub))
    if A.shape[0] > 0:
        zero_col = np.array(np.sum(A != 0, axis=0) == 0).flatten()
        x[np.logical_and(zero_col, c < 0)] = ub[np.logical_and(zero_col, c < 0)]
        x[np.logical_and(zero_col, c > 0)] = lb[np.logical_and(zero_col, c > 0)]
        if np.any(np.isinf(x)):
            status = 3
            message = 'If feasible, the problem is (trivially) unbounded due  to a zero column in the constraint matrices. If you wish to check whether the problem is infeasible, turn presolve off.'
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
        lb[np.logical_and(zero_col, c < 0)] = ub[np.logical_and(zero_col, c < 0)]
        ub[np.logical_and(zero_col, c > 0)] = lb[np.logical_and(zero_col, c > 0)]
    singleton_row = np.array(np.sum(A_eq != 0, axis=1) == 1).flatten()
    rows = where(singleton_row)[0]
    cols = where(A_eq[rows, :])[1]
    if len(rows) > 0:
        for (row, col) in zip(rows, cols):
            val = b_eq[row] / A_eq[row, col]
            if not lb[col] - tol <= val <= ub[col] + tol:
                status = 2
                message = 'The problem is (trivially) infeasible because a singleton row in the equality constraints is inconsistent with the bounds.'
                complete = True
                return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
            else:
                lb[col] = val
                ub[col] = val
        A_eq = A_eq[np.logical_not(singleton_row), :]
        b_eq = b_eq[np.logical_not(singleton_row)]
    singleton_row = np.array(np.sum(A_ub != 0, axis=1) == 1).flatten()
    cols = where(A_ub[singleton_row, :])[1]
    rows = where(singleton_row)[0]
    if len(rows) > 0:
        for (row, col) in zip(rows, cols):
            val = b_ub[row] / A_ub[row, col]
            if A_ub[row, col] > 0:
                if val < lb[col] - tol:
                    complete = True
                elif val < ub[col]:
                    ub[col] = val
            elif val > ub[col] + tol:
                complete = True
            elif val > lb[col]:
                lb[col] = val
            if complete:
                status = 2
                message = 'The problem is (trivially) infeasible because a singleton row in the upper bound constraints is inconsistent with the bounds.'
                return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
        A_ub = A_ub[np.logical_not(singleton_row), :]
        b_ub = b_ub[np.logical_not(singleton_row)]
    i_f = np.abs(lb - ub) < tol
    i_nf = np.logical_not(i_f)
    if np.all(i_f):
        residual = b_eq - A_eq.dot(lb)
        slack = b_ub - A_ub.dot(lb)
        if A_ub.size > 0 and np.any(slack < 0) or (A_eq.size > 0 and (not np.allclose(residual, 0))):
            status = 2
            message = 'The problem is (trivially) infeasible because the bounds fix all variables to values inconsistent with the constraints'
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
    ub_mod = ub
    lb_mod = lb
    if np.any(i_f):
        c0 += c[i_f].dot(lb[i_f])
        b_eq = b_eq - A_eq[:, i_f].dot(lb[i_f])
        b_ub = b_ub - A_ub[:, i_f].dot(lb[i_f])
        c = c[i_nf]
        x_undo = lb[i_f]
        x = x[i_nf]
        if x0 is not None:
            x0 = x0[i_nf]
        A_eq = A_eq[:, i_nf]
        A_ub = A_ub[:, i_nf]
        lb_mod = lb[i_nf]
        ub_mod = ub[i_nf]

        def rev(x_mod):
            if False:
                for i in range(10):
                    print('nop')
            i = np.flatnonzero(i_f)
            N = len(i)
            index_offset = np.arange(N)
            insert_indices = i - index_offset
            x_rev = np.insert(x_mod.astype(float), insert_indices, x_undo)
            return x_rev
        revstack.append(rev)
    if A_eq.size == 0 and A_ub.size == 0:
        b_eq = np.array([])
        b_ub = np.array([])
        if c.size == 0:
            status = 0
            message = 'The solution was determined in presolve as there are no non-trivial constraints.'
        elif np.any(np.logical_and(c < 0, ub_mod == np.inf)) or np.any(np.logical_and(c > 0, lb_mod == -np.inf)):
            status = 3
            message = 'The problem is (trivially) unbounded because there are no non-trivial constraints and a) at least one decision variable is unbounded above and its corresponding cost is negative, or b) at least one decision variable is unbounded below and its corresponding cost is positive. '
        else:
            status = 0
            message = 'The solution was determined in presolve as there are no non-trivial constraints.'
        complete = True
        x[c < 0] = ub_mod[c < 0]
        x[c > 0] = lb_mod[c > 0]
        x_zero_c = ub_mod[c == 0]
        x_zero_c[np.isinf(x_zero_c)] = ub_mod[c == 0][np.isinf(x_zero_c)]
        x_zero_c[np.isinf(x_zero_c)] = 0
        x[c == 0] = x_zero_c
    bounds = np.hstack((lb_mod[:, np.newaxis], ub_mod[:, np.newaxis]))
    n_rows_A = A_eq.shape[0]
    redundancy_warning = 'A_eq does not appear to be of full row rank. To improve performance, check the problem formulation for redundant equality constraints.'
    if sps.issparse(A_eq):
        if rr and A_eq.size > 0:
            rr_res = _remove_redundancy_pivot_sparse(A_eq, b_eq)
            (A_eq, b_eq, status, message) = rr_res
            if A_eq.shape[0] < n_rows_A:
                warn(redundancy_warning, OptimizeWarning, stacklevel=1)
            if status != 0:
                complete = True
        return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
    small_nullspace = 5
    if rr and A_eq.size > 0:
        try:
            rank = np.linalg.matrix_rank(A_eq)
        except Exception:
            rank = 0
    if rr and A_eq.size > 0 and (rank < A_eq.shape[0]):
        warn(redundancy_warning, OptimizeWarning, stacklevel=3)
        dim_row_nullspace = A_eq.shape[0] - rank
        if rr_method is None:
            if dim_row_nullspace <= small_nullspace:
                rr_res = _remove_redundancy_svd(A_eq, b_eq)
                (A_eq, b_eq, status, message) = rr_res
            if dim_row_nullspace > small_nullspace or status == 4:
                rr_res = _remove_redundancy_pivot_dense(A_eq, b_eq)
                (A_eq, b_eq, status, message) = rr_res
        else:
            rr_method = rr_method.lower()
            if rr_method == 'svd':
                rr_res = _remove_redundancy_svd(A_eq, b_eq)
                (A_eq, b_eq, status, message) = rr_res
            elif rr_method == 'pivot':
                rr_res = _remove_redundancy_pivot_dense(A_eq, b_eq)
                (A_eq, b_eq, status, message) = rr_res
            elif rr_method == 'id':
                rr_res = _remove_redundancy_id(A_eq, b_eq, rank)
                (A_eq, b_eq, status, message) = rr_res
            else:
                pass
        if A_eq.shape[0] < rank:
            message = 'Due to numerical issues, redundant equality constraints could not be removed automatically. Try providing your constraint matrices as sparse matrices to activate sparse presolve, try turning off redundancy removal, or try turning off presolve altogether.'
            status = 4
        if status != 0:
            complete = True
    return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)

def _parse_linprog(lp, options, meth):
    if False:
        return 10
    "\n    Parse the provided linear programming problem\n\n    ``_parse_linprog`` employs two main steps ``_check_sparse_inputs`` and\n    ``_clean_inputs``. ``_check_sparse_inputs`` checks for sparsity in the\n    provided constraints (``A_ub`` and ``A_eq) and if these match the provided\n    sparsity optional values.\n\n    ``_clean inputs`` checks of the provided inputs. If no violations are\n    identified the objective vector, upper bound constraints, equality\n    constraints, and simple bounds are returned in the expected format.\n\n    Parameters\n    ----------\n    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:\n\n        c : 1D array\n            The coefficients of the linear objective function to be minimized.\n        A_ub : 2D array, optional\n            The inequality constraint matrix. Each row of ``A_ub`` specifies the\n            coefficients of a linear inequality constraint on ``x``.\n        b_ub : 1D array, optional\n            The inequality constraint vector. Each element represents an\n            upper bound on the corresponding value of ``A_ub @ x``.\n        A_eq : 2D array, optional\n            The equality constraint matrix. Each row of ``A_eq`` specifies the\n            coefficients of a linear equality constraint on ``x``.\n        b_eq : 1D array, optional\n            The equality constraint vector. Each element of ``A_eq @ x`` must equal\n            the corresponding element of ``b_eq``.\n        bounds : various valid formats, optional\n            The bounds of ``x``, as ``min`` and ``max`` pairs.\n            If bounds are specified for all N variables separately, valid formats are:\n            * a 2D array (2 x N or N x 2);\n            * a sequence of N sequences, each with 2 values.\n            If all variables have the same bounds, a single pair of values can\n            be specified. Valid formats are:\n            * a sequence with 2 scalar values;\n            * a sequence with a single element containing 2 scalar values.\n            If all variables have a lower bound of 0 and no upper bound, the bounds\n            parameter can be omitted (or given as None).\n        x0 : 1D array, optional\n            Guess values of the decision variables, which will be refined by\n            the optimization algorithm. This argument is currently used only by the\n            'revised simplex' method, and can only be used if `x0` represents a\n            basic feasible solution.\n\n    options : dict\n        A dictionary of solver options. All methods accept the following\n        generic options:\n\n            maxiter : int\n                Maximum number of iterations to perform.\n            disp : bool\n                Set to True to print convergence messages.\n\n        For method-specific options, see :func:`show_options('linprog')`.\n\n    Returns\n    -------\n    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:\n\n        c : 1D array\n            The coefficients of the linear objective function to be minimized.\n        A_ub : 2D array, optional\n            The inequality constraint matrix. Each row of ``A_ub`` specifies the\n            coefficients of a linear inequality constraint on ``x``.\n        b_ub : 1D array, optional\n            The inequality constraint vector. Each element represents an\n            upper bound on the corresponding value of ``A_ub @ x``.\n        A_eq : 2D array, optional\n            The equality constraint matrix. Each row of ``A_eq`` specifies the\n            coefficients of a linear equality constraint on ``x``.\n        b_eq : 1D array, optional\n            The equality constraint vector. Each element of ``A_eq @ x`` must equal\n            the corresponding element of ``b_eq``.\n        bounds : 2D array\n            The bounds of ``x``, as ``min`` and ``max`` pairs, one for each of the N\n            elements of ``x``. The N x 2 array contains lower bounds in the first\n            column and upper bounds in the 2nd. Unbounded variables have lower\n            bound -np.inf and/or upper bound np.inf.\n        x0 : 1D array, optional\n            Guess values of the decision variables, which will be refined by\n            the optimization algorithm. This argument is currently used only by the\n            'revised simplex' method, and can only be used if `x0` represents a\n            basic feasible solution.\n\n    options : dict, optional\n        A dictionary of solver options. All methods accept the following\n        generic options:\n\n            maxiter : int\n                Maximum number of iterations to perform.\n            disp : bool\n                Set to True to print convergence messages.\n\n        For method-specific options, see :func:`show_options('linprog')`.\n\n    "
    if options is None:
        options = {}
    solver_options = {k: v for (k, v) in options.items()}
    (solver_options, A_ub, A_eq) = _check_sparse_inputs(solver_options, meth, lp.A_ub, lp.A_eq)
    lp = _clean_inputs(lp._replace(A_ub=A_ub, A_eq=A_eq))
    return (lp, solver_options)

def _get_Abc(lp, c0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a linear programming problem of the form:\n\n    Minimize::\n\n        c @ x\n\n    Subject to::\n\n        A_ub @ x <= b_ub\n        A_eq @ x == b_eq\n         lb <= x <= ub\n\n    where ``lb = 0`` and ``ub = None`` unless set in ``bounds``.\n\n    Return the problem in standard form:\n\n    Minimize::\n\n        c @ x\n\n    Subject to::\n\n        A @ x == b\n            x >= 0\n\n    by adding slack variables and making variable substitutions as necessary.\n\n    Parameters\n    ----------\n    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:\n\n        c : 1D array\n            The coefficients of the linear objective function to be minimized.\n        A_ub : 2D array, optional\n            The inequality constraint matrix. Each row of ``A_ub`` specifies the\n            coefficients of a linear inequality constraint on ``x``.\n        b_ub : 1D array, optional\n            The inequality constraint vector. Each element represents an\n            upper bound on the corresponding value of ``A_ub @ x``.\n        A_eq : 2D array, optional\n            The equality constraint matrix. Each row of ``A_eq`` specifies the\n            coefficients of a linear equality constraint on ``x``.\n        b_eq : 1D array, optional\n            The equality constraint vector. Each element of ``A_eq @ x`` must equal\n            the corresponding element of ``b_eq``.\n        bounds : 2D array\n            The bounds of ``x``, lower bounds in the 1st column, upper\n            bounds in the 2nd column. The bounds are possibly tightened\n            by the presolve procedure.\n        x0 : 1D array, optional\n            Guess values of the decision variables, which will be refined by\n            the optimization algorithm. This argument is currently used only by the\n            \'revised simplex\' method, and can only be used if `x0` represents a\n            basic feasible solution.\n\n    c0 : float\n        Constant term in objective function due to fixed (and eliminated)\n        variables.\n\n    Returns\n    -------\n    A : 2-D array\n        2-D array such that ``A`` @ ``x``, gives the values of the equality\n        constraints at ``x``.\n    b : 1-D array\n        1-D array of values representing the RHS of each equality constraint\n        (row) in A (for standard form problem).\n    c : 1-D array\n        Coefficients of the linear objective function to be minimized (for\n        standard form problem).\n    c0 : float\n        Constant term in objective function due to fixed (and eliminated)\n        variables.\n    x0 : 1-D array\n        Starting values of the independent variables, which will be refined by\n        the optimization algorithm\n\n    References\n    ----------\n    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear\n           programming." Athena Scientific 1 (1997): 997.\n\n    '
    (c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality) = lp
    if sps.issparse(A_eq):
        sparse = True
        A_eq = sps.csr_matrix(A_eq)
        A_ub = sps.csr_matrix(A_ub)

        def hstack(blocks):
            if False:
                return 10
            return sps.hstack(blocks, format='csr')

        def vstack(blocks):
            if False:
                while True:
                    i = 10
            return sps.vstack(blocks, format='csr')
        zeros = sps.csr_matrix
        eye = sps.eye
    else:
        sparse = False
        hstack = np.hstack
        vstack = np.vstack
        zeros = np.zeros
        eye = np.eye
    bounds = np.array(bounds, copy=True)
    lbs = bounds[:, 0]
    ubs = bounds[:, 1]
    (m_ub, n_ub) = A_ub.shape
    lb_none = np.equal(lbs, -np.inf)
    ub_none = np.equal(ubs, np.inf)
    lb_some = np.logical_not(lb_none)
    ub_some = np.logical_not(ub_none)
    l_nolb_someub = np.logical_and(lb_none, ub_some)
    i_nolb = np.nonzero(l_nolb_someub)[0]
    (lbs[l_nolb_someub], ubs[l_nolb_someub]) = (-ubs[l_nolb_someub], -lbs[l_nolb_someub])
    lb_none = np.equal(lbs, -np.inf)
    ub_none = np.equal(ubs, np.inf)
    lb_some = np.logical_not(lb_none)
    ub_some = np.logical_not(ub_none)
    c[i_nolb] *= -1
    if x0 is not None:
        x0[i_nolb] *= -1
    if len(i_nolb) > 0:
        if A_ub.shape[0] > 0:
            A_ub[:, i_nolb] *= -1
        if A_eq.shape[0] > 0:
            A_eq[:, i_nolb] *= -1
    (i_newub,) = ub_some.nonzero()
    ub_newub = ubs[ub_some]
    n_bounds = len(i_newub)
    if n_bounds > 0:
        shape = (n_bounds, A_ub.shape[1])
        if sparse:
            idxs = (np.arange(n_bounds), i_newub)
            A_ub = vstack((A_ub, sps.csr_matrix((np.ones(n_bounds), idxs), shape=shape)))
        else:
            A_ub = vstack((A_ub, np.zeros(shape)))
            A_ub[np.arange(m_ub, A_ub.shape[0]), i_newub] = 1
        b_ub = np.concatenate((b_ub, np.zeros(n_bounds)))
        b_ub[m_ub:] = ub_newub
    A1 = vstack((A_ub, A_eq))
    b = np.concatenate((b_ub, b_eq))
    c = np.concatenate((c, np.zeros((A_ub.shape[0],))))
    if x0 is not None:
        x0 = np.concatenate((x0, np.zeros((A_ub.shape[0],))))
    l_free = np.logical_and(lb_none, ub_none)
    i_free = np.nonzero(l_free)[0]
    n_free = len(i_free)
    c = np.concatenate((c, np.zeros(n_free)))
    if x0 is not None:
        x0 = np.concatenate((x0, np.zeros(n_free)))
    A1 = hstack((A1[:, :n_ub], -A1[:, i_free]))
    c[n_ub:n_ub + n_free] = -c[i_free]
    if x0 is not None:
        i_free_neg = x0[i_free] < 0
        x0[np.arange(n_ub, A1.shape[1])[i_free_neg]] = -x0[i_free[i_free_neg]]
        x0[i_free[i_free_neg]] = 0
    A2 = vstack([eye(A_ub.shape[0]), zeros((A_eq.shape[0], A_ub.shape[0]))])
    A = hstack([A1, A2])
    i_shift = np.nonzero(lb_some)[0]
    lb_shift = lbs[lb_some].astype(float)
    c0 += np.sum(lb_shift * c[i_shift])
    if sparse:
        b = b.reshape(-1, 1)
        A = A.tocsc()
        b -= (A[:, i_shift] * sps.diags(lb_shift)).sum(axis=1)
        b = b.ravel()
    else:
        b -= (A[:, i_shift] * lb_shift).sum(axis=1)
    if x0 is not None:
        x0[i_shift] -= lb_shift
    return (A, b, c, c0, x0)

def _round_to_power_of_two(x):
    if False:
        i = 10
        return i + 15
    '\n    Round elements of the array to the nearest power of two.\n    '
    return 2 ** np.around(np.log2(x))

def _autoscale(A, b, c, x0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Scales the problem according to equilibration from [12].\n    Also normalizes the right hand side vector by its maximum element.\n    '
    (m, n) = A.shape
    C = 1
    R = 1
    if A.size > 0:
        R = np.max(np.abs(A), axis=1)
        if sps.issparse(A):
            R = R.toarray().flatten()
        R[R == 0] = 1
        R = 1 / _round_to_power_of_two(R)
        A = sps.diags(R) * A if sps.issparse(A) else A * R.reshape(m, 1)
        b = b * R
        C = np.max(np.abs(A), axis=0)
        if sps.issparse(A):
            C = C.toarray().flatten()
        C[C == 0] = 1
        C = 1 / _round_to_power_of_two(C)
        A = A * sps.diags(C) if sps.issparse(A) else A * C
        c = c * C
    b_scale = np.max(np.abs(b)) if b.size > 0 else 1
    if b_scale == 0:
        b_scale = 1.0
    b = b / b_scale
    if x0 is not None:
        x0 = x0 / b_scale * (1 / C)
    return (A, b, c, x0, C, b_scale)

def _unscale(x, C, b_scale):
    if False:
        print('Hello World!')
    '\n    Converts solution to _autoscale problem -> solution to original problem.\n    '
    try:
        n = len(C)
    except TypeError:
        n = len(x)
    return x[:n] * b_scale * C

def _display_summary(message, status, fun, iteration):
    if False:
        for i in range(10):
            print('nop')
    '\n    Print the termination summary of the linear program\n\n    Parameters\n    ----------\n    message : str\n            A string descriptor of the exit status of the optimization.\n    status : int\n        An integer representing the exit status of the optimization::\n\n                0 : Optimization terminated successfully\n                1 : Iteration limit reached\n                2 : Problem appears to be infeasible\n                3 : Problem appears to be unbounded\n                4 : Serious numerical difficulties encountered\n\n    fun : float\n        Value of the objective function.\n    iteration : iteration\n        The number of iterations performed.\n    '
    print(message)
    if status in (0, 1):
        print(f'         Current function value: {fun: <12.6f}')
    print(f'         Iterations: {iteration:d}')

def _postsolve(x, postsolve_args, complete=False):
    if False:
        i = 10
        return i + 15
    "\n    Given solution x to presolved, standard form linear program x, add\n    fixed variables back into the problem and undo the variable substitutions\n    to get solution to original linear program. Also, calculate the objective\n    function value, slack in original upper bound constraints, and residuals\n    in original equality constraints.\n\n    Parameters\n    ----------\n    x : 1-D array\n        Solution vector to the standard-form problem.\n    postsolve_args : tuple\n        Data needed by _postsolve to convert the solution to the standard-form\n        problem into the solution to the original problem, including:\n\n    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:\n\n        c : 1D array\n            The coefficients of the linear objective function to be minimized.\n        A_ub : 2D array, optional\n            The inequality constraint matrix. Each row of ``A_ub`` specifies the\n            coefficients of a linear inequality constraint on ``x``.\n        b_ub : 1D array, optional\n            The inequality constraint vector. Each element represents an\n            upper bound on the corresponding value of ``A_ub @ x``.\n        A_eq : 2D array, optional\n            The equality constraint matrix. Each row of ``A_eq`` specifies the\n            coefficients of a linear equality constraint on ``x``.\n        b_eq : 1D array, optional\n            The equality constraint vector. Each element of ``A_eq @ x`` must equal\n            the corresponding element of ``b_eq``.\n        bounds : 2D array\n            The bounds of ``x``, lower bounds in the 1st column, upper\n            bounds in the 2nd column. The bounds are possibly tightened\n            by the presolve procedure.\n        x0 : 1D array, optional\n            Guess values of the decision variables, which will be refined by\n            the optimization algorithm. This argument is currently used only by the\n            'revised simplex' method, and can only be used if `x0` represents a\n            basic feasible solution.\n\n    revstack: list of functions\n        the functions in the list reverse the operations of _presolve()\n        the function signature is x_org = f(x_mod), where x_mod is the result\n        of a presolve step and x_org the value at the start of the step\n    complete : bool\n        Whether the solution is was determined in presolve (``True`` if so)\n\n    Returns\n    -------\n    x : 1-D array\n        Solution vector to original linear programming problem\n    fun: float\n        optimal objective value for original problem\n    slack : 1-D array\n        The (non-negative) slack in the upper bound constraints, that is,\n        ``b_ub - A_ub @ x``\n    con : 1-D array\n        The (nominally zero) residuals of the equality constraints, that is,\n        ``b - A_eq @ x``\n    "
    (c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality) = postsolve_args[0]
    (revstack, C, b_scale) = postsolve_args[1:]
    x = _unscale(x, C, b_scale)
    n_x = bounds.shape[0]
    if not complete and bounds is not None:
        n_unbounded = 0
        for (i, bi) in enumerate(bounds):
            lbi = bi[0]
            ubi = bi[1]
            if lbi == -np.inf and ubi == np.inf:
                n_unbounded += 1
                x[i] = x[i] - x[n_x + n_unbounded - 1]
            elif lbi == -np.inf:
                x[i] = ubi - x[i]
            else:
                x[i] += lbi
    x = x[:n_x]
    for rev in reversed(revstack):
        x = rev(x)
    fun = x.dot(c)
    slack = b_ub - A_ub.dot(x)
    con = b_eq - A_eq.dot(x)
    return (x, fun, slack, con)

def _check_result(x, fun, status, slack, con, bounds, tol, message, integrality):
    if False:
        print('Hello World!')
    '\n    Check the validity of the provided solution.\n\n    A valid (optimal) solution satisfies all bounds, all slack variables are\n    negative and all equality constraint residuals are strictly non-zero.\n    Further, the lower-bounds, upper-bounds, slack and residuals contain\n    no nan values.\n\n    Parameters\n    ----------\n    x : 1-D array\n        Solution vector to original linear programming problem\n    fun: float\n        optimal objective value for original problem\n    status : int\n        An integer representing the exit status of the optimization::\n\n             0 : Optimization terminated successfully\n             1 : Iteration limit reached\n             2 : Problem appears to be infeasible\n             3 : Problem appears to be unbounded\n             4 : Serious numerical difficulties encountered\n\n    slack : 1-D array\n        The (non-negative) slack in the upper bound constraints, that is,\n        ``b_ub - A_ub @ x``\n    con : 1-D array\n        The (nominally zero) residuals of the equality constraints, that is,\n        ``b - A_eq @ x``\n    bounds : 2D array\n        The bounds on the original variables ``x``\n    message : str\n        A string descriptor of the exit status of the optimization.\n    tol : float\n        Termination tolerance; see [1]_ Section 4.5.\n\n    Returns\n    -------\n    status : int\n        An integer representing the exit status of the optimization::\n\n             0 : Optimization terminated successfully\n             1 : Iteration limit reached\n             2 : Problem appears to be infeasible\n             3 : Problem appears to be unbounded\n             4 : Serious numerical difficulties encountered\n\n    message : str\n        A string descriptor of the exit status of the optimization.\n    '
    tol = np.sqrt(tol) * 10
    if x is None:
        if status == 0:
            status = 4
            message = 'The solver did not provide a solution nor did it report a failure. Please submit a bug report.'
        return (status, message)
    contains_nans = np.isnan(x).any() or np.isnan(fun) or np.isnan(slack).any() or np.isnan(con).any()
    if contains_nans:
        is_feasible = False
    else:
        if integrality is None:
            integrality = 0
        valid_bounds = (x >= bounds[:, 0] - tol) & (x <= bounds[:, 1] + tol)
        valid_bounds |= (integrality > 1) & np.isclose(x, 0, atol=tol)
        invalid_bounds = not np.all(valid_bounds)
        invalid_slack = status != 3 and (slack < -tol).any()
        invalid_con = status != 3 and (np.abs(con) > tol).any()
        is_feasible = not (invalid_bounds or invalid_slack or invalid_con)
    if status == 0 and (not is_feasible):
        status = 4
        message = 'The solution does not satisfy the constraints within the required tolerance of ' + f'{tol:.2E}' + ', yet no errors were raised and there is no certificate of infeasibility or unboundedness. Check whether the slack and constraint residuals are acceptable; if not, consider enabling presolve, adjusting the tolerance option(s), and/or using a different method. Please consider submitting a bug report.'
    elif status == 2 and is_feasible:
        status = 4
        message = 'The solution is feasible, but the solver did not report that the solution was optimal. Please try a different method.'
    return (status, message)