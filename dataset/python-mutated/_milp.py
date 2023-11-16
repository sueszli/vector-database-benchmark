import warnings
import numpy as np
from scipy.sparse import csc_array, vstack, issparse
from scipy._lib._util import VisibleDeprecationWarning
from ._highs._highs_wrapper import _highs_wrapper
from ._constraints import LinearConstraint, Bounds
from ._optimize import OptimizeResult
from ._linprog_highs import _highs_to_scipy_status_message

def _constraints_to_components(constraints):
    if False:
        print('Hello World!')
    '\n    Convert sequence of constraints to a single set of components A, b_l, b_u.\n\n    `constraints` could be\n\n    1. A LinearConstraint\n    2. A tuple representing a LinearConstraint\n    3. An invalid object\n    4. A sequence of composed entirely of objects of type 1/2\n    5. A sequence containing at least one object of type 3\n\n    We want to accept 1, 2, and 4 and reject 3 and 5.\n    '
    message = '`constraints` (or each element within `constraints`) must be convertible into an instance of `scipy.optimize.LinearConstraint`.'
    As = []
    b_ls = []
    b_us = []
    if isinstance(constraints, LinearConstraint):
        constraints = [constraints]
    else:
        try:
            iter(constraints)
        except TypeError as exc:
            raise ValueError(message) from exc
        if len(constraints) == 3:
            try:
                constraints = [LinearConstraint(*constraints)]
            except (TypeError, ValueError, VisibleDeprecationWarning):
                pass
    for constraint in constraints:
        if not isinstance(constraint, LinearConstraint):
            try:
                constraint = LinearConstraint(*constraint)
            except TypeError as exc:
                raise ValueError(message) from exc
        As.append(csc_array(constraint.A))
        b_ls.append(np.atleast_1d(constraint.lb).astype(np.float64))
        b_us.append(np.atleast_1d(constraint.ub).astype(np.float64))
    if len(As) > 1:
        A = vstack(As, format='csc')
        b_l = np.concatenate(b_ls)
        b_u = np.concatenate(b_us)
    else:
        A = As[0]
        b_l = b_ls[0]
        b_u = b_us[0]
    return (A, b_l, b_u)

def _milp_iv(c, integrality, bounds, constraints, options):
    if False:
        print('Hello World!')
    if issparse(c):
        raise ValueError('`c` must be a dense array.')
    c = np.atleast_1d(c).astype(np.float64)
    if c.ndim != 1 or c.size == 0 or (not np.all(np.isfinite(c))):
        message = '`c` must be a one-dimensional array of finite numbers with at least one element.'
        raise ValueError(message)
    if issparse(integrality):
        raise ValueError('`integrality` must be a dense array.')
    message = '`integrality` must contain integers 0-3 and be broadcastable to `c.shape`.'
    if integrality is None:
        integrality = 0
    try:
        integrality = np.broadcast_to(integrality, c.shape).astype(np.uint8)
    except ValueError:
        raise ValueError(message)
    if integrality.min() < 0 or integrality.max() > 3:
        raise ValueError(message)
    if bounds is None:
        bounds = Bounds(0, np.inf)
    elif not isinstance(bounds, Bounds):
        message = '`bounds` must be convertible into an instance of `scipy.optimize.Bounds`.'
        try:
            bounds = Bounds(*bounds)
        except TypeError as exc:
            raise ValueError(message) from exc
    try:
        lb = np.broadcast_to(bounds.lb, c.shape).astype(np.float64)
        ub = np.broadcast_to(bounds.ub, c.shape).astype(np.float64)
    except (ValueError, TypeError) as exc:
        message = '`bounds.lb` and `bounds.ub` must contain reals and be broadcastable to `c.shape`.'
        raise ValueError(message) from exc
    if not constraints:
        constraints = [LinearConstraint(np.empty((0, c.size)), np.empty((0,)), np.empty((0,)))]
    try:
        (A, b_l, b_u) = _constraints_to_components(constraints)
    except ValueError as exc:
        message = '`constraints` (or each element within `constraints`) must be convertible into an instance of `scipy.optimize.LinearConstraint`.'
        raise ValueError(message) from exc
    if A.shape != (b_l.size, c.size):
        message = 'The shape of `A` must be (len(b_l), len(c)).'
        raise ValueError(message)
    (indptr, indices, data) = (A.indptr, A.indices, A.data.astype(np.float64))
    options = options or {}
    supported_options = {'disp', 'presolve', 'time_limit', 'node_limit', 'mip_rel_gap'}
    unsupported_options = set(options).difference(supported_options)
    if unsupported_options:
        message = f'Unrecognized options detected: {unsupported_options}. These will be passed to HiGHS verbatim.'
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    options_iv = {'log_to_console': options.pop('disp', False), 'mip_max_nodes': options.pop('node_limit', None)}
    options_iv.update(options)
    return (c, integrality, lb, ub, indptr, indices, data, b_l, b_u, options_iv)

def milp(c, *, integrality=None, bounds=None, constraints=None, options=None):
    if False:
        i = 10
        return i + 15
    '\n    Mixed-integer linear programming\n\n    Solves problems of the following form:\n\n    .. math::\n\n        \\min_x \\ & c^T x \\\\\n        \\mbox{such that} \\ & b_l \\leq A x \\leq b_u,\\\\\n        & l \\leq x \\leq u, \\\\\n        & x_i \\in \\mathbb{Z}, i \\in X_i\n\n    where :math:`x` is a vector of decision variables;\n    :math:`c`, :math:`b_l`, :math:`b_u`, :math:`l`, and :math:`u` are vectors;\n    :math:`A` is a matrix, and :math:`X_i` is the set of indices of\n    decision variables that must be integral. (In this context, a\n    variable that can assume only integer values is said to be "integral";\n    it has an "integrality" constraint.)\n\n    Alternatively, that\'s:\n\n    minimize::\n\n        c @ x\n\n    such that::\n\n        b_l <= A @ x <= b_u\n        l <= x <= u\n        Specified elements of x must be integers\n\n    By default, ``l = 0`` and ``u = np.inf`` unless specified with\n    ``bounds``.\n\n    Parameters\n    ----------\n    c : 1D dense array_like\n        The coefficients of the linear objective function to be minimized.\n        `c` is converted to a double precision array before the problem is\n        solved.\n    integrality : 1D dense array_like, optional\n        Indicates the type of integrality constraint on each decision variable.\n\n        ``0`` : Continuous variable; no integrality constraint.\n\n        ``1`` : Integer variable; decision variable must be an integer\n        within `bounds`.\n\n        ``2`` : Semi-continuous variable; decision variable must be within\n        `bounds` or take value ``0``.\n\n        ``3`` : Semi-integer variable; decision variable must be an integer\n        within `bounds` or take value ``0``.\n\n        By default, all variables are continuous. `integrality` is converted\n        to an array of integers before the problem is solved.\n\n    bounds : scipy.optimize.Bounds, optional\n        Bounds on the decision variables. Lower and upper bounds are converted\n        to double precision arrays before the problem is solved. The\n        ``keep_feasible`` parameter of the `Bounds` object is ignored. If\n        not specified, all decision variables are constrained to be\n        non-negative.\n    constraints : sequence of scipy.optimize.LinearConstraint, optional\n        Linear constraints of the optimization problem. Arguments may be\n        one of the following:\n\n        1. A single `LinearConstraint` object\n        2. A single tuple that can be converted to a `LinearConstraint` object\n           as ``LinearConstraint(*constraints)``\n        3. A sequence composed entirely of objects of type 1. and 2.\n\n        Before the problem is solved, all values are converted to double\n        precision, and the matrices of constraint coefficients are converted to\n        instances of `scipy.sparse.csc_array`. The ``keep_feasible`` parameter\n        of `LinearConstraint` objects is ignored.\n    options : dict, optional\n        A dictionary of solver options. The following keys are recognized.\n\n        disp : bool (default: ``False``)\n            Set to ``True`` if indicators of optimization status are to be\n            printed to the console during optimization.\n        node_limit : int, optional\n            The maximum number of nodes (linear program relaxations) to solve\n            before stopping. Default is no maximum number of nodes.\n        presolve : bool (default: ``True``)\n            Presolve attempts to identify trivial infeasibilities,\n            identify trivial unboundedness, and simplify the problem before\n            sending it to the main solver.\n        time_limit : float, optional\n            The maximum number of seconds allotted to solve the problem.\n            Default is no time limit.\n        mip_rel_gap : float, optional\n            Termination criterion for MIP solver: solver will terminate when\n            the gap between the primal objective value and the dual objective\n            bound, scaled by the primal objective value, is <= mip_rel_gap.\n\n    Returns\n    -------\n    res : OptimizeResult\n        An instance of :class:`scipy.optimize.OptimizeResult`. The object\n        is guaranteed to have the following attributes.\n\n        status : int\n            An integer representing the exit status of the algorithm.\n\n            ``0`` : Optimal solution found.\n\n            ``1`` : Iteration or time limit reached.\n\n            ``2`` : Problem is infeasible.\n\n            ``3`` : Problem is unbounded.\n\n            ``4`` : Other; see message for details.\n\n        success : bool\n            ``True`` when an optimal solution is found and ``False`` otherwise.\n\n        message : str\n            A string descriptor of the exit status of the algorithm.\n\n        The following attributes will also be present, but the values may be\n        ``None``, depending on the solution status.\n\n        x : ndarray\n            The values of the decision variables that minimize the\n            objective function while satisfying the constraints.\n        fun : float\n            The optimal value of the objective function ``c @ x``.\n        mip_node_count : int\n            The number of subproblems or "nodes" solved by the MILP solver.\n        mip_dual_bound : float\n            The MILP solver\'s final estimate of the lower bound on the optimal\n            solution.\n        mip_gap : float\n            The difference between the primal objective value and the dual\n            objective bound, scaled by the primal objective value.\n\n    Notes\n    -----\n    `milp` is a wrapper of the HiGHS linear optimization software [1]_. The\n    algorithm is deterministic, and it typically finds the global optimum of\n    moderately challenging mixed-integer linear programs (when it exists).\n\n    References\n    ----------\n    .. [1] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.\n           "HiGHS - high performance software for linear optimization."\n           https://highs.dev/\n    .. [2] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised\n           simplex method." Mathematical Programming Computation, 10 (1),\n           119-142, 2018. DOI: 10.1007/s12532-017-0130-5\n\n    Examples\n    --------\n    Consider the problem at\n    https://en.wikipedia.org/wiki/Integer_programming#Example, which is\n    expressed as a maximization problem of two variables. Since `milp` requires\n    that the problem be expressed as a minimization problem, the objective\n    function coefficients on the decision variables are:\n\n    >>> import numpy as np\n    >>> c = -np.array([0, 1])\n\n    Note the negative sign: we maximize the original objective function\n    by minimizing the negative of the objective function.\n\n    We collect the coefficients of the constraints into arrays like:\n\n    >>> A = np.array([[-1, 1], [3, 2], [2, 3]])\n    >>> b_u = np.array([1, 12, 12])\n    >>> b_l = np.full_like(b_u, -np.inf)\n\n    Because there is no lower limit on these constraints, we have defined a\n    variable ``b_l`` full of values representing negative infinity. This may\n    be unfamiliar to users of `scipy.optimize.linprog`, which only accepts\n    "less than" (or "upper bound") inequality constraints of the form\n    ``A_ub @ x <= b_u``. By accepting both ``b_l`` and ``b_u`` of constraints\n    ``b_l <= A_ub @ x <= b_u``, `milp` makes it easy to specify "greater than"\n    inequality constraints, "less than" inequality constraints, and equality\n    constraints concisely.\n\n    These arrays are collected into a single `LinearConstraint` object like:\n\n    >>> from scipy.optimize import LinearConstraint\n    >>> constraints = LinearConstraint(A, b_l, b_u)\n\n    The non-negativity bounds on the decision variables are enforced by\n    default, so we do not need to provide an argument for `bounds`.\n\n    Finally, the problem states that both decision variables must be integers:\n\n    >>> integrality = np.ones_like(c)\n\n    We solve the problem like:\n\n    >>> from scipy.optimize import milp\n    >>> res = milp(c=c, constraints=constraints, integrality=integrality)\n    >>> res.x\n    [1.0, 2.0]\n\n    Note that had we solved the relaxed problem (without integrality\n    constraints):\n\n    >>> res = milp(c=c, constraints=constraints)  # OR:\n    >>> # from scipy.optimize import linprog; res = linprog(c, A, b_u)\n    >>> res.x\n    [1.8, 2.8]\n\n    we would not have obtained the correct solution by rounding to the nearest\n    integers.\n\n    Other examples are given :ref:`in the tutorial <tutorial-optimize_milp>`.\n\n    '
    args_iv = _milp_iv(c, integrality, bounds, constraints, options)
    (c, integrality, lb, ub, indptr, indices, data, b_l, b_u, options) = args_iv
    highs_res = _highs_wrapper(c, indptr, indices, data, b_l, b_u, lb, ub, integrality, options)
    res = {}
    highs_status = highs_res.get('status', None)
    highs_message = highs_res.get('message', None)
    (status, message) = _highs_to_scipy_status_message(highs_status, highs_message)
    res['status'] = status
    res['message'] = message
    res['success'] = status == 0
    x = highs_res.get('x', None)
    res['x'] = np.array(x) if x is not None else None
    res['fun'] = highs_res.get('fun', None)
    res['mip_node_count'] = highs_res.get('mip_node_count', None)
    res['mip_dual_bound'] = highs_res.get('mip_dual_bound', None)
    res['mip_gap'] = highs_res.get('mip_gap', None)
    return OptimizeResult(res)