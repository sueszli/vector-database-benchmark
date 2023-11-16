"""HiGHS Linear Optimization Methods

Interface to HiGHS linear optimization software.
https://highs.dev/

.. versionadded:: 1.5.0

References
----------
.. [1] Q. Huangfu and J.A.J. Hall. "Parallelizing the dual revised simplex
           method." Mathematical Programming Computation, 10 (1), 119-142,
           2018. DOI: 10.1007/s12532-017-0130-5

"""
import inspect
import numpy as np
from ._optimize import OptimizeWarning, OptimizeResult
from warnings import warn
from ._highs._highs_wrapper import _highs_wrapper
from ._highs._highs_constants import CONST_INF, MESSAGE_LEVEL_NONE, HIGHS_OBJECTIVE_SENSE_MINIMIZE, MODEL_STATUS_NOTSET, MODEL_STATUS_LOAD_ERROR, MODEL_STATUS_MODEL_ERROR, MODEL_STATUS_PRESOLVE_ERROR, MODEL_STATUS_SOLVE_ERROR, MODEL_STATUS_POSTSOLVE_ERROR, MODEL_STATUS_MODEL_EMPTY, MODEL_STATUS_OPTIMAL, MODEL_STATUS_INFEASIBLE, MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE, MODEL_STATUS_UNBOUNDED, MODEL_STATUS_REACHED_DUAL_OBJECTIVE_VALUE_UPPER_BOUND as MODEL_STATUS_RDOVUB, MODEL_STATUS_REACHED_OBJECTIVE_TARGET, MODEL_STATUS_REACHED_TIME_LIMIT, MODEL_STATUS_REACHED_ITERATION_LIMIT, HIGHS_SIMPLEX_STRATEGY_DUAL, HIGHS_SIMPLEX_CRASH_STRATEGY_OFF, HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_CHOOSE, HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_DANTZIG, HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_DEVEX, HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_STEEPEST_EDGE
from scipy.sparse import csc_matrix, vstack, issparse

def _highs_to_scipy_status_message(highs_status, highs_message):
    if False:
        while True:
            i = 10
    'Converts HiGHS status number/message to SciPy status number/message'
    scipy_statuses_messages = {None: (4, 'HiGHS did not provide a status code. '), MODEL_STATUS_NOTSET: (4, ''), MODEL_STATUS_LOAD_ERROR: (4, ''), MODEL_STATUS_MODEL_ERROR: (2, ''), MODEL_STATUS_PRESOLVE_ERROR: (4, ''), MODEL_STATUS_SOLVE_ERROR: (4, ''), MODEL_STATUS_POSTSOLVE_ERROR: (4, ''), MODEL_STATUS_MODEL_EMPTY: (4, ''), MODEL_STATUS_RDOVUB: (4, ''), MODEL_STATUS_REACHED_OBJECTIVE_TARGET: (4, ''), MODEL_STATUS_OPTIMAL: (0, 'Optimization terminated successfully. '), MODEL_STATUS_REACHED_TIME_LIMIT: (1, 'Time limit reached. '), MODEL_STATUS_REACHED_ITERATION_LIMIT: (1, 'Iteration limit reached. '), MODEL_STATUS_INFEASIBLE: (2, 'The problem is infeasible. '), MODEL_STATUS_UNBOUNDED: (3, 'The problem is unbounded. '), MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE: (4, 'The problem is unbounded or infeasible. ')}
    unrecognized = (4, 'The HiGHS status code was not recognized. ')
    (scipy_status, scipy_message) = scipy_statuses_messages.get(highs_status, unrecognized)
    scipy_message = f'{scipy_message}(HiGHS Status {highs_status}: {highs_message})'
    return (scipy_status, scipy_message)

def _replace_inf(x):
    if False:
        while True:
            i = 10
    infs = np.isinf(x)
    with np.errstate(invalid='ignore'):
        x[infs] = np.sign(x[infs]) * CONST_INF
    return x

def _convert_to_highs_enum(option, option_str, choices):
    if False:
        return 10
    try:
        return choices[option.lower()]
    except AttributeError:
        return choices[option]
    except KeyError:
        sig = inspect.signature(_linprog_highs)
        default_str = sig.parameters[option_str].default
        warn(f'Option {option_str} is {option}, but only values in {set(choices.keys())} are allowed. Using default: {default_str}.', OptimizeWarning, stacklevel=3)
        return choices[default_str]

def _linprog_highs(lp, solver, time_limit=None, presolve=True, disp=False, maxiter=None, dual_feasibility_tolerance=None, primal_feasibility_tolerance=None, ipm_optimality_tolerance=None, simplex_dual_edge_weight_strategy=None, mip_rel_gap=None, mip_max_nodes=None, **unknown_options):
    if False:
        return 10
    '\n    Solve the following linear programming problem using one of the HiGHS\n    solvers:\n\n    User-facing documentation is in _linprog_doc.py.\n\n    Parameters\n    ----------\n    lp :  _LPProblem\n        A ``scipy.optimize._linprog_util._LPProblem`` ``namedtuple``.\n    solver : "ipm" or "simplex" or None\n        Which HiGHS solver to use.  If ``None``, "simplex" will be used.\n\n    Options\n    -------\n    maxiter : int\n        The maximum number of iterations to perform in either phase. For\n        ``solver=\'ipm\'``, this does not include the number of crossover\n        iterations.  Default is the largest possible value for an ``int``\n        on the platform.\n    disp : bool\n        Set to ``True`` if indicators of optimization status are to be printed\n        to the console each iteration; default ``False``.\n    time_limit : float\n        The maximum time in seconds allotted to solve the problem; default is\n        the largest possible value for a ``double`` on the platform.\n    presolve : bool\n        Presolve attempts to identify trivial infeasibilities,\n        identify trivial unboundedness, and simplify the problem before\n        sending it to the main solver. It is generally recommended\n        to keep the default setting ``True``; set to ``False`` if presolve is\n        to be disabled.\n    dual_feasibility_tolerance : double\n        Dual feasibility tolerance.  Default is 1e-07.\n        The minimum of this and ``primal_feasibility_tolerance``\n        is used for the feasibility tolerance when ``solver=\'ipm\'``.\n    primal_feasibility_tolerance : double\n        Primal feasibility tolerance.  Default is 1e-07.\n        The minimum of this and ``dual_feasibility_tolerance``\n        is used for the feasibility tolerance when ``solver=\'ipm\'``.\n    ipm_optimality_tolerance : double\n        Optimality tolerance for ``solver=\'ipm\'``.  Default is 1e-08.\n        Minimum possible value is 1e-12 and must be smaller than the largest\n        possible value for a ``double`` on the platform.\n    simplex_dual_edge_weight_strategy : str (default: None)\n        Strategy for simplex dual edge weights. The default, ``None``,\n        automatically selects one of the following.\n\n        ``\'dantzig\'`` uses Dantzig\'s original strategy of choosing the most\n        negative reduced cost.\n\n        ``\'devex\'`` uses the strategy described in [15]_.\n\n        ``steepest`` uses the exact steepest edge strategy as described in\n        [16]_.\n\n        ``\'steepest-devex\'`` begins with the exact steepest edge strategy\n        until the computation is too costly or inexact and then switches to\n        the devex method.\n\n        Currently, using ``None`` always selects ``\'steepest-devex\'``, but this\n        may change as new options become available.\n\n    mip_max_nodes : int\n        The maximum number of nodes allotted to solve the problem; default is\n        the largest possible value for a ``HighsInt`` on the platform.\n        Ignored if not using the MIP solver.\n    unknown_options : dict\n        Optional arguments not used by this particular solver. If\n        ``unknown_options`` is non-empty, a warning is issued listing all\n        unused options.\n\n    Returns\n    -------\n    sol : dict\n        A dictionary consisting of the fields:\n\n            x : 1D array\n                The values of the decision variables that minimizes the\n                objective function while satisfying the constraints.\n            fun : float\n                The optimal value of the objective function ``c @ x``.\n            slack : 1D array\n                The (nominally positive) values of the slack,\n                ``b_ub - A_ub @ x``.\n            con : 1D array\n                The (nominally zero) residuals of the equality constraints,\n                ``b_eq - A_eq @ x``.\n            success : bool\n                ``True`` when the algorithm succeeds in finding an optimal\n                solution.\n            status : int\n                An integer representing the exit status of the algorithm.\n\n                ``0`` : Optimization terminated successfully.\n\n                ``1`` : Iteration or time limit reached.\n\n                ``2`` : Problem appears to be infeasible.\n\n                ``3`` : Problem appears to be unbounded.\n\n                ``4`` : The HiGHS solver ran into a problem.\n\n            message : str\n                A string descriptor of the exit status of the algorithm.\n            nit : int\n                The total number of iterations performed.\n                For ``solver=\'simplex\'``, this includes iterations in all\n                phases. For ``solver=\'ipm\'``, this does not include\n                crossover iterations.\n            crossover_nit : int\n                The number of primal/dual pushes performed during the\n                crossover routine for ``solver=\'ipm\'``.  This is ``0``\n                for ``solver=\'simplex\'``.\n            ineqlin : OptimizeResult\n                Solution and sensitivity information corresponding to the\n                inequality constraints, `b_ub`. A dictionary consisting of the\n                fields:\n\n                residual : np.ndnarray\n                    The (nominally positive) values of the slack variables,\n                    ``b_ub - A_ub @ x``.  This quantity is also commonly\n                    referred to as "slack".\n\n                marginals : np.ndarray\n                    The sensitivity (partial derivative) of the objective\n                    function with respect to the right-hand side of the\n                    inequality constraints, `b_ub`.\n\n            eqlin : OptimizeResult\n                Solution and sensitivity information corresponding to the\n                equality constraints, `b_eq`.  A dictionary consisting of the\n                fields:\n\n                residual : np.ndarray\n                    The (nominally zero) residuals of the equality constraints,\n                    ``b_eq - A_eq @ x``.\n\n                marginals : np.ndarray\n                    The sensitivity (partial derivative) of the objective\n                    function with respect to the right-hand side of the\n                    equality constraints, `b_eq`.\n\n            lower, upper : OptimizeResult\n                Solution and sensitivity information corresponding to the\n                lower and upper bounds on decision variables, `bounds`.\n\n                residual : np.ndarray\n                    The (nominally positive) values of the quantity\n                    ``x - lb`` (lower) or ``ub - x`` (upper).\n\n                marginals : np.ndarray\n                    The sensitivity (partial derivative) of the objective\n                    function with respect to the lower and upper\n                    `bounds`.\n\n            mip_node_count : int\n                The number of subproblems or "nodes" solved by the MILP\n                solver. Only present when `integrality` is not `None`.\n\n            mip_dual_bound : float\n                The MILP solver\'s final estimate of the lower bound on the\n                optimal solution. Only present when `integrality` is not\n                `None`.\n\n            mip_gap : float\n                The difference between the final objective function value\n                and the final dual bound, scaled by the final objective\n                function value. Only present when `integrality` is not\n                `None`.\n\n    Notes\n    -----\n    The result fields `ineqlin`, `eqlin`, `lower`, and `upper` all contain\n    `marginals`, or partial derivatives of the objective function with respect\n    to the right-hand side of each constraint. These partial derivatives are\n    also referred to as "Lagrange multipliers", "dual values", and\n    "shadow prices". The sign convention of `marginals` is opposite that\n    of Lagrange multipliers produced by many nonlinear solvers.\n\n    References\n    ----------\n    .. [15] Harris, Paula MJ. "Pivot selection methods of the Devex LP code."\n            Mathematical programming 5.1 (1973): 1-28.\n    .. [16] Goldfarb, Donald, and John Ker Reid. "A practicable steepest-edge\n            simplex algorithm." Mathematical Programming 12.1 (1977): 361-371.\n    '
    if unknown_options:
        message = f'Unrecognized options detected: {unknown_options}. These will be passed to HiGHS verbatim.'
        warn(message, OptimizeWarning, stacklevel=3)
    simplex_dual_edge_weight_strategy_enum = _convert_to_highs_enum(simplex_dual_edge_weight_strategy, 'simplex_dual_edge_weight_strategy', choices={'dantzig': HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_DANTZIG, 'devex': HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_DEVEX, 'steepest-devex': HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_CHOOSE, 'steepest': HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_STEEPEST_EDGE, None: None})
    (c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality) = lp
    (lb, ub) = bounds.T.copy()
    with np.errstate(invalid='ignore'):
        lhs_ub = -np.ones_like(b_ub) * np.inf
    rhs_ub = b_ub
    lhs_eq = b_eq
    rhs_eq = b_eq
    lhs = np.concatenate((lhs_ub, lhs_eq))
    rhs = np.concatenate((rhs_ub, rhs_eq))
    if issparse(A_ub) or issparse(A_eq):
        A = vstack((A_ub, A_eq))
    else:
        A = np.vstack((A_ub, A_eq))
    A = csc_matrix(A)
    options = {'presolve': presolve, 'sense': HIGHS_OBJECTIVE_SENSE_MINIMIZE, 'solver': solver, 'time_limit': time_limit, 'highs_debug_level': MESSAGE_LEVEL_NONE, 'dual_feasibility_tolerance': dual_feasibility_tolerance, 'ipm_optimality_tolerance': ipm_optimality_tolerance, 'log_to_console': disp, 'mip_max_nodes': mip_max_nodes, 'output_flag': disp, 'primal_feasibility_tolerance': primal_feasibility_tolerance, 'simplex_dual_edge_weight_strategy': simplex_dual_edge_weight_strategy_enum, 'simplex_strategy': HIGHS_SIMPLEX_STRATEGY_DUAL, 'simplex_crash_strategy': HIGHS_SIMPLEX_CRASH_STRATEGY_OFF, 'ipm_iteration_limit': maxiter, 'simplex_iteration_limit': maxiter, 'mip_rel_gap': mip_rel_gap}
    options.update(unknown_options)
    rhs = _replace_inf(rhs)
    lhs = _replace_inf(lhs)
    lb = _replace_inf(lb)
    ub = _replace_inf(ub)
    if integrality is None or np.sum(integrality) == 0:
        integrality = np.empty(0)
    else:
        integrality = np.array(integrality)
    res = _highs_wrapper(c, A.indptr, A.indices, A.data, lhs, rhs, lb, ub, integrality.astype(np.uint8), options)
    if 'slack' in res:
        slack = res['slack']
        con = np.array(slack[len(b_ub):])
        slack = np.array(slack[:len(b_ub)])
    else:
        (slack, con) = (None, None)
    if 'lambda' in res:
        lamda = res['lambda']
        marg_ineqlin = np.array(lamda[:len(b_ub)])
        marg_eqlin = np.array(lamda[len(b_ub):])
        marg_upper = np.array(res['marg_bnds'][1, :])
        marg_lower = np.array(res['marg_bnds'][0, :])
    else:
        (marg_ineqlin, marg_eqlin) = (None, None)
        (marg_upper, marg_lower) = (None, None)
    highs_status = res.get('status', None)
    highs_message = res.get('message', None)
    (status, message) = _highs_to_scipy_status_message(highs_status, highs_message)
    x = np.array(res['x']) if 'x' in res else None
    sol = {'x': x, 'slack': slack, 'con': con, 'ineqlin': OptimizeResult({'residual': slack, 'marginals': marg_ineqlin}), 'eqlin': OptimizeResult({'residual': con, 'marginals': marg_eqlin}), 'lower': OptimizeResult({'residual': None if x is None else x - lb, 'marginals': marg_lower}), 'upper': OptimizeResult({'residual': None if x is None else ub - x, 'marginals': marg_upper}), 'fun': res.get('fun'), 'status': status, 'success': res['status'] == MODEL_STATUS_OPTIMAL, 'message': message, 'nit': res.get('simplex_nit', 0) or res.get('ipm_nit', 0), 'crossover_nit': res.get('crossover_nit')}
    if np.any(x) and integrality is not None:
        sol.update({'mip_node_count': res.get('mip_node_count', 0), 'mip_dual_bound': res.get('mip_dual_bound', 0.0), 'mip_gap': res.get('mip_gap', 0.0)})
    return sol