import time
import numpy as np
from scipy.sparse.linalg import LinearOperator
from .._differentiable_functions import VectorFunction
from .._constraints import NonlinearConstraint, LinearConstraint, PreparedConstraint, Bounds, strict_bounds
from .._hessian_update_strategy import BFGS
from .._optimize import OptimizeResult
from .._differentiable_functions import ScalarFunction
from .equality_constrained_sqp import equality_constrained_sqp
from .canonical_constraint import CanonicalConstraint, initial_constraints_as_canonical
from .tr_interior_point import tr_interior_point
from .report import BasicReport, SQPReport, IPReport
TERMINATION_MESSAGES = {0: 'The maximum number of function evaluations is exceeded.', 1: '`gtol` termination condition is satisfied.', 2: '`xtol` termination condition is satisfied.', 3: '`callback` function requested termination.'}

class HessianLinearOperator:
    """Build LinearOperator from hessp"""

    def __init__(self, hessp, n):
        if False:
            print('Hello World!')
        self.hessp = hessp
        self.n = n

    def __call__(self, x, *args):
        if False:
            for i in range(10):
                print('nop')

        def matvec(p):
            if False:
                print('Hello World!')
            return self.hessp(x, p, *args)
        return LinearOperator((self.n, self.n), matvec=matvec)

class LagrangianHessian:
    """The Hessian of the Lagrangian as LinearOperator.

    The Lagrangian is computed as the objective function plus all the
    constraints multiplied with some numbers (Lagrange multipliers).
    """

    def __init__(self, n, objective_hess, constraints_hess):
        if False:
            return 10
        self.n = n
        self.objective_hess = objective_hess
        self.constraints_hess = constraints_hess

    def __call__(self, x, v_eq=np.empty(0), v_ineq=np.empty(0)):
        if False:
            while True:
                i = 10
        H_objective = self.objective_hess(x)
        H_constraints = self.constraints_hess(x, v_eq, v_ineq)

        def matvec(p):
            if False:
                return 10
            return H_objective.dot(p) + H_constraints.dot(p)
        return LinearOperator((self.n, self.n), matvec)

def update_state_sqp(state, x, last_iteration_failed, objective, prepared_constraints, start_time, tr_radius, constr_penalty, cg_info):
    if False:
        i = 10
        return i + 15
    state.nit += 1
    state.nfev = objective.nfev
    state.njev = objective.ngev
    state.nhev = objective.nhev
    state.constr_nfev = [c.fun.nfev if isinstance(c.fun, VectorFunction) else 0 for c in prepared_constraints]
    state.constr_njev = [c.fun.njev if isinstance(c.fun, VectorFunction) else 0 for c in prepared_constraints]
    state.constr_nhev = [c.fun.nhev if isinstance(c.fun, VectorFunction) else 0 for c in prepared_constraints]
    if not last_iteration_failed:
        state.x = x
        state.fun = objective.f
        state.grad = objective.g
        state.v = [c.fun.v for c in prepared_constraints]
        state.constr = [c.fun.f for c in prepared_constraints]
        state.jac = [c.fun.J for c in prepared_constraints]
        state.lagrangian_grad = np.copy(state.grad)
        for c in prepared_constraints:
            state.lagrangian_grad += c.fun.J.T.dot(c.fun.v)
        state.optimality = np.linalg.norm(state.lagrangian_grad, np.inf)
        state.constr_violation = 0
        for i in range(len(prepared_constraints)):
            (lb, ub) = prepared_constraints[i].bounds
            c = state.constr[i]
            state.constr_violation = np.max([state.constr_violation, np.max(lb - c), np.max(c - ub)])
    state.execution_time = time.time() - start_time
    state.tr_radius = tr_radius
    state.constr_penalty = constr_penalty
    state.cg_niter += cg_info['niter']
    state.cg_stop_cond = cg_info['stop_cond']
    return state

def update_state_ip(state, x, last_iteration_failed, objective, prepared_constraints, start_time, tr_radius, constr_penalty, cg_info, barrier_parameter, barrier_tolerance):
    if False:
        print('Hello World!')
    state = update_state_sqp(state, x, last_iteration_failed, objective, prepared_constraints, start_time, tr_radius, constr_penalty, cg_info)
    state.barrier_parameter = barrier_parameter
    state.barrier_tolerance = barrier_tolerance
    return state

def _minimize_trustregion_constr(fun, x0, args, grad, hess, hessp, bounds, constraints, xtol=1e-08, gtol=1e-08, barrier_tol=1e-08, sparse_jacobian=None, callback=None, maxiter=1000, verbose=0, finite_diff_rel_step=None, initial_constr_penalty=1.0, initial_tr_radius=1.0, initial_barrier_parameter=0.1, initial_barrier_tolerance=0.1, factorization_method=None, disp=False):
    if False:
        print('Hello World!')
    "Minimize a scalar function subject to constraints.\n\n    Parameters\n    ----------\n    gtol : float, optional\n        Tolerance for termination by the norm of the Lagrangian gradient.\n        The algorithm will terminate when both the infinity norm (i.e., max\n        abs value) of the Lagrangian gradient and the constraint violation\n        are smaller than ``gtol``. Default is 1e-8.\n    xtol : float, optional\n        Tolerance for termination by the change of the independent variable.\n        The algorithm will terminate when ``tr_radius < xtol``, where\n        ``tr_radius`` is the radius of the trust region used in the algorithm.\n        Default is 1e-8.\n    barrier_tol : float, optional\n        Threshold on the barrier parameter for the algorithm termination.\n        When inequality constraints are present, the algorithm will terminate\n        only when the barrier parameter is less than `barrier_tol`.\n        Default is 1e-8.\n    sparse_jacobian : {bool, None}, optional\n        Determines how to represent Jacobians of the constraints. If bool,\n        then Jacobians of all the constraints will be converted to the\n        corresponding format. If None (default), then Jacobians won't be\n        converted, but the algorithm can proceed only if they all have the\n        same format.\n    initial_tr_radius: float, optional\n        Initial trust radius. The trust radius gives the maximum distance\n        between solution points in consecutive iterations. It reflects the\n        trust the algorithm puts in the local approximation of the optimization\n        problem. For an accurate local approximation the trust-region should be\n        large and for an  approximation valid only close to the current point it\n        should be a small one. The trust radius is automatically updated throughout\n        the optimization process, with ``initial_tr_radius`` being its initial value.\n        Default is 1 (recommended in [1]_, p. 19).\n    initial_constr_penalty : float, optional\n        Initial constraints penalty parameter. The penalty parameter is used for\n        balancing the requirements of decreasing the objective function\n        and satisfying the constraints. It is used for defining the merit function:\n        ``merit_function(x) = fun(x) + constr_penalty * constr_norm_l2(x)``,\n        where ``constr_norm_l2(x)`` is the l2 norm of a vector containing all\n        the constraints. The merit function is used for accepting or rejecting\n        trial points and ``constr_penalty`` weights the two conflicting goals\n        of reducing objective function and constraints. The penalty is automatically\n        updated throughout the optimization  process, with\n        ``initial_constr_penalty`` being its  initial value. Default is 1\n        (recommended in [1]_, p 19).\n    initial_barrier_parameter, initial_barrier_tolerance: float, optional\n        Initial barrier parameter and initial tolerance for the barrier subproblem.\n        Both are used only when inequality constraints are present. For dealing with\n        optimization problems ``min_x f(x)`` subject to inequality constraints\n        ``c(x) <= 0`` the algorithm introduces slack variables, solving the problem\n        ``min_(x,s) f(x) + barrier_parameter*sum(ln(s))`` subject to the equality\n        constraints  ``c(x) + s = 0`` instead of the original problem. This subproblem\n        is solved for decreasing values of ``barrier_parameter`` and with decreasing\n        tolerances for the termination, starting with ``initial_barrier_parameter``\n        for the barrier parameter and ``initial_barrier_tolerance`` for the\n        barrier tolerance. Default is 0.1 for both values (recommended in [1]_ p. 19).\n        Also note that ``barrier_parameter`` and ``barrier_tolerance`` are updated\n        with the same prefactor.\n    factorization_method : string or None, optional\n        Method to factorize the Jacobian of the constraints. Use None (default)\n        for the auto selection or one of:\n\n            - 'NormalEquation' (requires scikit-sparse)\n            - 'AugmentedSystem'\n            - 'QRFactorization'\n            - 'SVDFactorization'\n\n        The methods 'NormalEquation' and 'AugmentedSystem' can be used only\n        with sparse constraints. The projections required by the algorithm\n        will be computed using, respectively, the normal equation  and the\n        augmented system approaches explained in [1]_. 'NormalEquation'\n        computes the Cholesky factorization of ``A A.T`` and 'AugmentedSystem'\n        performs the LU factorization of an augmented system. They usually\n        provide similar results. 'AugmentedSystem' is used by default for\n        sparse matrices.\n\n        The methods 'QRFactorization' and 'SVDFactorization' can be used\n        only with dense constraints. They compute the required projections\n        using, respectively, QR and SVD factorizations. The 'SVDFactorization'\n        method can cope with Jacobian matrices with deficient row rank and will\n        be used whenever other factorization methods fail (which may imply the\n        conversion of sparse matrices to a dense format when required).\n        By default, 'QRFactorization' is used for dense matrices.\n    finite_diff_rel_step : None or array_like, optional\n        Relative step size for the finite difference approximation.\n    maxiter : int, optional\n        Maximum number of algorithm iterations. Default is 1000.\n    verbose : {0, 1, 2}, optional\n        Level of algorithm's verbosity:\n\n            * 0 (default) : work silently.\n            * 1 : display a termination report.\n            * 2 : display progress during iterations.\n            * 3 : display progress during iterations (more complete report).\n\n    disp : bool, optional\n        If True (default), then `verbose` will be set to 1 if it was 0.\n\n    Returns\n    -------\n    `OptimizeResult` with the fields documented below. Note the following:\n\n        1. All values corresponding to the constraints are ordered as they\n           were passed to the solver. And values corresponding to `bounds`\n           constraints are put *after* other constraints.\n        2. All numbers of function, Jacobian or Hessian evaluations correspond\n           to numbers of actual Python function calls. It means, for example,\n           that if a Jacobian is estimated by finite differences, then the\n           number of Jacobian evaluations will be zero and the number of\n           function evaluations will be incremented by all calls during the\n           finite difference estimation.\n\n    x : ndarray, shape (n,)\n        Solution found.\n    optimality : float\n        Infinity norm of the Lagrangian gradient at the solution.\n    constr_violation : float\n        Maximum constraint violation at the solution.\n    fun : float\n        Objective function at the solution.\n    grad : ndarray, shape (n,)\n        Gradient of the objective function at the solution.\n    lagrangian_grad : ndarray, shape (n,)\n        Gradient of the Lagrangian function at the solution.\n    nit : int\n        Total number of iterations.\n    nfev : integer\n        Number of the objective function evaluations.\n    njev : integer\n        Number of the objective function gradient evaluations.\n    nhev : integer\n        Number of the objective function Hessian evaluations.\n    cg_niter : int\n        Total number of the conjugate gradient method iterations.\n    method : {'equality_constrained_sqp', 'tr_interior_point'}\n        Optimization method used.\n    constr : list of ndarray\n        List of constraint values at the solution.\n    jac : list of {ndarray, sparse matrix}\n        List of the Jacobian matrices of the constraints at the solution.\n    v : list of ndarray\n        List of the Lagrange multipliers for the constraints at the solution.\n        For an inequality constraint a positive multiplier means that the upper\n        bound is active, a negative multiplier means that the lower bound is\n        active and if a multiplier is zero it means the constraint is not\n        active.\n    constr_nfev : list of int\n        Number of constraint evaluations for each of the constraints.\n    constr_njev : list of int\n        Number of Jacobian matrix evaluations for each of the constraints.\n    constr_nhev : list of int\n        Number of Hessian evaluations for each of the constraints.\n    tr_radius : float\n        Radius of the trust region at the last iteration.\n    constr_penalty : float\n        Penalty parameter at the last iteration, see `initial_constr_penalty`.\n    barrier_tolerance : float\n        Tolerance for the barrier subproblem at the last iteration.\n        Only for problems with inequality constraints.\n    barrier_parameter : float\n        Barrier parameter at the last iteration. Only for problems\n        with inequality constraints.\n    execution_time : float\n        Total execution time.\n    message : str\n        Termination message.\n    status : {0, 1, 2, 3}\n        Termination status:\n\n            * 0 : The maximum number of function evaluations is exceeded.\n            * 1 : `gtol` termination condition is satisfied.\n            * 2 : `xtol` termination condition is satisfied.\n            * 3 : `callback` function requested termination.\n\n    cg_stop_cond : int\n        Reason for CG subproblem termination at the last iteration:\n\n            * 0 : CG subproblem not evaluated.\n            * 1 : Iteration limit was reached.\n            * 2 : Reached the trust-region boundary.\n            * 3 : Negative curvature detected.\n            * 4 : Tolerance was satisfied.\n\n    References\n    ----------\n    .. [1] Conn, A. R., Gould, N. I., & Toint, P. L.\n           Trust region methods. 2000. Siam. pp. 19.\n    "
    x0 = np.atleast_1d(x0).astype(float)
    n_vars = np.size(x0)
    if hess is None:
        if callable(hessp):
            hess = HessianLinearOperator(hessp, n_vars)
        else:
            hess = BFGS()
    if disp and verbose == 0:
        verbose = 1
    if bounds is not None:
        modified_lb = np.nextafter(bounds.lb, -np.inf, where=bounds.lb > -np.inf)
        modified_ub = np.nextafter(bounds.ub, np.inf, where=bounds.ub < np.inf)
        modified_lb = np.where(np.isfinite(bounds.lb), modified_lb, bounds.lb)
        modified_ub = np.where(np.isfinite(bounds.ub), modified_ub, bounds.ub)
        bounds = Bounds(modified_lb, modified_ub, keep_feasible=bounds.keep_feasible)
        finite_diff_bounds = strict_bounds(bounds.lb, bounds.ub, bounds.keep_feasible, n_vars)
    else:
        finite_diff_bounds = (-np.inf, np.inf)
    objective = ScalarFunction(fun, x0, args, grad, hess, finite_diff_rel_step, finite_diff_bounds)
    if isinstance(constraints, (NonlinearConstraint, LinearConstraint)):
        constraints = [constraints]
    prepared_constraints = [PreparedConstraint(c, x0, sparse_jacobian, finite_diff_bounds) for c in constraints]
    n_sparse = sum((c.fun.sparse_jacobian for c in prepared_constraints))
    if 0 < n_sparse < len(prepared_constraints):
        raise ValueError('All constraints must have the same kind of the Jacobian --- either all sparse or all dense. You can set the sparsity globally by setting `sparse_jacobian` to either True of False.')
    if prepared_constraints:
        sparse_jacobian = n_sparse > 0
    if bounds is not None:
        if sparse_jacobian is None:
            sparse_jacobian = True
        prepared_constraints.append(PreparedConstraint(bounds, x0, sparse_jacobian))
    (c_eq0, c_ineq0, J_eq0, J_ineq0) = initial_constraints_as_canonical(n_vars, prepared_constraints, sparse_jacobian)
    canonical_all = [CanonicalConstraint.from_PreparedConstraint(c) for c in prepared_constraints]
    if len(canonical_all) == 0:
        canonical = CanonicalConstraint.empty(n_vars)
    elif len(canonical_all) == 1:
        canonical = canonical_all[0]
    else:
        canonical = CanonicalConstraint.concatenate(canonical_all, sparse_jacobian)
    lagrangian_hess = LagrangianHessian(n_vars, objective.hess, canonical.hess)
    if canonical.n_ineq == 0:
        method = 'equality_constrained_sqp'
    else:
        method = 'tr_interior_point'
    state = OptimizeResult(nit=0, nfev=0, njev=0, nhev=0, cg_niter=0, cg_stop_cond=0, fun=objective.f, grad=objective.g, lagrangian_grad=np.copy(objective.g), constr=[c.fun.f for c in prepared_constraints], jac=[c.fun.J for c in prepared_constraints], constr_nfev=[0 for c in prepared_constraints], constr_njev=[0 for c in prepared_constraints], constr_nhev=[0 for c in prepared_constraints], v=[c.fun.v for c in prepared_constraints], method=method)
    start_time = time.time()
    if method == 'equality_constrained_sqp':

        def stop_criteria(state, x, last_iteration_failed, optimality, constr_violation, tr_radius, constr_penalty, cg_info):
            if False:
                i = 10
                return i + 15
            state = update_state_sqp(state, x, last_iteration_failed, objective, prepared_constraints, start_time, tr_radius, constr_penalty, cg_info)
            if verbose == 2:
                BasicReport.print_iteration(state.nit, state.nfev, state.cg_niter, state.fun, state.tr_radius, state.optimality, state.constr_violation)
            elif verbose > 2:
                SQPReport.print_iteration(state.nit, state.nfev, state.cg_niter, state.fun, state.tr_radius, state.optimality, state.constr_violation, state.constr_penalty, state.cg_stop_cond)
            state.status = None
            state.niter = state.nit
            if callback is not None:
                callback_stop = False
                try:
                    callback_stop = callback(state)
                except StopIteration:
                    callback_stop = True
                if callback_stop:
                    state.status = 3
                    return True
            if state.optimality < gtol and state.constr_violation < gtol:
                state.status = 1
            elif state.tr_radius < xtol:
                state.status = 2
            elif state.nit >= maxiter:
                state.status = 0
            return state.status in (0, 1, 2, 3)
    elif method == 'tr_interior_point':

        def stop_criteria(state, x, last_iteration_failed, tr_radius, constr_penalty, cg_info, barrier_parameter, barrier_tolerance):
            if False:
                print('Hello World!')
            state = update_state_ip(state, x, last_iteration_failed, objective, prepared_constraints, start_time, tr_radius, constr_penalty, cg_info, barrier_parameter, barrier_tolerance)
            if verbose == 2:
                BasicReport.print_iteration(state.nit, state.nfev, state.cg_niter, state.fun, state.tr_radius, state.optimality, state.constr_violation)
            elif verbose > 2:
                IPReport.print_iteration(state.nit, state.nfev, state.cg_niter, state.fun, state.tr_radius, state.optimality, state.constr_violation, state.constr_penalty, state.barrier_parameter, state.cg_stop_cond)
            state.status = None
            state.niter = state.nit
            if callback is not None:
                callback_stop = False
                try:
                    callback_stop = callback(state)
                except StopIteration:
                    callback_stop = True
                if callback_stop:
                    state.status = 3
                    return True
            if state.optimality < gtol and state.constr_violation < gtol:
                state.status = 1
            elif state.tr_radius < xtol and state.barrier_parameter < barrier_tol:
                state.status = 2
            elif state.nit >= maxiter:
                state.status = 0
            return state.status in (0, 1, 2, 3)
    if verbose == 2:
        BasicReport.print_header()
    elif verbose > 2:
        if method == 'equality_constrained_sqp':
            SQPReport.print_header()
        elif method == 'tr_interior_point':
            IPReport.print_header()
    if method == 'equality_constrained_sqp':

        def fun_and_constr(x):
            if False:
                for i in range(10):
                    print('nop')
            f = objective.fun(x)
            (c_eq, _) = canonical.fun(x)
            return (f, c_eq)

        def grad_and_jac(x):
            if False:
                while True:
                    i = 10
            g = objective.grad(x)
            (J_eq, _) = canonical.jac(x)
            return (g, J_eq)
        (_, result) = equality_constrained_sqp(fun_and_constr, grad_and_jac, lagrangian_hess, x0, objective.f, objective.g, c_eq0, J_eq0, stop_criteria, state, initial_constr_penalty, initial_tr_radius, factorization_method)
    elif method == 'tr_interior_point':
        (_, result) = tr_interior_point(objective.fun, objective.grad, lagrangian_hess, n_vars, canonical.n_ineq, canonical.n_eq, canonical.fun, canonical.jac, x0, objective.f, objective.g, c_ineq0, J_ineq0, c_eq0, J_eq0, stop_criteria, canonical.keep_feasible, xtol, state, initial_barrier_parameter, initial_barrier_tolerance, initial_constr_penalty, initial_tr_radius, factorization_method)
    result.success = True if result.status in (1, 2) else False
    result.message = TERMINATION_MESSAGES[result.status]
    result.niter = result.nit
    if verbose == 2:
        BasicReport.print_footer()
    elif verbose > 2:
        if method == 'equality_constrained_sqp':
            SQPReport.print_footer()
        elif method == 'tr_interior_point':
            IPReport.print_footer()
    if verbose >= 1:
        print(result.message)
        print('Number of iterations: {}, function evaluations: {}, CG iterations: {}, optimality: {:.2e}, constraint violation: {:.2e}, execution time: {:4.2} s.'.format(result.nit, result.nfev, result.cg_niter, result.optimality, result.constr_violation, result.execution_time))
    return result