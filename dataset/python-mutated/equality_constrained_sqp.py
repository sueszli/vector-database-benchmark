"""Byrd-Omojokun Trust-Region SQP method."""
from scipy.sparse import eye as speye
from .projections import projections
from .qp_subproblem import modified_dogleg, projected_cg, box_intersections
import numpy as np
from numpy.linalg import norm
__all__ = ['equality_constrained_sqp']

def default_scaling(x):
    if False:
        i = 10
        return i + 15
    (n,) = np.shape(x)
    return speye(n)

def equality_constrained_sqp(fun_and_constr, grad_and_jac, lagr_hess, x0, fun0, grad0, constr0, jac0, stop_criteria, state, initial_penalty, initial_trust_radius, factorization_method, trust_lb=None, trust_ub=None, scaling=default_scaling):
    if False:
        print('Hello World!')
    'Solve nonlinear equality-constrained problem using trust-region SQP.\n\n    Solve optimization problem:\n\n        minimize fun(x)\n        subject to: constr(x) = 0\n\n    using Byrd-Omojokun Trust-Region SQP method described in [1]_. Several\n    implementation details are based on [2]_ and [3]_, p. 549.\n\n    References\n    ----------\n    .. [1] Lalee, Marucha, Jorge Nocedal, and Todd Plantenga. "On the\n           implementation of an algorithm for large-scale equality\n           constrained optimization." SIAM Journal on\n           Optimization 8.3 (1998): 682-706.\n    .. [2] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.\n           "An interior point algorithm for large-scale nonlinear\n           programming." SIAM Journal on Optimization 9.4 (1999): 877-900.\n    .. [3] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"\n           Second Edition (2006).\n    '
    PENALTY_FACTOR = 0.3
    LARGE_REDUCTION_RATIO = 0.9
    INTERMEDIARY_REDUCTION_RATIO = 0.3
    SUFFICIENT_REDUCTION_RATIO = 1e-08
    TRUST_ENLARGEMENT_FACTOR_L = 7.0
    TRUST_ENLARGEMENT_FACTOR_S = 2.0
    MAX_TRUST_REDUCTION = 0.5
    MIN_TRUST_REDUCTION = 0.1
    SOC_THRESHOLD = 0.1
    TR_FACTOR = 0.8
    BOX_FACTOR = 0.5
    (n,) = np.shape(x0)
    if trust_lb is None:
        trust_lb = np.full(n, -np.inf)
    if trust_ub is None:
        trust_ub = np.full(n, np.inf)
    x = np.copy(x0)
    trust_radius = initial_trust_radius
    penalty = initial_penalty
    f = fun0
    c = grad0
    b = constr0
    A = jac0
    S = scaling(x)
    (Z, LS, Y) = projections(A, factorization_method)
    v = -LS.dot(c)
    H = lagr_hess(x, v)
    optimality = norm(c + A.T.dot(v), np.inf)
    constr_violation = norm(b, np.inf) if len(b) > 0 else 0
    cg_info = {'niter': 0, 'stop_cond': 0, 'hits_boundary': False}
    last_iteration_failed = False
    while not stop_criteria(state, x, last_iteration_failed, optimality, constr_violation, trust_radius, penalty, cg_info):
        dn = modified_dogleg(A, Y, b, TR_FACTOR * trust_radius, BOX_FACTOR * trust_lb, BOX_FACTOR * trust_ub)
        c_t = H.dot(dn) + c
        b_t = np.zeros_like(b)
        trust_radius_t = np.sqrt(trust_radius ** 2 - np.linalg.norm(dn) ** 2)
        lb_t = trust_lb - dn
        ub_t = trust_ub - dn
        (dt, cg_info) = projected_cg(H, c_t, Z, Y, b_t, trust_radius_t, lb_t, ub_t)
        d = dn + dt
        quadratic_model = 1 / 2 * H.dot(d).dot(d) + c.T.dot(d)
        linearized_constr = A.dot(d) + b
        vpred = norm(b) - norm(linearized_constr)
        vpred = max(1e-16, vpred)
        previous_penalty = penalty
        if quadratic_model > 0:
            new_penalty = quadratic_model / ((1 - PENALTY_FACTOR) * vpred)
            penalty = max(penalty, new_penalty)
        predicted_reduction = -quadratic_model + penalty * vpred
        merit_function = f + penalty * norm(b)
        x_next = x + S.dot(d)
        (f_next, b_next) = fun_and_constr(x_next)
        merit_function_next = f_next + penalty * norm(b_next)
        actual_reduction = merit_function - merit_function_next
        reduction_ratio = actual_reduction / predicted_reduction
        if reduction_ratio < SUFFICIENT_REDUCTION_RATIO and norm(dn) <= SOC_THRESHOLD * norm(dt):
            y = -Y.dot(b_next)
            (_, t, intersect) = box_intersections(d, y, trust_lb, trust_ub)
            x_soc = x + S.dot(d + t * y)
            (f_soc, b_soc) = fun_and_constr(x_soc)
            merit_function_soc = f_soc + penalty * norm(b_soc)
            actual_reduction_soc = merit_function - merit_function_soc
            reduction_ratio_soc = actual_reduction_soc / predicted_reduction
            if intersect and reduction_ratio_soc >= SUFFICIENT_REDUCTION_RATIO:
                x_next = x_soc
                f_next = f_soc
                b_next = b_soc
                reduction_ratio = reduction_ratio_soc
        if reduction_ratio >= LARGE_REDUCTION_RATIO:
            trust_radius = max(TRUST_ENLARGEMENT_FACTOR_L * norm(d), trust_radius)
        elif reduction_ratio >= INTERMEDIARY_REDUCTION_RATIO:
            trust_radius = max(TRUST_ENLARGEMENT_FACTOR_S * norm(d), trust_radius)
        elif reduction_ratio < SUFFICIENT_REDUCTION_RATIO:
            trust_reduction = (1 - SUFFICIENT_REDUCTION_RATIO) / (1 - reduction_ratio)
            new_trust_radius = trust_reduction * norm(d)
            if new_trust_radius >= MAX_TRUST_REDUCTION * trust_radius:
                trust_radius *= MAX_TRUST_REDUCTION
            elif new_trust_radius >= MIN_TRUST_REDUCTION * trust_radius:
                trust_radius = new_trust_radius
            else:
                trust_radius *= MIN_TRUST_REDUCTION
        if reduction_ratio >= SUFFICIENT_REDUCTION_RATIO:
            x = x_next
            (f, b) = (f_next, b_next)
            (c, A) = grad_and_jac(x)
            S = scaling(x)
            (Z, LS, Y) = projections(A, factorization_method)
            v = -LS.dot(c)
            H = lagr_hess(x, v)
            last_iteration_failed = False
            optimality = norm(c + A.T.dot(v), np.inf)
            constr_violation = norm(b, np.inf) if len(b) > 0 else 0
        else:
            penalty = previous_penalty
            last_iteration_failed = True
    return (x, state)