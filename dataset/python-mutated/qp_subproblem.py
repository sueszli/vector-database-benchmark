"""Equality-constrained quadratic programming solvers."""
from scipy.sparse import linalg, bmat, csc_matrix
from math import copysign
import numpy as np
from numpy.linalg import norm
__all__ = ['eqp_kktfact', 'sphere_intersections', 'box_intersections', 'box_sphere_intersections', 'inside_box_boundaries', 'modified_dogleg', 'projected_cg']

def eqp_kktfact(H, c, A, b):
    if False:
        while True:
            i = 10
    'Solve equality-constrained quadratic programming (EQP) problem.\n\n    Solve ``min 1/2 x.T H x + x.t c`` subject to ``A x + b = 0``\n    using direct factorization of the KKT system.\n\n    Parameters\n    ----------\n    H : sparse matrix, shape (n, n)\n        Hessian matrix of the EQP problem.\n    c : array_like, shape (n,)\n        Gradient of the quadratic objective function.\n    A : sparse matrix\n        Jacobian matrix of the EQP problem.\n    b : array_like, shape (m,)\n        Right-hand side of the constraint equation.\n\n    Returns\n    -------\n    x : array_like, shape (n,)\n        Solution of the KKT problem.\n    lagrange_multipliers : ndarray, shape (m,)\n        Lagrange multipliers of the KKT problem.\n    '
    (n,) = np.shape(c)
    (m,) = np.shape(b)
    kkt_matrix = csc_matrix(bmat([[H, A.T], [A, None]]))
    kkt_vec = np.hstack([-c, -b])
    lu = linalg.splu(kkt_matrix)
    kkt_sol = lu.solve(kkt_vec)
    x = kkt_sol[:n]
    lagrange_multipliers = -kkt_sol[n:n + m]
    return (x, lagrange_multipliers)

def sphere_intersections(z, d, trust_radius, entire_line=False):
    if False:
        while True:
            i = 10
    'Find the intersection between segment (or line) and spherical constraints.\n\n    Find the intersection between the segment (or line) defined by the\n    parametric  equation ``x(t) = z + t*d`` and the ball\n    ``||x|| <= trust_radius``.\n\n    Parameters\n    ----------\n    z : array_like, shape (n,)\n        Initial point.\n    d : array_like, shape (n,)\n        Direction.\n    trust_radius : float\n        Ball radius.\n    entire_line : bool, optional\n        When ``True``, the function returns the intersection between the line\n        ``x(t) = z + t*d`` (``t`` can assume any value) and the ball\n        ``||x|| <= trust_radius``. When ``False``, the function returns the intersection\n        between the segment ``x(t) = z + t*d``, ``0 <= t <= 1``, and the ball.\n\n    Returns\n    -------\n    ta, tb : float\n        The line/segment ``x(t) = z + t*d`` is inside the ball for\n        for ``ta <= t <= tb``.\n    intersect : bool\n        When ``True``, there is a intersection between the line/segment\n        and the sphere. On the other hand, when ``False``, there is no\n        intersection.\n    '
    if norm(d) == 0:
        return (0, 0, False)
    if np.isinf(trust_radius):
        if entire_line:
            ta = -np.inf
            tb = np.inf
        else:
            ta = 0
            tb = 1
        intersect = True
        return (ta, tb, intersect)
    a = np.dot(d, d)
    b = 2 * np.dot(z, d)
    c = np.dot(z, z) - trust_radius ** 2
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        intersect = False
        return (0, 0, intersect)
    sqrt_discriminant = np.sqrt(discriminant)
    aux = b + copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux
    (ta, tb) = sorted([ta, tb])
    if entire_line:
        intersect = True
    elif tb < 0 or ta > 1:
        intersect = False
        ta = 0
        tb = 0
    else:
        intersect = True
        ta = max(0, ta)
        tb = min(1, tb)
    return (ta, tb, intersect)

def box_intersections(z, d, lb, ub, entire_line=False):
    if False:
        return 10
    'Find the intersection between segment (or line) and box constraints.\n\n    Find the intersection between the segment (or line) defined by the\n    parametric  equation ``x(t) = z + t*d`` and the rectangular box\n    ``lb <= x <= ub``.\n\n    Parameters\n    ----------\n    z : array_like, shape (n,)\n        Initial point.\n    d : array_like, shape (n,)\n        Direction.\n    lb : array_like, shape (n,)\n        Lower bounds to each one of the components of ``x``. Used\n        to delimit the rectangular box.\n    ub : array_like, shape (n, )\n        Upper bounds to each one of the components of ``x``. Used\n        to delimit the rectangular box.\n    entire_line : bool, optional\n        When ``True``, the function returns the intersection between the line\n        ``x(t) = z + t*d`` (``t`` can assume any value) and the rectangular\n        box. When ``False``, the function returns the intersection between the segment\n        ``x(t) = z + t*d``, ``0 <= t <= 1``, and the rectangular box.\n\n    Returns\n    -------\n    ta, tb : float\n        The line/segment ``x(t) = z + t*d`` is inside the box for\n        for ``ta <= t <= tb``.\n    intersect : bool\n        When ``True``, there is a intersection between the line (or segment)\n        and the rectangular box. On the other hand, when ``False``, there is no\n        intersection.\n    '
    z = np.asarray(z)
    d = np.asarray(d)
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    if norm(d) == 0:
        return (0, 0, False)
    zero_d = d == 0
    if (z[zero_d] < lb[zero_d]).any() or (z[zero_d] > ub[zero_d]).any():
        intersect = False
        return (0, 0, intersect)
    not_zero_d = np.logical_not(zero_d)
    z = z[not_zero_d]
    d = d[not_zero_d]
    lb = lb[not_zero_d]
    ub = ub[not_zero_d]
    t_lb = (lb - z) / d
    t_ub = (ub - z) / d
    ta = max(np.minimum(t_lb, t_ub))
    tb = min(np.maximum(t_lb, t_ub))
    if ta <= tb:
        intersect = True
    else:
        intersect = False
    if not entire_line:
        if tb < 0 or ta > 1:
            intersect = False
            ta = 0
            tb = 0
        else:
            ta = max(0, ta)
            tb = min(1, tb)
    return (ta, tb, intersect)

def box_sphere_intersections(z, d, lb, ub, trust_radius, entire_line=False, extra_info=False):
    if False:
        for i in range(10):
            print('nop')
    'Find the intersection between segment (or line) and box/sphere constraints.\n\n    Find the intersection between the segment (or line) defined by the\n    parametric  equation ``x(t) = z + t*d``, the rectangular box\n    ``lb <= x <= ub`` and the ball ``||x|| <= trust_radius``.\n\n    Parameters\n    ----------\n    z : array_like, shape (n,)\n        Initial point.\n    d : array_like, shape (n,)\n        Direction.\n    lb : array_like, shape (n,)\n        Lower bounds to each one of the components of ``x``. Used\n        to delimit the rectangular box.\n    ub : array_like, shape (n, )\n        Upper bounds to each one of the components of ``x``. Used\n        to delimit the rectangular box.\n    trust_radius : float\n        Ball radius.\n    entire_line : bool, optional\n        When ``True``, the function returns the intersection between the line\n        ``x(t) = z + t*d`` (``t`` can assume any value) and the constraints.\n        When ``False``, the function returns the intersection between the segment\n        ``x(t) = z + t*d``, ``0 <= t <= 1`` and the constraints.\n    extra_info : bool, optional\n        When ``True``, the function returns ``intersect_sphere`` and ``intersect_box``.\n\n    Returns\n    -------\n    ta, tb : float\n        The line/segment ``x(t) = z + t*d`` is inside the rectangular box and\n        inside the ball for ``ta <= t <= tb``.\n    intersect : bool\n        When ``True``, there is a intersection between the line (or segment)\n        and both constraints. On the other hand, when ``False``, there is no\n        intersection.\n    sphere_info : dict, optional\n        Dictionary ``{ta, tb, intersect}`` containing the interval ``[ta, tb]``\n        for which the line intercepts the ball. And a boolean value indicating\n        whether the sphere is intersected by the line.\n    box_info : dict, optional\n        Dictionary ``{ta, tb, intersect}`` containing the interval ``[ta, tb]``\n        for which the line intercepts the box. And a boolean value indicating\n        whether the box is intersected by the line.\n    '
    (ta_b, tb_b, intersect_b) = box_intersections(z, d, lb, ub, entire_line)
    (ta_s, tb_s, intersect_s) = sphere_intersections(z, d, trust_radius, entire_line)
    ta = np.maximum(ta_b, ta_s)
    tb = np.minimum(tb_b, tb_s)
    if intersect_b and intersect_s and (ta <= tb):
        intersect = True
    else:
        intersect = False
    if extra_info:
        sphere_info = {'ta': ta_s, 'tb': tb_s, 'intersect': intersect_s}
        box_info = {'ta': ta_b, 'tb': tb_b, 'intersect': intersect_b}
        return (ta, tb, intersect, sphere_info, box_info)
    else:
        return (ta, tb, intersect)

def inside_box_boundaries(x, lb, ub):
    if False:
        return 10
    'Check if lb <= x <= ub.'
    return (lb <= x).all() and (x <= ub).all()

def reinforce_box_boundaries(x, lb, ub):
    if False:
        return 10
    'Return clipped value of x'
    return np.minimum(np.maximum(x, lb), ub)

def modified_dogleg(A, Y, b, trust_radius, lb, ub):
    if False:
        return 10
    'Approximately  minimize ``1/2*|| A x + b ||^2`` inside trust-region.\n\n    Approximately solve the problem of minimizing ``1/2*|| A x + b ||^2``\n    subject to ``||x|| < Delta`` and ``lb <= x <= ub`` using a modification\n    of the classical dogleg approach.\n\n    Parameters\n    ----------\n    A : LinearOperator (or sparse matrix or ndarray), shape (m, n)\n        Matrix ``A`` in the minimization problem. It should have\n        dimension ``(m, n)`` such that ``m < n``.\n    Y : LinearOperator (or sparse matrix or ndarray), shape (n, m)\n        LinearOperator that apply the projection matrix\n        ``Q = A.T inv(A A.T)`` to the vector. The obtained vector\n        ``y = Q x`` being the minimum norm solution of ``A y = x``.\n    b : array_like, shape (m,)\n        Vector ``b``in the minimization problem.\n    trust_radius: float\n        Trust radius to be considered. Delimits a sphere boundary\n        to the problem.\n    lb : array_like, shape (n,)\n        Lower bounds to each one of the components of ``x``.\n        It is expected that ``lb <= 0``, otherwise the algorithm\n        may fail. If ``lb[i] = -Inf``, the lower\n        bound for the ith component is just ignored.\n    ub : array_like, shape (n, )\n        Upper bounds to each one of the components of ``x``.\n        It is expected that ``ub >= 0``, otherwise the algorithm\n        may fail. If ``ub[i] = Inf``, the upper bound for the ith\n        component is just ignored.\n\n    Returns\n    -------\n    x : array_like, shape (n,)\n        Solution to the problem.\n\n    Notes\n    -----\n    Based on implementations described in pp. 885-886 from [1]_.\n\n    References\n    ----------\n    .. [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.\n           "An interior point algorithm for large-scale nonlinear\n           programming." SIAM Journal on Optimization 9.4 (1999): 877-900.\n    '
    newton_point = -Y.dot(b)
    if inside_box_boundaries(newton_point, lb, ub) and norm(newton_point) <= trust_radius:
        x = newton_point
        return x
    g = A.T.dot(b)
    A_g = A.dot(g)
    cauchy_point = -np.dot(g, g) / np.dot(A_g, A_g) * g
    origin_point = np.zeros_like(cauchy_point)
    z = cauchy_point
    p = newton_point - cauchy_point
    (_, alpha, intersect) = box_sphere_intersections(z, p, lb, ub, trust_radius)
    if intersect:
        x1 = z + alpha * p
    else:
        z = origin_point
        p = cauchy_point
        (_, alpha, _) = box_sphere_intersections(z, p, lb, ub, trust_radius)
        x1 = z + alpha * p
    z = origin_point
    p = newton_point
    (_, alpha, _) = box_sphere_intersections(z, p, lb, ub, trust_radius)
    x2 = z + alpha * p
    if norm(A.dot(x1) + b) < norm(A.dot(x2) + b):
        return x1
    else:
        return x2

def projected_cg(H, c, Z, Y, b, trust_radius=np.inf, lb=None, ub=None, tol=None, max_iter=None, max_infeasible_iter=None, return_all=False):
    if False:
        i = 10
        return i + 15
    'Solve EQP problem with projected CG method.\n\n    Solve equality-constrained quadratic programming problem\n    ``min 1/2 x.T H x + x.t c``  subject to ``A x + b = 0`` and,\n    possibly, to trust region constraints ``||x|| < trust_radius``\n    and box constraints ``lb <= x <= ub``.\n\n    Parameters\n    ----------\n    H : LinearOperator (or sparse matrix or ndarray), shape (n, n)\n        Operator for computing ``H v``.\n    c : array_like, shape (n,)\n        Gradient of the quadratic objective function.\n    Z : LinearOperator (or sparse matrix or ndarray), shape (n, n)\n        Operator for projecting ``x`` into the null space of A.\n    Y : LinearOperator,  sparse matrix, ndarray, shape (n, m)\n        Operator that, for a given a vector ``b``, compute smallest\n        norm solution of ``A x + b = 0``.\n    b : array_like, shape (m,)\n        Right-hand side of the constraint equation.\n    trust_radius : float, optional\n        Trust radius to be considered. By default, uses ``trust_radius=inf``,\n        which means no trust radius at all.\n    lb : array_like, shape (n,), optional\n        Lower bounds to each one of the components of ``x``.\n        If ``lb[i] = -Inf`` the lower bound for the i-th\n        component is just ignored (default).\n    ub : array_like, shape (n, ), optional\n        Upper bounds to each one of the components of ``x``.\n        If ``ub[i] = Inf`` the upper bound for the i-th\n        component is just ignored (default).\n    tol : float, optional\n        Tolerance used to interrupt the algorithm.\n    max_iter : int, optional\n        Maximum algorithm iterations. Where ``max_inter <= n-m``.\n        By default, uses ``max_iter = n-m``.\n    max_infeasible_iter : int, optional\n        Maximum infeasible (regarding box constraints) iterations the\n        algorithm is allowed to take.\n        By default, uses ``max_infeasible_iter = n-m``.\n    return_all : bool, optional\n        When ``true``, return the list of all vectors through the iterations.\n\n    Returns\n    -------\n    x : array_like, shape (n,)\n        Solution of the EQP problem.\n    info : Dict\n        Dictionary containing the following:\n\n            - niter : Number of iterations.\n            - stop_cond : Reason for algorithm termination:\n                1. Iteration limit was reached;\n                2. Reached the trust-region boundary;\n                3. Negative curvature detected;\n                4. Tolerance was satisfied.\n            - allvecs : List containing all intermediary vectors (optional).\n            - hits_boundary : True if the proposed step is on the boundary\n              of the trust region.\n\n    Notes\n    -----\n    Implementation of Algorithm 6.2 on [1]_.\n\n    In the absence of spherical and box constraints, for sufficient\n    iterations, the method returns a truly optimal result.\n    In the presence of those constraints, the value returned is only\n    a inexpensive approximation of the optimal value.\n\n    References\n    ----------\n    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.\n           "On the solution of equality constrained quadratic\n            programming problems arising in optimization."\n            SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.\n    '
    CLOSE_TO_ZERO = 1e-25
    (n,) = np.shape(c)
    (m,) = np.shape(b)
    x = Y.dot(-b)
    r = Z.dot(H.dot(x) + c)
    g = Z.dot(r)
    p = -g
    if return_all:
        allvecs = [x]
    H_p = H.dot(p)
    rt_g = norm(g) ** 2
    tr_distance = trust_radius - norm(x)
    if tr_distance < 0:
        raise ValueError('Trust region problem does not have a solution.')
    elif tr_distance < CLOSE_TO_ZERO:
        info = {'niter': 0, 'stop_cond': 2, 'hits_boundary': True}
        if return_all:
            allvecs.append(x)
            info['allvecs'] = allvecs
        return (x, info)
    if tol is None:
        tol = max(min(0.01 * np.sqrt(rt_g), 0.1 * rt_g), CLOSE_TO_ZERO)
    if lb is None:
        lb = np.full(n, -np.inf)
    if ub is None:
        ub = np.full(n, np.inf)
    if max_iter is None:
        max_iter = n - m
    max_iter = min(max_iter, n - m)
    if max_infeasible_iter is None:
        max_infeasible_iter = n - m
    hits_boundary = False
    stop_cond = 1
    counter = 0
    last_feasible_x = np.zeros_like(x)
    k = 0
    for i in range(max_iter):
        if rt_g < tol:
            stop_cond = 4
            break
        k += 1
        pt_H_p = H_p.dot(p)
        if pt_H_p <= 0:
            if np.isinf(trust_radius):
                raise ValueError('Negative curvature not allowed for unrestricted problems.')
            else:
                (_, alpha, intersect) = box_sphere_intersections(x, p, lb, ub, trust_radius, entire_line=True)
                if intersect:
                    x = x + alpha * p
                x = reinforce_box_boundaries(x, lb, ub)
                stop_cond = 3
                hits_boundary = True
                break
        alpha = rt_g / pt_H_p
        x_next = x + alpha * p
        if np.linalg.norm(x_next) >= trust_radius:
            (_, theta, intersect) = box_sphere_intersections(x, alpha * p, lb, ub, trust_radius)
            if intersect:
                x = x + theta * alpha * p
            x = reinforce_box_boundaries(x, lb, ub)
            stop_cond = 2
            hits_boundary = True
            break
        if inside_box_boundaries(x_next, lb, ub):
            counter = 0
        else:
            counter += 1
        if counter > 0:
            (_, theta, intersect) = box_sphere_intersections(x, alpha * p, lb, ub, trust_radius)
            if intersect:
                last_feasible_x = x + theta * alpha * p
                last_feasible_x = reinforce_box_boundaries(last_feasible_x, lb, ub)
                counter = 0
        if counter > max_infeasible_iter:
            break
        if return_all:
            allvecs.append(x_next)
        r_next = r + alpha * H_p
        g_next = Z.dot(r_next)
        rt_g_next = norm(g_next) ** 2
        beta = rt_g_next / rt_g
        p = -g_next + beta * p
        x = x_next
        g = g_next
        r = g_next
        rt_g = norm(g) ** 2
        H_p = H.dot(p)
    if not inside_box_boundaries(x, lb, ub):
        x = last_feasible_x
        hits_boundary = True
    info = {'niter': k, 'stop_cond': stop_cond, 'hits_boundary': hits_boundary}
    if return_all:
        info['allvecs'] = allvecs
    return (x, info)