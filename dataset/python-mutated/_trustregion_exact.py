"""Nearly exact trust-region optimization subproblem."""
import numpy as np
from scipy.linalg import norm, get_lapack_funcs, solve_triangular, cho_solve
from ._trustregion import _minimize_trust_region, BaseQuadraticSubproblem
__all__ = ['_minimize_trustregion_exact', 'estimate_smallest_singular_value', 'singular_leading_submatrix', 'IterativeSubproblem']

def _minimize_trustregion_exact(fun, x0, args=(), jac=None, hess=None, **trust_region_options):
    if False:
        while True:
            i = 10
    '\n    Minimization of scalar function of one or more variables using\n    a nearly exact trust-region algorithm.\n\n    Options\n    -------\n    initial_trust_radius : float\n        Initial trust-region radius.\n    max_trust_radius : float\n        Maximum value of the trust-region radius. No steps that are longer\n        than this value will be proposed.\n    eta : float\n        Trust region related acceptance stringency for proposed steps.\n    gtol : float\n        Gradient norm must be less than ``gtol`` before successful\n        termination.\n    '
    if jac is None:
        raise ValueError('Jacobian is required for trust region exact minimization.')
    if not callable(hess):
        raise ValueError('Hessian matrix is required for trust region exact minimization.')
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess, subproblem=IterativeSubproblem, **trust_region_options)

def estimate_smallest_singular_value(U):
    if False:
        i = 10
        return i + 15
    'Given upper triangular matrix ``U`` estimate the smallest singular\n    value and the correspondent right singular vector in O(n**2) operations.\n\n    Parameters\n    ----------\n    U : ndarray\n        Square upper triangular matrix.\n\n    Returns\n    -------\n    s_min : float\n        Estimated smallest singular value of the provided matrix.\n    z_min : ndarray\n        Estimatied right singular vector.\n\n    Notes\n    -----\n    The procedure is based on [1]_ and is done in two steps. First, it finds\n    a vector ``e`` with components selected from {+1, -1} such that the\n    solution ``w`` from the system ``U.T w = e`` is as large as possible.\n    Next it estimate ``U v = w``. The smallest singular value is close\n    to ``norm(w)/norm(v)`` and the right singular vector is close\n    to ``v/norm(v)``.\n\n    The estimation will be better more ill-conditioned is the matrix.\n\n    References\n    ----------\n    .. [1] Cline, A. K., Moler, C. B., Stewart, G. W., Wilkinson, J. H.\n           An estimate for the condition number of a matrix.  1979.\n           SIAM Journal on Numerical Analysis, 16(2), 368-375.\n    '
    U = np.atleast_2d(U)
    (m, n) = U.shape
    if m != n:
        raise ValueError('A square triangular matrix should be provided.')
    p = np.zeros(n)
    w = np.empty(n)
    for k in range(n):
        wp = (1 - p[k]) / U.T[k, k]
        wm = (-1 - p[k]) / U.T[k, k]
        pp = p[k + 1:] + U.T[k + 1:, k] * wp
        pm = p[k + 1:] + U.T[k + 1:, k] * wm
        if abs(wp) + norm(pp, 1) >= abs(wm) + norm(pm, 1):
            w[k] = wp
            p[k + 1:] = pp
        else:
            w[k] = wm
            p[k + 1:] = pm
    v = solve_triangular(U, w)
    v_norm = norm(v)
    w_norm = norm(w)
    s_min = w_norm / v_norm
    z_min = v / v_norm
    return (s_min, z_min)

def gershgorin_bounds(H):
    if False:
        return 10
    '\n    Given a square matrix ``H`` compute upper\n    and lower bounds for its eigenvalues (Gregoshgorin Bounds).\n    Defined ref. [1].\n\n    References\n    ----------\n    .. [1] Conn, A. R., Gould, N. I., & Toint, P. L.\n           Trust region methods. 2000. Siam. pp. 19.\n    '
    H_diag = np.diag(H)
    H_diag_abs = np.abs(H_diag)
    H_row_sums = np.sum(np.abs(H), axis=1)
    lb = np.min(H_diag + H_diag_abs - H_row_sums)
    ub = np.max(H_diag - H_diag_abs + H_row_sums)
    return (lb, ub)

def singular_leading_submatrix(A, U, k):
    if False:
        while True:
            i = 10
    '\n    Compute term that makes the leading ``k`` by ``k``\n    submatrix from ``A`` singular.\n\n    Parameters\n    ----------\n    A : ndarray\n        Symmetric matrix that is not positive definite.\n    U : ndarray\n        Upper triangular matrix resulting of an incomplete\n        Cholesky decomposition of matrix ``A``.\n    k : int\n        Positive integer such that the leading k by k submatrix from\n        `A` is the first non-positive definite leading submatrix.\n\n    Returns\n    -------\n    delta : float\n        Amount that should be added to the element (k, k) of the\n        leading k by k submatrix of ``A`` to make it singular.\n    v : ndarray\n        A vector such that ``v.T B v = 0``. Where B is the matrix A after\n        ``delta`` is added to its element (k, k).\n    '
    delta = np.sum(U[:k - 1, k - 1] ** 2) - A[k - 1, k - 1]
    n = len(A)
    v = np.zeros(n)
    v[k - 1] = 1
    if k != 1:
        v[:k - 1] = solve_triangular(U[:k - 1, :k - 1], -U[:k - 1, k - 1])
    return (delta, v)

class IterativeSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by nearly exact iterative method.

    Notes
    -----
    This subproblem solver was based on [1]_, [2]_ and [3]_,
    which implement similar algorithms. The algorithm is basically
    that of [1]_ but ideas from [2]_ and [3]_ were also used.

    References
    ----------
    .. [1] A.R. Conn, N.I. Gould, and P.L. Toint, "Trust region methods",
           Siam, pp. 169-200, 2000.
    .. [2] J. Nocedal and  S. Wright, "Numerical optimization",
           Springer Science & Business Media. pp. 83-91, 2006.
    .. [3] J.J. More and D.C. Sorensen, "Computing a trust region step",
           SIAM Journal on Scientific and Statistical Computing, vol. 4(3),
           pp. 553-572, 1983.
    """
    UPDATE_COEFF = 0.01
    EPS = np.finfo(float).eps

    def __init__(self, x, fun, jac, hess, hessp=None, k_easy=0.1, k_hard=0.2):
        if False:
            while True:
                i = 10
        super().__init__(x, fun, jac, hess)
        self.previous_tr_radius = -1
        self.lambda_lb = None
        self.niter = 0
        self.k_easy = k_easy
        self.k_hard = k_hard
        (self.cholesky,) = get_lapack_funcs(('potrf',), (self.hess,))
        self.dimension = len(self.hess)
        (self.hess_gershgorin_lb, self.hess_gershgorin_ub) = gershgorin_bounds(self.hess)
        self.hess_inf = norm(self.hess, np.inf)
        self.hess_fro = norm(self.hess, 'fro')
        self.CLOSE_TO_ZERO = self.dimension * self.EPS * self.hess_inf

    def _initial_values(self, tr_radius):
        if False:
            return 10
        'Given a trust radius, return a good initial guess for\n        the damping factor, the lower bound and the upper bound.\n        The values were chosen accordingly to the guidelines on\n        section 7.3.8 (p. 192) from [1]_.\n        '
        lambda_ub = max(0, self.jac_mag / tr_radius + min(-self.hess_gershgorin_lb, self.hess_fro, self.hess_inf))
        lambda_lb = max(0, -min(self.hess.diagonal()), self.jac_mag / tr_radius - min(self.hess_gershgorin_ub, self.hess_fro, self.hess_inf))
        if tr_radius < self.previous_tr_radius:
            lambda_lb = max(self.lambda_lb, lambda_lb)
        if lambda_lb == 0:
            lambda_initial = 0
        else:
            lambda_initial = max(np.sqrt(lambda_lb * lambda_ub), lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb))
        return (lambda_initial, lambda_lb, lambda_ub)

    def solve(self, tr_radius):
        if False:
            i = 10
            return i + 15
        'Solve quadratic subproblem'
        (lambda_current, lambda_lb, lambda_ub) = self._initial_values(tr_radius)
        n = self.dimension
        hits_boundary = True
        already_factorized = False
        self.niter = 0
        while True:
            if already_factorized:
                already_factorized = False
            else:
                H = self.hess + lambda_current * np.eye(n)
                (U, info) = self.cholesky(H, lower=False, overwrite_a=False, clean=True)
            self.niter += 1
            if info == 0 and self.jac_mag > self.CLOSE_TO_ZERO:
                p = cho_solve((U, False), -self.jac)
                p_norm = norm(p)
                if p_norm <= tr_radius and lambda_current == 0:
                    hits_boundary = False
                    break
                w = solve_triangular(U, p, trans='T')
                w_norm = norm(w)
                delta_lambda = (p_norm / w_norm) ** 2 * (p_norm - tr_radius) / tr_radius
                lambda_new = lambda_current + delta_lambda
                if p_norm < tr_radius:
                    (s_min, z_min) = estimate_smallest_singular_value(U)
                    (ta, tb) = self.get_boundaries_intersections(p, z_min, tr_radius)
                    step_len = min([ta, tb], key=abs)
                    quadratic_term = np.dot(p, np.dot(H, p))
                    relative_error = step_len ** 2 * s_min ** 2 / (quadratic_term + lambda_current * tr_radius ** 2)
                    if relative_error <= self.k_hard:
                        p += step_len * z_min
                        break
                    lambda_ub = lambda_current
                    lambda_lb = max(lambda_lb, lambda_current - s_min ** 2)
                    H = self.hess + lambda_new * np.eye(n)
                    (c, info) = self.cholesky(H, lower=False, overwrite_a=False, clean=True)
                    if info == 0:
                        lambda_current = lambda_new
                        already_factorized = True
                    else:
                        lambda_lb = max(lambda_lb, lambda_new)
                        lambda_current = max(np.sqrt(lambda_lb * lambda_ub), lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb))
                else:
                    relative_error = abs(p_norm - tr_radius) / tr_radius
                    if relative_error <= self.k_easy:
                        break
                    lambda_lb = lambda_current
                    lambda_current = lambda_new
            elif info == 0 and self.jac_mag <= self.CLOSE_TO_ZERO:
                if lambda_current == 0:
                    p = np.zeros(n)
                    hits_boundary = False
                    break
                (s_min, z_min) = estimate_smallest_singular_value(U)
                step_len = tr_radius
                if step_len ** 2 * s_min ** 2 <= self.k_hard * lambda_current * tr_radius ** 2:
                    p = step_len * z_min
                    break
                lambda_ub = lambda_current
                lambda_lb = max(lambda_lb, lambda_current - s_min ** 2)
                lambda_current = max(np.sqrt(lambda_lb * lambda_ub), lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb))
            else:
                (delta, v) = singular_leading_submatrix(H, U, info)
                v_norm = norm(v)
                lambda_lb = max(lambda_lb, lambda_current + delta / v_norm ** 2)
                lambda_current = max(np.sqrt(lambda_lb * lambda_ub), lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb))
        self.lambda_lb = lambda_lb
        self.lambda_current = lambda_current
        self.previous_tr_radius = tr_radius
        return (p, hits_boundary)