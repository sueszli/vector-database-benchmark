"""
Functions
---------
.. autosummary::
   :toctree: generated/

    line_search_armijo
    line_search_wolfe1
    line_search_wolfe2
    scalar_search_wolfe1
    scalar_search_wolfe2

"""
from warnings import warn
from scipy.optimize import _minpack2 as minpack2
from ._dcsrch import DCSRCH
import numpy as np
__all__ = ['LineSearchWarning', 'line_search_wolfe1', 'line_search_wolfe2', 'scalar_search_wolfe1', 'scalar_search_wolfe2', 'line_search_armijo']

class LineSearchWarning(RuntimeWarning):
    pass

def _check_c1_c2(c1, c2):
    if False:
        while True:
            i = 10
    if not 0 < c1 < c2 < 1:
        raise ValueError("'c1' and 'c2' do not satisfy'0 < c1 < c2 < 1'.")

def line_search_wolfe1(f, fprime, xk, pk, gfk=None, old_fval=None, old_old_fval=None, args=(), c1=0.0001, c2=0.9, amax=50, amin=1e-08, xtol=1e-14):
    if False:
        while True:
            i = 10
    '\n    As `scalar_search_wolfe1` but do a line search to direction `pk`\n\n    Parameters\n    ----------\n    f : callable\n        Function `f(x)`\n    fprime : callable\n        Gradient of `f`\n    xk : array_like\n        Current point\n    pk : array_like\n        Search direction\n    gfk : array_like, optional\n        Gradient of `f` at point `xk`\n    old_fval : float, optional\n        Value of `f` at point `xk`\n    old_old_fval : float, optional\n        Value of `f` at point preceding `xk`\n\n    The rest of the parameters are the same as for `scalar_search_wolfe1`.\n\n    Returns\n    -------\n    stp, f_count, g_count, fval, old_fval\n        As in `line_search_wolfe1`\n    gval : array\n        Gradient of `f` at the final point\n\n    Notes\n    -----\n    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.\n\n    '
    if gfk is None:
        gfk = fprime(xk, *args)
    gval = [gfk]
    gc = [0]
    fc = [0]

    def phi(s):
        if False:
            return 10
        fc[0] += 1
        return f(xk + s * pk, *args)

    def derphi(s):
        if False:
            print('Hello World!')
        gval[0] = fprime(xk + s * pk, *args)
        gc[0] += 1
        return np.dot(gval[0], pk)
    derphi0 = np.dot(gfk, pk)
    (stp, fval, old_fval) = scalar_search_wolfe1(phi, derphi, old_fval, old_old_fval, derphi0, c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)
    return (stp, fc[0], gc[0], fval, old_fval, gval[0])

def scalar_search_wolfe1(phi, derphi, phi0=None, old_phi0=None, derphi0=None, c1=0.0001, c2=0.9, amax=50, amin=1e-08, xtol=1e-14):
    if False:
        print('Hello World!')
    "\n    Scalar function search for alpha that satisfies strong Wolfe conditions\n\n    alpha > 0 is assumed to be a descent direction.\n\n    Parameters\n    ----------\n    phi : callable phi(alpha)\n        Function at point `alpha`\n    derphi : callable phi'(alpha)\n        Objective function derivative. Returns a scalar.\n    phi0 : float, optional\n        Value of phi at 0\n    old_phi0 : float, optional\n        Value of phi at previous point\n    derphi0 : float, optional\n        Value derphi at 0\n    c1 : float, optional\n        Parameter for Armijo condition rule.\n    c2 : float, optional\n        Parameter for curvature condition rule.\n    amax, amin : float, optional\n        Maximum and minimum step size\n    xtol : float, optional\n        Relative tolerance for an acceptable step.\n\n    Returns\n    -------\n    alpha : float\n        Step size, or None if no suitable step was found\n    phi : float\n        Value of `phi` at the new point `alpha`\n    phi0 : float\n        Value of `phi` at `alpha=0`\n\n    Notes\n    -----\n    Uses routine DCSRCH from MINPACK.\n    \n    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1`` as described in [1]_.\n\n    References\n    ----------\n    \n    .. [1] Nocedal, J., & Wright, S. J. (2006). Numerical optimization.\n       In Springer Series in Operations Research and Financial Engineering.\n       (Springer Series in Operations Research and Financial Engineering).\n       Springer Nature.\n\n    "
    _check_c1_c2(c1, c2)
    if phi0 is None:
        phi0 = phi(0.0)
    if derphi0 is None:
        derphi0 = derphi(0.0)
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01 * 2 * (phi0 - old_phi0) / derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0
    maxiter = 100
    dcsrch = DCSRCH(phi, derphi, c1, c2, xtol, amin, amax)
    (stp, phi1, phi0, task) = dcsrch(alpha1, phi0=phi0, derphi0=derphi0, maxiter=maxiter)
    return (stp, phi1, phi0)
line_search = line_search_wolfe1

def line_search_wolfe2(f, myfprime, xk, pk, gfk=None, old_fval=None, old_old_fval=None, args=(), c1=0.0001, c2=0.9, amax=None, extra_condition=None, maxiter=10):
    if False:
        i = 10
        return i + 15
    "Find alpha that satisfies strong Wolfe conditions.\n\n    Parameters\n    ----------\n    f : callable f(x,*args)\n        Objective function.\n    myfprime : callable f'(x,*args)\n        Objective function gradient.\n    xk : ndarray\n        Starting point.\n    pk : ndarray\n        Search direction. The search direction must be a descent direction\n        for the algorithm to converge.\n    gfk : ndarray, optional\n        Gradient value for x=xk (xk being the current parameter\n        estimate). Will be recomputed if omitted.\n    old_fval : float, optional\n        Function value for x=xk. Will be recomputed if omitted.\n    old_old_fval : float, optional\n        Function value for the point preceding x=xk.\n    args : tuple, optional\n        Additional arguments passed to objective function.\n    c1 : float, optional\n        Parameter for Armijo condition rule.\n    c2 : float, optional\n        Parameter for curvature condition rule.\n    amax : float, optional\n        Maximum step size\n    extra_condition : callable, optional\n        A callable of the form ``extra_condition(alpha, x, f, g)``\n        returning a boolean. Arguments are the proposed step ``alpha``\n        and the corresponding ``x``, ``f`` and ``g`` values. The line search\n        accepts the value of ``alpha`` only if this\n        callable returns ``True``. If the callable returns ``False``\n        for the step length, the algorithm will continue with\n        new iterates. The callable is only called for iterates\n        satisfying the strong Wolfe conditions.\n    maxiter : int, optional\n        Maximum number of iterations to perform.\n\n    Returns\n    -------\n    alpha : float or None\n        Alpha for which ``x_new = x0 + alpha * pk``,\n        or None if the line search algorithm did not converge.\n    fc : int\n        Number of function evaluations made.\n    gc : int\n        Number of gradient evaluations made.\n    new_fval : float or None\n        New function value ``f(x_new)=f(x0+alpha*pk)``,\n        or None if the line search algorithm did not converge.\n    old_fval : float\n        Old function value ``f(x0)``.\n    new_slope : float or None\n        The local slope along the search direction at the\n        new value ``<myfprime(x_new), pk>``,\n        or None if the line search algorithm did not converge.\n\n\n    Notes\n    -----\n    Uses the line search algorithm to enforce strong Wolfe\n    conditions. See Wright and Nocedal, 'Numerical Optimization',\n    1999, pp. 59-61.\n\n    The search direction `pk` must be a descent direction (e.g.\n    ``-myfprime(xk)``) to find a step length that satisfies the strong Wolfe\n    conditions. If the search direction is not a descent direction (e.g.\n    ``myfprime(xk)``), then `alpha`, `new_fval`, and `new_slope` will be None.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.optimize import line_search\n\n    A objective function and its gradient are defined.\n\n    >>> def obj_func(x):\n    ...     return (x[0])**2+(x[1])**2\n    >>> def obj_grad(x):\n    ...     return [2*x[0], 2*x[1]]\n\n    We can find alpha that satisfies strong Wolfe conditions.\n\n    >>> start_point = np.array([1.8, 1.7])\n    >>> search_gradient = np.array([-1.0, -1.0])\n    >>> line_search(obj_func, obj_grad, start_point, search_gradient)\n    (1.0, 2, 1, 1.1300000000000001, 6.13, [1.6, 1.4])\n\n    "
    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]

    def phi(alpha):
        if False:
            return 10
        fc[0] += 1
        return f(xk + alpha * pk, *args)
    fprime = myfprime

    def derphi(alpha):
        if False:
            while True:
                i = 10
        gc[0] += 1
        gval[0] = fprime(xk + alpha * pk, *args)
        gval_alpha[0] = alpha
        return np.dot(gval[0], pk)
    if gfk is None:
        gfk = fprime(xk, *args)
    derphi0 = np.dot(gfk, pk)
    if extra_condition is not None:

        def extra_condition2(alpha, phi):
            if False:
                i = 10
                return i + 15
            if gval_alpha[0] != alpha:
                derphi(alpha)
            x = xk + alpha * pk
            return extra_condition(alpha, x, phi, gval[0])
    else:
        extra_condition2 = None
    (alpha_star, phi_star, old_fval, derphi_star) = scalar_search_wolfe2(phi, derphi, old_fval, old_old_fval, derphi0, c1, c2, amax, extra_condition2, maxiter=maxiter)
    if derphi_star is None:
        warn('The line search algorithm did not converge', LineSearchWarning)
    else:
        derphi_star = gval[0]
    return (alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star)

def scalar_search_wolfe2(phi, derphi, phi0=None, old_phi0=None, derphi0=None, c1=0.0001, c2=0.9, amax=None, extra_condition=None, maxiter=10):
    if False:
        while True:
            i = 10
    "Find alpha that satisfies strong Wolfe conditions.\n\n    alpha > 0 is assumed to be a descent direction.\n\n    Parameters\n    ----------\n    phi : callable phi(alpha)\n        Objective scalar function.\n    derphi : callable phi'(alpha)\n        Objective function derivative. Returns a scalar.\n    phi0 : float, optional\n        Value of phi at 0.\n    old_phi0 : float, optional\n        Value of phi at previous point.\n    derphi0 : float, optional\n        Value of derphi at 0\n    c1 : float, optional\n        Parameter for Armijo condition rule.\n    c2 : float, optional\n        Parameter for curvature condition rule.\n    amax : float, optional\n        Maximum step size.\n    extra_condition : callable, optional\n        A callable of the form ``extra_condition(alpha, phi_value)``\n        returning a boolean. The line search accepts the value\n        of ``alpha`` only if this callable returns ``True``.\n        If the callable returns ``False`` for the step length,\n        the algorithm will continue with new iterates.\n        The callable is only called for iterates satisfying\n        the strong Wolfe conditions.\n    maxiter : int, optional\n        Maximum number of iterations to perform.\n\n    Returns\n    -------\n    alpha_star : float or None\n        Best alpha, or None if the line search algorithm did not converge.\n    phi_star : float\n        phi at alpha_star.\n    phi0 : float\n        phi at 0.\n    derphi_star : float or None\n        derphi at alpha_star, or None if the line search algorithm\n        did not converge.\n\n    Notes\n    -----\n    Uses the line search algorithm to enforce strong Wolfe\n    conditions. See Wright and Nocedal, 'Numerical Optimization',\n    1999, pp. 59-61.\n\n    "
    _check_c1_c2(c1, c2)
    if phi0 is None:
        phi0 = phi(0.0)
    if derphi0 is None:
        derphi0 = derphi(0.0)
    alpha0 = 0
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01 * 2 * (phi0 - old_phi0) / derphi0)
    else:
        alpha1 = 1.0
    if alpha1 < 0:
        alpha1 = 1.0
    if amax is not None:
        alpha1 = min(alpha1, amax)
    phi_a1 = phi(alpha1)
    phi_a0 = phi0
    derphi_a0 = derphi0
    if extra_condition is None:

        def extra_condition(alpha, phi):
            if False:
                return 10
            return True
    for i in range(maxiter):
        if alpha1 == 0 or (amax is not None and alpha0 == amax):
            alpha_star = None
            phi_star = phi0
            phi0 = old_phi0
            derphi_star = None
            if alpha1 == 0:
                msg = 'Rounding errors prevent the line search from converging'
            else:
                msg = 'The line search algorithm could not find a solution ' + 'less than or equal to amax: %s' % amax
            warn(msg, LineSearchWarning)
            break
        not_first_iteration = i > 0
        if phi_a1 > phi0 + c1 * alpha1 * derphi0 or (phi_a1 >= phi_a0 and not_first_iteration):
            (alpha_star, phi_star, derphi_star) = _zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi, derphi, phi0, derphi0, c1, c2, extra_condition)
            break
        derphi_a1 = derphi(alpha1)
        if abs(derphi_a1) <= -c2 * derphi0:
            if extra_condition(alpha1, phi_a1):
                alpha_star = alpha1
                phi_star = phi_a1
                derphi_star = derphi_a1
                break
        if derphi_a1 >= 0:
            (alpha_star, phi_star, derphi_star) = _zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi, derphi, phi0, derphi0, c1, c2, extra_condition)
            break
        alpha2 = 2 * alpha1
        if amax is not None:
            alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1
    else:
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None
        warn('The line search algorithm did not converge', LineSearchWarning)
    return (alpha_star, phi_star, phi0, derphi_star)

def _cubicmin(a, fa, fpa, b, fb, c, fc):
    if False:
        i = 10
        return i + 15
    '\n    Finds the minimizer for a cubic polynomial that goes through the\n    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.\n\n    If no minimizer can be found, return None.\n\n    '
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db, fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin

def _quadmin(a, fa, fpa, b, fb):
    if False:
        print('Hello World!')
    '\n    Finds the minimizer for a quadratic polynomial that goes through\n    the points (a,fa), (b,fb) with derivative at a of fpa.\n\n    '
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin

def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, derphi, phi0, derphi0, c1, c2, extra_condition):
    if False:
        return 10
    "Zoom stage of approximate linesearch satisfying strong Wolfe conditions.\n\n    Part of the optimization algorithm in `scalar_search_wolfe2`.\n\n    Notes\n    -----\n    Implements Algorithm 3.6 (zoom) in Wright and Nocedal,\n    'Numerical Optimization', 1999, pp. 61.\n\n    "
    maxiter = 10
    i = 0
    delta1 = 0.2
    delta2 = 0.1
    phi_rec = phi0
    a_rec = 0
    while True:
        dalpha = a_hi - a_lo
        if dalpha < 0:
            (a, b) = (a_hi, a_lo)
        else:
            (a, b) = (a_lo, a_hi)
        if i > 0:
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
        if i == 0 or a_j is None or a_j > b - cchk or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if a_j is None or a_j > b - qchk or a_j < a + qchk:
                a_j = a_lo + 0.5 * dalpha
        phi_aj = phi(a_j)
        if phi_aj > phi0 + c1 * a_j * derphi0 or phi_aj >= phi_lo:
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= -c2 * derphi0 and extra_condition(a_j, phi_aj):
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj * (a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if i > maxiter:
            a_star = None
            val_star = None
            valprime_star = None
            break
    return (a_star, val_star, valprime_star)

def line_search_armijo(f, xk, pk, gfk, old_fval, args=(), c1=0.0001, alpha0=1):
    if False:
        return 10
    "Minimize over alpha, the function ``f(xk+alpha pk)``.\n\n    Parameters\n    ----------\n    f : callable\n        Function to be minimized.\n    xk : array_like\n        Current point.\n    pk : array_like\n        Search direction.\n    gfk : array_like\n        Gradient of `f` at point `xk`.\n    old_fval : float\n        Value of `f` at point `xk`.\n    args : tuple, optional\n        Optional arguments.\n    c1 : float, optional\n        Value to control stopping criterion.\n    alpha0 : scalar, optional\n        Value of `alpha` at start of the optimization.\n\n    Returns\n    -------\n    alpha\n    f_count\n    f_val_at_alpha\n\n    Notes\n    -----\n    Uses the interpolation algorithm (Armijo backtracking) as suggested by\n    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57\n\n    "
    xk = np.atleast_1d(xk)
    fc = [0]

    def phi(alpha1):
        if False:
            while True:
                i = 10
        fc[0] += 1
        return f(xk + alpha1 * pk, *args)
    if old_fval is None:
        phi0 = phi(0.0)
    else:
        phi0 = old_fval
    derphi0 = np.dot(gfk, pk)
    (alpha, phi1) = scalar_search_armijo(phi, phi0, derphi0, c1=c1, alpha0=alpha0)
    return (alpha, fc[0], phi1)

def line_search_BFGS(f, xk, pk, gfk, old_fval, args=(), c1=0.0001, alpha0=1):
    if False:
        while True:
            i = 10
    '\n    Compatibility wrapper for `line_search_armijo`\n    '
    r = line_search_armijo(f, xk, pk, gfk, old_fval, args=args, c1=c1, alpha0=alpha0)
    return (r[0], r[1], 0, r[2])

def scalar_search_armijo(phi, phi0, derphi0, c1=0.0001, alpha0=1, amin=0):
    if False:
        while True:
            i = 10
    "Minimize over alpha, the function ``phi(alpha)``.\n\n    Uses the interpolation algorithm (Armijo backtracking) as suggested by\n    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57\n\n    alpha > 0 is assumed to be a descent direction.\n\n    Returns\n    -------\n    alpha\n    phi1\n\n    "
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return (alpha0, phi_a0)
    alpha1 = -derphi0 * alpha0 ** 2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)
    if phi_a1 <= phi0 + c1 * alpha1 * derphi0:
        return (alpha1, phi_a1)
    while alpha1 > amin:
        factor = alpha0 ** 2 * alpha1 ** 2 * (alpha1 - alpha0)
        a = alpha0 ** 2 * (phi_a1 - phi0 - derphi0 * alpha1) - alpha1 ** 2 * (phi_a0 - phi0 - derphi0 * alpha0)
        a = a / factor
        b = -alpha0 ** 3 * (phi_a1 - phi0 - derphi0 * alpha1) + alpha1 ** 3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor
        alpha2 = (-b + np.sqrt(abs(b ** 2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2)
        if phi_a2 <= phi0 + c1 * alpha2 * derphi0:
            return (alpha2, phi_a2)
        if alpha1 - alpha2 > alpha1 / 2.0 or 1 - alpha2 / alpha1 < 0.96:
            alpha2 = alpha1 / 2.0
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2
    return (None, phi_a1)

def _nonmonotone_line_search_cruz(f, x_k, d, prev_fs, eta, gamma=0.0001, tau_min=0.1, tau_max=0.5):
    if False:
        return 10
    '\n    Nonmonotone backtracking line search as described in [1]_\n\n    Parameters\n    ----------\n    f : callable\n        Function returning a tuple ``(f, F)`` where ``f`` is the value\n        of a merit function and ``F`` the residual.\n    x_k : ndarray\n        Initial position.\n    d : ndarray\n        Search direction.\n    prev_fs : float\n        List of previous merit function values. Should have ``len(prev_fs) <= M``\n        where ``M`` is the nonmonotonicity window parameter.\n    eta : float\n        Allowed merit function increase, see [1]_\n    gamma, tau_min, tau_max : float, optional\n        Search parameters, see [1]_\n\n    Returns\n    -------\n    alpha : float\n        Step length\n    xp : ndarray\n        Next position\n    fp : float\n        Merit function value at next position\n    Fp : ndarray\n        Residual at next position\n\n    References\n    ----------\n    [1] "Spectral residual method without gradient information for solving\n        large-scale nonlinear systems of equations." W. La Cruz,\n        J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).\n\n    '
    f_k = prev_fs[-1]
    f_bar = max(prev_fs)
    alpha_p = 1
    alpha_m = 1
    alpha = 1
    while True:
        xp = x_k + alpha_p * d
        (fp, Fp) = f(xp)
        if fp <= f_bar + eta - gamma * alpha_p ** 2 * f_k:
            alpha = alpha_p
            break
        alpha_tp = alpha_p ** 2 * f_k / (fp + (2 * alpha_p - 1) * f_k)
        xp = x_k - alpha_m * d
        (fp, Fp) = f(xp)
        if fp <= f_bar + eta - gamma * alpha_m ** 2 * f_k:
            alpha = -alpha_m
            break
        alpha_tm = alpha_m ** 2 * f_k / (fp + (2 * alpha_m - 1) * f_k)
        alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
        alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)
    return (alpha, xp, fp, Fp)

def _nonmonotone_line_search_cheng(f, x_k, d, f_k, C, Q, eta, gamma=0.0001, tau_min=0.1, tau_max=0.5, nu=0.85):
    if False:
        for i in range(10):
            print('nop')
    "\n    Nonmonotone line search from [1]\n\n    Parameters\n    ----------\n    f : callable\n        Function returning a tuple ``(f, F)`` where ``f`` is the value\n        of a merit function and ``F`` the residual.\n    x_k : ndarray\n        Initial position.\n    d : ndarray\n        Search direction.\n    f_k : float\n        Initial merit function value.\n    C, Q : float\n        Control parameters. On the first iteration, give values\n        Q=1.0, C=f_k\n    eta : float\n        Allowed merit function increase, see [1]_\n    nu, gamma, tau_min, tau_max : float, optional\n        Search parameters, see [1]_\n\n    Returns\n    -------\n    alpha : float\n        Step length\n    xp : ndarray\n        Next position\n    fp : float\n        Merit function value at next position\n    Fp : ndarray\n        Residual at next position\n    C : float\n        New value for the control parameter C\n    Q : float\n        New value for the control parameter Q\n\n    References\n    ----------\n    .. [1] W. Cheng & D.-H. Li, ''A derivative-free nonmonotone line\n           search and its application to the spectral residual\n           method'', IMA J. Numer. Anal. 29, 814 (2009).\n\n    "
    alpha_p = 1
    alpha_m = 1
    alpha = 1
    while True:
        xp = x_k + alpha_p * d
        (fp, Fp) = f(xp)
        if fp <= C + eta - gamma * alpha_p ** 2 * f_k:
            alpha = alpha_p
            break
        alpha_tp = alpha_p ** 2 * f_k / (fp + (2 * alpha_p - 1) * f_k)
        xp = x_k - alpha_m * d
        (fp, Fp) = f(xp)
        if fp <= C + eta - gamma * alpha_m ** 2 * f_k:
            alpha = -alpha_m
            break
        alpha_tm = alpha_m ** 2 * f_k / (fp + (2 * alpha_m - 1) * f_k)
        alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
        alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)
    Q_next = nu * Q + 1
    C = (nu * Q * (C + eta) + fp) / Q_next
    Q = Q_next
    return (alpha, xp, fp, Fp, C, Q)