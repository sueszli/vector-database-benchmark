import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
_iter = 100
_xtol = 2e-12
_rtol = 4 * np.finfo(float).eps
__all__ = ['newton', 'bisect', 'ridder', 'brentq', 'brenth', 'toms748', 'RootResults']
_ECONVERGED = 0
_ESIGNERR = -1
_EERRORINCREASE = -1
_ELIMITS = -1
_ECONVERR = -2
_EVALUEERR = -3
_ECALLBACK = -4
_EINPROGRESS = 1
_ESTOPONESIDE = 2
CONVERGED = 'converged'
SIGNERR = 'sign error'
CONVERR = 'convergence error'
VALUEERR = 'value error'
INPROGRESS = 'No error'
flag_map = {_ECONVERGED: CONVERGED, _ESIGNERR: SIGNERR, _ECONVERR: CONVERR, _EVALUEERR: VALUEERR, _EINPROGRESS: INPROGRESS}

class RootResults(OptimizeResult):
    """Represents the root finding result.

    Attributes
    ----------
    root : float
        Estimated root location.
    iterations : int
        Number of iterations needed to find the root.
    function_calls : int
        Number of times the function was called.
    converged : bool
        True if the routine converged.
    flag : str
        Description of the cause of termination.
    method : str
        Root finding method used.

    """

    def __init__(self, root, iterations, function_calls, flag, method):
        if False:
            return 10
        self.root = root
        self.iterations = iterations
        self.function_calls = function_calls
        self.converged = flag == _ECONVERGED
        if flag in flag_map:
            self.flag = flag_map[flag]
        else:
            self.flag = flag
        self.method = method

def results_c(full_output, r, method):
    if False:
        while True:
            i = 10
    if full_output:
        (x, funcalls, iterations, flag) = r
        results = RootResults(root=x, iterations=iterations, function_calls=funcalls, flag=flag, method=method)
        return (x, results)
    else:
        return r

def _results_select(full_output, r, method):
    if False:
        for i in range(10):
            print('nop')
    'Select from a tuple of (root, funccalls, iterations, flag)'
    (x, funcalls, iterations, flag) = r
    if full_output:
        results = RootResults(root=x, iterations=iterations, function_calls=funcalls, flag=flag, method=method)
        return (x, results)
    return x

def _wrap_nan_raise(f):
    if False:
        while True:
            i = 10

    def f_raise(x, *args):
        if False:
            i = 10
            return i + 15
        fx = f(x, *args)
        f_raise._function_calls += 1
        if np.isnan(fx):
            msg = f'The function value at x={x} is NaN; solver cannot continue.'
            err = ValueError(msg)
            err._x = x
            err._function_calls = f_raise._function_calls
            raise err
        return fx
    f_raise._function_calls = 0
    return f_raise

def newton(func, x0, fprime=None, args=(), tol=1.48e-08, maxiter=50, fprime2=None, x1=None, rtol=0.0, full_output=False, disp=True):
    if False:
        i = 10
        return i + 15
    "\n    Find a root of a real or complex function using the Newton-Raphson\n    (or secant or Halley's) method.\n\n    Find a root of the scalar-valued function `func` given a nearby scalar\n    starting point `x0`.\n    The Newton-Raphson method is used if the derivative `fprime` of `func`\n    is provided, otherwise the secant method is used. If the second order\n    derivative `fprime2` of `func` is also provided, then Halley's method is\n    used.\n\n    If `x0` is a sequence with more than one item, `newton` returns an array:\n    the roots of the function from each (scalar) starting point in `x0`.\n    In this case, `func` must be vectorized to return a sequence or array of\n    the same shape as its first argument. If `fprime` (`fprime2`) is given,\n    then its return must also have the same shape: each element is the first\n    (second) derivative of `func` with respect to its only variable evaluated\n    at each element of its first argument.\n\n    `newton` is for finding roots of a scalar-valued functions of a single\n    variable. For problems involving several variables, see `root`.\n\n    Parameters\n    ----------\n    func : callable\n        The function whose root is wanted. It must be a function of a\n        single variable of the form ``f(x,a,b,c...)``, where ``a,b,c...``\n        are extra arguments that can be passed in the `args` parameter.\n    x0 : float, sequence, or ndarray\n        An initial estimate of the root that should be somewhere near the\n        actual root. If not scalar, then `func` must be vectorized and return\n        a sequence or array of the same shape as its first argument.\n    fprime : callable, optional\n        The derivative of the function when available and convenient. If it\n        is None (default), then the secant method is used.\n    args : tuple, optional\n        Extra arguments to be used in the function call.\n    tol : float, optional\n        The allowable error of the root's value. If `func` is complex-valued,\n        a larger `tol` is recommended as both the real and imaginary parts\n        of `x` contribute to ``|x - x0|``.\n    maxiter : int, optional\n        Maximum number of iterations.\n    fprime2 : callable, optional\n        The second order derivative of the function when available and\n        convenient. If it is None (default), then the normal Newton-Raphson\n        or the secant method is used. If it is not None, then Halley's method\n        is used.\n    x1 : float, optional\n        Another estimate of the root that should be somewhere near the\n        actual root. Used if `fprime` is not provided.\n    rtol : float, optional\n        Tolerance (relative) for termination.\n    full_output : bool, optional\n        If `full_output` is False (default), the root is returned.\n        If True and `x0` is scalar, the return value is ``(x, r)``, where ``x``\n        is the root and ``r`` is a `RootResults` object.\n        If True and `x0` is non-scalar, the return value is ``(x, converged,\n        zero_der)`` (see Returns section for details).\n    disp : bool, optional\n        If True, raise a RuntimeError if the algorithm didn't converge, with\n        the error message containing the number of iterations and current\n        function value. Otherwise, the convergence status is recorded in a\n        `RootResults` return object.\n        Ignored if `x0` is not scalar.\n        *Note: this has little to do with displaying, however,\n        the `disp` keyword cannot be renamed for backwards compatibility.*\n\n    Returns\n    -------\n    root : float, sequence, or ndarray\n        Estimated location where function is zero.\n    r : `RootResults`, optional\n        Present if ``full_output=True`` and `x0` is scalar.\n        Object containing information about the convergence. In particular,\n        ``r.converged`` is True if the routine converged.\n    converged : ndarray of bool, optional\n        Present if ``full_output=True`` and `x0` is non-scalar.\n        For vector functions, indicates which elements converged successfully.\n    zero_der : ndarray of bool, optional\n        Present if ``full_output=True`` and `x0` is non-scalar.\n        For vector functions, indicates which elements had a zero derivative.\n\n    See Also\n    --------\n    root_scalar : interface to root solvers for scalar functions\n    root : interface to root solvers for multi-input, multi-output functions\n\n    Notes\n    -----\n    The convergence rate of the Newton-Raphson method is quadratic,\n    the Halley method is cubic, and the secant method is\n    sub-quadratic. This means that if the function is well-behaved\n    the actual error in the estimated root after the nth iteration\n    is approximately the square (cube for Halley) of the error\n    after the (n-1)th step. However, the stopping criterion used\n    here is the step size and there is no guarantee that a root\n    has been found. Consequently, the result should be verified.\n    Safer algorithms are brentq, brenth, ridder, and bisect,\n    but they all require that the root first be bracketed in an\n    interval where the function changes sign. The brentq algorithm\n    is recommended for general use in one dimensional problems\n    when such an interval has been found.\n\n    When `newton` is used with arrays, it is best suited for the following\n    types of problems:\n\n    * The initial guesses, `x0`, are all relatively the same distance from\n      the roots.\n    * Some or all of the extra arguments, `args`, are also arrays so that a\n      class of similar problems can be solved together.\n    * The size of the initial guesses, `x0`, is larger than O(100) elements.\n      Otherwise, a naive loop may perform as well or better than a vector.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy import optimize\n\n    >>> def f(x):\n    ...     return (x**3 - 1)  # only one real root at x = 1\n\n    ``fprime`` is not provided, use the secant method:\n\n    >>> root = optimize.newton(f, 1.5)\n    >>> root\n    1.0000000000000016\n    >>> root = optimize.newton(f, 1.5, fprime2=lambda x: 6 * x)\n    >>> root\n    1.0000000000000016\n\n    Only ``fprime`` is provided, use the Newton-Raphson method:\n\n    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2)\n    >>> root\n    1.0\n\n    Both ``fprime2`` and ``fprime`` are provided, use Halley's method:\n\n    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2,\n    ...                        fprime2=lambda x: 6 * x)\n    >>> root\n    1.0\n\n    When we want to find roots for a set of related starting values and/or\n    function parameters, we can provide both of those as an array of inputs:\n\n    >>> f = lambda x, a: x**3 - a\n    >>> fder = lambda x, a: 3 * x**2\n    >>> rng = np.random.default_rng()\n    >>> x = rng.standard_normal(100)\n    >>> a = np.arange(-50, 50)\n    >>> vec_res = optimize.newton(f, x, fprime=fder, args=(a, ), maxiter=200)\n\n    The above is the equivalent of solving for each value in ``(x, a)``\n    separately in a for-loop, just faster:\n\n    >>> loop_res = [optimize.newton(f, x0, fprime=fder, args=(a0,),\n    ...                             maxiter=200)\n    ...             for x0, a0 in zip(x, a)]\n    >>> np.allclose(vec_res, loop_res)\n    True\n\n    Plot the results found for all values of ``a``:\n\n    >>> analytical_result = np.sign(a) * np.abs(a)**(1/3)\n    >>> fig, ax = plt.subplots()\n    >>> ax.plot(a, analytical_result, 'o')\n    >>> ax.plot(a, vec_res, '.')\n    >>> ax.set_xlabel('$a$')\n    >>> ax.set_ylabel('$x$ where $f(x, a)=0$')\n    >>> plt.show()\n\n    "
    if tol <= 0:
        raise ValueError('tol too small (%g <= 0)' % tol)
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError('maxiter must be greater than 0')
    if np.size(x0) > 1:
        return _array_newton(func, x0, fprime, args, tol, maxiter, fprime2, full_output)
    x0 = np.asarray(x0)[()] * 1.0
    p0 = x0
    funcalls = 0
    if fprime is not None:
        method = 'newton'
        for itr in range(maxiter):
            fval = func(p0, *args)
            funcalls += 1
            if fval == 0:
                return _results_select(full_output, (p0, funcalls, itr, _ECONVERGED), method)
            fder = fprime(p0, *args)
            funcalls += 1
            if fder == 0:
                msg = 'Derivative was zero.'
                if disp:
                    msg += ' Failed to converge after %d iterations, value is %s.' % (itr + 1, p0)
                    raise RuntimeError(msg)
                warnings.warn(msg, RuntimeWarning)
                return _results_select(full_output, (p0, funcalls, itr + 1, _ECONVERR), method)
            newton_step = fval / fder
            if fprime2:
                fder2 = fprime2(p0, *args)
                funcalls += 1
                method = 'halley'
                adj = newton_step * fder2 / fder / 2
                if np.abs(adj) < 1:
                    newton_step /= 1.0 - adj
            p = p0 - newton_step
            if np.isclose(p, p0, rtol=rtol, atol=tol):
                return _results_select(full_output, (p, funcalls, itr + 1, _ECONVERGED), method)
            p0 = p
    else:
        method = 'secant'
        if x1 is not None:
            if x1 == x0:
                raise ValueError('x1 and x0 must be different')
            p1 = x1
        else:
            eps = 0.0001
            p1 = x0 * (1 + eps)
            p1 += eps if p1 >= 0 else -eps
        q0 = func(p0, *args)
        funcalls += 1
        q1 = func(p1, *args)
        funcalls += 1
        if abs(q1) < abs(q0):
            (p0, p1, q0, q1) = (p1, p0, q1, q0)
        for itr in range(maxiter):
            if q1 == q0:
                if p1 != p0:
                    msg = 'Tolerance of %s reached.' % (p1 - p0)
                    if disp:
                        msg += ' Failed to converge after %d iterations, value is %s.' % (itr + 1, p1)
                        raise RuntimeError(msg)
                    warnings.warn(msg, RuntimeWarning)
                p = (p1 + p0) / 2.0
                return _results_select(full_output, (p, funcalls, itr + 1, _ECONVERR), method)
            elif abs(q1) > abs(q0):
                p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
            else:
                p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
            if np.isclose(p, p1, rtol=rtol, atol=tol):
                return _results_select(full_output, (p, funcalls, itr + 1, _ECONVERGED), method)
            (p0, q0) = (p1, q1)
            p1 = p
            q1 = func(p1, *args)
            funcalls += 1
    if disp:
        msg = 'Failed to converge after %d iterations, value is %s.' % (itr + 1, p)
        raise RuntimeError(msg)
    return _results_select(full_output, (p, funcalls, itr + 1, _ECONVERR), method)

def _array_newton(func, x0, fprime, args, tol, maxiter, fprime2, full_output):
    if False:
        while True:
            i = 10
    '\n    A vectorized version of Newton, Halley, and secant methods for arrays.\n\n    Do not use this method directly. This method is called from `newton`\n    when ``np.size(x0) > 1`` is ``True``. For docstring, see `newton`.\n    '
    p = np.array(x0, copy=True)
    failures = np.ones_like(p, dtype=bool)
    nz_der = np.ones_like(failures)
    if fprime is not None:
        for iteration in range(maxiter):
            fval = np.asarray(func(p, *args))
            if not fval.any():
                failures = fval.astype(bool)
                break
            fder = np.asarray(fprime(p, *args))
            nz_der = fder != 0
            if not nz_der.any():
                break
            dp = fval[nz_der] / fder[nz_der]
            if fprime2 is not None:
                fder2 = np.asarray(fprime2(p, *args))
                dp = dp / (1.0 - 0.5 * dp * fder2[nz_der] / fder[nz_der])
            p = np.asarray(p, dtype=np.result_type(p, dp, np.float64))
            p[nz_der] -= dp
            failures[nz_der] = np.abs(dp) >= tol
            if not failures[nz_der].any():
                break
    else:
        dx = np.finfo(float).eps ** 0.33
        p1 = p * (1 + dx) + np.where(p >= 0, dx, -dx)
        q0 = np.asarray(func(p, *args))
        q1 = np.asarray(func(p1, *args))
        active = np.ones_like(p, dtype=bool)
        for iteration in range(maxiter):
            nz_der = q1 != q0
            if not nz_der.any():
                p = (p1 + p) / 2.0
                break
            dp = (q1 * (p1 - p))[nz_der] / (q1 - q0)[nz_der]
            p = np.asarray(p, dtype=np.result_type(p, p1, dp, np.float64))
            p[nz_der] = p1[nz_der] - dp
            active_zero_der = ~nz_der & active
            p[active_zero_der] = (p1 + p)[active_zero_der] / 2.0
            active &= nz_der
            failures[nz_der] = np.abs(dp) >= tol
            if not failures[nz_der].any():
                break
            (p1, p) = (p, p1)
            q0 = q1
            q1 = np.asarray(func(p1, *args))
    zero_der = ~nz_der & failures
    if zero_der.any():
        if fprime is None:
            nonzero_dp = p1 != p
            zero_der_nz_dp = zero_der & nonzero_dp
            if zero_der_nz_dp.any():
                rms = np.sqrt(sum((p1[zero_der_nz_dp] - p[zero_der_nz_dp]) ** 2))
                warnings.warn(f'RMS of {rms:g} reached', RuntimeWarning)
        else:
            all_or_some = 'all' if zero_der.all() else 'some'
            msg = f'{all_or_some:s} derivatives were zero'
            warnings.warn(msg, RuntimeWarning)
    elif failures.any():
        all_or_some = 'all' if failures.all() else 'some'
        msg = '{:s} failed to converge after {:d} iterations'.format(all_or_some, maxiter)
        if failures.all():
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning)
    if full_output:
        result = namedtuple('result', ('root', 'converged', 'zero_der'))
        p = result(p, ~failures, zero_der)
    return p

def bisect(f, a, b, args=(), xtol=_xtol, rtol=_rtol, maxiter=_iter, full_output=False, disp=True):
    if False:
        return 10
    "\n    Find root of a function within an interval using bisection.\n\n    Basic bisection routine to find a root of the function `f` between the\n    arguments `a` and `b`. `f(a)` and `f(b)` cannot have the same signs.\n    Slow but sure.\n\n    Parameters\n    ----------\n    f : function\n        Python function returning a number.  `f` must be continuous, and\n        f(a) and f(b) must have opposite signs.\n    a : scalar\n        One end of the bracketing interval [a,b].\n    b : scalar\n        The other end of the bracketing interval [a,b].\n    xtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter must be positive.\n    rtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter cannot be smaller than its default value of\n        ``4*np.finfo(float).eps``.\n    maxiter : int, optional\n        If convergence is not achieved in `maxiter` iterations, an error is\n        raised. Must be >= 0.\n    args : tuple, optional\n        Containing extra arguments for the function `f`.\n        `f` is called by ``apply(f, (x)+args)``.\n    full_output : bool, optional\n        If `full_output` is False, the root is returned. If `full_output` is\n        True, the return value is ``(x, r)``, where x is the root, and r is\n        a `RootResults` object.\n    disp : bool, optional\n        If True, raise RuntimeError if the algorithm didn't converge.\n        Otherwise, the convergence status is recorded in a `RootResults`\n        return object.\n\n    Returns\n    -------\n    root : float\n        Root of `f` between `a` and `b`.\n    r : `RootResults` (present if ``full_output = True``)\n        Object containing information about the convergence. In particular,\n        ``r.converged`` is True if the routine converged.\n\n    Examples\n    --------\n\n    >>> def f(x):\n    ...     return (x**2 - 1)\n\n    >>> from scipy import optimize\n\n    >>> root = optimize.bisect(f, 0, 2)\n    >>> root\n    1.0\n\n    >>> root = optimize.bisect(f, -2, 0)\n    >>> root\n    -1.0\n\n    See Also\n    --------\n    brentq, brenth, bisect, newton\n    fixed_point : scalar fixed-point finder\n    fsolve : n-dimensional root-finding\n\n    "
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError('xtol too small (%g <= 0)' % xtol)
    if rtol < _rtol:
        raise ValueError(f'rtol too small ({rtol:g} < {_rtol:g})')
    f = _wrap_nan_raise(f)
    r = _zeros._bisect(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, 'bisect')

def ridder(f, a, b, args=(), xtol=_xtol, rtol=_rtol, maxiter=_iter, full_output=False, disp=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find a root of a function in an interval using Ridder\'s method.\n\n    Parameters\n    ----------\n    f : function\n        Python function returning a number. f must be continuous, and f(a) and\n        f(b) must have opposite signs.\n    a : scalar\n        One end of the bracketing interval [a,b].\n    b : scalar\n        The other end of the bracketing interval [a,b].\n    xtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter must be positive.\n    rtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter cannot be smaller than its default value of\n        ``4*np.finfo(float).eps``.\n    maxiter : int, optional\n        If convergence is not achieved in `maxiter` iterations, an error is\n        raised. Must be >= 0.\n    args : tuple, optional\n        Containing extra arguments for the function `f`.\n        `f` is called by ``apply(f, (x)+args)``.\n    full_output : bool, optional\n        If `full_output` is False, the root is returned. If `full_output` is\n        True, the return value is ``(x, r)``, where `x` is the root, and `r` is\n        a `RootResults` object.\n    disp : bool, optional\n        If True, raise RuntimeError if the algorithm didn\'t converge.\n        Otherwise, the convergence status is recorded in any `RootResults`\n        return object.\n\n    Returns\n    -------\n    root : float\n        Root of `f` between `a` and `b`.\n    r : `RootResults` (present if ``full_output = True``)\n        Object containing information about the convergence.\n        In particular, ``r.converged`` is True if the routine converged.\n\n    See Also\n    --------\n    brentq, brenth, bisect, newton : 1-D root-finding\n    fixed_point : scalar fixed-point finder\n\n    Notes\n    -----\n    Uses [Ridders1979]_ method to find a root of the function `f` between the\n    arguments `a` and `b`. Ridders\' method is faster than bisection, but not\n    generally as fast as the Brent routines. [Ridders1979]_ provides the\n    classic description and source of the algorithm. A description can also be\n    found in any recent edition of Numerical Recipes.\n\n    The routine used here diverges slightly from standard presentations in\n    order to be a bit more careful of tolerance.\n\n    References\n    ----------\n    .. [Ridders1979]\n       Ridders, C. F. J. "A New Algorithm for Computing a\n       Single Root of a Real Continuous Function."\n       IEEE Trans. Circuits Systems 26, 979-980, 1979.\n\n    Examples\n    --------\n\n    >>> def f(x):\n    ...     return (x**2 - 1)\n\n    >>> from scipy import optimize\n\n    >>> root = optimize.ridder(f, 0, 2)\n    >>> root\n    1.0\n\n    >>> root = optimize.ridder(f, -2, 0)\n    >>> root\n    -1.0\n    '
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError('xtol too small (%g <= 0)' % xtol)
    if rtol < _rtol:
        raise ValueError(f'rtol too small ({rtol:g} < {_rtol:g})')
    f = _wrap_nan_raise(f)
    r = _zeros._ridder(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, 'ridder')

def brentq(f, a, b, args=(), xtol=_xtol, rtol=_rtol, maxiter=_iter, full_output=False, disp=True):
    if False:
        i = 10
        return i + 15
    '\n    Find a root of a function in a bracketing interval using Brent\'s method.\n\n    Uses the classic Brent\'s method to find a root of the function `f` on\n    the sign changing interval [a , b]. Generally considered the best of the\n    rootfinding routines here. It is a safe version of the secant method that\n    uses inverse quadratic extrapolation. Brent\'s method combines root\n    bracketing, interval bisection, and inverse quadratic interpolation. It is\n    sometimes known as the van Wijngaarden-Dekker-Brent method. Brent (1973)\n    claims convergence is guaranteed for functions computable within [a,b].\n\n    [Brent1973]_ provides the classic description of the algorithm. Another\n    description can be found in a recent edition of Numerical Recipes, including\n    [PressEtal1992]_. A third description is at\n    http://mathworld.wolfram.com/BrentsMethod.html. It should be easy to\n    understand the algorithm just by reading our code. Our code diverges a bit\n    from standard presentations: we choose a different formula for the\n    extrapolation step.\n\n    Parameters\n    ----------\n    f : function\n        Python function returning a number. The function :math:`f`\n        must be continuous, and :math:`f(a)` and :math:`f(b)` must\n        have opposite signs.\n    a : scalar\n        One end of the bracketing interval :math:`[a, b]`.\n    b : scalar\n        The other end of the bracketing interval :math:`[a, b]`.\n    xtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter must be positive. For nice functions, Brent\'s\n        method will often satisfy the above condition with ``xtol/2``\n        and ``rtol/2``. [Brent1973]_\n    rtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter cannot be smaller than its default value of\n        ``4*np.finfo(float).eps``. For nice functions, Brent\'s\n        method will often satisfy the above condition with ``xtol/2``\n        and ``rtol/2``. [Brent1973]_\n    maxiter : int, optional\n        If convergence is not achieved in `maxiter` iterations, an error is\n        raised. Must be >= 0.\n    args : tuple, optional\n        Containing extra arguments for the function `f`.\n        `f` is called by ``apply(f, (x)+args)``.\n    full_output : bool, optional\n        If `full_output` is False, the root is returned. If `full_output` is\n        True, the return value is ``(x, r)``, where `x` is the root, and `r` is\n        a `RootResults` object.\n    disp : bool, optional\n        If True, raise RuntimeError if the algorithm didn\'t converge.\n        Otherwise, the convergence status is recorded in any `RootResults`\n        return object.\n\n    Returns\n    -------\n    root : float\n        Root of `f` between `a` and `b`.\n    r : `RootResults` (present if ``full_output = True``)\n        Object containing information about the convergence. In particular,\n        ``r.converged`` is True if the routine converged.\n\n    Notes\n    -----\n    `f` must be continuous.  f(a) and f(b) must have opposite signs.\n\n    Related functions fall into several classes:\n\n    multivariate local optimizers\n      `fmin`, `fmin_powell`, `fmin_cg`, `fmin_bfgs`, `fmin_ncg`\n    nonlinear least squares minimizer\n      `leastsq`\n    constrained multivariate optimizers\n      `fmin_l_bfgs_b`, `fmin_tnc`, `fmin_cobyla`\n    global optimizers\n      `basinhopping`, `brute`, `differential_evolution`\n    local scalar minimizers\n      `fminbound`, `brent`, `golden`, `bracket`\n    N-D root-finding\n      `fsolve`\n    1-D root-finding\n      `brenth`, `ridder`, `bisect`, `newton`\n    scalar fixed-point finder\n      `fixed_point`\n\n    References\n    ----------\n    .. [Brent1973]\n       Brent, R. P.,\n       *Algorithms for Minimization Without Derivatives*.\n       Englewood Cliffs, NJ: Prentice-Hall, 1973. Ch. 3-4.\n\n    .. [PressEtal1992]\n       Press, W. H.; Flannery, B. P.; Teukolsky, S. A.; and Vetterling, W. T.\n       *Numerical Recipes in FORTRAN: The Art of Scientific Computing*, 2nd ed.\n       Cambridge, England: Cambridge University Press, pp. 352-355, 1992.\n       Section 9.3:  "Van Wijngaarden-Dekker-Brent Method."\n\n    Examples\n    --------\n    >>> def f(x):\n    ...     return (x**2 - 1)\n\n    >>> from scipy import optimize\n\n    >>> root = optimize.brentq(f, -2, 0)\n    >>> root\n    -1.0\n\n    >>> root = optimize.brentq(f, 0, 2)\n    >>> root\n    1.0\n    '
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError('xtol too small (%g <= 0)' % xtol)
    if rtol < _rtol:
        raise ValueError(f'rtol too small ({rtol:g} < {_rtol:g})')
    f = _wrap_nan_raise(f)
    r = _zeros._brentq(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, 'brentq')

def brenth(f, a, b, args=(), xtol=_xtol, rtol=_rtol, maxiter=_iter, full_output=False, disp=True):
    if False:
        i = 10
        return i + 15
    'Find a root of a function in a bracketing interval using Brent\'s\n    method with hyperbolic extrapolation.\n\n    A variation on the classic Brent routine to find a root of the function f\n    between the arguments a and b that uses hyperbolic extrapolation instead of\n    inverse quadratic extrapolation. Bus & Dekker (1975) guarantee convergence\n    for this method, claiming that the upper bound of function evaluations here\n    is 4 or 5 times that of bisection.\n    f(a) and f(b) cannot have the same signs. Generally, on a par with the\n    brent routine, but not as heavily tested. It is a safe version of the\n    secant method that uses hyperbolic extrapolation.\n    The version here is by Chuck Harris, and implements Algorithm M of\n    [BusAndDekker1975]_, where further details (convergence properties,\n    additional remarks and such) can be found\n\n    Parameters\n    ----------\n    f : function\n        Python function returning a number. f must be continuous, and f(a) and\n        f(b) must have opposite signs.\n    a : scalar\n        One end of the bracketing interval [a,b].\n    b : scalar\n        The other end of the bracketing interval [a,b].\n    xtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter must be positive. As with `brentq`, for nice\n        functions the method will often satisfy the above condition\n        with ``xtol/2`` and ``rtol/2``.\n    rtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter cannot be smaller than its default value of\n        ``4*np.finfo(float).eps``. As with `brentq`, for nice functions\n        the method will often satisfy the above condition with\n        ``xtol/2`` and ``rtol/2``.\n    maxiter : int, optional\n        If convergence is not achieved in `maxiter` iterations, an error is\n        raised. Must be >= 0.\n    args : tuple, optional\n        Containing extra arguments for the function `f`.\n        `f` is called by ``apply(f, (x)+args)``.\n    full_output : bool, optional\n        If `full_output` is False, the root is returned. If `full_output` is\n        True, the return value is ``(x, r)``, where `x` is the root, and `r` is\n        a `RootResults` object.\n    disp : bool, optional\n        If True, raise RuntimeError if the algorithm didn\'t converge.\n        Otherwise, the convergence status is recorded in any `RootResults`\n        return object.\n\n    Returns\n    -------\n    root : float\n        Root of `f` between `a` and `b`.\n    r : `RootResults` (present if ``full_output = True``)\n        Object containing information about the convergence. In particular,\n        ``r.converged`` is True if the routine converged.\n\n    See Also\n    --------\n    fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg : multivariate local optimizers\n    leastsq : nonlinear least squares minimizer\n    fmin_l_bfgs_b, fmin_tnc, fmin_cobyla : constrained multivariate optimizers\n    basinhopping, differential_evolution, brute : global optimizers\n    fminbound, brent, golden, bracket : local scalar minimizers\n    fsolve : N-D root-finding\n    brentq, brenth, ridder, bisect, newton : 1-D root-finding\n    fixed_point : scalar fixed-point finder\n\n    References\n    ----------\n    .. [BusAndDekker1975]\n       Bus, J. C. P., Dekker, T. J.,\n       "Two Efficient Algorithms with Guaranteed Convergence for Finding a Zero\n       of a Function", ACM Transactions on Mathematical Software, Vol. 1, Issue\n       4, Dec. 1975, pp. 330-345. Section 3: "Algorithm M".\n       :doi:`10.1145/355656.355659`\n\n    Examples\n    --------\n    >>> def f(x):\n    ...     return (x**2 - 1)\n\n    >>> from scipy import optimize\n\n    >>> root = optimize.brenth(f, -2, 0)\n    >>> root\n    -1.0\n\n    >>> root = optimize.brenth(f, 0, 2)\n    >>> root\n    1.0\n\n    '
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError('xtol too small (%g <= 0)' % xtol)
    if rtol < _rtol:
        raise ValueError(f'rtol too small ({rtol:g} < {_rtol:g})')
    f = _wrap_nan_raise(f)
    r = _zeros._brenth(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, 'brenth')

def _notclose(fs, rtol=_rtol, atol=_xtol):
    if False:
        return 10
    notclosefvals = all(fs) and all(np.isfinite(fs)) and (not any((any(np.isclose(_f, fs[i + 1:], rtol=rtol, atol=atol)) for (i, _f) in enumerate(fs[:-1]))))
    return notclosefvals

def _secant(xvals, fvals):
    if False:
        return 10
    'Perform a secant step, taking a little care'
    (x0, x1) = xvals[:2]
    (f0, f1) = fvals[:2]
    if f0 == f1:
        return np.nan
    if np.abs(f1) > np.abs(f0):
        x2 = (-f0 / f1 * x1 + x0) / (1 - f0 / f1)
    else:
        x2 = (-f1 / f0 * x0 + x1) / (1 - f1 / f0)
    return x2

def _update_bracket(ab, fab, c, fc):
    if False:
        print('Hello World!')
    'Update a bracket given (c, fc), return the discarded endpoints.'
    (fa, fb) = fab
    idx = 0 if np.sign(fa) * np.sign(fc) > 0 else 1
    (rx, rfx) = (ab[idx], fab[idx])
    fab[idx] = fc
    ab[idx] = c
    return (rx, rfx)

def _compute_divided_differences(xvals, fvals, N=None, full=True, forward=True):
    if False:
        while True:
            i = 10
    'Return a matrix of divided differences for the xvals, fvals pairs\n\n    DD[i, j] = f[x_{i-j}, ..., x_i] for 0 <= j <= i\n\n    If full is False, just return the main diagonal(or last row):\n      f[a], f[a, b] and f[a, b, c].\n    If forward is False, return f[c], f[b, c], f[a, b, c].'
    if full:
        if forward:
            xvals = np.asarray(xvals)
        else:
            xvals = np.array(xvals)[::-1]
        M = len(xvals)
        N = M if N is None else min(N, M)
        DD = np.zeros([M, N])
        DD[:, 0] = fvals[:]
        for i in range(1, N):
            DD[i:, i] = np.diff(DD[i - 1:, i - 1]) / (xvals[i:] - xvals[:M - i])
        return DD
    xvals = np.asarray(xvals)
    dd = np.array(fvals)
    row = np.array(fvals)
    idx2Use = 0 if forward else -1
    dd[0] = fvals[idx2Use]
    for i in range(1, len(xvals)):
        denom = xvals[i:i + len(row) - 1] - xvals[:len(row) - 1]
        row = np.diff(row)[:] / denom
        dd[i] = row[idx2Use]
    return dd

def _interpolated_poly(xvals, fvals, x):
    if False:
        while True:
            i = 10
    "Compute p(x) for the polynomial passing through the specified locations.\n\n    Use Neville's algorithm to compute p(x) where p is the minimal degree\n    polynomial passing through the points xvals, fvals"
    xvals = np.asarray(xvals)
    N = len(xvals)
    Q = np.zeros([N, N])
    D = np.zeros([N, N])
    Q[:, 0] = fvals[:]
    D[:, 0] = fvals[:]
    for k in range(1, N):
        alpha = D[k:, k - 1] - Q[k - 1:N - 1, k - 1]
        diffik = xvals[0:N - k] - xvals[k:N]
        Q[k:, k] = (xvals[k:] - x) / diffik * alpha
        D[k:, k] = (xvals[:N - k] - x) / diffik * alpha
    return np.sum(Q[-1, 1:]) + Q[-1, 0]

def _inverse_poly_zero(a, b, c, d, fa, fb, fc, fd):
    if False:
        print('Hello World!')
    'Inverse cubic interpolation f-values -> x-values\n\n    Given four points (fa, a), (fb, b), (fc, c), (fd, d) with\n    fa, fb, fc, fd all distinct, find poly IP(y) through the 4 points\n    and compute x=IP(0).\n    '
    return _interpolated_poly([fa, fb, fc, fd], [a, b, c, d], 0)

def _newton_quadratic(ab, fab, d, fd, k):
    if False:
        print('Hello World!')
    "Apply Newton-Raphson like steps, using divided differences to approximate f'\n\n    ab is a real interval [a, b] containing a root,\n    fab holds the real values of f(a), f(b)\n    d is a real number outside [ab, b]\n    k is the number of steps to apply\n    "
    (a, b) = ab
    (fa, fb) = fab
    (_, B, A) = _compute_divided_differences([a, b, d], [fa, fb, fd], forward=True, full=False)

    def _P(x):
        if False:
            while True:
                i = 10
        return (A * (x - b) + B) * (x - a) + fa
    if A == 0:
        r = a - fa / B
    else:
        r = a if np.sign(A) * np.sign(fa) > 0 else b
        for i in range(k):
            r1 = r - _P(r) / (B + A * (2 * r - a - b))
            if not ab[0] < r1 < ab[1]:
                if ab[0] < r < ab[1]:
                    return r
                r = sum(ab) / 2.0
                break
            r = r1
    return r

class TOMS748Solver:
    """Solve f(x, *args) == 0 using Algorithm748 of Alefeld, Potro & Shi.
    """
    _MU = 0.5
    _K_MIN = 1
    _K_MAX = 100

    def __init__(self):
        if False:
            return 10
        self.f = None
        self.args = None
        self.function_calls = 0
        self.iterations = 0
        self.k = 2
        self.ab = [np.nan, np.nan]
        self.fab = [np.nan, np.nan]
        self.d = None
        self.fd = None
        self.e = None
        self.fe = None
        self.disp = False
        self.xtol = _xtol
        self.rtol = _rtol
        self.maxiter = _iter

    def configure(self, xtol, rtol, maxiter, disp, k):
        if False:
            i = 10
            return i + 15
        self.disp = disp
        self.xtol = xtol
        self.rtol = rtol
        self.maxiter = maxiter
        self.k = max(k, self._K_MIN)
        if self.k > self._K_MAX:
            msg = 'toms748: Overriding k: ->%d' % self._K_MAX
            warnings.warn(msg, RuntimeWarning)
            self.k = self._K_MAX

    def _callf(self, x, error=True):
        if False:
            while True:
                i = 10
        'Call the user-supplied function, update book-keeping'
        fx = self.f(x, *self.args)
        self.function_calls += 1
        if not np.isfinite(fx) and error:
            raise ValueError(f'Invalid function value: f({x:f}) -> {fx} ')
        return fx

    def get_result(self, x, flag=_ECONVERGED):
        if False:
            while True:
                i = 10
        'Package the result and statistics into a tuple.'
        return (x, self.function_calls, self.iterations, flag)

    def _update_bracket(self, c, fc):
        if False:
            while True:
                i = 10
        return _update_bracket(self.ab, self.fab, c, fc)

    def start(self, f, a, b, args=()):
        if False:
            while True:
                i = 10
        'Prepare for the iterations.'
        self.function_calls = 0
        self.iterations = 0
        self.f = f
        self.args = args
        self.ab[:] = [a, b]
        if not np.isfinite(a) or np.imag(a) != 0:
            raise ValueError('Invalid x value: %s ' % a)
        if not np.isfinite(b) or np.imag(b) != 0:
            raise ValueError('Invalid x value: %s ' % b)
        fa = self._callf(a)
        if not np.isfinite(fa) or np.imag(fa) != 0:
            raise ValueError(f'Invalid function value: f({a:f}) -> {fa} ')
        if fa == 0:
            return (_ECONVERGED, a)
        fb = self._callf(b)
        if not np.isfinite(fb) or np.imag(fb) != 0:
            raise ValueError(f'Invalid function value: f({b:f}) -> {fb} ')
        if fb == 0:
            return (_ECONVERGED, b)
        if np.sign(fb) * np.sign(fa) > 0:
            raise ValueError('f(a) and f(b) must have different signs, but f({:e})={:e}, f({:e})={:e} '.format(a, fa, b, fb))
        self.fab[:] = [fa, fb]
        return (_EINPROGRESS, sum(self.ab) / 2.0)

    def get_status(self):
        if False:
            print('Hello World!')
        'Determine the current status.'
        (a, b) = self.ab[:2]
        if np.isclose(a, b, rtol=self.rtol, atol=self.xtol):
            return (_ECONVERGED, sum(self.ab) / 2.0)
        if self.iterations >= self.maxiter:
            return (_ECONVERR, sum(self.ab) / 2.0)
        return (_EINPROGRESS, sum(self.ab) / 2.0)

    def iterate(self):
        if False:
            while True:
                i = 10
        'Perform one step in the algorithm.\n\n        Implements Algorithm 4.1(k=1) or 4.2(k=2) in [APS1995]\n        '
        self.iterations += 1
        eps = np.finfo(float).eps
        (d, fd, e, fe) = (self.d, self.fd, self.e, self.fe)
        ab_width = self.ab[1] - self.ab[0]
        c = None
        for nsteps in range(2, self.k + 2):
            if _notclose(self.fab + [fd, fe], rtol=0, atol=32 * eps):
                c0 = _inverse_poly_zero(self.ab[0], self.ab[1], d, e, self.fab[0], self.fab[1], fd, fe)
                if self.ab[0] < c0 < self.ab[1]:
                    c = c0
            if c is None:
                c = _newton_quadratic(self.ab, self.fab, d, fd, nsteps)
            fc = self._callf(c)
            if fc == 0:
                return (_ECONVERGED, c)
            (e, fe) = (d, fd)
            (d, fd) = self._update_bracket(c, fc)
        uix = 0 if np.abs(self.fab[0]) < np.abs(self.fab[1]) else 1
        (u, fu) = (self.ab[uix], self.fab[uix])
        (_, A) = _compute_divided_differences(self.ab, self.fab, forward=uix == 0, full=False)
        c = u - 2 * fu / A
        if np.abs(c - u) > 0.5 * (self.ab[1] - self.ab[0]):
            c = sum(self.ab) / 2.0
        elif np.isclose(c, u, rtol=eps, atol=0):
            frs = np.frexp(self.fab)[1]
            if frs[uix] < frs[1 - uix] - 50:
                c = (31 * self.ab[uix] + self.ab[1 - uix]) / 32
            else:
                mm = 1 if uix == 0 else -1
                adj = mm * np.abs(c) * self.rtol + mm * self.xtol
                c = u + adj
            if not self.ab[0] < c < self.ab[1]:
                c = sum(self.ab) / 2.0
        fc = self._callf(c)
        if fc == 0:
            return (_ECONVERGED, c)
        (e, fe) = (d, fd)
        (d, fd) = self._update_bracket(c, fc)
        if self.ab[1] - self.ab[0] > self._MU * ab_width:
            (e, fe) = (d, fd)
            z = sum(self.ab) / 2.0
            fz = self._callf(z)
            if fz == 0:
                return (_ECONVERGED, z)
            (d, fd) = self._update_bracket(z, fz)
        (self.d, self.fd) = (d, fd)
        (self.e, self.fe) = (e, fe)
        (status, xn) = self.get_status()
        return (status, xn)

    def solve(self, f, a, b, args=(), xtol=_xtol, rtol=_rtol, k=2, maxiter=_iter, disp=True):
        if False:
            i = 10
            return i + 15
        'Solve f(x) = 0 given an interval containing a root.'
        self.configure(xtol=xtol, rtol=rtol, maxiter=maxiter, disp=disp, k=k)
        (status, xn) = self.start(f, a, b, args)
        if status == _ECONVERGED:
            return self.get_result(xn)
        c = _secant(self.ab, self.fab)
        if not self.ab[0] < c < self.ab[1]:
            c = sum(self.ab) / 2.0
        fc = self._callf(c)
        if fc == 0:
            return self.get_result(c)
        (self.d, self.fd) = self._update_bracket(c, fc)
        (self.e, self.fe) = (None, None)
        self.iterations += 1
        while True:
            (status, xn) = self.iterate()
            if status == _ECONVERGED:
                return self.get_result(xn)
            if status == _ECONVERR:
                fmt = 'Failed to converge after %d iterations, bracket is %s'
                if disp:
                    msg = fmt % (self.iterations + 1, self.ab)
                    raise RuntimeError(msg)
                return self.get_result(xn, _ECONVERR)

def toms748(f, a, b, args=(), k=1, xtol=_xtol, rtol=_rtol, maxiter=_iter, full_output=False, disp=True):
    if False:
        i = 10
        return i + 15
    '\n    Find a root using TOMS Algorithm 748 method.\n\n    Implements the Algorithm 748 method of Alefeld, Potro and Shi to find a\n    root of the function `f` on the interval `[a , b]`, where `f(a)` and\n    `f(b)` must have opposite signs.\n\n    It uses a mixture of inverse cubic interpolation and\n    "Newton-quadratic" steps. [APS1995].\n\n    Parameters\n    ----------\n    f : function\n        Python function returning a scalar. The function :math:`f`\n        must be continuous, and :math:`f(a)` and :math:`f(b)`\n        have opposite signs.\n    a : scalar,\n        lower boundary of the search interval\n    b : scalar,\n        upper boundary of the search interval\n    args : tuple, optional\n        containing extra arguments for the function `f`.\n        `f` is called by ``f(x, *args)``.\n    k : int, optional\n        The number of Newton quadratic steps to perform each\n        iteration. ``k>=1``.\n    xtol : scalar, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter must be positive.\n    rtol : scalar, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root.\n    maxiter : int, optional\n        If convergence is not achieved in `maxiter` iterations, an error is\n        raised. Must be >= 0.\n    full_output : bool, optional\n        If `full_output` is False, the root is returned. If `full_output` is\n        True, the return value is ``(x, r)``, where `x` is the root, and `r` is\n        a `RootResults` object.\n    disp : bool, optional\n        If True, raise RuntimeError if the algorithm didn\'t converge.\n        Otherwise, the convergence status is recorded in the `RootResults`\n        return object.\n\n    Returns\n    -------\n    root : float\n        Approximate root of `f`\n    r : `RootResults` (present if ``full_output = True``)\n        Object containing information about the convergence. In particular,\n        ``r.converged`` is True if the routine converged.\n\n    See Also\n    --------\n    brentq, brenth, ridder, bisect, newton\n    fsolve : find roots in N dimensions.\n\n    Notes\n    -----\n    `f` must be continuous.\n    Algorithm 748 with ``k=2`` is asymptotically the most efficient\n    algorithm known for finding roots of a four times continuously\n    differentiable function.\n    In contrast with Brent\'s algorithm, which may only decrease the length of\n    the enclosing bracket on the last step, Algorithm 748 decreases it each\n    iteration with the same asymptotic efficiency as it finds the root.\n\n    For easy statement of efficiency indices, assume that `f` has 4\n    continuouous deriviatives.\n    For ``k=1``, the convergence order is at least 2.7, and with about\n    asymptotically 2 function evaluations per iteration, the efficiency\n    index is approximately 1.65.\n    For ``k=2``, the order is about 4.6 with asymptotically 3 function\n    evaluations per iteration, and the efficiency index 1.66.\n    For higher values of `k`, the efficiency index approaches\n    the kth root of ``(3k-2)``, hence ``k=1`` or ``k=2`` are\n    usually appropriate.\n\n    References\n    ----------\n    .. [APS1995]\n       Alefeld, G. E. and Potra, F. A. and Shi, Yixun,\n       *Algorithm 748: Enclosing Zeros of Continuous Functions*,\n       ACM Trans. Math. Softw. Volume 221(1995)\n       doi = {10.1145/210089.210111}\n\n    Examples\n    --------\n    >>> def f(x):\n    ...     return (x**3 - 1)  # only one real root at x = 1\n\n    >>> from scipy import optimize\n    >>> root, results = optimize.toms748(f, 0, 2, full_output=True)\n    >>> root\n    1.0\n    >>> results\n          converged: True\n               flag: converged\n     function_calls: 11\n         iterations: 5\n               root: 1.0\n             method: toms748\n    '
    if xtol <= 0:
        raise ValueError('xtol too small (%g <= 0)' % xtol)
    if rtol < _rtol / 4:
        raise ValueError(f'rtol too small ({rtol:g} < {_rtol / 4:g})')
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError('maxiter must be greater than 0')
    if not np.isfinite(a):
        raise ValueError('a is not finite %s' % a)
    if not np.isfinite(b):
        raise ValueError('b is not finite %s' % b)
    if a >= b:
        raise ValueError(f'a and b are not an interval [{a}, {b}]')
    if not k >= 1:
        raise ValueError('k too small (%s < 1)' % k)
    if not isinstance(args, tuple):
        args = (args,)
    f = _wrap_nan_raise(f)
    solver = TOMS748Solver()
    result = solver.solve(f, a, b, args=args, k=k, xtol=xtol, rtol=rtol, maxiter=maxiter, disp=disp)
    (x, function_calls, iterations, flag) = result
    return _results_select(full_output, (x, function_calls, iterations, flag), 'toms748')

def _bracket_root_iv(func, a, b, min, max, factor, args, maxiter):
    if False:
        i = 10
        return i + 15
    if not callable(func):
        raise ValueError('`func` must be callable.')
    if not np.iterable(args):
        args = (args,)
    a = np.asarray(a)[()]
    if not np.issubdtype(a.dtype, np.number) or np.iscomplex(a).any():
        raise ValueError('`a` must be numeric and real.')
    b = a + 1 if b is None else b
    min = -np.inf if min is None else min
    max = np.inf if max is None else max
    factor = 2.0 if factor is None else factor
    (a, b, min, max, factor) = np.broadcast_arrays(a, b, min, max, factor)
    if not np.issubdtype(b.dtype, np.number) or np.iscomplex(b).any():
        raise ValueError('`b` must be numeric and real.')
    if not np.issubdtype(min.dtype, np.number) or np.iscomplex(min).any():
        raise ValueError('`min` must be numeric and real.')
    if not np.issubdtype(max.dtype, np.number) or np.iscomplex(max).any():
        raise ValueError('`max` must be numeric and real.')
    if not np.issubdtype(factor.dtype, np.number) or np.iscomplex(factor).any():
        raise ValueError('`factor` must be numeric and real.')
    if not np.all(factor > 1):
        raise ValueError('All elements of `factor` must be greater than 1.')
    maxiter = np.asarray(maxiter)
    message = '`maxiter` must be a non-negative integer.'
    if not np.issubdtype(maxiter.dtype, np.number) or maxiter.shape != tuple() or np.iscomplex(maxiter):
        raise ValueError(message)
    maxiter_int = int(maxiter[()])
    if not maxiter == maxiter_int or maxiter < 0:
        raise ValueError(message)
    if not np.all((min <= a) & (a < b) & (b <= max)):
        raise ValueError('`min <= a < b <= max` must be True (elementwise).')
    return (func, a, b, min, max, factor, args, maxiter)

def _bracket_root(func, a, b=None, *, min=None, max=None, factor=None, args=(), maxiter=1000):
    if False:
        print('Hello World!')
    'Bracket the root of a monotonic scalar function of one variable\n\n    This function works elementwise when `a`, `b`, `min`, `max`, `factor`, and\n    the elements of `args` are broadcastable arrays.\n\n    Parameters\n    ----------\n    func : callable\n        The function for which the root is to be bracketed.\n        The signature must be::\n\n            func(x: ndarray, *args) -> ndarray\n\n        where each element of ``x`` is a finite real and ``args`` is a tuple,\n        which may contain an arbitrary number of arrays that are broadcastable\n        with `x`. ``func`` must be an elementwise function: each element\n        ``func(x)[i]`` must equal ``func(x[i])`` for all indices ``i``.\n    a, b : float array_like\n        Starting guess of bracket, which need not contain a root. If `b` is\n        not provided, ``b = a + 1``. Must be broadcastable with one another.\n    min, max : float array_like, optional\n        Minimum and maximum allowable endpoints of the bracket, inclusive. Must\n        be broadcastable with `a` and `b`.\n    factor : float array_like, default: 2\n        The factor used to grow the bracket. See notes for details.\n    args : tuple, optional\n        Additional positional arguments to be passed to `func`.  Must be arrays\n        broadcastable with `a`, `b`, `min`, and `max`. If the callable to be\n        bracketed requires arguments that are not broadcastable with these\n        arrays, wrap that callable with `func` such that `func` accepts\n        only `x` and broadcastable arrays.\n    maxiter : int, optional\n        The maximum number of iterations of the algorithm to perform.\n\n    Returns\n    -------\n    res : OptimizeResult\n        An instance of `scipy.optimize.OptimizeResult` with the following\n        attributes. The descriptions are written as though the values will be\n        scalars; however, if `func` returns an array, the outputs will be\n        arrays of the same shape.\n\n        xl, xr : float\n            The lower and upper ends of the bracket, if the algorithm\n            terminated successfully.\n        fl, fr : float\n            The function value at the lower and upper ends of the bracket.\n        nfev : int\n            The number of function evaluations required to find the bracket.\n            This is distinct from the number of times `func` is *called*\n            because the function may evaluated at multiple points in a single\n            call.\n        nit : int\n            The number of iterations of the algorithm that were performed.\n        status : int\n            An integer representing the exit status of the algorithm.\n\n            - ``0`` : The algorithm produced a valid bracket.\n            - ``-1`` : The bracket expanded to the allowable limits without finding a bracket.\n            - ``-2`` : The maximum number of iterations was reached.\n            - ``-3`` : A non-finite value was encountered.\n            - ``-4`` : Iteration was terminated by `callback`.\n            - ``1`` : The algorithm is proceeding normally (in `callback` only).\n            - ``2`` : A bracket was found in the opposite search direction (in `callback` only).\n\n        success : bool\n            ``True`` when the algorithm terminated successfully (status ``0``).\n\n    Notes\n    -----\n    This function generalizes an algorithm found in pieces throughout\n    `scipy.stats`. The strategy is to iteratively grow the bracket `(l, r)`\n     until ``func(l) < 0 < func(r)``. The bracket grows to the left as follows.\n\n    - If `min` is not provided, the distance between `b` and `l` is iteratively\n      increased by `factor`.\n    - If `min` is provided, the distance between `min` and `l` is iteratively\n      decreased by `factor`. Note that this also *increases* the bracket size.\n\n    Growth of the bracket to the right is analogous.\n\n    Growth of the bracket in one direction stops when the endpoint is no longer\n    finite, the function value at the endpoint is no longer finite, or the\n    endpoint reaches its limiting value (`min` or `max`). Iteration terminates\n    when the bracket stops growing in both directions, the bracket surrounds\n    the root, or a root is found (accidentally).\n\n    If two brackets are found - that is, a bracket is found on both sides in\n    the same iteration, the smaller of the two is returned.\n    If roots of the function are found, both `l` and `r` are set to the\n    leftmost root.\n\n    '
    callback = None
    temp = _bracket_root_iv(func, a, b, min, max, factor, args, maxiter)
    (func, a, b, min, max, factor, args, maxiter) = temp
    xs = (a, b)
    temp = _scalar_optimization_initialize(func, xs, args)
    (xs, fs, args, shape, dtype) = temp
    x = np.concatenate(xs)
    f = np.concatenate(fs)
    n = len(x) // 2
    x_last = np.concatenate((x[n:], x[:n]))
    f_last = np.concatenate((f[n:], f[:n]))
    x0 = x_last
    min = np.broadcast_to(min, shape).astype(dtype, copy=False).ravel()
    max = np.broadcast_to(max, shape).astype(dtype, copy=False).ravel()
    limit = np.concatenate((min, max))
    factor = np.broadcast_to(factor, shape).astype(dtype, copy=False).ravel()
    factor = np.concatenate((factor, factor))
    active = np.arange(2 * n)
    args = [np.concatenate((arg, arg)) for arg in args]
    shape = shape + (2,)
    i = np.isinf(limit)
    ni = ~i
    d = np.zeros_like(x)
    d[i] = x[i] - x0[i]
    d[ni] = limit[ni] - x[ni]
    status = np.full_like(x, _EINPROGRESS, dtype=int)
    (nit, nfev) = (0, 1)
    work = OptimizeResult(x=x, x0=x0, f=f, limit=limit, factor=factor, active=active, d=d, x_last=x_last, f_last=f_last, nit=nit, nfev=nfev, status=status, args=args, xl=None, xr=None, fl=None, fr=None, n=n)
    res_work_pairs = [('status', 'status'), ('xl', 'xl'), ('xr', 'xr'), ('nit', 'nit'), ('nfev', 'nfev'), ('fl', 'fl'), ('fr', 'fr'), ('x', 'x'), ('f', 'f'), ('x_last', 'x_last'), ('f_last', 'f_last')]

    def pre_func_eval(work):
        if False:
            i = 10
            return i + 15
        x = np.zeros_like(work.x)
        i = np.isinf(work.limit)
        work.d[i] *= work.factor[i]
        x[i] = work.x0[i] + work.d[i]
        ni = ~i
        work.d[ni] /= work.factor[ni]
        x[ni] = work.limit[ni] - work.d[ni]
        return x

    def post_func_eval(x, f, work):
        if False:
            while True:
                i = 10
        work.x_last = work.x
        work.f_last = work.f
        work.x = x
        work.f = f

    def check_termination(work):
        if False:
            while True:
                i = 10
        stop = np.zeros_like(work.x, dtype=bool)
        sf = np.sign(work.f)
        sf_last = np.sign(work.f_last)
        i = (sf_last == -sf) | (sf_last == 0) | (sf == 0)
        work.status[i] = _ECONVERGED
        stop[i] = True
        also_stop = (work.active[i] + work.n) % (2 * work.n)
        j = np.searchsorted(work.active, also_stop)
        j = j[j < len(work.active)]
        j = j[also_stop == work.active[j]]
        i = np.zeros_like(stop)
        i[j] = True
        i = i & ~stop
        work.status[i] = _ESTOPONESIDE
        stop[i] = True
        i = (work.x == work.limit) & ~stop
        work.status[i] = _ELIMITS
        stop[i] = True
        i = ~(np.isfinite(work.x) & np.isfinite(work.f)) & ~stop
        work.status[i] = _EVALUEERR
        stop[i] = True
        return stop

    def post_termination_check(work):
        if False:
            for i in range(10):
                print('nop')
        pass

    def customize_result(res, shape):
        if False:
            return 10
        n = len(res['x']) // 2
        xal = res['x'][:n]
        xar = res['x_last'][:n]
        xbl = res['x_last'][n:]
        xbr = res['x'][n:]
        fal = res['f'][:n]
        far = res['f_last'][:n]
        fbl = res['f_last'][n:]
        fbr = res['f'][n:]
        xl = xal.copy()
        fl = fal.copy()
        xr = xbr.copy()
        fr = fbr.copy()
        sa = res['status'][:n]
        sb = res['status'][n:]
        da = xar - xal
        db = xbr - xbl
        i1 = (da <= db) & (sa == 0) | (sa == 0) & (sb != 0)
        i2 = (db <= da) & (sb == 0) | (sb == 0) & (sa != 0)
        xr[i1] = xar[i1]
        fr[i1] = far[i1]
        xl[i2] = xbl[i2]
        fl[i2] = fbl[i2]
        res['xl'] = xl
        res['xr'] = xr
        res['fl'] = fl
        res['fr'] = fr
        res['nit'] = np.maximum(res['nit'][:n], res['nit'][n:])
        res['nfev'] = res['nfev'][:n] + res['nfev'][n:]
        res['status'] = np.choose(sa == 0, (sb, sa))
        res['success'] = res['status'] == 0
        del res['x']
        del res['f']
        del res['x_last']
        del res['f_last']
        return shape[:-1]
    return _scalar_optimization_loop(work, callback, shape, maxiter, func, args, dtype, pre_func_eval, post_func_eval, check_termination, post_termination_check, customize_result, res_work_pairs)

def _chandrupatla(func, a, b, *, args=(), xatol=_xtol, xrtol=_rtol, fatol=None, frtol=0, maxiter=_iter, callback=None):
    if False:
        return 10
    'Find the root of an elementwise function using Chandrupatla\'s algorithm.\n\n    For each element of the output of `func`, `chandrupatla` seeks the scalar\n    root that makes the element 0. This function allows for `a`, `b`, and the\n    output of `func` to be of any broadcastable shapes.\n\n    Parameters\n    ----------\n    func : callable\n        The function whose root is desired. The signature must be::\n\n            func(x: ndarray, *args) -> ndarray\n\n         where each element of ``x`` is a finite real and ``args`` is a tuple,\n         which may contain an arbitrary number of components of any type(s).\n         ``func`` must be an elementwise function: each element ``func(x)[i]``\n         must equal ``func(x[i])`` for all indices ``i``. `_chandrupatla`\n         seeks an array ``x`` such that ``func(x)`` is an array of zeros.\n    a, b : array_like\n        The lower and upper bounds of the root of the function. Must be\n        broadcastable with one another.\n    args : tuple, optional\n        Additional positional arguments to be passed to `func`.\n    xatol, xrtol, fatol, frtol : float, optional\n        Absolute and relative tolerances on the root and function value.\n        See Notes for details.\n    maxiter : int, optional\n        The maximum number of iterations of the algorithm to perform.\n    callback : callable, optional\n        An optional user-supplied function to be called before the first\n        iteration and after each iteration.\n        Called as ``callback(res)``, where ``res`` is an ``OptimizeResult``\n        similar to that returned by `_chandrupatla` (but containing the current\n        iterate\'s values of all variables). If `callback` raises a\n        ``StopIteration``, the algorithm will terminate immediately and\n        `_chandrupatla` will return a result.\n\n    Returns\n    -------\n    res : OptimizeResult\n        An instance of `scipy.optimize.OptimizeResult` with the following\n        attributes. The descriptions are written as though the values will be\n        scalars; however, if `func` returns an array, the outputs will be\n        arrays of the same shape.\n\n        x : float\n            The root of the function, if the algorithm terminated successfully.\n        nfev : int\n            The number of times the function was called to find the root.\n        nit : int\n            The number of iterations of Chandrupatla\'s algorithm performed.\n        status : int\n            An integer representing the exit status of the algorithm.\n            ``0`` : The algorithm converged to the specified tolerances.\n            ``-1`` : The algorithm encountered an invalid bracket.\n            ``-2`` : The maximum number of iterations was reached.\n            ``-3`` : A non-finite value was encountered.\n            ``-4`` : Iteration was terminated by `callback`.\n            ``1`` : The algorithm is proceeding normally (in `callback` only).\n        success : bool\n            ``True`` when the algorithm terminated successfully (status ``0``).\n        fun : float\n            The value of `func` evaluated at `x`.\n        xl, xr : float\n            The lower and upper ends of the bracket.\n        fl, fr : float\n            The function value at the lower and upper ends of the bracket.\n\n    Notes\n    -----\n    Implemented based on Chandrupatla\'s original paper [1]_.\n\n    If ``xl`` and ``xr`` are the left and right ends of the bracket,\n    ``xmin = xl if abs(func(xl)) <= abs(func(xr)) else xr``,\n    and ``fmin0 = min(func(a), func(b))``, then the algorithm is considered to\n    have converged when ``abs(xr - xl) < xatol + abs(xmin) * xrtol`` or\n    ``fun(xmin) <= fatol + abs(fmin0) * frtol``. This is equivalent to the\n    termination condition described in [1]_ with ``xrtol = 4e-10``,\n    ``xatol = 1e-5``, and ``fatol = frtol = 0``. The default values are\n    ``xatol = 2e-12``, ``xrtol = 4 * np.finfo(float).eps``, ``frtol = 0``,\n    and ``fatol`` is the smallest normal number of the ``dtype`` returned\n    by ``func``.\n\n    References\n    ----------\n\n    .. [1] Chandrupatla, Tirupathi R.\n        "A new hybrid quadratic/bisection algorithm for finding the zero of a\n        nonlinear function without using derivatives".\n        Advances in Engineering Software, 28(3), 145-149.\n        https://doi.org/10.1016/s0965-9978(96)00051-8\n\n    See Also\n    --------\n    brentq, brenth, ridder, bisect, newton\n\n    Examples\n    --------\n    >>> from scipy import optimize\n    >>> def f(x, c):\n    ...     return x**3 - 2*x - c\n    >>> c = 5\n    >>> res = optimize._zeros_py._chandrupatla(f, 0, 3, args=(c,))\n    >>> res.x\n    2.0945514818937463\n\n    >>> c = [3, 4, 5]\n    >>> res = optimize._zeros_py._chandrupatla(f, 0, 3, args=(c,))\n    >>> res.x\n    array([1.8932892 , 2.        , 2.09455148])\n\n    '
    res = _chandrupatla_iv(func, args, xatol, xrtol, fatol, frtol, maxiter, callback)
    (func, args, xatol, xrtol, fatol, frtol, maxiter, callback) = res
    (xs, fs, args, shape, dtype) = _scalar_optimization_initialize(func, (a, b), args)
    (x1, x2) = xs
    (f1, f2) = fs
    status = np.full_like(x1, _EINPROGRESS, dtype=int)
    (nit, nfev) = (0, 2)
    xatol = _xtol if xatol is None else xatol
    xrtol = _rtol if xrtol is None else xrtol
    fatol = np.finfo(dtype).tiny if fatol is None else fatol
    frtol = frtol * np.minimum(np.abs(f1), np.abs(f2))
    work = OptimizeResult(x1=x1, f1=f1, x2=x2, f2=f2, x3=None, f3=None, t=0.5, xatol=xatol, xrtol=xrtol, fatol=fatol, frtol=frtol, nit=nit, nfev=nfev, status=status)
    res_work_pairs = [('status', 'status'), ('x', 'xmin'), ('fun', 'fmin'), ('nit', 'nit'), ('nfev', 'nfev'), ('xl', 'x1'), ('fl', 'f1'), ('xr', 'x2'), ('fr', 'f2')]

    def pre_func_eval(work):
        if False:
            for i in range(10):
                print('nop')
        x = work.x1 + work.t * (work.x2 - work.x1)
        return x

    def post_func_eval(x, f, work):
        if False:
            i = 10
            return i + 15
        (work.x3, work.f3) = (work.x2.copy(), work.f2.copy())
        j = np.sign(f) == np.sign(work.f1)
        nj = ~j
        (work.x3[j], work.f3[j]) = (work.x1[j], work.f1[j])
        (work.x2[nj], work.f2[nj]) = (work.x1[nj], work.f1[nj])
        (work.x1, work.f1) = (x, f)

    def check_termination(work):
        if False:
            print('Hello World!')
        i = np.abs(work.f1) < np.abs(work.f2)
        work.xmin = np.choose(i, (work.x2, work.x1))
        work.fmin = np.choose(i, (work.f2, work.f1))
        stop = np.zeros_like(work.x1, dtype=bool)
        work.dx = abs(work.x2 - work.x1)
        work.tol = abs(work.xmin) * work.xrtol + work.xatol
        i = work.dx < work.tol
        i |= np.abs(work.fmin) <= work.fatol + work.frtol
        work.status[i] = _ECONVERGED
        stop[i] = True
        i = (np.sign(work.f1) == np.sign(work.f2)) & ~stop
        (work.xmin[i], work.fmin[i], work.status[i]) = (np.nan, np.nan, _ESIGNERR)
        stop[i] = True
        i = ~(np.isfinite(work.x1) & np.isfinite(work.x2) & np.isfinite(work.f1) & np.isfinite(work.f2) | stop)
        (work.xmin[i], work.fmin[i], work.status[i]) = (np.nan, np.nan, _EVALUEERR)
        stop[i] = True
        return stop

    def post_termination_check(work):
        if False:
            return 10
        xi1 = (work.x1 - work.x2) / (work.x3 - work.x2)
        phi1 = (work.f1 - work.f2) / (work.f3 - work.f2)
        alpha = (work.x3 - work.x1) / (work.x2 - work.x1)
        j = (1 - np.sqrt(1 - xi1) < phi1) & (phi1 < np.sqrt(xi1))
        (f1j, f2j, f3j, alphaj) = (work.f1[j], work.f2[j], work.f3[j], alpha[j])
        t = np.full_like(alpha, 0.5)
        t[j] = f1j / (f1j - f2j) * f3j / (f3j - f2j) - alphaj * f1j / (f3j - f1j) * f2j / (f2j - f3j)
        tl = 0.5 * work.tol / work.dx
        work.t = np.clip(t, tl, 1 - tl)

    def customize_result(res, shape):
        if False:
            print('Hello World!')
        (xl, xr, fl, fr) = (res['xl'], res['xr'], res['fl'], res['fr'])
        i = res['xl'] < res['xr']
        res['xl'] = np.choose(i, (xr, xl))
        res['xr'] = np.choose(i, (xl, xr))
        res['fl'] = np.choose(i, (fr, fl))
        res['fr'] = np.choose(i, (fl, fr))
        return shape
    return _scalar_optimization_loop(work, callback, shape, maxiter, func, args, dtype, pre_func_eval, post_func_eval, check_termination, post_termination_check, customize_result, res_work_pairs)

def _scalar_optimization_loop(work, callback, shape, maxiter, func, args, dtype, pre_func_eval, post_func_eval, check_termination, post_termination_check, customize_result, res_work_pairs):
    if False:
        for i in range(10):
            print('nop')
    'Main loop of a vectorized scalar optimization algorithm\n\n    Parameters\n    ----------\n    work : OptimizeResult\n        All variables that need to be retained between iterations. Must\n        contain attributes `nit`, `nfev`, and `success`\n    callback : callable\n        User-specified callback function\n    shape : tuple of ints\n        The shape of all output arrays\n    maxiter :\n        Maximum number of iterations of the algorithm\n    func : callable\n        The user-specified callable that is being optimized or solved\n    args : tuple\n        Additional positional arguments to be passed to `func`.\n    dtype : NumPy dtype\n        The common dtype of all abscissae and function values\n    pre_func_eval : callable\n        A function that accepts `work` and returns `x`, the active elements\n        of `x` at which `func` will be evaluated. May modify attributes\n        of `work` with any algorithmic steps that need to happen\n         at the beginning of an iteration, before `func` is evaluated,\n    post_func_eval : callable\n        A function that accepts `x`, `func(x)`, and `work`. May modify\n        attributes of `work` with any algorithmic steps that need to happen\n         in the middle of an iteration, after `func` is evaluated but before\n         the termination check.\n    check_termination : callable\n        A function that accepts `work` and returns `stop`, a boolean array\n        indicating which of the active elements have met a termination\n        condition.\n    post_termination_check : callable\n        A function that accepts `work`. May modify `work` with any algorithmic\n        steps that need to happen after the termination check and before the\n        end of the iteration.\n    customize_result : callable\n        A function that accepts `res` and `shape` and returns `shape`. May\n        modify `res` (in-place) according to preferences (e.g. rearrange\n        elements between attributes) and modify `shape` if needed.\n    res_work_pairs : list of (str, str)\n        Identifies correspondence between attributes of `res` and attributes\n        of `work`; i.e., attributes of active elements of `work` will be\n        copied to the appropriate indices of `res` when appropriate. The order\n        determines the order in which OptimizeResult attributes will be\n        pretty-printed.\n\n    Returns\n    -------\n    res : OptimizeResult\n        The final result object\n\n    Notes\n    -----\n    Besides providing structure, this framework provides several important\n    services for a vectorized optimization algorithm.\n\n    - It handles common tasks involving iteration count, function evaluation\n      count, a user-specified callback, and associated termination conditions.\n    - It compresses the attributes of `work` to eliminate unnecessary\n      computation on elements that have already converged.\n\n    '
    cb_terminate = False
    n_elements = int(np.prod(shape))
    active = np.arange(n_elements)
    res_dict = {i: np.zeros(n_elements, dtype=dtype) for (i, j) in res_work_pairs}
    res_dict['success'] = np.zeros(n_elements, dtype=bool)
    res_dict['status'] = np.full(n_elements, _EINPROGRESS)
    res_dict['nit'] = np.zeros(n_elements, dtype=int)
    res_dict['nfev'] = np.zeros(n_elements, dtype=int)
    res = OptimizeResult(res_dict)
    work.args = args
    active = _scalar_optimization_check_termination(work, res, res_work_pairs, active, check_termination)
    if callback is not None:
        temp = _scalar_optimization_prepare_result(work, res, res_work_pairs, active, shape, customize_result)
        if _call_callback_maybe_halt(callback, temp):
            cb_terminate = True
    while work.nit < maxiter and active.size and (not cb_terminate) and n_elements:
        x = pre_func_eval(work)
        if work.args and work.args[0].ndim != x.ndim:
            dims = np.arange(x.ndim, dtype=np.int64)
            work.args = [np.expand_dims(arg, tuple(dims[arg.ndim:])) for arg in work.args]
        f = func(x, *work.args)
        f = np.asarray(f, dtype=dtype)
        work.nfev += 1 if x.ndim == 1 else x.shape[-1]
        post_func_eval(x, f, work)
        work.nit += 1
        active = _scalar_optimization_check_termination(work, res, res_work_pairs, active, check_termination)
        if callback is not None:
            temp = _scalar_optimization_prepare_result(work, res, res_work_pairs, active, shape, customize_result)
            if _call_callback_maybe_halt(callback, temp):
                cb_terminate = True
                break
        if active.size == 0:
            break
        post_termination_check(work)
    work.status[:] = _ECALLBACK if cb_terminate else _ECONVERR
    return _scalar_optimization_prepare_result(work, res, res_work_pairs, active, shape, customize_result)

def _chandrupatla_iv(func, args, xatol, xrtol, fatol, frtol, maxiter, callback):
    if False:
        print('Hello World!')
    if not callable(func):
        raise ValueError('`func` must be callable.')
    if not np.iterable(args):
        args = (args,)
    tols = np.asarray([xatol if xatol is not None else 1, xrtol if xrtol is not None else 1, fatol if fatol is not None else 1, frtol if frtol is not None else 1])
    if not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0) or np.any(np.isnan(tols)) or (tols.shape != (4,)):
        raise ValueError('Tolerances must be non-negative scalars.')
    maxiter_int = int(maxiter)
    if maxiter != maxiter_int or maxiter < 0:
        raise ValueError('`maxiter` must be a non-negative integer.')
    if callback is not None and (not callable(callback)):
        raise ValueError('`callback` must be callable.')
    return (func, args, xatol, xrtol, fatol, frtol, maxiter, callback)

def _scalar_optimization_initialize(func, xs, args, complex_ok=False):
    if False:
        return 10
    'Initialize abscissa, function, and args arrays for elementwise function\n\n    Parameters\n    ----------\n    func : callable\n        An elementwise function with signature\n\n            func(x: ndarray, *args) -> ndarray\n\n        where each element of ``x`` is a finite real and ``args`` is a tuple,\n        which may contain an arbitrary number of arrays that are broadcastable\n        with ``x``.\n    xs : tuple of arrays\n        Finite real abscissa arrays. Must be broadcastable.\n    args : tuple, optional\n        Additional positional arguments to be passed to `func`.\n\n    Returns\n    -------\n    xs, fs, args : tuple of arrays\n        Broadcasted, writeable, 1D abscissa and function value arrays (or\n        NumPy floats, if appropriate). The dtypes of the `xs` and `fs` are\n        `xfat`; the dtype of the `args` are unchanged.\n    shape : tuple of ints\n        Original shape of broadcasted arrays.\n    xfat : NumPy dtype\n        Result dtype of abscissae, function values, and args determined using\n        `np.result_type`, except integer types are promoted to `np.float64`.\n\n    Raises\n    ------\n    ValueError\n        If the result dtype is not that of a real scalar\n\n    Notes\n    -----\n    Useful for initializing the input of SciPy functions that accept\n    an elementwise callable, abscissae, and arguments; e.g.\n    `scipy.optimize._chandrupatla`.\n    '
    nx = len(xs)
    xas = np.broadcast_arrays(*xs, *args)
    xat = np.result_type(*[xa.dtype for xa in xas])
    xat = np.float64 if np.issubdtype(xat, np.integer) else xat
    (xs, args) = (xas[:nx], xas[nx:])
    xs = [x.astype(xat, copy=False)[()] for x in xs]
    fs = [np.asarray(func(x, *args)) for x in xs]
    shape = xs[0].shape
    message = 'The shape of the array returned by `func` must be the same as the broadcasted shape of `x` and all other `args`.'
    shapes_equal = [f.shape == shape for f in fs]
    if not np.all(shapes_equal):
        raise ValueError(message)
    xfat = np.result_type(*[f.dtype for f in fs] + [xat])
    if not complex_ok and (not np.issubdtype(xfat, np.floating)):
        raise ValueError('Abscissae and function output must be real numbers.')
    xs = [x.astype(xfat, copy=True)[()] for x in xs]
    fs = [f.astype(xfat, copy=True)[()] for f in fs]
    xs = [x.ravel() for x in xs]
    fs = [f.ravel() for f in fs]
    args = [arg.flatten() for arg in args]
    return (xs, fs, args, shape, xfat)

def _scalar_optimization_check_termination(work, res, res_work_pairs, active, check_termination):
    if False:
        print('Hello World!')
    stop = check_termination(work)
    if np.any(stop):
        _scalar_optimization_update_active(work, res, res_work_pairs, active, stop)
        proceed = ~stop
        active = active[proceed]
        for (key, val) in work.items():
            work[key] = val[proceed] if isinstance(val, np.ndarray) else val
        work.args = [arg[proceed] for arg in work.args]
    return active

def _scalar_optimization_update_active(work, res, res_work_pairs, active, mask=None):
    if False:
        while True:
            i = 10
    update_dict = {key1: work[key2] for (key1, key2) in res_work_pairs}
    update_dict['success'] = work.status == 0
    if mask is not None:
        active_mask = active[mask]
        for (key, val) in update_dict.items():
            res[key][active_mask] = val[mask] if np.size(val) > 1 else val
    else:
        for (key, val) in update_dict.items():
            res[key][active] = val

def _scalar_optimization_prepare_result(work, res, res_work_pairs, active, shape, customize_result):
    if False:
        while True:
            i = 10
    res = res.copy()
    _scalar_optimization_update_active(work, res, res_work_pairs, active)
    shape = customize_result(res, shape)
    for (key, val) in res.items():
        res[key] = np.reshape(val, shape)[()]
    res['_order_keys'] = ['success'] + [i for (i, j) in res_work_pairs]
    return OptimizeResult(**res)

def _differentiate_iv(func, x, args, atol, rtol, maxiter, order, initial_step, step_factor, step_direction, callback):
    if False:
        while True:
            i = 10
    if not callable(func):
        raise ValueError('`func` must be callable.')
    x = np.asarray(x)
    dtype = x.dtype if np.issubdtype(x.dtype, np.inexact) else np.float64
    if not np.iterable(args):
        args = (args,)
    if atol is None:
        atol = np.finfo(dtype).tiny
    if rtol is None:
        rtol = np.sqrt(np.finfo(dtype).eps)
    message = 'Tolerances and step parameters must be non-negative scalars.'
    tols = np.asarray([atol, rtol, initial_step, step_factor])
    if not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0) or tols.shape != (4,):
        raise ValueError(message)
    (initial_step, step_factor) = tols[2:].astype(dtype)
    maxiter_int = int(maxiter)
    if maxiter != maxiter_int or maxiter <= 0:
        raise ValueError('`maxiter` must be a positive integer.')
    order_int = int(order)
    if order_int != order or order <= 0:
        raise ValueError('`order` must be a positive integer.')
    step_direction = np.sign(step_direction).astype(dtype)
    (x, step_direction) = np.broadcast_arrays(x, step_direction)
    (x, step_direction) = (x[()], step_direction[()])
    if callback is not None and (not callable(callback)):
        raise ValueError('`callback` must be callable.')
    return (func, x, args, atol, rtol, maxiter_int, order_int, initial_step, step_factor, step_direction, callback)

def _differentiate(func, x, *, args=(), atol=None, rtol=None, maxiter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0, callback=None):
    if False:
        i = 10
        return i + 15
    'Evaluate the derivative of an elementwise scalar function numerically.\n\n    Parameters\n    ----------\n    func : callable\n        The function whose derivative is desired. The signature must be::\n\n            func(x: ndarray, *args) -> ndarray\n\n         where each element of ``x`` is a finite real and ``args`` is a tuple,\n         which may contain an arbitrary number of arrays that are broadcastable\n         with `x`. ``func`` must be an elementwise function: each element\n         ``func(x)[i]`` must equal ``func(x[i])`` for all indices ``i``.\n    x : array_like\n        Abscissae at which to evaluate the derivative.\n    args : tuple, optional\n        Additional positional arguments to be passed to `func`. Must be arrays\n        broadcastable with `x`. If the callable to be differentiated requires\n        arguments that are not broadcastable with `x`, wrap that callable with\n        `func`. See Examples.\n    atol, rtol : float, optional\n        Absolute and relative tolerances for the stopping condition: iteration\n        will stop when ``res.error < atol + rtol * abs(res.df)``. The default\n        `atol` is the smallest normal number of the appropriate dtype, and\n        the default `rtol` is the square root of the precision of the\n        appropriate dtype.\n    order : int, default: 8\n        The (positive integer) order of the finite difference formula to be\n        used. Odd integers will be rounded up to the next even integer.\n    initial_step : float, default: 0.5\n        The (absolute) initial step size for the finite difference derivative\n        approximation.\n    step_factor : float, default: 2.0\n        The factor by which the step size is *reduced* in each iteration; i.e.\n        the step size in iteration 1 is ``initial_step/step_factor``. If\n        ``step_factor < 1``, subsequent steps will be greater than the initial\n        step; this may be useful if steps smaller than some threshold are\n        undesirable (e.g. due to subtractive cancellation error).\n    maxiter : int, default: 10\n        The maximum number of iterations of the algorithm to perform. See\n        notes.\n    step_direction : array_like\n        An array representing the direction of the finite difference steps (for\n        use when `x` lies near to the boundary of the domain of the function.)\n        Must be broadcastable with `x` and all `args`.\n        Where 0 (default), central differences are used; where negative (e.g.\n        -1), steps are non-positive; and where positive (e.g. 1), all steps are\n        non-negative.\n    callback : callable, optional\n        An optional user-supplied function to be called before the first\n        iteration and after each iteration.\n        Called as ``callback(res)``, where ``res`` is an ``OptimizeResult``\n        similar to that returned by `_differentiate` (but containing the\n        current iterate\'s values of all variables). If `callback` raises a\n        ``StopIteration``, the algorithm will terminate immediately and\n        `_differentiate` will return a result.\n\n    Returns\n    -------\n    res : OptimizeResult\n        An instance of `scipy.optimize.OptimizeResult` with the following\n        attributes. (The descriptions are written as though the values will be\n        scalars; however, if `func` returns an array, the outputs will be\n        arrays of the same shape.)\n\n        success : bool\n            ``True`` when the algorithm terminated successfully (status ``0``).\n        status : int\n            An integer representing the exit status of the algorithm.\n            ``0`` : The algorithm converged to the specified tolerances.\n            ``-1`` : The error estimate increased, so iteration was terminated.\n            ``-2`` : The maximum number of iterations was reached.\n            ``-3`` : A non-finite value was encountered.\n            ``-4`` : Iteration was terminated by `callback`.\n            ``1`` : The algorithm is proceeding normally (in `callback` only).\n        df : float\n            The derivative of `func` at `x`, if the algorithm terminated\n            successfully.\n        error : float\n            An estimate of the error: the magnitude of the difference between\n            the current estimate of the derivative and the estimate in the\n            previous iteration.\n        nit : int\n            The number of iterations performed.\n        nfev : int\n            The number of points at which `func` was evaluated.\n        x : float\n            The value at which the derivative of `func` was evaluated\n            (after broadcasting with `args` and `step_direction`).\n\n    Notes\n    -----\n    The implementation was inspired by jacobi [1]_, numdifftools [2]_, and\n    DERIVEST [3]_, but the implementation follows the theory of Taylor series\n    more straightforwardly (and arguably naively so).\n    In the first iteration, the derivative is estimated using a finite\n    difference formula of order `order` with maximum step size `initial_step`.\n    Each subsequent iteration, the maximum step size is reduced by\n    `step_factor`, and the derivative is estimated again until a termination\n    condition is reached. The error estimate is the magnitude of the difference\n    between the current derivative approximation and that of the previous\n    iteration.\n\n    The stencils of the finite difference formulae are designed such that\n    abscissae are "nested": after `func` is evaluated at ``order + 1``\n    points in the first iteration, `func` is evaluated at only two new points\n    in each subsequent iteration; ``order - 1`` previously evaluated function\n    values required by the finite difference formula are reused, and two\n    function values (evaluations at the points furthest from `x`) are unused.\n\n    Step sizes are absolute. When the step size is small relative to the\n    magnitude of `x`, precision is lost; for example, if `x` is ``1e20``, the\n    default initial step size of ``0.5`` cannot be resolved. Accordingly,\n    consider using larger initial step sizes for large magnitudes of `x`.\n\n    The default tolerances are challenging to satisfy at points where the\n    true derivative is exactly zero. If the derivative may be exactly zero,\n    consider specifying an absolute tolerance (e.g. ``atol=1e-16``) to\n    improve convergence.\n\n    References\n    ----------\n    [1]_ Hans Dembinski (@HDembinski). jacobi.\n         https://github.com/HDembinski/jacobi\n    [2]_ Per A. Brodtkorb and John D\'Errico. numdifftools.\n         https://numdifftools.readthedocs.io/en/latest/\n    [3]_ John D\'Errico. DERIVEST: Adaptive Robust Numerical Differentiation.\n         https://www.mathworks.com/matlabcentral/fileexchange/13490-adaptive-robust-numerical-differentiation\n    [4]_ Numerical Differentition. Wikipedia.\n         https://en.wikipedia.org/wiki/Numerical_differentiation\n\n    Examples\n    --------\n    Evaluate the derivative of ``np.exp`` at several points ``x``.\n\n    >>> import numpy as np\n    >>> from scipy.optimize._zeros_py import _differentiate\n    >>> f = np.exp\n    >>> df = np.exp  # true derivative\n    >>> x = np.linspace(1, 2, 5)\n    >>> res = _differentiate(f, x)\n    >>> res.df  # approximation of the derivative\n    array([2.71828183, 3.49034296, 4.48168907, 5.75460268, 7.3890561 ])\n    >>> res.error  # estimate of the error\n    array([7.12940817e-12, 9.16688947e-12, 1.17594823e-11, 1.50972568e-11, 1.93942640e-11])\n    >>> abs(res.df - df(x))  # true error\n    array([3.06421555e-14, 3.01980663e-14, 5.06261699e-14, 6.30606678e-14, 8.34887715e-14])\n\n    Show the convergence of the approximation as the step size is reduced.\n    Each iteration, the step size is reduced by `step_factor`, so for\n    sufficiently small initial step, each iteration reduces the error by a\n    factor of ``1/step_factor**order`` until finite precision arithmetic\n    inhibits further improvement.\n\n    >>> iter = list(range(1, 12))  # maximum iterations\n    >>> hfac = 2  # step size reduction per iteration\n    >>> hdir = [-1, 0, 1]  # compare left-, central-, and right- steps\n    >>> order = 4  # order of differentiation formula\n    >>> x = 1\n    >>> ref = df(x)\n    >>> errors = []  # true error\n    >>> for i in iter:\n    ...     res = _differentiate(f, x, maxiter=i, step_factor=hfac,\n    ...                          step_direction=hdir, order=order,\n    ...                          atol=0, rtol=0)  # prevent early termination\n    ...     errors.append(abs(res.df - ref))\n    >>> errors = np.array(errors)\n    >>> plt.semilogy(iter, errors[:, 0], label=\'left differences\')\n    >>> plt.semilogy(iter, errors[:, 1], label=\'central differences\')\n    >>> plt.semilogy(iter, errors[:, 2], label=\'right differences\')\n    >>> plt.xlabel(\'iteration\')\n    >>> plt.ylabel(\'error\')\n    >>> plt.legend()\n    >>> plt.show()\n    >>> (errors[1, 1] / errors[0, 1], 1 / hfac**order)\n    (0.06215223140159822, 0.0625)\n\n    The implementation is vectorized over `x`, `step_direction`, and `args`.\n    The function is evaluated once before the first iteration to perform input\n    validation and standardization, and once per iteration thereafter.\n\n    >>> def f(x, p):\n    ...     print(\'here\')\n    ...     f.nit += 1\n    ...     return x**p\n    >>> f.nit = 0\n    >>> def df(x, p):\n    ...     return p*x**(p-1)\n    >>> x = np.arange(1, 5)\n    >>> p = np.arange(1, 6).reshape((-1, 1))\n    >>> hdir = np.arange(-1, 2).reshape((-1, 1, 1))\n    >>> res = _differentiate(f, x, args=(p,), step_direction=hdir, maxiter=1)\n    >>> np.allclose(res.df, df(x, p))\n    True\n    >>> res.df.shape\n    (3, 5, 4)\n    >>> f.nit\n    2\n\n    '
    res = _differentiate_iv(func, x, args, atol, rtol, maxiter, order, initial_step, step_factor, step_direction, callback)
    (func, x, args, atol, rtol, maxiter, order, h0, fac, hdir, callback) = res
    (xs, fs, args, shape, dtype) = _scalar_optimization_initialize(func, (x,), args)
    (x, f) = (xs[0], fs[0])
    df = np.full_like(f, np.nan)
    hdir = np.broadcast_to(hdir, shape).flatten()
    status = np.full_like(x, _EINPROGRESS, dtype=int)
    (nit, nfev) = (0, 1)
    il = hdir < 0
    ic = hdir == 0
    ir = hdir > 0
    io = il | ir
    work = OptimizeResult(x=x, df=df, fs=f[:, np.newaxis], error=np.nan, h=h0, df_last=np.nan, error_last=np.nan, h0=h0, fac=fac, atol=atol, rtol=rtol, nit=nit, nfev=nfev, status=status, dtype=dtype, terms=(order + 1) // 2, hdir=hdir, il=il, ic=ic, ir=ir, io=io)
    res_work_pairs = [('status', 'status'), ('df', 'df'), ('error', 'error'), ('nit', 'nit'), ('nfev', 'nfev'), ('x', 'x')]

    def pre_func_eval(work):
        if False:
            while True:
                i = 10
        'Determine the abscissae at which the function needs to be evaluated.\n\n        See `_differentiate_weights` for a description of the stencil (pattern\n        of the abscissae).\n\n        In the first iteration, there is only one stored function value in\n        `work.fs`, `f(x)`, so we need to evaluate at `order` new points. In\n        subsequent iterations, we evaluate at two new points. Note that\n        `work.x` is always flattened into a 1D array after broadcasting with\n        all `args`, so we add a new axis at the end and evaluate all point\n        in one call to the function.\n\n        For improvement:\n        - Consider measuring the step size actually taken, since `(x + h) - x`\n          is not identically equal to `h` with floating point arithmetic.\n        - Adjust the step size automatically if `x` is too big to resolve the\n          step.\n        - We could probably save some work if there are no central difference\n          steps or no one-sided steps.\n        '
        n = work.terms
        h = work.h
        c = work.fac
        d = c ** 0.5
        if work.nit == 0:
            hc = h / c ** np.arange(n)
            hc = np.concatenate((-hc[::-1], hc))
        else:
            hc = np.asarray([-h, h]) / c ** (n - 1)
        if work.nit == 0:
            hr = h / d ** np.arange(2 * n)
        else:
            hr = np.asarray([h, h / d]) / c ** (n - 1)
        n_new = 2 * n if work.nit == 0 else 2
        x_eval = np.zeros((len(work.hdir), n_new), dtype=work.dtype)
        (il, ic, ir) = (work.il, work.ic, work.ir)
        x_eval[ir] = work.x[ir, np.newaxis] + hr
        x_eval[ic] = work.x[ic, np.newaxis] + hc
        x_eval[il] = work.x[il, np.newaxis] - hr
        return x_eval

    def post_func_eval(x, f, work):
        if False:
            i = 10
            return i + 15
        ' Estimate the derivative and error from the function evaluations\n\n        As in `pre_func_eval`: in the first iteration, there is only one stored\n        function value in `work.fs`, `f(x)`, so we need to add the `order` new\n        points. In subsequent iterations, we add two new points. The tricky\n        part is getting the order to match that of the weights, which is\n        described in `_differentiate_weights`.\n\n        For improvement:\n        - Change the order of the weights (and steps in `pre_func_eval`) to\n          simplify `work_fc` concatenation and eliminate `fc` concatenation.\n        - It would be simple to do one-step Richardson extrapolation with `df`\n          and `df_last` to increase the order of the estimate and/or improve\n          the error estimate.\n        - Process the function evaluations in a more numerically favorable\n          way. For instance, combining the pairs of central difference evals\n          into a second-order approximation and using Richardson extrapolation\n          to produce a higher order approximation seemed to retain accuracy up\n          to very high order.\n        - Alternatively, we could use `polyfit` like Jacobi. An advantage of\n          fitting polynomial to more points than necessary is improved noise\n          tolerance.\n        '
        n = work.terms
        n_new = n if work.nit == 0 else 1
        (il, ic, io) = (work.il, work.ic, work.io)
        work_fc = (f[ic, :n_new], work.fs[ic, :], f[ic, -n_new:])
        work_fc = np.concatenate(work_fc, axis=-1)
        if work.nit == 0:
            fc = work_fc
        else:
            fc = (work_fc[:, :n], work_fc[:, n:n + 1], work_fc[:, -n:])
            fc = np.concatenate(fc, axis=-1)
        work_fo = np.concatenate((work.fs[io, :], f[io, :]), axis=-1)
        if work.nit == 0:
            fo = work_fo
        else:
            fo = np.concatenate((work_fo[:, 0:1], work_fo[:, -2 * n:]), axis=-1)
        work.fs = np.zeros((len(ic), work.fs.shape[-1] + 2 * n_new))
        work.fs[ic] = work_fc
        work.fs[io] = work_fo
        (wc, wo) = _differentiate_weights(work, n)
        work.df_last = work.df.copy()
        work.df[ic] = fc @ wc / work.h
        work.df[io] = fo @ wo / work.h
        work.df[il] *= -1
        work.h /= work.fac
        work.error_last = work.error
        work.error = abs(work.df - work.df_last)

    def check_termination(work):
        if False:
            print('Hello World!')
        'Terminate due to convergence, non-finite values, or error increase'
        stop = np.zeros_like(work.df).astype(bool)
        i = work.error < work.atol + work.rtol * abs(work.df)
        work.status[i] = _ECONVERGED
        stop[i] = True
        if work.nit > 0:
            i = ~(np.isfinite(work.x) & np.isfinite(work.df) | stop)
            (work.df[i], work.status[i]) = (np.nan, _EVALUEERR)
            stop[i] = True
        i = (work.error > work.error_last * 10) & ~stop
        work.status[i] = _EERRORINCREASE
        stop[i] = True
        return stop

    def post_termination_check(work):
        if False:
            while True:
                i = 10
        return

    def customize_result(res, shape):
        if False:
            return 10
        return shape
    return _scalar_optimization_loop(work, callback, shape, maxiter, func, args, dtype, pre_func_eval, post_func_eval, check_termination, post_termination_check, customize_result, res_work_pairs)

def _differentiate_weights(work, n):
    if False:
        i = 10
        return i + 15
    fac = work.fac.astype(np.float64)
    if fac != _differentiate_weights.fac:
        _differentiate_weights.central = []
        _differentiate_weights.right = []
        _differentiate_weights.fac = fac
    if len(_differentiate_weights.central) != 2 * n + 1:
        i = np.arange(-n, n + 1)
        p = np.abs(i) - 1.0
        s = np.sign(i)
        h = s / fac ** p
        A = np.vander(h, increasing=True).T
        b = np.zeros(2 * n + 1)
        b[1] = 1
        weights = np.linalg.solve(A, b)
        weights[n] = 0
        for i in range(n):
            weights[-i - 1] = -weights[i]
        _differentiate_weights.central = weights
        i = np.arange(2 * n + 1)
        p = i - 1.0
        s = np.sign(i)
        h = s / np.sqrt(fac) ** p
        A = np.vander(h, increasing=True).T
        b = np.zeros(2 * n + 1)
        b[1] = 1
        weights = np.linalg.solve(A, b)
        _differentiate_weights.right = weights
    return (_differentiate_weights.central.astype(work.dtype, copy=False), _differentiate_weights.right.astype(work.dtype, copy=False))
_differentiate_weights.central = []
_differentiate_weights.right = []
_differentiate_weights.fac = None