from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, cast
import numpy as np
import math
import warnings
from collections import namedtuple
from scipy.special import roots_legendre
from scipy.special import gammaln, logsumexp
from scipy._lib._util import _rng_spawn
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
__all__ = ['fixed_quad', 'quadrature', 'romberg', 'romb', 'trapezoid', 'trapz', 'simps', 'simpson', 'cumulative_trapezoid', 'cumtrapz', 'newton_cotes', 'qmc_quad', 'AccuracyWarning']

def trapezoid(y, x=None, dx=1.0, axis=-1):
    if False:
        i = 10
        return i + 15
    '\n    Integrate along the given axis using the composite trapezoidal rule.\n\n    If `x` is provided, the integration happens in sequence along its\n    elements - they are not sorted.\n\n    Integrate `y` (`x`) along each 1d slice on the given axis, compute\n    :math:`\\int y(x) dx`.\n    When `x` is specified, this integrates along the parametric curve,\n    computing :math:`\\int_t y(t) dt =\n    \\int_t y(t) \\left.\\frac{dx}{dt}\\right|_{x=x(t)} dt`.\n\n    Parameters\n    ----------\n    y : array_like\n        Input array to integrate.\n    x : array_like, optional\n        The sample points corresponding to the `y` values. If `x` is None,\n        the sample points are assumed to be evenly spaced `dx` apart. The\n        default is None.\n    dx : scalar, optional\n        The spacing between sample points when `x` is None. The default is 1.\n    axis : int, optional\n        The axis along which to integrate.\n\n    Returns\n    -------\n    trapezoid : float or ndarray\n        Definite integral of `y` = n-dimensional array as approximated along\n        a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,\n        then the result is a float. If `n` is greater than 1, then the result\n        is an `n`-1 dimensional array.\n\n    See Also\n    --------\n    cumulative_trapezoid, simpson, romb\n\n    Notes\n    -----\n    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points\n    will be taken from `y` array, by default x-axis distances between\n    points will be 1.0, alternatively they can be provided with `x` array\n    or with `dx` scalar.  Return value will be equal to combined area under\n    the red lines.\n\n    References\n    ----------\n    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule\n\n    .. [2] Illustration image:\n           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png\n\n    Examples\n    --------\n    Use the trapezoidal rule on evenly spaced points:\n\n    >>> import numpy as np\n    >>> from scipy import integrate\n    >>> integrate.trapezoid([1, 2, 3])\n    4.0\n\n    The spacing between sample points can be selected by either the\n    ``x`` or ``dx`` arguments:\n\n    >>> integrate.trapezoid([1, 2, 3], x=[4, 6, 8])\n    8.0\n    >>> integrate.trapezoid([1, 2, 3], dx=2)\n    8.0\n\n    Using a decreasing ``x`` corresponds to integrating in reverse:\n\n    >>> integrate.trapezoid([1, 2, 3], x=[8, 6, 4])\n    -8.0\n\n    More generally ``x`` is used to integrate along a parametric curve. We can\n    estimate the integral :math:`\\int_0^1 x^2 = 1/3` using:\n\n    >>> x = np.linspace(0, 1, num=50)\n    >>> y = x**2\n    >>> integrate.trapezoid(y, x)\n    0.33340274885464394\n\n    Or estimate the area of a circle, noting we repeat the sample which closes\n    the curve:\n\n    >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)\n    >>> integrate.trapezoid(np.cos(theta), x=np.sin(theta))\n    3.141571941375841\n\n    ``trapezoid`` can be applied along a specified axis to do multiple\n    computations in one call:\n\n    >>> a = np.arange(6).reshape(2, 3)\n    >>> a\n    array([[0, 1, 2],\n           [3, 4, 5]])\n    >>> integrate.trapezoid(a, axis=0)\n    array([1.5, 2.5, 3.5])\n    >>> integrate.trapezoid(a, axis=1)\n    array([2.,  8.])\n    '
    y = np.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = np.diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    except ValueError:
        d = np.asarray(d)
        y = np.asarray(y)
        ret = np.add.reduce(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
    return ret

def trapz(y, x=None, dx=1.0, axis=-1):
    if False:
        while True:
            i = 10
    'An alias of `trapezoid`.\n\n    `trapz` is kept for backwards compatibility. For new code, prefer\n    `trapezoid` instead.\n    '
    msg = "'scipy.integrate.trapz' is deprecated in favour of 'scipy.integrate.trapezoid' and will be removed in SciPy 1.14.0"
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return trapezoid(y, x=x, dx=dx, axis=axis)

class AccuracyWarning(Warning):
    pass
if TYPE_CHECKING:
    from typing import Protocol

    class CacheAttributes(Protocol):
        cache: dict[int, tuple[Any, Any]]
else:
    CacheAttributes = Callable

def cache_decorator(func: Callable) -> CacheAttributes:
    if False:
        i = 10
        return i + 15
    return cast(CacheAttributes, func)

@cache_decorator
def _cached_roots_legendre(n):
    if False:
        i = 10
        return i + 15
    '\n    Cache roots_legendre results to speed up calls of the fixed_quad\n    function.\n    '
    if n in _cached_roots_legendre.cache:
        return _cached_roots_legendre.cache[n]
    _cached_roots_legendre.cache[n] = roots_legendre(n)
    return _cached_roots_legendre.cache[n]
_cached_roots_legendre.cache = dict()

def fixed_quad(func, a, b, args=(), n=5):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute a definite integral using fixed-order Gaussian quadrature.\n\n    Integrate `func` from `a` to `b` using Gaussian quadrature of\n    order `n`.\n\n    Parameters\n    ----------\n    func : callable\n        A Python function or method to integrate (must accept vector inputs).\n        If integrating a vector-valued function, the returned array must have\n        shape ``(..., len(x))``.\n    a : float\n        Lower limit of integration.\n    b : float\n        Upper limit of integration.\n    args : tuple, optional\n        Extra arguments to pass to function, if any.\n    n : int, optional\n        Order of quadrature integration. Default is 5.\n\n    Returns\n    -------\n    val : float\n        Gaussian quadrature approximation to the integral\n    none : None\n        Statically returned value of None\n\n    See Also\n    --------\n    quad : adaptive quadrature using QUADPACK\n    dblquad : double integrals\n    tplquad : triple integrals\n    romberg : adaptive Romberg quadrature\n    quadrature : adaptive Gaussian quadrature\n    romb : integrators for sampled data\n    simpson : integrators for sampled data\n    cumulative_trapezoid : cumulative integration for sampled data\n    ode : ODE integrator\n    odeint : ODE integrator\n\n    Examples\n    --------\n    >>> from scipy import integrate\n    >>> import numpy as np\n    >>> f = lambda x: x**8\n    >>> integrate.fixed_quad(f, 0.0, 1.0, n=4)\n    (0.1110884353741496, None)\n    >>> integrate.fixed_quad(f, 0.0, 1.0, n=5)\n    (0.11111111111111102, None)\n    >>> print(1/9.0)  # analytical result\n    0.1111111111111111\n\n    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=4)\n    (0.9999999771971152, None)\n    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=5)\n    (1.000000000039565, None)\n    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result\n    1.0\n\n    '
    (x, w) = _cached_roots_legendre(n)
    x = np.real(x)
    if np.isinf(a) or np.isinf(b):
        raise ValueError('Gaussian quadrature is only available for finite limits.')
    y = (b - a) * (x + 1) / 2.0 + a
    return ((b - a) / 2.0 * np.sum(w * func(y, *args), axis=-1), None)

def vectorize1(func, args=(), vec_func=False):
    if False:
        while True:
            i = 10
    'Vectorize the call to a function.\n\n    This is an internal utility function used by `romberg` and\n    `quadrature` to create a vectorized version of a function.\n\n    If `vec_func` is True, the function `func` is assumed to take vector\n    arguments.\n\n    Parameters\n    ----------\n    func : callable\n        User defined function.\n    args : tuple, optional\n        Extra arguments for the function.\n    vec_func : bool, optional\n        True if the function func takes vector arguments.\n\n    Returns\n    -------\n    vfunc : callable\n        A function that will take a vector argument and return the\n        result.\n\n    '
    if vec_func:

        def vfunc(x):
            if False:
                print('Hello World!')
            return func(x, *args)
    else:

        def vfunc(x):
            if False:
                for i in range(10):
                    print('nop')
            if np.isscalar(x):
                return func(x, *args)
            x = np.asarray(x)
            y0 = func(x[0], *args)
            n = len(x)
            dtype = getattr(y0, 'dtype', type(y0))
            output = np.empty((n,), dtype=dtype)
            output[0] = y0
            for i in range(1, n):
                output[i] = func(x[i], *args)
            return output
    return vfunc

def quadrature(func, a, b, args=(), tol=1.49e-08, rtol=1.49e-08, maxiter=50, vec_func=True, miniter=1):
    if False:
        i = 10
        return i + 15
    '\n    Compute a definite integral using fixed-tolerance Gaussian quadrature.\n\n    Integrate `func` from `a` to `b` using Gaussian quadrature\n    with absolute tolerance `tol`.\n\n    Parameters\n    ----------\n    func : function\n        A Python function or method to integrate.\n    a : float\n        Lower limit of integration.\n    b : float\n        Upper limit of integration.\n    args : tuple, optional\n        Extra arguments to pass to function.\n    tol, rtol : float, optional\n        Iteration stops when error between last two iterates is less than\n        `tol` OR the relative change is less than `rtol`.\n    maxiter : int, optional\n        Maximum order of Gaussian quadrature.\n    vec_func : bool, optional\n        True or False if func handles arrays as arguments (is\n        a "vector" function). Default is True.\n    miniter : int, optional\n        Minimum order of Gaussian quadrature.\n\n    Returns\n    -------\n    val : float\n        Gaussian quadrature approximation (within tolerance) to integral.\n    err : float\n        Difference between last two estimates of the integral.\n\n    See Also\n    --------\n    romberg : adaptive Romberg quadrature\n    fixed_quad : fixed-order Gaussian quadrature\n    quad : adaptive quadrature using QUADPACK\n    dblquad : double integrals\n    tplquad : triple integrals\n    romb : integrator for sampled data\n    simpson : integrator for sampled data\n    cumulative_trapezoid : cumulative integration for sampled data\n    ode : ODE integrator\n    odeint : ODE integrator\n\n    Examples\n    --------\n    >>> from scipy import integrate\n    >>> import numpy as np\n    >>> f = lambda x: x**8\n    >>> integrate.quadrature(f, 0.0, 1.0)\n    (0.11111111111111106, 4.163336342344337e-17)\n    >>> print(1/9.0)  # analytical result\n    0.1111111111111111\n\n    >>> integrate.quadrature(np.cos, 0.0, np.pi/2)\n    (0.9999999999999536, 3.9611425250996035e-11)\n    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result\n    1.0\n\n    '
    if not isinstance(args, tuple):
        args = (args,)
    vfunc = vectorize1(func, args, vec_func=vec_func)
    val = np.inf
    err = np.inf
    maxiter = max(miniter + 1, maxiter)
    for n in range(miniter, maxiter + 1):
        newval = fixed_quad(vfunc, a, b, (), n)[0]
        err = abs(newval - val)
        val = newval
        if err < tol or err < rtol * abs(val):
            break
    else:
        warnings.warn('maxiter (%d) exceeded. Latest difference = %e' % (maxiter, err), AccuracyWarning)
    return (val, err)

def tupleset(t, i, value):
    if False:
        for i in range(10):
            print('nop')
    l = list(t)
    l[i] = value
    return tuple(l)

def cumtrapz(y, x=None, dx=1.0, axis=-1, initial=None):
    if False:
        return 10
    'An alias of `cumulative_trapezoid`.\n\n    `cumtrapz` is kept for backwards compatibility. For new code, prefer\n    `cumulative_trapezoid` instead.\n    '
    msg = "'scipy.integrate.cumtrapz' is deprecated in favour of 'scipy.integrate.cumulative_trapezoid' and will be removed in SciPy 1.14.0"
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=initial)

def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    if False:
        while True:
            i = 10
    "\n    Cumulatively integrate y(x) using the composite trapezoidal rule.\n\n    Parameters\n    ----------\n    y : array_like\n        Values to integrate.\n    x : array_like, optional\n        The coordinate to integrate along. If None (default), use spacing `dx`\n        between consecutive elements in `y`.\n    dx : float, optional\n        Spacing between elements of `y`. Only used if `x` is None.\n    axis : int, optional\n        Specifies the axis to cumulate. Default is -1 (last axis).\n    initial : scalar, optional\n        If given, insert this value at the beginning of the returned result.\n        0 or None are the only values accepted. Default is None, which means\n        `res` has one element less than `y` along the axis of integration.\n\n        .. deprecated:: 1.12.0\n            The option for non-zero inputs for `initial` will be deprecated in\n            SciPy 1.14.0. After this time, a ValueError will be raised if\n            `initial` is not None or 0.\n\n    Returns\n    -------\n    res : ndarray\n        The result of cumulative integration of `y` along `axis`.\n        If `initial` is None, the shape is such that the axis of integration\n        has one less value than `y`. If `initial` is given, the shape is equal\n        to that of `y`.\n\n    See Also\n    --------\n    numpy.cumsum, numpy.cumprod\n    quad : adaptive quadrature using QUADPACK\n    romberg : adaptive Romberg quadrature\n    quadrature : adaptive Gaussian quadrature\n    fixed_quad : fixed-order Gaussian quadrature\n    dblquad : double integrals\n    tplquad : triple integrals\n    romb : integrators for sampled data\n    ode : ODE integrators\n    odeint : ODE integrators\n\n    Examples\n    --------\n    >>> from scipy import integrate\n    >>> import numpy as np\n    >>> import matplotlib.pyplot as plt\n\n    >>> x = np.linspace(-2, 2, num=20)\n    >>> y = x\n    >>> y_int = integrate.cumulative_trapezoid(y, x, initial=0)\n    >>> plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')\n    >>> plt.show()\n\n    "
    y = np.asarray(y)
    if x is None:
        d = dx
    else:
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError('If given, shape of x must be 1-D or the same as y.')
        else:
            d = np.diff(x, axis=axis)
        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError('If given, length of x along axis must be the same as y.')
    nd = len(y.shape)
    slice1 = tupleset((slice(None),) * nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),) * nd, axis, slice(None, -1))
    res = np.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)
    if initial is not None:
        if initial != 0:
            warnings.warn('The option for values for `initial` other than None or 0 is deprecated as of SciPy 1.12.0 and will raise a value error in SciPy 1.14.0.', DeprecationWarning, stacklevel=2)
        if not np.isscalar(initial):
            raise ValueError('`initial` parameter should be a scalar.')
        shape = list(res.shape)
        shape[axis] = 1
        res = np.concatenate([np.full(shape, initial, dtype=res.dtype), res], axis=axis)
    return res

def _basic_simpson(y, start, stop, x, dx, axis):
    if False:
        return 10
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),) * nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
    slice2 = tupleset(slice_all, axis, slice(start + 2, stop + 2, step))
    if x is None:
        result = np.sum(y[slice0] + 4.0 * y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        h = np.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
        h0 = h[sl0].astype(float, copy=False)
        h1 = h[sl1].astype(float, copy=False)
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = np.true_divide(h0, h1, out=np.zeros_like(h0), where=h1 != 0)
        tmp = hsum / 6.0 * (y[slice0] * (2.0 - np.true_divide(1.0, h0divh1, out=np.zeros_like(h0divh1), where=h0divh1 != 0)) + y[slice1] * (hsum * np.true_divide(hsum, hprod, out=np.zeros_like(hsum), where=hprod != 0)) + y[slice2] * (2.0 - h0divh1))
        result = np.sum(tmp, axis=axis)
    return result

def simps(y, x=None, dx=1.0, axis=-1, even=_NoValue):
    if False:
        i = 10
        return i + 15
    'An alias of `simpson`.\n\n    `simps` is kept for backwards compatibility. For new code, prefer\n    `simpson` instead.\n    '
    msg = "'scipy.integrate.simps' is deprecated in favour of 'scipy.integrate.simpson' and will be removed in SciPy 1.14.0"
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return simpson(y, x=x, dx=dx, axis=axis, even=even)

@_deprecate_positional_args(version='1.14')
def simpson(y, *, x=None, dx=1.0, axis=-1, even=_NoValue):
    if False:
        while True:
            i = 10
    "\n    Integrate y(x) using samples along the given axis and the composite\n    Simpson's rule. If x is None, spacing of dx is assumed.\n\n    If there are an even number of samples, N, then there are an odd\n    number of intervals (N-1), but Simpson's rule requires an even number\n    of intervals. The parameter 'even' controls how this is handled.\n\n    Parameters\n    ----------\n    y : array_like\n        Array to be integrated.\n    x : array_like, optional\n        If given, the points at which `y` is sampled.\n    dx : float, optional\n        Spacing of integration points along axis of `x`. Only used when\n        `x` is None. Default is 1.\n    axis : int, optional\n        Axis along which to integrate. Default is the last axis.\n    even : {None, 'simpson', 'avg', 'first', 'last'}, optional\n        'avg' : Average two results:\n            1) use the first N-2 intervals with\n               a trapezoidal rule on the last interval and\n            2) use the last\n               N-2 intervals with a trapezoidal rule on the first interval.\n\n        'first' : Use Simpson's rule for the first N-2 intervals with\n                a trapezoidal rule on the last interval.\n\n        'last' : Use Simpson's rule for the last N-2 intervals with a\n               trapezoidal rule on the first interval.\n\n        None : equivalent to 'simpson' (default)\n\n        'simpson' : Use Simpson's rule for the first N-2 intervals with the\n                  addition of a 3-point parabolic segment for the last\n                  interval using equations outlined by Cartwright [1]_.\n                  If the axis to be integrated over only has two points then\n                  the integration falls back to a trapezoidal integration.\n\n                  .. versionadded:: 1.11.0\n\n        .. versionchanged:: 1.11.0\n            The newly added 'simpson' option is now the default as it is more\n            accurate in most situations.\n\n        .. deprecated:: 1.11.0\n            Parameter `even` is deprecated and will be removed in SciPy\n            1.14.0. After this time the behaviour for an even number of\n            points will follow that of `even='simpson'`.\n\n    Returns\n    -------\n    float\n        The estimated integral computed with the composite Simpson's rule.\n\n    See Also\n    --------\n    quad : adaptive quadrature using QUADPACK\n    romberg : adaptive Romberg quadrature\n    quadrature : adaptive Gaussian quadrature\n    fixed_quad : fixed-order Gaussian quadrature\n    dblquad : double integrals\n    tplquad : triple integrals\n    romb : integrators for sampled data\n    cumulative_trapezoid : cumulative integration for sampled data\n    ode : ODE integrators\n    odeint : ODE integrators\n\n    Notes\n    -----\n    For an odd number of samples that are equally spaced the result is\n    exact if the function is a polynomial of order 3 or less. If\n    the samples are not equally spaced, then the result is exact only\n    if the function is a polynomial of order 2 or less.\n\n    References\n    ----------\n    .. [1] Cartwright, Kenneth V. Simpson's Rule Cumulative Integration with\n           MS Excel and Irregularly-spaced Data. Journal of Mathematical\n           Sciences and Mathematics Education. 12 (2): 1-9\n\n    Examples\n    --------\n    >>> from scipy import integrate\n    >>> import numpy as np\n    >>> x = np.arange(0, 10)\n    >>> y = np.arange(0, 10)\n\n    >>> integrate.simpson(y, x)\n    40.5\n\n    >>> y = np.power(x, 3)\n    >>> integrate.simpson(y, x)\n    1640.5\n    >>> integrate.quad(lambda x: x**3, 0, 9)[0]\n    1640.25\n\n    >>> integrate.simpson(y, x, even='first')\n    1644.5\n\n    "
    y = np.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    first_dx = dx
    returnshape = 0
    if x is not None:
        x = np.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError('If given, shape of x must be 1-D or the same as y.')
        if x.shape[axis] != N:
            raise ValueError('If given, length of x along axis must be the same as y.')
    if even is not _NoValue:
        warnings.warn("The 'even' keyword is deprecated as of SciPy 1.11.0 and will be removed in SciPy 1.14.0", DeprecationWarning, stacklevel=2)
    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice_all = (slice(None),) * nd
        even = even if even not in (_NoValue, None) else 'simpson'
        if even not in ['avg', 'last', 'first', 'simpson']:
            raise ValueError("Parameter 'even' must be 'simpson', 'avg', 'last', or 'first'.")
        if N == 2:
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])
            even = None
        if even == 'simpson':
            result = _basic_simpson(y, 0, N - 3, x, dx, axis)
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            slice3 = tupleset(slice_all, axis, -3)
            h = np.asarray([dx, dx], dtype=np.float64)
            if x is not None:
                hm2 = tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = tupleset(slice_all, axis, slice(-1, None, 1))
                diffs = np.float64(np.diff(x, axis=axis))
                h = [np.squeeze(diffs[hm2], axis=axis), np.squeeze(diffs[hm1], axis=axis)]
            num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
            den = 6 * (h[1] + h[0])
            alpha = np.true_divide(num, den, out=np.zeros_like(den), where=den != 0)
            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = np.true_divide(num, den, out=np.zeros_like(den), where=den != 0)
            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = np.true_divide(num, den, out=np.zeros_like(den), where=den != 0)
            result += alpha * y[slice1] + beta * y[slice2] - eta * y[slice3]
        if even in ['avg', 'first']:
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])
            result = _basic_simpson(y, 0, N - 3, x, dx, axis)
        if even in ['avg', 'last']:
            slice1 = tupleset(slice_all, axis, 0)
            slice2 = tupleset(slice_all, axis, 1)
            if x is not None:
                first_dx = x[tuple(slice2)] - x[tuple(slice1)]
            val += 0.5 * first_dx * (y[slice2] + y[slice1])
            result += _basic_simpson(y, 1, N - 2, x, dx, axis)
        if even == 'avg':
            val /= 2.0
            result /= 2.0
        result = result + val
    else:
        result = _basic_simpson(y, 0, N - 2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result

def romb(y, dx=1.0, axis=-1, show=False):
    if False:
        while True:
            i = 10
    '\n    Romberg integration using samples of a function.\n\n    Parameters\n    ----------\n    y : array_like\n        A vector of ``2**k + 1`` equally-spaced samples of a function.\n    dx : float, optional\n        The sample spacing. Default is 1.\n    axis : int, optional\n        The axis along which to integrate. Default is -1 (last axis).\n    show : bool, optional\n        When `y` is a single 1-D array, then if this argument is True\n        print the table showing Richardson extrapolation from the\n        samples. Default is False.\n\n    Returns\n    -------\n    romb : ndarray\n        The integrated result for `axis`.\n\n    See Also\n    --------\n    quad : adaptive quadrature using QUADPACK\n    romberg : adaptive Romberg quadrature\n    quadrature : adaptive Gaussian quadrature\n    fixed_quad : fixed-order Gaussian quadrature\n    dblquad : double integrals\n    tplquad : triple integrals\n    simpson : integrators for sampled data\n    cumulative_trapezoid : cumulative integration for sampled data\n    ode : ODE integrators\n    odeint : ODE integrators\n\n    Examples\n    --------\n    >>> from scipy import integrate\n    >>> import numpy as np\n    >>> x = np.arange(10, 14.25, 0.25)\n    >>> y = np.arange(3, 12)\n\n    >>> integrate.romb(y)\n    56.0\n\n    >>> y = np.sin(np.power(x, 2.5))\n    >>> integrate.romb(y)\n    -0.742561336672229\n\n    >>> integrate.romb(y, show=True)\n    Richardson Extrapolation Table for Romberg Integration\n    ======================================================\n    -0.81576\n     4.63862  6.45674\n    -1.10581 -3.02062 -3.65245\n    -2.57379 -3.06311 -3.06595 -3.05664\n    -1.34093 -0.92997 -0.78776 -0.75160 -0.74256\n    ======================================================\n    -0.742561336672229  # may vary\n\n    '
    y = np.asarray(y)
    nd = len(y.shape)
    Nsamps = y.shape[axis]
    Ninterv = Nsamps - 1
    n = 1
    k = 0
    while n < Ninterv:
        n <<= 1
        k += 1
    if n != Ninterv:
        raise ValueError('Number of samples must be one plus a non-negative power of 2.')
    R = {}
    slice_all = (slice(None),) * nd
    slice0 = tupleset(slice_all, axis, 0)
    slicem1 = tupleset(slice_all, axis, -1)
    h = Ninterv * np.asarray(dx, dtype=float)
    R[0, 0] = (y[slice0] + y[slicem1]) / 2.0 * h
    slice_R = slice_all
    start = stop = step = Ninterv
    for i in range(1, k + 1):
        start >>= 1
        slice_R = tupleset(slice_R, axis, slice(start, stop, step))
        step >>= 1
        R[i, 0] = 0.5 * (R[i - 1, 0] + h * y[slice_R].sum(axis=axis))
        for j in range(1, i + 1):
            prev = R[i, j - 1]
            R[i, j] = prev + (prev - R[i - 1, j - 1]) / ((1 << 2 * j) - 1)
        h /= 2.0
    if show:
        if not np.isscalar(R[0, 0]):
            print('*** Printing table only supported for integrals' + ' of a single data set.')
        else:
            try:
                precis = show[0]
            except (TypeError, IndexError):
                precis = 5
            try:
                width = show[1]
            except (TypeError, IndexError):
                width = 8
            formstr = '%%%d.%df' % (width, precis)
            title = 'Richardson Extrapolation Table for Romberg Integration'
            print(title, '=' * len(title), sep='\n', end='\n')
            for i in range(k + 1):
                for j in range(i + 1):
                    print(formstr % R[i, j], end=' ')
                print()
            print('=' * len(title))
    return R[k, k]

def _difftrap(function, interval, numtraps):
    if False:
        for i in range(10):
            print('nop')
    "\n    Perform part of the trapezoidal rule to integrate a function.\n    Assume that we had called difftrap with all lower powers-of-2\n    starting with 1. Calling difftrap only returns the summation\n    of the new ordinates. It does _not_ multiply by the width\n    of the trapezoids. This must be performed by the caller.\n        'function' is the function to evaluate (must accept vector arguments).\n        'interval' is a sequence with lower and upper limits\n                   of integration.\n        'numtraps' is the number of trapezoids to use (must be a\n                   power-of-2).\n    "
    if numtraps <= 0:
        raise ValueError('numtraps must be > 0 in difftrap().')
    elif numtraps == 1:
        return 0.5 * (function(interval[0]) + function(interval[1]))
    else:
        numtosum = numtraps / 2
        h = float(interval[1] - interval[0]) / numtosum
        lox = interval[0] + 0.5 * h
        points = lox + h * np.arange(numtosum)
        s = np.sum(function(points), axis=0)
        return s

def _romberg_diff(b, c, k):
    if False:
        while True:
            i = 10
    '\n    Compute the differences for the Romberg quadrature corrections.\n    See Forman Acton\'s "Real Computing Made Real," p 143.\n    '
    tmp = 4.0 ** k
    return (tmp * c - b) / (tmp - 1.0)

def _printresmat(function, interval, resmat):
    if False:
        i = 10
        return i + 15
    i = j = 0
    print('Romberg integration of', repr(function), end=' ')
    print('from', interval)
    print('')
    print('%6s %9s %9s' % ('Steps', 'StepSize', 'Results'))
    for i in range(len(resmat)):
        print('%6d %9f' % (2 ** i, (interval[1] - interval[0]) / 2.0 ** i), end=' ')
        for j in range(i + 1):
            print('%9f' % resmat[i][j], end=' ')
        print('')
    print('')
    print('The final result is', resmat[i][j], end=' ')
    print('after', 2 ** (len(resmat) - 1) + 1, 'function evaluations.')

def romberg(function, a, b, args=(), tol=1.48e-08, rtol=1.48e-08, show=False, divmax=10, vec_func=False):
    if False:
        i = 10
        return i + 15
    '\n    Romberg integration of a callable function or method.\n\n    Returns the integral of `function` (a function of one variable)\n    over the interval (`a`, `b`).\n\n    If `show` is 1, the triangular array of the intermediate results\n    will be printed. If `vec_func` is True (default is False), then\n    `function` is assumed to support vector arguments.\n\n    Parameters\n    ----------\n    function : callable\n        Function to be integrated.\n    a : float\n        Lower limit of integration.\n    b : float\n        Upper limit of integration.\n\n    Returns\n    -------\n    results : float\n        Result of the integration.\n\n    Other Parameters\n    ----------------\n    args : tuple, optional\n        Extra arguments to pass to function. Each element of `args` will\n        be passed as a single argument to `func`. Default is to pass no\n        extra arguments.\n    tol, rtol : float, optional\n        The desired absolute and relative tolerances. Defaults are 1.48e-8.\n    show : bool, optional\n        Whether to print the results. Default is False.\n    divmax : int, optional\n        Maximum order of extrapolation. Default is 10.\n    vec_func : bool, optional\n        Whether `func` handles arrays as arguments (i.e., whether it is a\n        "vector" function). Default is False.\n\n    See Also\n    --------\n    fixed_quad : Fixed-order Gaussian quadrature.\n    quad : Adaptive quadrature using QUADPACK.\n    dblquad : Double integrals.\n    tplquad : Triple integrals.\n    romb : Integrators for sampled data.\n    simpson : Integrators for sampled data.\n    cumulative_trapezoid : Cumulative integration for sampled data.\n    ode : ODE integrator.\n    odeint : ODE integrator.\n\n    References\n    ----------\n    .. [1] \'Romberg\'s method\' https://en.wikipedia.org/wiki/Romberg%27s_method\n\n    Examples\n    --------\n    Integrate a gaussian from 0 to 1 and compare to the error function.\n\n    >>> from scipy import integrate\n    >>> from scipy.special import erf\n    >>> import numpy as np\n    >>> gaussian = lambda x: 1/np.sqrt(np.pi) * np.exp(-x**2)\n    >>> result = integrate.romberg(gaussian, 0, 1, show=True)\n    Romberg integration of <function vfunc at ...> from [0, 1]\n\n    ::\n\n       Steps  StepSize  Results\n           1  1.000000  0.385872\n           2  0.500000  0.412631  0.421551\n           4  0.250000  0.419184  0.421368  0.421356\n           8  0.125000  0.420810  0.421352  0.421350  0.421350\n          16  0.062500  0.421215  0.421350  0.421350  0.421350  0.421350\n          32  0.031250  0.421317  0.421350  0.421350  0.421350  0.421350  0.421350\n\n    The final result is 0.421350396475 after 33 function evaluations.\n\n    >>> print("%g %g" % (2*result, erf(1)))\n    0.842701 0.842701\n\n    '
    if np.isinf(a) or np.isinf(b):
        raise ValueError('Romberg integration only available for finite limits.')
    vfunc = vectorize1(function, args, vec_func=vec_func)
    n = 1
    interval = [a, b]
    intrange = b - a
    ordsum = _difftrap(vfunc, interval, n)
    result = intrange * ordsum
    resmat = [[result]]
    err = np.inf
    last_row = resmat[0]
    for i in range(1, divmax + 1):
        n *= 2
        ordsum += _difftrap(vfunc, interval, n)
        row = [intrange * ordsum / n]
        for k in range(i):
            row.append(_romberg_diff(last_row[k], row[k], k + 1))
        result = row[i]
        lastresult = last_row[i - 1]
        if show:
            resmat.append(row)
        err = abs(result - lastresult)
        if err < tol or err < rtol * abs(result):
            break
        last_row = row
    else:
        warnings.warn('divmax (%d) exceeded. Latest difference = %e' % (divmax, err), AccuracyWarning)
    if show:
        _printresmat(vfunc, interval, resmat)
    return result
_builtincoeffs = {1: (1, 2, [1, 1], -1, 12), 2: (1, 3, [1, 4, 1], -1, 90), 3: (3, 8, [1, 3, 3, 1], -3, 80), 4: (2, 45, [7, 32, 12, 32, 7], -8, 945), 5: (5, 288, [19, 75, 50, 50, 75, 19], -275, 12096), 6: (1, 140, [41, 216, 27, 272, 27, 216, 41], -9, 1400), 7: (7, 17280, [751, 3577, 1323, 2989, 2989, 1323, 3577, 751], -8183, 518400), 8: (4, 14175, [989, 5888, -928, 10496, -4540, 10496, -928, 5888, 989], -2368, 467775), 9: (9, 89600, [2857, 15741, 1080, 19344, 5778, 5778, 19344, 1080, 15741, 2857], -4671, 394240), 10: (5, 299376, [16067, 106300, -48525, 272400, -260550, 427368, -260550, 272400, -48525, 106300, 16067], -673175, 163459296), 11: (11, 87091200, [2171465, 13486539, -3237113, 25226685, -9595542, 15493566, 15493566, -9595542, 25226685, -3237113, 13486539, 2171465], -2224234463, 237758976000), 12: (1, 5255250, [1364651, 9903168, -7587864, 35725120, -51491295, 87516288, -87797136, 87516288, -51491295, 35725120, -7587864, 9903168, 1364651], -3012, 875875), 13: (13, 402361344000, [8181904909, 56280729661, -31268252574, 156074417954, -151659573325, 206683437987, -43111992612, -43111992612, 206683437987, -151659573325, 156074417954, -31268252574, 56280729661, 8181904909], -2639651053, 344881152000), 14: (7, 2501928000, [90241897, 710986864, -770720657, 3501442784, -6625093363, 12630121616, -16802270373, 19534438464, -16802270373, 12630121616, -6625093363, 3501442784, -770720657, 710986864, 90241897], -3740727473, 1275983280000)}

def newton_cotes(rn, equal=0):
    if False:
        print('Hello World!')
    "\n    Return weights and error coefficient for Newton-Cotes integration.\n\n    Suppose we have (N+1) samples of f at the positions\n    x_0, x_1, ..., x_N. Then an N-point Newton-Cotes formula for the\n    integral between x_0 and x_N is:\n\n    :math:`\\int_{x_0}^{x_N} f(x)dx = \\Delta x \\sum_{i=0}^{N} a_i f(x_i)\n    + B_N (\\Delta x)^{N+2} f^{N+1} (\\xi)`\n\n    where :math:`\\xi \\in [x_0,x_N]`\n    and :math:`\\Delta x = \\frac{x_N-x_0}{N}` is the average samples spacing.\n\n    If the samples are equally-spaced and N is even, then the error\n    term is :math:`B_N (\\Delta x)^{N+3} f^{N+2}(\\xi)`.\n\n    Parameters\n    ----------\n    rn : int\n        The integer order for equally-spaced data or the relative positions of\n        the samples with the first sample at 0 and the last at N, where N+1 is\n        the length of `rn`. N is the order of the Newton-Cotes integration.\n    equal : int, optional\n        Set to 1 to enforce equally spaced data.\n\n    Returns\n    -------\n    an : ndarray\n        1-D array of weights to apply to the function at the provided sample\n        positions.\n    B : float\n        Error coefficient.\n\n    Notes\n    -----\n    Normally, the Newton-Cotes rules are used on smaller integration\n    regions and a composite rule is used to return the total integral.\n\n    Examples\n    --------\n    Compute the integral of sin(x) in [0, :math:`\\pi`]:\n\n    >>> from scipy.integrate import newton_cotes\n    >>> import numpy as np\n    >>> def f(x):\n    ...     return np.sin(x)\n    >>> a = 0\n    >>> b = np.pi\n    >>> exact = 2\n    >>> for N in [2, 4, 6, 8, 10]:\n    ...     x = np.linspace(a, b, N + 1)\n    ...     an, B = newton_cotes(N, 1)\n    ...     dx = (b - a) / N\n    ...     quad = dx * np.sum(an * f(x))\n    ...     error = abs(quad - exact)\n    ...     print('{:2d}  {:10.9f}  {:.5e}'.format(N, quad, error))\n    ...\n     2   2.094395102   9.43951e-02\n     4   1.998570732   1.42927e-03\n     6   2.000017814   1.78136e-05\n     8   1.999999835   1.64725e-07\n    10   2.000000001   1.14677e-09\n\n    "
    try:
        N = len(rn) - 1
        if equal:
            rn = np.arange(N + 1)
        elif np.all(np.diff(rn) == 1):
            equal = 1
    except Exception:
        N = rn
        rn = np.arange(N + 1)
        equal = 1
    if equal and N in _builtincoeffs:
        (na, da, vi, nb, db) = _builtincoeffs[N]
        an = na * np.array(vi, dtype=float) / da
        return (an, float(nb) / db)
    if rn[0] != 0 or rn[-1] != N:
        raise ValueError('The sample positions must start at 0 and end at N')
    yi = rn / float(N)
    ti = 2 * yi - 1
    nvec = np.arange(N + 1)
    C = ti ** nvec[:, np.newaxis]
    Cinv = np.linalg.inv(C)
    for i in range(2):
        Cinv = 2 * Cinv - Cinv.dot(C).dot(Cinv)
    vec = 2.0 / (nvec[::2] + 1)
    ai = Cinv[:, ::2].dot(vec) * (N / 2.0)
    if N % 2 == 0 and equal:
        BN = N / (N + 3.0)
        power = N + 2
    else:
        BN = N / (N + 2.0)
        power = N + 1
    BN = BN - np.dot(yi ** power, ai)
    p1 = power + 1
    fac = power * math.log(N) - gammaln(p1)
    fac = math.exp(fac)
    return (ai, BN * fac)

def _qmc_quad_iv(func, a, b, n_points, n_estimates, qrng, log):
    if False:
        i = 10
        return i + 15
    if not hasattr(qmc_quad, 'qmc'):
        from scipy import stats
        qmc_quad.stats = stats
    else:
        stats = qmc_quad.stats
    if not callable(func):
        message = '`func` must be callable.'
        raise TypeError(message)
    a = np.atleast_1d(a).copy()
    b = np.atleast_1d(b).copy()
    (a, b) = np.broadcast_arrays(a, b)
    dim = a.shape[0]
    try:
        func((a + b) / 2)
    except Exception as e:
        message = '`func` must evaluate the integrand at points within the integration range; e.g. `func( (a + b) / 2)` must return the integrand at the centroid of the integration volume.'
        raise ValueError(message) from e
    try:
        func(np.array([a, b]).T)
        vfunc = func
    except Exception as e:
        message = f'Exception encountered when attempting vectorized call to `func`: {e}. For better performance, `func` should accept two-dimensional array `x` with shape `(len(a), n_points)` and return an array of the integrand value at each of the `n_points.'
        warnings.warn(message, stacklevel=3)

        def vfunc(x):
            if False:
                print('Hello World!')
            return np.apply_along_axis(func, axis=-1, arr=x)
    n_points_int = np.int64(n_points)
    if n_points != n_points_int:
        message = '`n_points` must be an integer.'
        raise TypeError(message)
    n_estimates_int = np.int64(n_estimates)
    if n_estimates != n_estimates_int:
        message = '`n_estimates` must be an integer.'
        raise TypeError(message)
    if qrng is None:
        qrng = stats.qmc.Halton(dim)
    elif not isinstance(qrng, stats.qmc.QMCEngine):
        message = '`qrng` must be an instance of scipy.stats.qmc.QMCEngine.'
        raise TypeError(message)
    if qrng.d != a.shape[0]:
        message = '`qrng` must be initialized with dimensionality equal to the number of variables in `a`, i.e., `qrng.random().shape[-1]` must equal `a.shape[0]`.'
        raise ValueError(message)
    rng_seed = getattr(qrng, 'rng_seed', None)
    rng = stats._qmc.check_random_state(rng_seed)
    if log not in {True, False}:
        message = '`log` must be boolean (`True` or `False`).'
        raise TypeError(message)
    return (vfunc, a, b, n_points_int, n_estimates_int, qrng, rng, log, stats)
QMCQuadResult = namedtuple('QMCQuadResult', ['integral', 'standard_error'])

def qmc_quad(func, a, b, *, n_estimates=8, n_points=1024, qrng=None, log=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute an integral in N-dimensions using Quasi-Monte Carlo quadrature.\n\n    Parameters\n    ----------\n    func : callable\n        The integrand. Must accept a single argument ``x``, an array which\n        specifies the point(s) at which to evaluate the scalar-valued\n        integrand, and return the value(s) of the integrand.\n        For efficiency, the function should be vectorized to accept an array of\n        shape ``(d, n_points)``, where ``d`` is the number of variables (i.e.\n        the dimensionality of the function domain) and `n_points` is the number\n        of quadrature points, and return an array of shape ``(n_points,)``,\n        the integrand at each quadrature point.\n    a, b : array-like\n        One-dimensional arrays specifying the lower and upper integration\n        limits, respectively, of each of the ``d`` variables.\n    n_estimates, n_points : int, optional\n        `n_estimates` (default: 8) statistically independent QMC samples, each\n        of `n_points` (default: 1024) points, will be generated by `qrng`.\n        The total number of points at which the integrand `func` will be\n        evaluated is ``n_points * n_estimates``. See Notes for details.\n    qrng : `~scipy.stats.qmc.QMCEngine`, optional\n        An instance of the QMCEngine from which to sample QMC points.\n        The QMCEngine must be initialized to a number of dimensions ``d``\n        corresponding with the number of variables ``x1, ..., xd`` passed to\n        `func`.\n        The provided QMCEngine is used to produce the first integral estimate.\n        If `n_estimates` is greater than one, additional QMCEngines are\n        spawned from the first (with scrambling enabled, if it is an option.)\n        If a QMCEngine is not provided, the default `scipy.stats.qmc.Halton`\n        will be initialized with the number of dimensions determine from\n        the length of `a`.\n    log : boolean, default: False\n        When set to True, `func` returns the log of the integrand, and\n        the result object contains the log of the integral.\n\n    Returns\n    -------\n    result : object\n        A result object with attributes:\n\n        integral : float\n            The estimate of the integral.\n        standard_error :\n            The error estimate. See Notes for interpretation.\n\n    Notes\n    -----\n    Values of the integrand at each of the `n_points` points of a QMC sample\n    are used to produce an estimate of the integral. This estimate is drawn\n    from a population of possible estimates of the integral, the value of\n    which we obtain depends on the particular points at which the integral\n    was evaluated. We perform this process `n_estimates` times, each time\n    evaluating the integrand at different scrambled QMC points, effectively\n    drawing i.i.d. random samples from the population of integral estimates.\n    The sample mean :math:`m` of these integral estimates is an\n    unbiased estimator of the true value of the integral, and the standard\n    error of the mean :math:`s` of these estimates may be used to generate\n    confidence intervals using the t distribution with ``n_estimates - 1``\n    degrees of freedom. Perhaps counter-intuitively, increasing `n_points`\n    while keeping the total number of function evaluation points\n    ``n_points * n_estimates`` fixed tends to reduce the actual error, whereas\n    increasing `n_estimates` tends to decrease the error estimate.\n\n    Examples\n    --------\n    QMC quadrature is particularly useful for computing integrals in higher\n    dimensions. An example integrand is the probability density function\n    of a multivariate normal distribution.\n\n    >>> import numpy as np\n    >>> from scipy import stats\n    >>> dim = 8\n    >>> mean = np.zeros(dim)\n    >>> cov = np.eye(dim)\n    >>> def func(x):\n    ...     # `multivariate_normal` expects the _last_ axis to correspond with\n    ...     # the dimensionality of the space, so `x` must be transposed\n    ...     return stats.multivariate_normal.pdf(x.T, mean, cov)\n\n    To compute the integral over the unit hypercube:\n\n    >>> from scipy.integrate import qmc_quad\n    >>> a = np.zeros(dim)\n    >>> b = np.ones(dim)\n    >>> rng = np.random.default_rng()\n    >>> qrng = stats.qmc.Halton(d=dim, seed=rng)\n    >>> n_estimates = 8\n    >>> res = qmc_quad(func, a, b, n_estimates=n_estimates, qrng=qrng)\n    >>> res.integral, res.standard_error\n    (0.00018429555666024108, 1.0389431116001344e-07)\n\n    A two-sided, 99% confidence interval for the integral may be estimated\n    as:\n\n    >>> t = stats.t(df=n_estimates-1, loc=res.integral,\n    ...             scale=res.standard_error)\n    >>> t.interval(0.99)\n    (0.0001839319802536469, 0.00018465913306683527)\n\n    Indeed, the value reported by `scipy.stats.multivariate_normal` is\n    within this range.\n\n    >>> stats.multivariate_normal.cdf(b, mean, cov, lower_limit=a)\n    0.00018430867675187443\n\n    '
    args = _qmc_quad_iv(func, a, b, n_points, n_estimates, qrng, log)
    (func, a, b, n_points, n_estimates, qrng, rng, log, stats) = args

    def sum_product(integrands, dA, log=False):
        if False:
            for i in range(10):
                print('nop')
        if log:
            return logsumexp(integrands) + np.log(dA)
        else:
            return np.sum(integrands * dA)

    def mean(estimates, log=False):
        if False:
            i = 10
            return i + 15
        if log:
            return logsumexp(estimates) - np.log(n_estimates)
        else:
            return np.mean(estimates)

    def std(estimates, m=None, ddof=0, log=False):
        if False:
            print('Hello World!')
        m = m or mean(estimates, log)
        if log:
            (estimates, m) = np.broadcast_arrays(estimates, m)
            temp = np.vstack((estimates, m + np.pi * 1j))
            diff = logsumexp(temp, axis=0)
            return np.real(0.5 * (logsumexp(2 * diff) - np.log(n_estimates - ddof)))
        else:
            return np.std(estimates, ddof=ddof)

    def sem(estimates, m=None, s=None, log=False):
        if False:
            i = 10
            return i + 15
        m = m or mean(estimates, log)
        s = s or std(estimates, m, ddof=1, log=log)
        if log:
            return s - 0.5 * np.log(n_estimates)
        else:
            return s / np.sqrt(n_estimates)
    if np.any(a == b):
        message = 'A lower limit was equal to an upper limit, so the value of the integral is zero by definition.'
        warnings.warn(message, stacklevel=2)
        return QMCQuadResult(-np.inf if log else 0, 0)
    i_swap = b < a
    sign = (-1) ** i_swap.sum(axis=-1)
    (a[i_swap], b[i_swap]) = (b[i_swap], a[i_swap])
    A = np.prod(b - a)
    dA = A / n_points
    estimates = np.zeros(n_estimates)
    rngs = _rng_spawn(qrng.rng, n_estimates)
    for i in range(n_estimates):
        sample = qrng.random(n_points)
        x = stats.qmc.scale(sample, a, b).T
        integrands = func(x)
        estimates[i] = sum_product(integrands, dA, log)
        qrng = type(qrng)(seed=rngs[i], **qrng._init_quad)
    integral = mean(estimates, log)
    standard_error = sem(estimates, m=integral, log=log)
    integral = integral + np.pi * 1j if log and sign < 0 else integral * sign
    return QMCQuadResult(integral, standard_error)