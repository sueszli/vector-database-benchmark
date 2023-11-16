import numpy as np
from scipy import special
from scipy.optimize import OptimizeResult
from scipy.optimize._zeros_py import _scalar_optimization_initialize, _scalar_optimization_loop, _ECONVERGED, _ESIGNERR, _ECONVERR, _EVALUEERR, _ECALLBACK, _EINPROGRESS

def _tanhsinh(f, a, b, *, args=(), log=False, maxfun=None, maxlevel=None, minlevel=2, atol=None, rtol=None, callback=None):
    if False:
        return 10
    'Evaluate a convergent integral numerically using tanh-sinh quadrature.\n\n    In practice, tanh-sinh quadrature achieves quadratic convergence for\n    many integrands: the number of accurate *digits* scales roughly linearly\n    with the number of function evaluations [1]_.\n\n    Either or both of the limits of integration may be infinite, and\n    singularities at the endpoints are acceptable. Divergent integrals and\n    integrands with non-finite derivatives or singularities within an interval\n    are out of scope, but the latter may be evaluated be calling `_tanhsinh` on\n    each sub-interval separately.\n\n    Parameters\n    ----------\n    f : callable\n        The function to be integrated. The signature must be::\n            func(x: ndarray, *args) -> ndarray\n         where each element of ``x`` is a finite real and ``args`` is a tuple,\n         which may contain an arbitrary number of arrays that are broadcastable\n         with `x`. ``func`` must be an elementwise function: each element\n         ``func(x)[i]`` must equal ``func(x[i])`` for all indices ``i``.\n         If ``func`` returns a value with complex dtype when evaluated at\n         either endpoint, subsequent arguments ``x`` will have complex dtype\n         (but zero imaginary part).\n    a, b : array_like\n        Real lower and upper limits of integration. Must be broadcastable.\n        Elements may be infinite.\n    args : tuple, optional\n        Additional positional arguments to be passed to `func`. Must be arrays\n        broadcastable with `a` and `b`. If the callable to be integrated\n        requires arguments that are not broadcastable with `a` and `b`, wrap\n        that callable with `f`. See Examples.\n    log : bool, default: False\n        Setting to True indicates that `f` returns the log of the integrand\n        and that `atol` and `rtol` are expressed as the logs of the absolute\n        and relative errors. In this case, the result object will contain the\n        log of the integral and error. This is useful for integrands for which\n        numerical underflow or overflow would lead to inaccuracies.\n        When ``log=True``, the integrand (the exponential of `f`) must be real,\n        but it may be negative, in which case the log of the integrand is a\n        complex number with an imaginary part that is an odd multiple of π.\n    maxlevel : int, default: 10\n        The maximum refinement level of the algorithm.\n\n        At the zeroth level, `f` is called once, performing 16 function\n        evaluations. At each subsequent level, `f` is called once more,\n        approximately doubling the number of function evaluations that have\n        been performed. Accordingly, for many integrands, each successive level\n        will double the number of accurate digits in the result (up to the\n        limits of floating point precision).\n\n        The algorithm will terminate after completing level `maxlevel` or after\n        another termination condition is satisfied, whichever comes first.\n    minlevel : int, default: 2\n        The level at which to begin iteration (default: 2). This does not\n        change the total number of function evaluations or the abscissae at\n        which the function is evaluated; it changes only the *number of times*\n        `f` is called. If ``minlevel=k``, then the integrand is evaluated at\n        all abscissae from levels ``0`` through ``k`` in a single call.\n        Note that if `minlevel` exceeds `maxlevel`, the provided `minlevel` is\n        ignored, and `minlevel` is set equal to `maxlevel`.\n    atol, rtol : float, optional\n        Absolute termination tolerance (default: 0) and relative termination\n        tolerance (default: ``eps**0.75``, where ``eps`` is the precision of\n        the result dtype), respectively. The error estimate is as\n        described in [1]_ Section 5. While not theoretically rigorous or\n        conservative, it is said to work well in practice. Must be non-negative\n        and finite if `log` is False, and must be expressed as the log of a\n        non-negative and finite number if `log` is True.\n    callback : callable, optional\n        An optional user-supplied function to be called before the first\n        iteration and after each iteration.\n        Called as ``callback(res)``, where ``res`` is an ``OptimizeResult``\n        similar to that returned by `_differentiate` (but containing the\n        current iterate\'s values of all variables). If `callback` raises a\n        ``StopIteration``, the algorithm will terminate immediately and\n        `_tanhsinh` will return a result object.\n\n    Returns\n    -------\n    res : OptimizeResult\n        An instance of `scipy.optimize.OptimizeResult` with the following\n        attributes. (The descriptions are written as though the values will be\n        scalars; however, if `func` returns an array, the outputs will be\n        arrays of the same shape.)\n        success : bool\n            ``True`` when the algorithm terminated successfully (status ``0``).\n        status : int\n            An integer representing the exit status of the algorithm.\n            ``0`` : The algorithm converged to the specified tolerances.\n            ``-1`` : (unused)\n            ``-2`` : The maximum number of iterations was reached.\n            ``-3`` : A non-finite value was encountered.\n            ``-4`` : Iteration was terminated by `callback`.\n            ``1`` : The algorithm is proceeding normally (in `callback` only).\n        integral : float\n            An estimate of the integral\n        error : float\n            An estimate of the error. Only available if level two or higher\n            has been completed; otherwise NaN.\n        nit : int\n            The number of iterations performed.\n        nfev : int\n            The number of points at which `func` was evaluated.\n\n    See Also\n    --------\n    quad, quadrature\n\n    Notes\n    -----\n    Implements the algorithm as described in [1]_ with minor adaptations for\n    finite-precision arithmetic, including some described by [2]_ and [3]_. The\n    tanh-sinh scheme was originally introduced in [4]_.\n\n    Due floating-point error in the abscissae, the function may be evaluated\n    at the endpoints of the interval during iterations. The values returned by\n    the function at the endpoints will be ignored.\n\n    References\n    ----------\n    [1] Bailey, David H., Karthik Jeyabalan, and Xiaoye S. Li. "A comparison of\n        three high-precision quadrature schemes." Experimental Mathematics 14.3\n        (2005): 317-329.\n    [2] Vanherck, Joren, Bart Sorée, and Wim Magnus. "Tanh-sinh quadrature for\n        single and multiple integration using floating-point arithmetic."\n        arXiv preprint arXiv:2007.15057 (2020).\n    [3] van Engelen, Robert A.  "Improving the Double Exponential Quadrature\n        Tanh-Sinh, Sinh-Sinh and Exp-Sinh Formulas."\n        https://www.genivia.com/files/qthsh.pdf\n    [4] Takahasi, Hidetosi, and Masatake Mori. "Double exponential formulas for\n        numerical integration." Publications of the Research Institute for\n        Mathematical Sciences 9.3 (1974): 721-741.\n\n    Example\n    -------\n    Evaluate the Gaussian integral:\n\n    >>> import numpy as np\n    >>> from scipy.integrate._tanhsinh import _tanhsinh\n    >>> def f(x):\n    ...     return np.exp(-x**2)\n    >>> res = _tanhsinh(f, -np.inf, np.inf)\n    >>> res.integral  # true value is np.sqrt(np.pi), 1.7724538509055159\n     1.7724538509055159\n    >>> res.error  # actual error is 0\n    4.0007963937534104e-16\n\n    The value of the Gaussian function (bell curve) is nearly zero for\n    arguments sufficiently far from zero, so the value of the integral\n    over a finite interval is nearly the same.\n\n    >>> _tanhsinh(f, -20, 20).integral\n    1.772453850905518\n\n    However, with unfavorable integration limits, the integration scheme\n    may not be able to find the important region.\n\n    >>> _tanhsinh(f, -np.inf, 1000).integral\n    4.500490856620352\n\n    In such cases, or when there are singularities within the interval,\n    break the integral into parts with endpoints at the important points.\n\n    >>> _tanhsinh(f, -np.inf, 0).integral + _tanhsinh(f, 0, 1000).integral\n    1.772453850905404\n\n    For integration involving very large or very small magnitudes, use\n    log-integration. (For illustrative purposes, the following example shows a\n    case in which both regular and log-integration work, but for more extreme\n    limits of integration, log-integration would avoid the underflow\n    experienced when evaluating the integral normally.)\n\n    >>> res = _tanhsinh(f, 20, 30, rtol=1e-10)\n    >>> res.integral, res.error\n    4.7819613911309014e-176, 4.670364401645202e-187\n    >>> def log_f(x):\n    ...     return -x**2\n    >>> np.exp(res.integral), np.exp(res.error)\n    4.7819613911306924e-176, 4.670364401645093e-187\n\n    The limits of integration and elements of `args` may be broadcastable\n    arrays, and integration is performed elementwise.\n\n    >>> from scipy import stats\n    >>> dist = stats.gausshyper(13.8, 3.12, 2.51, 5.18)\n    >>> a, b = dist.support()\n    >>> x = np.linspace(a, b, 100)\n    >>> res = _tanhsinh(dist.pdf, a, x)\n    >>> ref = dist.cdf(x)\n    >>> np.allclose(res.integral, ref)\n\n    '
    tmp = (f, a, b, log, maxfun, maxlevel, minlevel, atol, rtol, args, callback)
    tmp = _tanhsinh_iv(*tmp)
    (f, a, b, log, maxfun, maxlevel, minlevel, atol, rtol, args, callback) = tmp
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        c = ((a.ravel() + b.ravel()) / 2).reshape(a.shape)
        c[np.isinf(a)] = b[np.isinf(a)]
        c[np.isinf(b)] = a[np.isinf(b)]
        c[np.isnan(c)] = 0
        tmp = _scalar_optimization_initialize(f, (c,), args, complex_ok=True)
    (xs, fs, args, shape, dtype) = tmp
    a = np.broadcast_to(a, shape).astype(dtype).ravel()
    b = np.broadcast_to(b, shape).astype(dtype).ravel()
    (a, b, a0, negative, abinf, ainf, binf) = _transform_integrals(a, b)
    (nit, nfev) = (0, 1)
    zero = -np.inf if log else 0
    pi = dtype.type(np.pi)
    maxiter = maxlevel - minlevel + 1
    eps = np.finfo(dtype).eps
    if rtol is None:
        rtol = 0.75 * np.log(eps) if log else eps ** 0.75
    Sn = np.full(shape, zero, dtype=dtype).ravel()
    Sn[np.isnan(a) | np.isnan(b) | np.isnan(fs[0])] = np.nan
    Sk = np.empty_like(Sn).reshape(-1, 1)[:, 0:0]
    aerr = np.full(shape, np.nan, dtype=dtype).ravel()
    status = np.full(shape, _EINPROGRESS, dtype=int).ravel()
    h0 = np.real(_get_base_step(dtype=dtype))
    xr0 = np.full(shape, -np.inf, dtype=dtype).ravel()
    fr0 = np.full(shape, np.nan, dtype=dtype).ravel()
    wr0 = np.zeros(shape, dtype=dtype).ravel()
    xl0 = np.full(shape, np.inf, dtype=dtype).ravel()
    fl0 = np.full(shape, np.nan, dtype=dtype).ravel()
    wl0 = np.zeros(shape, dtype=dtype).ravel()
    d4 = np.zeros(shape, dtype=dtype).ravel()
    work = OptimizeResult(Sn=Sn, Sk=Sk, aerr=aerr, h=h0, log=log, dtype=dtype, pi=pi, eps=eps, a=a.reshape(-1, 1), b=b.reshape(-1, 1), n=minlevel, nit=nit, nfev=nfev, status=status, xr0=xr0, fr0=fr0, wr0=wr0, xl0=xl0, fl0=fl0, wl0=wl0, d4=d4, ainf=ainf, binf=binf, abinf=abinf, a0=a0.reshape(-1, 1))
    res_work_pairs = [('status', 'status'), ('integral', 'Sn'), ('error', 'aerr'), ('nit', 'nit'), ('nfev', 'nfev')]

    def pre_func_eval(work):
        if False:
            for i in range(10):
                print('nop')
        work.h = h0 / 2 ** work.n
        (xjc, wj) = _get_pairs(work.n, h0, dtype=work.dtype, inclusive=work.n == minlevel)
        (work.xj, work.wj) = _transform_to_limits(xjc, wj, work.a, work.b)
        xj = work.xj.copy()
        xj[work.abinf] = xj[work.abinf] / (1 - xj[work.abinf] ** 2)
        xj[work.binf] = 1 / xj[work.binf] - 1 + work.a0[work.binf]
        xj[work.ainf] *= -1
        return xj

    def post_func_eval(x, fj, work):
        if False:
            return 10
        if work.log:
            fj[work.abinf] += np.log(1 + work.xj[work.abinf] ** 2) - 2 * np.log(1 - work.xj[work.abinf] ** 2)
            fj[work.binf] -= 2 * np.log(work.xj[work.binf])
        else:
            fj[work.abinf] *= (1 + work.xj[work.abinf] ** 2) / (1 - work.xj[work.abinf] ** 2) ** 2
            fj[work.binf] *= work.xj[work.binf] ** (-2.0)
        (fjwj, Sn) = _euler_maclaurin_sum(fj, work)
        if work.Sk.shape[-1]:
            Snm1 = work.Sk[:, -1]
            Sn = special.logsumexp([Snm1 - np.log(2), Sn], axis=0) if log else Snm1 / 2 + Sn
        work.fjwj = fjwj
        work.Sn = Sn

    def check_termination(work):
        if False:
            return 10
        'Terminate due to convergence or encountering non-finite values'
        stop = np.zeros(work.Sn.shape, dtype=bool)
        if work.nit == 0:
            i = (work.a == work.b).ravel()
            zero = -np.inf if log else 0
            work.Sn[i] = zero
            work.aerr[i] = zero
            work.status[i] = _ECONVERGED
            stop[i] = True
        else:
            (work.rerr, work.aerr) = _estimate_error(work)
            i = (work.rerr < rtol) | (work.rerr + np.real(work.Sn) < atol) if log else (work.rerr < rtol) | (work.rerr * abs(work.Sn) < atol)
            work.status[i] = _ECONVERGED
            stop[i] = True
        if log:
            i = (np.isposinf(np.real(work.Sn)) | np.isnan(work.Sn)) & ~stop
        else:
            i = ~np.isfinite(work.Sn) & ~stop
        work.status[i] = _EVALUEERR
        stop[i] = True
        return stop

    def post_termination_check(work):
        if False:
            while True:
                i = 10
        work.n += 1
        work.Sk = np.concatenate((work.Sk, work.Sn[:, np.newaxis]), axis=-1)
        return

    def customize_result(res, shape):
        if False:
            return 10
        if log and np.any(negative):
            pi = res['integral'].dtype.type(np.pi)
            j = np.complex64(1j)
            res['integral'] = res['integral'] + negative * pi * j
        else:
            res['integral'][negative] *= -1
        res['maxlevel'] = minlevel + res['nit'] - 1
        res['maxlevel'][res['nit'] == 0] = -1
        del res['nit']
        return shape
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        res = _scalar_optimization_loop(work, callback, shape, maxiter, f, args, dtype, pre_func_eval, post_func_eval, check_termination, post_termination_check, customize_result, res_work_pairs)
    return res

def _get_base_step(dtype=np.float64):
    if False:
        return 10
    fmin = 4 * np.finfo(dtype).tiny
    tmax = np.arcsinh(np.log(2 / fmin - 1) / np.pi)
    h0 = tmax / _N_BASE_STEPS
    return h0.astype(dtype)
_N_BASE_STEPS = 8

def _compute_pair(k, h0):
    if False:
        while True:
            i = 10
    h = h0 / 2 ** k
    max = _N_BASE_STEPS * 2 ** k
    j = np.arange(max + 1) if k == 0 else np.arange(1, max + 1, 2)
    jh = j * h
    pi_2 = np.pi / 2
    u1 = pi_2 * np.cosh(jh)
    u2 = pi_2 * np.sinh(jh)
    wj = u1 / np.cosh(u2) ** 2
    xjc = 1 / (np.exp(u2) * np.cosh(u2))
    wj[0] = wj[0] / 2 if k == 0 else wj[0]
    return (xjc, wj)

def _pair_cache(k, h0):
    if False:
        for i in range(10):
            print('nop')
    if h0 != _pair_cache.h0:
        _pair_cache.xjc = np.empty(0)
        _pair_cache.wj = np.empty(0)
        _pair_cache.indices = [0]
    xjcs = [_pair_cache.xjc]
    wjs = [_pair_cache.wj]
    for i in range(len(_pair_cache.indices) - 1, k + 1):
        (xjc, wj) = _compute_pair(i, h0)
        xjcs.append(xjc)
        wjs.append(wj)
        _pair_cache.indices.append(_pair_cache.indices[-1] + len(xjc))
    _pair_cache.xjc = np.concatenate(xjcs)
    _pair_cache.wj = np.concatenate(wjs)
    _pair_cache.h0 = h0
_pair_cache.xjc = np.empty(0)
_pair_cache.wj = np.empty(0)
_pair_cache.indices = [0]
_pair_cache.h0 = None

def _get_pairs(k, h0, inclusive=False, dtype=np.float64):
    if False:
        i = 10
        return i + 15
    if len(_pair_cache.indices) <= k + 2 or h0 != _pair_cache.h0:
        _pair_cache(k, h0)
    xjc = _pair_cache.xjc
    wj = _pair_cache.wj
    indices = _pair_cache.indices
    start = 0 if inclusive else indices[k]
    end = indices[k + 1]
    return (xjc[start:end].astype(dtype), wj[start:end].astype(dtype))

def _transform_to_limits(xjc, wj, a, b):
    if False:
        while True:
            i = 10
    alpha = (b - a) / 2
    xj = np.concatenate((-alpha * xjc + b, alpha * xjc + a), axis=-1)
    wj = wj * alpha
    wj = np.concatenate((wj, wj), axis=-1)
    invalid = (xj <= a) | (xj >= b)
    wj[invalid] = 0
    return (xj, wj)

def _euler_maclaurin_sum(fj, work):
    if False:
        i = 10
        return i + 15
    (xr0, fr0, wr0) = (work.xr0, work.fr0, work.wr0)
    (xl0, fl0, wl0) = (work.xl0, work.fl0, work.wl0)
    (xj, fj, wj) = (work.xj.T, fj.T, work.wj.T)
    (n_x, n_active) = xj.shape
    (xr, xl) = xj.reshape(2, n_x // 2, n_active).copy()
    (fr, fl) = fj.reshape(2, n_x // 2, n_active)
    (wr, wl) = wj.reshape(2, n_x // 2, n_active)
    invalid_r = ~np.isfinite(fr) | (wr == 0)
    invalid_l = ~np.isfinite(fl) | (wl == 0)
    xr[invalid_r] = -np.inf
    ir = np.argmax(xr, axis=0, keepdims=True)
    xr_max = np.take_along_axis(xr, ir, axis=0)[0]
    fr_max = np.take_along_axis(fr, ir, axis=0)[0]
    wr_max = np.take_along_axis(wr, ir, axis=0)[0]
    j = xr_max > xr0
    xr0[j] = xr_max[j]
    fr0[j] = fr_max[j]
    wr0[j] = wr_max[j]
    xl[invalid_l] = np.inf
    il = np.argmin(xl, axis=0, keepdims=True)
    xl_min = np.take_along_axis(xl, il, axis=0)[0]
    fl_min = np.take_along_axis(fl, il, axis=0)[0]
    wl_min = np.take_along_axis(wl, il, axis=0)[0]
    j = xl_min < xl0
    xl0[j] = xl_min[j]
    fl0[j] = fl_min[j]
    wl0[j] = wl_min[j]
    fj = fj.T
    flwl0 = fl0 + np.log(wl0) if work.log else fl0 * wl0
    frwr0 = fr0 + np.log(wr0) if work.log else fr0 * wr0
    magnitude = np.real if work.log else np.abs
    work.d4 = np.maximum(magnitude(flwl0), magnitude(frwr0))
    fr0b = np.broadcast_to(fr0[np.newaxis, :], fr.shape)
    fl0b = np.broadcast_to(fl0[np.newaxis, :], fl.shape)
    fr[invalid_r] = fr0b[invalid_r]
    fl[invalid_l] = fl0b[invalid_l]
    fjwj = fj + np.log(work.wj) if work.log else fj * work.wj
    Sn = special.logsumexp(fjwj + np.log(work.h), axis=-1) if work.log else np.sum(fjwj, axis=-1) * work.h
    (work.xr0, work.fr0, work.wr0) = (xr0, fr0, wr0)
    (work.xl0, work.fl0, work.wl0) = (xl0, fl0, wl0)
    return (fjwj, Sn)

def _estimate_error(work):
    if False:
        i = 10
        return i + 15
    if work.n == 0 or work.nit == 0:
        nan = np.full_like(work.Sn, np.nan)
        return (nan, nan)
    indices = _pair_cache.indices
    n_active = len(work.Sn)
    axis_kwargs = dict(axis=-1, keepdims=True)
    if work.Sk.shape[-1] == 0:
        h = 2 * work.h
        n_x = indices[work.n]
        fjwj_rl = work.fjwj.reshape(n_active, 2, -1)
        fjwj = fjwj_rl[:, :, :n_x].reshape(n_active, 2 * n_x)
        Snm1 = special.logsumexp(fjwj, **axis_kwargs) + np.log(h) if work.log else np.sum(fjwj, **axis_kwargs) * h
        work.Sk = np.concatenate((Snm1, work.Sk), axis=-1)
    if work.n == 1:
        nan = np.full_like(work.Sn, np.nan)
        return (nan, nan)
    if work.Sk.shape[-1] < 2:
        h = 4 * work.h
        n_x = indices[work.n - 1]
        fjwj_rl = work.fjwj.reshape(len(work.Sn), 2, -1)
        fjwj = fjwj_rl[..., :n_x].reshape(n_active, 2 * n_x)
        Snm2 = special.logsumexp(fjwj, **axis_kwargs) + np.log(h) if work.log else np.sum(fjwj, **axis_kwargs) * h
        work.Sk = np.concatenate((Snm2, work.Sk), axis=-1)
    Snm2 = work.Sk[..., -2]
    Snm1 = work.Sk[..., -1]
    e1 = work.eps
    if work.log:
        log_e1 = np.log(e1)
        d1 = np.real(special.logsumexp([work.Sn, Snm1 + work.pi * 1j], axis=0))
        d2 = np.real(special.logsumexp([work.Sn, Snm2 + work.pi * 1j], axis=0))
        d3 = log_e1 + np.max(np.real(work.fjwj), axis=-1)
        d4 = work.d4
        aerr = np.max([d1 ** 2 / d2, 2 * d1, d3, d4], axis=0)
        rerr = np.maximum(log_e1, aerr - np.real(work.Sn))
    else:
        d1 = np.abs(work.Sn - Snm1)
        d2 = np.abs(work.Sn - Snm2)
        d3 = e1 * np.max(np.abs(work.fjwj), axis=-1)
        d4 = work.d4
        aerr = np.max([d1 ** (np.log(d1) / np.log(d2)), d1 ** 2, d3, d4], axis=0)
        rerr = np.maximum(e1, aerr / np.abs(work.Sn))
    return (rerr, aerr.reshape(work.Sn.shape))

def _transform_integrals(a, b):
    if False:
        while True:
            i = 10
    negative = b < a
    (a[negative], b[negative]) = (b[negative], a[negative])
    abinf = np.isinf(a) & np.isinf(b)
    (a[abinf], b[abinf]) = (-1, 1)
    ainf = np.isinf(a)
    (a[ainf], b[ainf]) = (-b[ainf], -a[ainf])
    binf = np.isinf(b)
    a0 = a.copy()
    (a[binf], b[binf]) = (0, 1)
    return (a, b, a0, negative, abinf, ainf, binf)

def _tanhsinh_iv(f, a, b, log, maxfun, maxlevel, minlevel, atol, rtol, args, callback):
    if False:
        for i in range(10):
            print('nop')
    message = '`f` must be callable.'
    if not callable(f):
        raise ValueError(message)
    message = 'All elements of `a` and `b` must be real numbers.'
    (a, b) = np.broadcast_arrays(a, b)
    if np.any(np.iscomplex(a)) or np.any(np.iscomplex(b)):
        raise ValueError(message)
    message = '`log` must be True or False.'
    if log not in {True, False}:
        raise ValueError(message)
    log = bool(log)
    if atol is None:
        atol = -np.inf if log else 0
    rtol_temp = rtol if rtol is not None else 0.0
    params = np.asarray([atol, rtol_temp, 0.0])
    message = '`atol` and `rtol` must be real numbers.'
    if not np.issubdtype(params.dtype, np.floating):
        raise ValueError(message)
    if log:
        message = '`atol` and `rtol` may not be positive infinity.'
        if np.any(np.isposinf(params)):
            raise ValueError(message)
    else:
        message = '`atol` and `rtol` must be non-negative and finite.'
        if np.any(params < 0) or np.any(np.isinf(params)):
            raise ValueError(message)
    atol = params[0]
    rtol = rtol if rtol is None else params[1]
    BIGINT = float(2 ** 62)
    if maxfun is None and maxlevel is None:
        maxlevel = 10
    maxfun = BIGINT if maxfun is None else maxfun
    maxlevel = BIGINT if maxlevel is None else maxlevel
    message = '`maxfun`, `maxlevel`, and `minlevel` must be integers.'
    params = np.asarray([maxfun, maxlevel, minlevel])
    if not (np.issubdtype(params.dtype, np.number) and np.all(np.isreal(params)) and np.all(params.astype(np.int64) == params)):
        raise ValueError(message)
    message = '`maxfun`, `maxlevel`, and `minlevel` must be non-negative.'
    if np.any(params < 0):
        raise ValueError(message)
    (maxfun, maxlevel, minlevel) = params.astype(np.int64)
    minlevel = min(minlevel, maxlevel)
    if not np.iterable(args):
        args = (args,)
    if callback is not None and (not callable(callback)):
        raise ValueError('`callback` must be callable.')
    return (f, a, b, log, maxfun, maxlevel, minlevel, atol, rtol, args, callback)