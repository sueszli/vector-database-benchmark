import sys
import copy
import heapq
import collections
import functools
import numpy as np
from scipy._lib._util import MapWrapper, _FunctionWrapper

class LRUDict(collections.OrderedDict):

    def __init__(self, max_size):
        if False:
            print('Hello World!')
        self.__max_size = max_size

    def __setitem__(self, key, value):
        if False:
            return 10
        existing_key = key in self
        super().__setitem__(key, value)
        if existing_key:
            self.move_to_end(key)
        elif len(self) > self.__max_size:
            self.popitem(last=False)

    def update(self, other):
        if False:
            return 10
        raise NotImplementedError()

class SemiInfiniteFunc:
    """
    Argument transform from (start, +-oo) to (0, 1)
    """

    def __init__(self, func, start, infty):
        if False:
            i = 10
            return i + 15
        self._func = func
        self._start = start
        self._sgn = -1 if infty < 0 else 1
        self._tmin = sys.float_info.min ** 0.5

    def get_t(self, x):
        if False:
            return 10
        z = self._sgn * (x - self._start) + 1
        if z == 0:
            return np.inf
        return 1 / z

    def __call__(self, t):
        if False:
            print('Hello World!')
        if t < self._tmin:
            return 0.0
        else:
            x = self._start + self._sgn * (1 - t) / t
            f = self._func(x)
            return self._sgn * (f / t) / t

class DoubleInfiniteFunc:
    """
    Argument transform from (-oo, oo) to (-1, 1)
    """

    def __init__(self, func):
        if False:
            i = 10
            return i + 15
        self._func = func
        self._tmin = sys.float_info.min ** 0.5

    def get_t(self, x):
        if False:
            while True:
                i = 10
        s = -1 if x < 0 else 1
        return s / (abs(x) + 1)

    def __call__(self, t):
        if False:
            while True:
                i = 10
        if abs(t) < self._tmin:
            return 0.0
        else:
            x = (1 - abs(t)) / t
            f = self._func(x)
            return f / t / t

def _max_norm(x):
    if False:
        return 10
    return np.amax(abs(x))

def _get_sizeof(obj):
    if False:
        for i in range(10):
            print('nop')
    try:
        return sys.getsizeof(obj)
    except TypeError:
        if hasattr(obj, '__sizeof__'):
            return int(obj.__sizeof__())
        return 64

class _Bunch:

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.__keys = kwargs.keys()
        self.__dict__.update(**kwargs)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '_Bunch({})'.format(', '.join((f'{k}={repr(self.__dict__[k])}' for k in self.__keys)))

def quad_vec(f, a, b, epsabs=1e-200, epsrel=1e-08, norm='2', cache_size=100000000.0, limit=10000, workers=1, points=None, quadrature=None, full_output=False, *, args=()):
    if False:
        for i in range(10):
            print('nop')
    'Adaptive integration of a vector-valued function.\n\n    Parameters\n    ----------\n    f : callable\n        Vector-valued function f(x) to integrate.\n    a : float\n        Initial point.\n    b : float\n        Final point.\n    epsabs : float, optional\n        Absolute tolerance.\n    epsrel : float, optional\n        Relative tolerance.\n    norm : {\'max\', \'2\'}, optional\n        Vector norm to use for error estimation.\n    cache_size : int, optional\n        Number of bytes to use for memoization.\n    limit : float or int, optional\n        An upper bound on the number of subintervals used in the adaptive\n        algorithm.\n    workers : int or map-like callable, optional\n        If `workers` is an integer, part of the computation is done in\n        parallel subdivided to this many tasks (using\n        :class:`python:multiprocessing.pool.Pool`).\n        Supply `-1` to use all cores available to the Process.\n        Alternatively, supply a map-like callable, such as\n        :meth:`python:multiprocessing.pool.Pool.map` for evaluating the\n        population in parallel.\n        This evaluation is carried out as ``workers(func, iterable)``.\n    points : list, optional\n        List of additional breakpoints.\n    quadrature : {\'gk21\', \'gk15\', \'trapezoid\'}, optional\n        Quadrature rule to use on subintervals.\n        Options: \'gk21\' (Gauss-Kronrod 21-point rule),\n        \'gk15\' (Gauss-Kronrod 15-point rule),\n        \'trapezoid\' (composite trapezoid rule).\n        Default: \'gk21\' for finite intervals and \'gk15\' for (semi-)infinite\n    full_output : bool, optional\n        Return an additional ``info`` dictionary.\n    args : tuple, optional\n        Extra arguments to pass to function, if any.\n\n        .. versionadded:: 1.8.0\n\n    Returns\n    -------\n    res : {float, array-like}\n        Estimate for the result\n    err : float\n        Error estimate for the result in the given norm\n    info : dict\n        Returned only when ``full_output=True``.\n        Info dictionary. Is an object with the attributes:\n\n            success : bool\n                Whether integration reached target precision.\n            status : int\n                Indicator for convergence, success (0),\n                failure (1), and failure due to rounding error (2).\n            neval : int\n                Number of function evaluations.\n            intervals : ndarray, shape (num_intervals, 2)\n                Start and end points of subdivision intervals.\n            integrals : ndarray, shape (num_intervals, ...)\n                Integral for each interval.\n                Note that at most ``cache_size`` values are recorded,\n                and the array may contains *nan* for missing items.\n            errors : ndarray, shape (num_intervals,)\n                Estimated integration error for each interval.\n\n    Notes\n    -----\n    The algorithm mainly follows the implementation of QUADPACK\'s\n    DQAG* algorithms, implementing global error control and adaptive\n    subdivision.\n\n    The algorithm here has some differences to the QUADPACK approach:\n\n    Instead of subdividing one interval at a time, the algorithm\n    subdivides N intervals with largest errors at once. This enables\n    (partial) parallelization of the integration.\n\n    The logic of subdividing "next largest" intervals first is then\n    not implemented, and we rely on the above extension to avoid\n    concentrating on "small" intervals only.\n\n    The Wynn epsilon table extrapolation is not used (QUADPACK uses it\n    for infinite intervals). This is because the algorithm here is\n    supposed to work on vector-valued functions, in an user-specified\n    norm, and the extension of the epsilon algorithm to this case does\n    not appear to be widely agreed. For max-norm, using elementwise\n    Wynn epsilon could be possible, but we do not do this here with\n    the hope that the epsilon extrapolation is mainly useful in\n    special cases.\n\n    References\n    ----------\n    [1] R. Piessens, E. de Doncker, QUADPACK (1983).\n\n    Examples\n    --------\n    We can compute integrations of a vector-valued function:\n\n    >>> from scipy.integrate import quad_vec\n    >>> import numpy as np\n    >>> import matplotlib.pyplot as plt\n    >>> alpha = np.linspace(0.0, 2.0, num=30)\n    >>> f = lambda x: x**alpha\n    >>> x0, x1 = 0, 2\n    >>> y, err = quad_vec(f, x0, x1)\n    >>> plt.plot(alpha, y)\n    >>> plt.xlabel(r"$\\alpha$")\n    >>> plt.ylabel(r"$\\int_{0}^{2} x^\\alpha dx$")\n    >>> plt.show()\n\n    '
    a = float(a)
    b = float(b)
    if args:
        if not isinstance(args, tuple):
            args = (args,)
        f = _FunctionWrapper(f, args)
    kwargs = dict(epsabs=epsabs, epsrel=epsrel, norm=norm, cache_size=cache_size, limit=limit, workers=workers, points=points, quadrature='gk15' if quadrature is None else quadrature, full_output=full_output)
    if np.isfinite(a) and np.isinf(b):
        f2 = SemiInfiniteFunc(f, start=a, infty=b)
        if points is not None:
            kwargs['points'] = tuple((f2.get_t(xp) for xp in points))
        return quad_vec(f2, 0, 1, **kwargs)
    elif np.isfinite(b) and np.isinf(a):
        f2 = SemiInfiniteFunc(f, start=b, infty=a)
        if points is not None:
            kwargs['points'] = tuple((f2.get_t(xp) for xp in points))
        res = quad_vec(f2, 0, 1, **kwargs)
        return (-res[0],) + res[1:]
    elif np.isinf(a) and np.isinf(b):
        sgn = -1 if b < a else 1
        f2 = DoubleInfiniteFunc(f)
        if points is not None:
            kwargs['points'] = (0,) + tuple((f2.get_t(xp) for xp in points))
        else:
            kwargs['points'] = (0,)
        if a != b:
            res = quad_vec(f2, -1, 1, **kwargs)
        else:
            res = quad_vec(f2, 1, 1, **kwargs)
        return (res[0] * sgn,) + res[1:]
    elif not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError(f'invalid integration bounds a={a}, b={b}')
    norm_funcs = {None: _max_norm, 'max': _max_norm, '2': np.linalg.norm}
    if callable(norm):
        norm_func = norm
    else:
        norm_func = norm_funcs[norm]
    parallel_count = 128
    min_intervals = 2
    try:
        _quadrature = {None: _quadrature_gk21, 'gk21': _quadrature_gk21, 'gk15': _quadrature_gk15, 'trapz': _quadrature_trapezoid, 'trapezoid': _quadrature_trapezoid}[quadrature]
    except KeyError as e:
        raise ValueError(f'unknown quadrature {quadrature!r}') from e
    if points is None:
        initial_intervals = [(a, b)]
    else:
        prev = a
        initial_intervals = []
        for p in sorted(points):
            p = float(p)
            if not a < p < b or p == prev:
                continue
            initial_intervals.append((prev, p))
            prev = p
        initial_intervals.append((prev, b))
    global_integral = None
    global_error = None
    rounding_error = None
    interval_cache = None
    intervals = []
    neval = 0
    for (x1, x2) in initial_intervals:
        (ig, err, rnd) = _quadrature(x1, x2, f, norm_func)
        neval += _quadrature.num_eval
        if global_integral is None:
            if isinstance(ig, (float, complex)):
                if norm_func in (_max_norm, np.linalg.norm):
                    norm_func = abs
            global_integral = ig
            global_error = float(err)
            rounding_error = float(rnd)
            cache_count = cache_size // _get_sizeof(ig)
            interval_cache = LRUDict(cache_count)
        else:
            global_integral += ig
            global_error += err
            rounding_error += rnd
        interval_cache[x1, x2] = copy.copy(ig)
        intervals.append((-err, x1, x2))
    heapq.heapify(intervals)
    CONVERGED = 0
    NOT_CONVERGED = 1
    ROUNDING_ERROR = 2
    NOT_A_NUMBER = 3
    status_msg = {CONVERGED: 'Target precision reached.', NOT_CONVERGED: 'Target precision not reached.', ROUNDING_ERROR: 'Target precision could not be reached due to rounding error.', NOT_A_NUMBER: 'Non-finite values encountered.'}
    with MapWrapper(workers) as mapwrapper:
        ier = NOT_CONVERGED
        while intervals and len(intervals) < limit:
            tol = max(epsabs, epsrel * norm_func(global_integral))
            to_process = []
            err_sum = 0
            for j in range(parallel_count):
                if not intervals:
                    break
                if j > 0 and err_sum > global_error - tol / 8:
                    break
                interval = heapq.heappop(intervals)
                (neg_old_err, a, b) = interval
                old_int = interval_cache.pop((a, b), None)
                to_process.append(((-neg_old_err, a, b, old_int), f, norm_func, _quadrature))
                err_sum += -neg_old_err
            for (dint, derr, dround_err, subint, dneval) in mapwrapper(_subdivide_interval, to_process):
                neval += dneval
                global_integral += dint
                global_error += derr
                rounding_error += dround_err
                for x in subint:
                    (x1, x2, ig, err) = x
                    interval_cache[x1, x2] = ig
                    heapq.heappush(intervals, (-err, x1, x2))
            if len(intervals) >= min_intervals:
                tol = max(epsabs, epsrel * norm_func(global_integral))
                if global_error < tol / 8:
                    ier = CONVERGED
                    break
                if global_error < rounding_error:
                    ier = ROUNDING_ERROR
                    break
            if not (np.isfinite(global_error) and np.isfinite(rounding_error)):
                ier = NOT_A_NUMBER
                break
    res = global_integral
    err = global_error + rounding_error
    if full_output:
        res_arr = np.asarray(res)
        dummy = np.full(res_arr.shape, np.nan, dtype=res_arr.dtype)
        integrals = np.array([interval_cache.get((z[1], z[2]), dummy) for z in intervals], dtype=res_arr.dtype)
        errors = np.array([-z[0] for z in intervals])
        intervals = np.array([[z[1], z[2]] for z in intervals])
        info = _Bunch(neval=neval, success=ier == CONVERGED, status=ier, message=status_msg[ier], intervals=intervals, integrals=integrals, errors=errors)
        return (res, err, info)
    else:
        return (res, err)

def _subdivide_interval(args):
    if False:
        i = 10
        return i + 15
    (interval, f, norm_func, _quadrature) = args
    (old_err, a, b, old_int) = interval
    c = 0.5 * (a + b)
    if getattr(_quadrature, 'cache_size', 0) > 0:
        f = functools.lru_cache(_quadrature.cache_size)(f)
    (s1, err1, round1) = _quadrature(a, c, f, norm_func)
    dneval = _quadrature.num_eval
    (s2, err2, round2) = _quadrature(c, b, f, norm_func)
    dneval += _quadrature.num_eval
    if old_int is None:
        (old_int, _, _) = _quadrature(a, b, f, norm_func)
        dneval += _quadrature.num_eval
    if getattr(_quadrature, 'cache_size', 0) > 0:
        dneval = f.cache_info().misses
    dint = s1 + s2 - old_int
    derr = err1 + err2 - old_err
    dround_err = round1 + round2
    subintervals = ((a, c, s1, err1), (c, b, s2, err2))
    return (dint, derr, dround_err, subintervals, dneval)

def _quadrature_trapezoid(x1, x2, f, norm_func):
    if False:
        return 10
    '\n    Composite trapezoid quadrature\n    '
    x3 = 0.5 * (x1 + x2)
    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)
    s2 = 0.25 * (x2 - x1) * (f1 + 2 * f3 + f2)
    round_err = 0.25 * abs(x2 - x1) * (float(norm_func(f1)) + 2 * float(norm_func(f3)) + float(norm_func(f2))) * 2e-16
    s1 = 0.5 * (x2 - x1) * (f1 + f2)
    err = 1 / 3 * float(norm_func(s1 - s2))
    return (s2, err, round_err)
_quadrature_trapezoid.cache_size = 3 * 3
_quadrature_trapezoid.num_eval = 3

def _quadrature_gk(a, b, f, norm_func, x, w, v):
    if False:
        i = 10
        return i + 15
    '\n    Generic Gauss-Kronrod quadrature\n    '
    fv = [0.0] * len(x)
    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    s_k = 0.0
    s_k_abs = 0.0
    for i in range(len(x)):
        ff = f(c + h * x[i])
        fv[i] = ff
        vv = v[i]
        s_k += vv * ff
        s_k_abs += vv * abs(ff)
    s_g = 0.0
    for i in range(len(w)):
        s_g += w[i] * fv[2 * i + 1]
    s_k_dabs = 0.0
    y0 = s_k / 2.0
    for i in range(len(x)):
        s_k_dabs += v[i] * abs(fv[i] - y0)
    err = float(norm_func((s_k - s_g) * h))
    dabs = float(norm_func(s_k_dabs * h))
    if dabs != 0 and err != 0:
        err = dabs * min(1.0, (200 * err / dabs) ** 1.5)
    eps = sys.float_info.epsilon
    round_err = float(norm_func(50 * eps * h * s_k_abs))
    if round_err > sys.float_info.min:
        err = max(err, round_err)
    return (h * s_k, err, round_err)

def _quadrature_gk21(a, b, f, norm_func):
    if False:
        print('Hello World!')
    '\n    Gauss-Kronrod 21 quadrature with error estimate\n    '
    x = (0.9956571630258081, 0.9739065285171717, 0.9301574913557082, 0.8650633666889845, 0.7808177265864169, 0.6794095682990244, 0.5627571346686047, 0.4333953941292472, 0.2943928627014602, 0.14887433898163122, 0, -0.14887433898163122, -0.2943928627014602, -0.4333953941292472, -0.5627571346686047, -0.6794095682990244, -0.7808177265864169, -0.8650633666889845, -0.9301574913557082, -0.9739065285171717, -0.9956571630258081)
    w = (0.06667134430868814, 0.1494513491505806, 0.21908636251598204, 0.26926671930999635, 0.29552422471475287, 0.29552422471475287, 0.26926671930999635, 0.21908636251598204, 0.1494513491505806, 0.06667134430868814)
    v = (0.011694638867371874, 0.032558162307964725, 0.054755896574351995, 0.07503967481091996, 0.0931254545836976, 0.10938715880229764, 0.12349197626206584, 0.13470921731147334, 0.14277593857706009, 0.14773910490133849, 0.1494455540029169, 0.14773910490133849, 0.14277593857706009, 0.13470921731147334, 0.12349197626206584, 0.10938715880229764, 0.0931254545836976, 0.07503967481091996, 0.054755896574351995, 0.032558162307964725, 0.011694638867371874)
    return _quadrature_gk(a, b, f, norm_func, x, w, v)
_quadrature_gk21.num_eval = 21

def _quadrature_gk15(a, b, f, norm_func):
    if False:
        i = 10
        return i + 15
    '\n    Gauss-Kronrod 15 quadrature with error estimate\n    '
    x = (0.9914553711208126, 0.9491079123427585, 0.8648644233597691, 0.7415311855993945, 0.5860872354676911, 0.4058451513773972, 0.20778495500789848, 0.0, -0.20778495500789848, -0.4058451513773972, -0.5860872354676911, -0.7415311855993945, -0.8648644233597691, -0.9491079123427585, -0.9914553711208126)
    w = (0.1294849661688697, 0.27970539148927664, 0.3818300505051189, 0.4179591836734694, 0.3818300505051189, 0.27970539148927664, 0.1294849661688697)
    v = (0.022935322010529224, 0.06309209262997856, 0.10479001032225019, 0.14065325971552592, 0.1690047266392679, 0.19035057806478542, 0.20443294007529889, 0.20948214108472782, 0.20443294007529889, 0.19035057806478542, 0.1690047266392679, 0.14065325971552592, 0.10479001032225019, 0.06309209262997856, 0.022935322010529224)
    return _quadrature_gk(a, b, f, norm_func, x, w, v)
_quadrature_gk15.num_eval = 15