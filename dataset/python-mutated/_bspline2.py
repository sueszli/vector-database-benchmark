import operator
from numpy.core.multiarray import normalize_axis_index
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.interpolate._bspline import _get_module_func, INTERVAL_MODULE, D_BOOR_MODULE, BSpline

def _get_dtype(dtype):
    if False:
        print('Hello World!')
    'Return np.complex128 for complex dtypes, np.float64 otherwise.'
    if cupy.issubdtype(dtype, cupy.complexfloating):
        return cupy.complex_
    else:
        return cupy.float_

def _as_float_array(x, check_finite=False):
    if False:
        print('Hello World!')
    'Convert the input into a C contiguous float array.\n\n    NB: Upcasts half- and single-precision floats to double precision.\n    '
    x = cupy.asarray(x)
    x = cupy.ascontiguousarray(x)
    dtyp = _get_dtype(x.dtype)
    x = x.astype(dtyp, copy=False)
    if check_finite and (not cupy.isfinite(x).all()):
        raise ValueError('Array must not contain infs or nans.')
    return x

def prod(iterable):
    if False:
        while True:
            i = 10
    '\n    Product of a sequence of numbers.\n    Faster than np.prod for short lists like array shapes, and does\n    not overflow if using Python integers.\n    '
    product = 1
    for x in iterable:
        product *= x
    return product

def _not_a_knot(x, k):
    if False:
        i = 10
        return i + 15
    'Given data x, construct the knot vector w/ not-a-knot BC.\n    cf de Boor, XIII(12).'
    x = cupy.asarray(x)
    if k % 2 != 1:
        raise ValueError('Odd degree for now only. Got %s.' % k)
    m = (k - 1) // 2
    t = x[m + 1:-m - 1]
    t = cupy.r_[(x[0],) * (k + 1), t, (x[-1],) * (k + 1)]
    return t

def _augknt(x, k):
    if False:
        print('Hello World!')
    'Construct a knot vector appropriate for the order-k interpolation.'
    return cupy.r_[(x[0],) * k, x, (x[-1],) * k]

def _periodic_knots(x, k):
    if False:
        i = 10
        return i + 15
    'Returns vector of nodes on a circle.'
    xc = cupy.copy(x)
    n = len(xc)
    if k % 2 == 0:
        dx = cupy.diff(xc)
        xc[1:-1] -= dx[:-1] / 2
    dx = cupy.diff(xc)
    t = cupy.zeros(n + 2 * k)
    t[k:-k] = xc
    for i in range(0, k):
        t[k - i - 1] = t[k - i] - dx[-(i % (n - 1)) - 1]
        t[-k + i] = t[-k + i - 1] + dx[i % (n - 1)]
    return t

def _convert_string_aliases(deriv, target_shape):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(deriv, str):
        if deriv == 'clamped':
            deriv = [(1, cupy.zeros(target_shape))]
        elif deriv == 'natural':
            deriv = [(2, cupy.zeros(target_shape))]
        else:
            raise ValueError('Unknown boundary condition : %s' % deriv)
    return deriv

def _process_deriv_spec(deriv):
    if False:
        return 10
    if deriv is not None:
        try:
            (ords, vals) = zip(*deriv)
        except TypeError as e:
            msg = 'Derivatives, `bc_type`, should be specified as a pair of iterables of pairs of (order, value).'
            raise ValueError(msg) from e
    else:
        (ords, vals) = ([], [])
    return cupy.atleast_1d(ords, vals)

def make_interp_spline(x, y, k=3, t=None, bc_type=None, axis=0, check_finite=True):
    if False:
        print('Hello World!')
    'Compute the (coefficients of) interpolating B-spline.\n\n    Parameters\n    ----------\n    x : array_like, shape (n,)\n        Abscissas.\n    y : array_like, shape (n, ...)\n        Ordinates.\n    k : int, optional\n        B-spline degree. Default is cubic, ``k = 3``.\n    t : array_like, shape (nt + k + 1,), optional.\n        Knots.\n        The number of knots needs to agree with the number of data points and\n        the number of derivatives at the edges. Specifically, ``nt - n`` must\n        equal ``len(deriv_l) + len(deriv_r)``.\n    bc_type : 2-tuple or None\n        Boundary conditions.\n        Default is None, which means choosing the boundary conditions\n        automatically. Otherwise, it must be a length-two tuple where the first\n        element (``deriv_l``) sets the boundary conditions at ``x[0]`` and\n        the second element (``deriv_r``) sets the boundary conditions at\n        ``x[-1]``. Each of these must be an iterable of pairs\n        ``(order, value)`` which gives the values of derivatives of specified\n        orders at the given edge of the interpolation interval.\n        Alternatively, the following string aliases are recognized:\n\n        * ``"clamped"``: The first derivatives at the ends are zero. This is\n           equivalent to ``bc_type=([(1, 0.0)], [(1, 0.0)])``.\n        * ``"natural"``: The second derivatives at ends are zero. This is\n          equivalent to ``bc_type=([(2, 0.0)], [(2, 0.0)])``.\n        * ``"not-a-knot"`` (default): The first and second segments are the\n          same polynomial. This is equivalent to having ``bc_type=None``.\n        * ``"periodic"``: The values and the first ``k-1`` derivatives at the\n          ends are equivalent.\n\n    axis : int, optional\n        Interpolation axis. Default is 0.\n    check_finite : bool, optional\n        Whether to check that the input arrays contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n        Default is True.\n\n    Returns\n    -------\n    b : a BSpline object of the degree ``k`` and with knots ``t``.\n\n    '
    if bc_type is None or bc_type == 'not-a-knot' or bc_type == 'periodic':
        (deriv_l, deriv_r) = (None, None)
    elif isinstance(bc_type, str):
        (deriv_l, deriv_r) = (bc_type, bc_type)
    else:
        try:
            (deriv_l, deriv_r) = bc_type
        except TypeError as e:
            raise ValueError('Unknown boundary condition: %s' % bc_type) from e
    y = cupy.asarray(y)
    axis = normalize_axis_index(axis, y.ndim)
    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)
    y = cupy.moveaxis(y, axis, 0)
    if bc_type == 'periodic' and (not cupy.allclose(y[0], y[-1], atol=1e-15)):
        raise ValueError('First and last points does not match while periodic case expected')
    if x.size != y.shape[0]:
        raise ValueError('Shapes of x {} and y {} are incompatible'.format(x.shape, y.shape))
    if (x[1:] == x[:-1]).any():
        raise ValueError('Expect x to not have duplicates')
    if x.ndim != 1 or (x[1:] < x[:-1]).any():
        raise ValueError('Expect x to be a 1D strictly increasing sequence.')
    if k == 0:
        if any((_ is not None for _ in (t, deriv_l, deriv_r))):
            raise ValueError('Too much info for k=0: t and bc_type can only be None.')
        t = cupy.r_[x, x[-1]]
        c = cupy.asarray(y)
        c = cupy.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        return BSpline.construct_fast(t, c, k, axis=axis)
    if k == 1 and t is None:
        if not (deriv_l is None and deriv_r is None):
            raise ValueError('Too much info for k=1: bc_type can only be None.')
        t = cupy.r_[x[0], x, x[-1]]
        c = cupy.asarray(y)
        c = cupy.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        return BSpline.construct_fast(t, c, k, axis=axis)
    k = operator.index(k)
    if bc_type == 'periodic' and t is not None:
        raise NotImplementedError('For periodic case t is constructed automatically and can not be passed manually')
    if t is None:
        if deriv_l is None and deriv_r is None:
            if bc_type == 'periodic':
                t = _periodic_knots(x, k)
            elif k == 2:
                t = (x[1:] + x[:-1]) / 2.0
                t = cupy.r_[(x[0],) * (k + 1), t[1:-1], (x[-1],) * (k + 1)]
            else:
                t = _not_a_knot(x, k)
        else:
            t = _augknt(x, k)
    t = _as_float_array(t, check_finite)
    if k < 0:
        raise ValueError('Expect non-negative k.')
    if t.ndim != 1 or (t[1:] < t[:-1]).any():
        raise ValueError('Expect t to be a 1-D sorted array_like.')
    if t.size < x.size + k + 1:
        raise ValueError('Got %d knots, need at least %d.' % (t.size, x.size + k + 1))
    if x[0] < t[k] or x[-1] > t[-k]:
        raise ValueError('Out of bounds w/ x = %s.' % x)
    if bc_type == 'periodic':
        return _make_periodic_spline(x, y, t, k, axis)
    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    (deriv_l_ords, deriv_l_vals) = _process_deriv_spec(deriv_l)
    nleft = deriv_l_ords.shape[0]
    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    (deriv_r_ords, deriv_r_vals) = _process_deriv_spec(deriv_r)
    nright = deriv_r_ords.shape[0]
    n = x.size
    nt = t.size - k - 1
    if nt - n != nleft + nright:
        raise ValueError('The number of derivatives at boundaries does not match: expected %s, got %s + %s' % (nt - n, nleft, nright))
    if y.size == 0:
        c = cupy.zeros((nt,) + y.shape[1:], dtype=float)
        return BSpline.construct_fast(t, c, k, axis=axis)
    matr = BSpline.design_matrix(x, t, k)
    if nleft > 0 or nright > 0:
        temp = cupy.zeros((nt,), dtype=float)
        num_c = 1
        dummy_c = cupy.empty((nt, num_c), dtype=float)
        out = cupy.empty((1, 1), dtype=dummy_c.dtype)
        d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', dummy_c)
        intervals_bc = cupy.empty(2, dtype=cupy.int64)
        interval_kernel = _get_module_func(INTERVAL_MODULE, 'find_interval')
        interval_kernel((1,), (2,), (t, cupy.r_[x[0], x[-1]], intervals_bc, k, nt, False, 2))
    if nleft > 0:
        x0 = cupy.array([x[0]], dtype=x.dtype)
        rows = cupy.zeros((nleft, nt), dtype=float)
        for (j, m) in enumerate(deriv_l_ords):
            d_boor_kernel((1,), (1,), (t, dummy_c, k, int(m), x0, intervals_bc, out, temp, num_c, 0, 1))
            left = intervals_bc[0]
            rows[j, left - k:left + 1] = temp[:k + 1]
        matr = sparse.vstack([sparse.csr_matrix(rows), matr])
    if nright > 0:
        intervals_bc[0] = intervals_bc[-1]
        x0 = cupy.array([x[-1]], dtype=x.dtype)
        rows = cupy.zeros((nright, nt), dtype=float)
        for (j, m) in enumerate(deriv_r_ords):
            d_boor_kernel((1,), (1,), (t, dummy_c, k, int(m), x0, intervals_bc, out, temp, num_c, 0, 1))
            left = intervals_bc[0]
            rows[j, left - k:left + 1] = temp[:k + 1]
        matr = sparse.vstack([matr, sparse.csr_matrix(rows)])
    extradim = prod(y.shape[1:])
    rhs = cupy.empty((nt, extradim), dtype=y.dtype)
    if nleft > 0:
        rhs[:nleft] = deriv_l_vals.reshape(-1, extradim)
    rhs[nleft:nt - nright] = y.reshape(-1, extradim)
    if nright > 0:
        rhs[nt - nright:] = deriv_r_vals.reshape(-1, extradim)
    if cupy.issubdtype(rhs.dtype, cupy.complexfloating):
        coef = spsolve(matr, rhs.real) + spsolve(matr, rhs.imag) * 1j
    else:
        coef = spsolve(matr, rhs)
    coef = cupy.ascontiguousarray(coef.reshape((nt,) + y.shape[1:]))
    return BSpline(t, coef, k)

def _make_interp_spline_full_matrix(x, y, k, t, bc_type):
    if False:
        return 10
    ' Construct the interpolating spline spl(x) = y with *full* linalg.\n\n        Only useful for testing, do not call directly!\n        This version is O(N**2) in memory and O(N**3) in flop count.\n    '
    if bc_type is None or bc_type == 'not-a-knot':
        (deriv_l, deriv_r) = (None, None)
    elif isinstance(bc_type, str):
        (deriv_l, deriv_r) = (bc_type, bc_type)
    else:
        try:
            (deriv_l, deriv_r) = bc_type
        except TypeError as e:
            raise ValueError('Unknown boundary condition: %s' % bc_type) from e
    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    (deriv_l_ords, deriv_l_vals) = _process_deriv_spec(deriv_l)
    nleft = deriv_l_ords.shape[0]
    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    (deriv_r_ords, deriv_r_vals) = _process_deriv_spec(deriv_r)
    nright = deriv_r_ords.shape[0]
    n = x.size
    nt = t.size - k - 1
    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    (deriv_l_ords, deriv_l_vals) = _process_deriv_spec(deriv_l)
    nleft = deriv_l_ords.shape[0]
    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    (deriv_r_ords, deriv_r_vals) = _process_deriv_spec(deriv_r)
    nright = deriv_r_ords.shape[0]
    n = x.size
    nt = t.size - k - 1
    assert nt - n == nleft + nright
    intervals = cupy.empty_like(x, dtype=cupy.int64)
    interval_kernel = _get_module_func(INTERVAL_MODULE, 'find_interval')
    interval_kernel(((x.shape[0] + 128 - 1) // 128,), (128,), (t, x, intervals, k, nt, False, x.shape[0]))
    dummy_c = cupy.empty((nt, 1), dtype=float)
    out = cupy.empty((len(x), prod(dummy_c.shape[1:])), dtype=dummy_c.dtype)
    num_c = prod(dummy_c.shape[1:])
    temp = cupy.empty(x.shape[0] * (2 * k + 1))
    d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', dummy_c)
    d_boor_kernel(((x.shape[0] + 128 - 1) // 128,), (128,), (t, dummy_c, k, 0, x, intervals, out, temp, num_c, 0, x.shape[0]))
    A = cupy.zeros((nt, nt), dtype=float)
    offset = nleft
    for j in range(len(x)):
        left = intervals[j]
        A[j + offset, left - k:left + 1] = temp[j * (2 * k + 1):j * (2 * k + 1) + k + 1]
    intervals_bc = cupy.empty(1, dtype=cupy.int64)
    if nleft > 0:
        intervals_bc[0] = intervals[0]
        x0 = cupy.array([x[0]], dtype=x.dtype)
        for (j, m) in enumerate(deriv_l_ords):
            d_boor_kernel((1,), (1,), (t, dummy_c, k, int(m), x0, intervals_bc, out, temp, num_c, 0, 1))
            left = intervals_bc[0]
            A[j, left - k:left + 1] = temp[:k + 1]
    if nright > 0:
        intervals_bc[0] = intervals[-1]
        x0 = cupy.array([x[-1]], dtype=x.dtype)
        for (j, m) in enumerate(deriv_r_ords):
            d_boor_kernel((1,), (1,), (t, dummy_c, k, int(m), x0, intervals_bc, out, temp, num_c, 0, 1))
            left = intervals_bc[0]
            row = nleft + len(x) + j
            A[row, left - k:left + 1] = temp[:k + 1]
    extradim = prod(y.shape[1:])
    rhs = cupy.empty((nt, extradim), dtype=y.dtype)
    if nleft > 0:
        rhs[:nleft] = deriv_l_vals.reshape(-1, extradim)
    rhs[nleft:nt - nright] = y.reshape(-1, extradim)
    if nright > 0:
        rhs[nt - nright:] = deriv_r_vals.reshape(-1, extradim)
    from cupy.linalg import solve
    coef = solve(A, rhs)
    coef = cupy.ascontiguousarray(coef.reshape((nt,) + y.shape[1:]))
    return BSpline(t, coef, k)

def _make_periodic_spline(x, y, t, k, axis):
    if False:
        for i in range(10):
            print('nop')
    n = x.size
    matr = BSpline.design_matrix(x, t, k)
    temp = cupy.zeros(2 * (2 * k + 1), dtype=float)
    num_c = 1
    dummy_c = cupy.empty((t.size - k - 1, num_c), dtype=float)
    out = cupy.empty((2, 1), dtype=dummy_c.dtype)
    d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', dummy_c)
    x0 = cupy.r_[x[0], x[-1]]
    intervals_bc = cupy.array([k, n + k - 1], dtype=cupy.int64)
    rows = cupy.zeros((k - 1, n + k - 1), dtype=float)
    for m in range(k - 1):
        d_boor_kernel((1,), (2,), (t, dummy_c, k, m + 1, x0, intervals_bc, out, temp, num_c, 0, 2))
        rows[m, :k + 1] = temp[:k + 1]
        rows[m, -k:] -= temp[2 * k + 1:2 * k + 1 + k + 1][:-1]
    matr_csr = sparse.vstack([sparse.csr_matrix(rows), matr])
    extradim = prod(y.shape[1:])
    rhs = cupy.empty((n + k - 1, extradim), dtype=float)
    rhs[:k - 1, :] = 0
    rhs[k - 1:, :] = y.reshape(n, 0) if y.size == 0 else y.reshape((-1, extradim))
    coef = spsolve(matr_csr, rhs)
    coef = cupy.ascontiguousarray(coef.reshape((n + k - 1,) + y.shape[1:]))
    return BSpline.construct_fast(t, coef, k, extrapolate='periodic', axis=axis)