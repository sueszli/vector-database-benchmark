import math
import cupy
from cupy.linalg import lstsq
from cupyx.scipy.ndimage import convolve1d
from ._arraytools import axis_slice

def float_factorial(n: int) -> float:
    if False:
        i = 10
        return i + 15
    'Compute the factorial and return as a float\n\n    Returns infinity when result is too large for a double\n    '
    return float(math.factorial(n)) if n < 171 else cupy.inf

def _polyval(p, x):
    if False:
        i = 10
        return i + 15
    p = cupy.asarray(p)
    x = cupy.asanyarray(x)
    y = cupy.zeros_like(x)
    for pv in p:
        y = y * x + pv
    return y

def savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0, pos=None, use='conv'):
    if False:
        for i in range(10):
            print('nop')
    "Compute the coefficients for a 1-D Savitzky-Golay FIR filter.\n\n    Parameters\n    ----------\n    window_length : int\n        The length of the filter window (i.e., the number of coefficients).\n    polyorder : int\n        The order of the polynomial used to fit the samples.\n        `polyorder` must be less than `window_length`.\n    deriv : int, optional\n        The order of the derivative to compute. This must be a\n        nonnegative integer. The default is 0, which means to filter\n        the data without differentiating.\n    delta : float, optional\n        The spacing of the samples to which the filter will be applied.\n        This is only used if deriv > 0.\n    pos : int or None, optional\n        If pos is not None, it specifies evaluation position within the\n        window. The default is the middle of the window.\n    use : str, optional\n        Either 'conv' or 'dot'. This argument chooses the order of the\n        coefficients. The default is 'conv', which means that the\n        coefficients are ordered to be used in a convolution. With\n        use='dot', the order is reversed, so the filter is applied by\n        dotting the coefficients with the data set.\n\n    Returns\n    -------\n    coeffs : 1-D ndarray\n        The filter coefficients.\n\n    See Also\n    --------\n    scipy.signal.savgol_coeffs\n    savgol_filter\n\n\n    References\n    ----------\n    A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by\n    Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8),\n    pp 1627-1639.\n    Jianwen Luo, Kui Ying, and Jing Bai. 2005. Savitzky-Golay smoothing and\n    differentiation filter for even number data. Signal Process.\n    85, 7 (July 2005), 1429-1434.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.signal import savgol_coeffs\n    >>> savgol_coeffs(5, 2)\n    array([-0.08571429,  0.34285714,  0.48571429,  0.34285714, -0.08571429])\n    >>> savgol_coeffs(5, 2, deriv=1)\n    array([ 2.00000000e-01,  1.00000000e-01,  2.07548111e-16, -1.00000000e-01,\n           -2.00000000e-01])\n\n    Note that use='dot' simply reverses the coefficients.\n\n    >>> savgol_coeffs(5, 2, pos=3)\n    array([ 0.25714286,  0.37142857,  0.34285714,  0.17142857, -0.14285714])\n    >>> savgol_coeffs(5, 2, pos=3, use='dot')\n    array([-0.14285714,  0.17142857,  0.34285714,  0.37142857,  0.25714286])\n    >>> savgol_coeffs(4, 2, pos=3, deriv=1, use='dot')\n    array([0.45,  -0.85,  -0.65,  1.05])\n\n    `x` contains data from the parabola x = t**2, sampled at\n    t = -1, 0, 1, 2, 3.  `c` holds the coefficients that will compute the\n    derivative at the last position.  When dotted with `x` the result should\n    be 6.\n\n    >>> x = np.array([1, 0, 1, 4, 9])\n    >>> c = savgol_coeffs(5, 2, pos=4, deriv=1, use='dot')\n    >>> c.dot(x)\n    6.0\n    "
    if polyorder >= window_length:
        raise ValueError('polyorder must be less than window_length.')
    (halflen, rem) = divmod(window_length, 2)
    if pos is None:
        if rem == 0:
            pos = halflen - 0.5
        else:
            pos = halflen
    if not 0 <= pos < window_length:
        raise ValueError('pos must be nonnegative and less than window_length.')
    if use not in ['conv', 'dot']:
        raise ValueError("`use` must be 'conv' or 'dot'")
    if deriv > polyorder:
        coeffs = cupy.zeros(window_length)
        return coeffs
    x = cupy.arange(-pos, window_length - pos, dtype=float)
    if use == 'conv':
        x = x[::-1]
    order = cupy.arange(polyorder + 1).reshape(-1, 1)
    A = x ** order
    y = cupy.zeros(polyorder + 1)
    y[deriv] = float_factorial(deriv) / delta ** deriv
    (coeffs, _, _, _) = lstsq(A, y, rcond=None)
    return coeffs

def _polyder(p, m):
    if False:
        print('Hello World!')
    "Differentiate polynomials represented with coefficients.\n\n    p must be a 1-D or 2-D array.  In the 2-D case, each column gives\n    the coefficients of a polynomial; the first row holds the coefficients\n    associated with the highest power. m must be a nonnegative integer.\n    (numpy.polyder doesn't handle the 2-D case.)\n    "
    if m == 0:
        result = p
    else:
        n = len(p)
        if n <= m:
            result = cupy.zeros_like(p[:1, ...])
        else:
            dp = p[:-m].copy()
            for k in range(m):
                rng = cupy.arange(n - k - 1, m - k - 1, -1)
                dp *= rng.reshape((n - m,) + (1,) * (p.ndim - 1))
            result = dp
    return result

def _fit_edge(x, window_start, window_stop, interp_start, interp_stop, axis, polyorder, deriv, delta, y):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given an N-d array `x` and the specification of a slice of `x` from\n    `window_start` to `window_stop` along `axis`, create an interpolating\n    polynomial of each 1-D slice, and evaluate that polynomial in the slice\n    from `interp_start` to `interp_stop`. Put the result into the\n    corresponding slice of `y`.\n    '
    x_edge = axis_slice(x, start=window_start, stop=window_stop, axis=axis)
    if axis == 0 or axis == -x.ndim:
        xx_edge = x_edge
        swapped = False
    else:
        xx_edge = x_edge.swapaxes(axis, 0)
        swapped = True
    xx_edge = xx_edge.reshape(xx_edge.shape[0], -1)
    poly_coeffs = cupy.polyfit(cupy.arange(0, window_stop - window_start), xx_edge, polyorder)
    if deriv > 0:
        poly_coeffs = _polyder(poly_coeffs, deriv)
    i = cupy.arange(interp_start - window_start, interp_stop - window_start)
    values = _polyval(poly_coeffs, i.reshape(-1, 1)) / delta ** deriv
    shp = list(y.shape)
    (shp[0], shp[axis]) = (shp[axis], shp[0])
    values = values.reshape(interp_stop - interp_start, *shp[1:])
    if swapped:
        values = values.swapaxes(0, axis)
    y_edge = axis_slice(y, start=interp_start, stop=interp_stop, axis=axis)
    y_edge[...] = values

def _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y):
    if False:
        i = 10
        return i + 15
    '\n    Use polynomial interpolation of x at the low and high ends of the axis\n    to fill in the halflen values in y.\n\n    This function just calls _fit_edge twice, once for each end of the axis.\n    '
    halflen = window_length // 2
    _fit_edge(x, 0, window_length, 0, halflen, axis, polyorder, deriv, delta, y)
    n = x.shape[axis]
    _fit_edge(x, n - window_length, n, n - halflen, n, axis, polyorder, deriv, delta, y)

def savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0):
    if False:
        return 10
    " Apply a Savitzky-Golay filter to an array.\n\n    This is a 1-D filter. If `x`  has dimension greater than 1, `axis`\n    determines the axis along which the filter is applied.\n\n    Parameters\n    ----------\n    x : array_like\n        The data to be filtered. If `x` is not a single or double precision\n        floating point array, it will be converted to type ``numpy.float64``\n        before filtering.\n    window_length : int\n        The length of the filter window (i.e., the number of coefficients).\n        If `mode` is 'interp', `window_length` must be less than or equal\n        to the size of `x`.\n    polyorder : int\n        The order of the polynomial used to fit the samples.\n        `polyorder` must be less than `window_length`.\n    deriv : int, optional\n        The order of the derivative to compute. This must be a\n        nonnegative integer. The default is 0, which means to filter\n        the data without differentiating.\n    delta : float, optional\n        The spacing of the samples to which the filter will be applied.\n        This is only used if deriv > 0. Default is 1.0.\n    axis : int, optional\n        The axis of the array `x` along which the filter is to be applied.\n        Default is -1.\n    mode : str, optional\n        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. This\n        determines the type of extension to use for the padded signal to\n        which the filter is applied.  When `mode` is 'constant', the padding\n        value is given by `cval`.  See the Notes for more details on 'mirror',\n        'constant', 'wrap', and 'nearest'.\n        When the 'interp' mode is selected (the default), no extension\n        is used.  Instead, a degree `polyorder` polynomial is fit to the\n        last `window_length` values of the edges, and this polynomial is\n        used to evaluate the last `window_length // 2` output values.\n    cval : scalar, optional\n        Value to fill past the edges of the input if `mode` is 'constant'.\n        Default is 0.0.\n\n    Returns\n    -------\n    y : ndarray, same shape as `x`\n        The filtered data.\n\n    See Also\n    --------\n    savgol_coeffs\n\n    Notes\n    -----\n    Details on the `mode` options:\n\n        'mirror':\n            Repeats the values at the edges in reverse order. The value\n            closest to the edge is not included.\n        'nearest':\n            The extension contains the nearest input value.\n        'constant':\n            The extension contains the value given by the `cval` argument.\n        'wrap':\n            The extension contains the values from the other end of the array.\n\n    For example, if the input is [1, 2, 3, 4, 5, 6, 7, 8], and\n    `window_length` is 7, the following shows the extended data for\n    the various `mode` options (assuming `cval` is 0)::\n\n        mode       |   Ext   |         Input          |   Ext\n        -----------+---------+------------------------+---------\n        'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5\n        'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8\n        'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0\n        'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3\n\n    .. versionadded:: 0.14.0\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.signal import savgol_filter\n    >>> np.set_printoptions(precision=2)  # For compact display.\n    >>> x = np.array([2, 2, 5, 2, 1, 0, 1, 4, 9])\n\n    Filter with a window length of 5 and a degree 2 polynomial.  Use\n    the defaults for all other parameters.\n\n    >>> savgol_filter(x, 5, 2)\n    array([1.66, 3.17, 3.54, 2.86, 0.66, 0.17, 1.  , 4.  , 9.  ])\n\n    Note that the last five values in x are samples of a parabola, so\n    when mode='interp' (the default) is used with polyorder=2, the last\n    three values are unchanged. Compare that to, for example,\n    `mode='nearest'`:\n\n    >>> savgol_filter(x, 5, 2, mode='nearest')\n    array([1.74, 3.03, 3.54, 2.86, 0.66, 0.17, 1.  , 4.6 , 7.97])\n\n    "
    if mode not in ['mirror', 'constant', 'nearest', 'interp', 'wrap']:
        raise ValueError("mode must be 'mirror', 'constant', 'nearest' 'wrap' or 'interp'.")
    x = cupy.asarray(x)
    if x.dtype != cupy.float64 and x.dtype != cupy.float32:
        x = x.astype(cupy.float64)
    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)
    if mode == 'interp':
        if window_length > x.shape[axis]:
            raise ValueError("If mode is 'interp', window_length must be less than or equal to the size of x.")
        y = convolve1d(x, coeffs, axis=axis, mode='constant')
        _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y)
    else:
        y = convolve1d(x, coeffs, axis=axis, mode=mode, cval=cval)
    return y