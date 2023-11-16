""" Basic functions for manipulating 2d arrays

"""
from __future__ import division, absolute_import, print_function
import functools
from numpy.core.numeric import absolute, asanyarray, arange, zeros, greater_equal, multiply, ones, asarray, where, int8, int16, int32, int64, empty, promote_types, diagonal, nonzero
from numpy.core.overrides import set_module
from numpy.core import overrides
from numpy.core import iinfo, transpose
__all__ = ['diag', 'diagflat', 'eye', 'fliplr', 'flipud', 'tri', 'triu', 'tril', 'vander', 'histogram2d', 'mask_indices', 'tril_indices', 'tril_indices_from', 'triu_indices', 'triu_indices_from']
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy')
i1 = iinfo(int8)
i2 = iinfo(int16)
i4 = iinfo(int32)

def _min_int(low, high):
    if False:
        print('Hello World!')
    ' get small int that fits the range '
    if high <= i1.max and low >= i1.min:
        return int8
    if high <= i2.max and low >= i2.min:
        return int16
    if high <= i4.max and low >= i4.min:
        return int32
    return int64

def _flip_dispatcher(m):
    if False:
        return 10
    return (m,)

@array_function_dispatch(_flip_dispatcher)
def fliplr(m):
    if False:
        while True:
            i = 10
    '\n    Flip array in the left/right direction.\n\n    Flip the entries in each row in the left/right direction.\n    Columns are preserved, but appear in a different order than before.\n\n    Parameters\n    ----------\n    m : array_like\n        Input array, must be at least 2-D.\n\n    Returns\n    -------\n    f : ndarray\n        A view of `m` with the columns reversed.  Since a view\n        is returned, this operation is :math:`\\mathcal O(1)`.\n\n    See Also\n    --------\n    flipud : Flip array in the up/down direction.\n    rot90 : Rotate array counterclockwise.\n\n    Notes\n    -----\n    Equivalent to m[:,::-1]. Requires the array to be at least 2-D.\n\n    Examples\n    --------\n    >>> A = np.diag([1.,2.,3.])\n    >>> A\n    array([[ 1.,  0.,  0.],\n           [ 0.,  2.,  0.],\n           [ 0.,  0.,  3.]])\n    >>> np.fliplr(A)\n    array([[ 0.,  0.,  1.],\n           [ 0.,  2.,  0.],\n           [ 3.,  0.,  0.]])\n\n    >>> A = np.random.randn(2,3,5)\n    >>> np.all(np.fliplr(A) == A[:,::-1,...])\n    True\n\n    '
    m = asanyarray(m)
    if m.ndim < 2:
        raise ValueError('Input must be >= 2-d.')
    return m[:, ::-1]

@array_function_dispatch(_flip_dispatcher)
def flipud(m):
    if False:
        i = 10
        return i + 15
    '\n    Flip array in the up/down direction.\n\n    Flip the entries in each column in the up/down direction.\n    Rows are preserved, but appear in a different order than before.\n\n    Parameters\n    ----------\n    m : array_like\n        Input array.\n\n    Returns\n    -------\n    out : array_like\n        A view of `m` with the rows reversed.  Since a view is\n        returned, this operation is :math:`\\mathcal O(1)`.\n\n    See Also\n    --------\n    fliplr : Flip array in the left/right direction.\n    rot90 : Rotate array counterclockwise.\n\n    Notes\n    -----\n    Equivalent to ``m[::-1,...]``.\n    Does not require the array to be two-dimensional.\n\n    Examples\n    --------\n    >>> A = np.diag([1.0, 2, 3])\n    >>> A\n    array([[ 1.,  0.,  0.],\n           [ 0.,  2.,  0.],\n           [ 0.,  0.,  3.]])\n    >>> np.flipud(A)\n    array([[ 0.,  0.,  3.],\n           [ 0.,  2.,  0.],\n           [ 1.,  0.,  0.]])\n\n    >>> A = np.random.randn(2,3,5)\n    >>> np.all(np.flipud(A) == A[::-1,...])\n    True\n\n    >>> np.flipud([1,2])\n    array([2, 1])\n\n    '
    m = asanyarray(m)
    if m.ndim < 1:
        raise ValueError('Input must be >= 1-d.')
    return m[::-1, ...]

@set_module('numpy')
def eye(N, M=None, k=0, dtype=float, order='C'):
    if False:
        return 10
    "\n    Return a 2-D array with ones on the diagonal and zeros elsewhere.\n\n    Parameters\n    ----------\n    N : int\n      Number of rows in the output.\n    M : int, optional\n      Number of columns in the output. If None, defaults to `N`.\n    k : int, optional\n      Index of the diagonal: 0 (the default) refers to the main diagonal,\n      a positive value refers to an upper diagonal, and a negative value\n      to a lower diagonal.\n    dtype : data-type, optional\n      Data-type of the returned array.\n    order : {'C', 'F'}, optional\n        Whether the output should be stored in row-major (C-style) or\n        column-major (Fortran-style) order in memory.\n\n        .. versionadded:: 1.14.0\n\n    Returns\n    -------\n    I : ndarray of shape (N,M)\n      An array where all elements are equal to zero, except for the `k`-th\n      diagonal, whose values are equal to one.\n\n    See Also\n    --------\n    identity : (almost) equivalent function\n    diag : diagonal 2-D array from a 1-D array specified by the user.\n\n    Examples\n    --------\n    >>> np.eye(2, dtype=int)\n    array([[1, 0],\n           [0, 1]])\n    >>> np.eye(3, k=1)\n    array([[ 0.,  1.,  0.],\n           [ 0.,  0.,  1.],\n           [ 0.,  0.,  0.]])\n\n    "
    if M is None:
        M = N
    m = zeros((N, M), dtype=dtype, order=order)
    if k >= M:
        return m
    if k >= 0:
        i = k
    else:
        i = -k * M
    m[:M - k].flat[i::M + 1] = 1
    return m

def _diag_dispatcher(v, k=None):
    if False:
        for i in range(10):
            print('nop')
    return (v,)

@array_function_dispatch(_diag_dispatcher)
def diag(v, k=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract a diagonal or construct a diagonal array.\n\n    See the more detailed documentation for ``numpy.diagonal`` if you use this\n    function to extract a diagonal and wish to write to the resulting array;\n    whether it returns a copy or a view depends on what version of numpy you\n    are using.\n\n    Parameters\n    ----------\n    v : array_like\n        If `v` is a 2-D array, return a copy of its `k`-th diagonal.\n        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th\n        diagonal.\n    k : int, optional\n        Diagonal in question. The default is 0. Use `k>0` for diagonals\n        above the main diagonal, and `k<0` for diagonals below the main\n        diagonal.\n\n    Returns\n    -------\n    out : ndarray\n        The extracted diagonal or constructed diagonal array.\n\n    See Also\n    --------\n    diagonal : Return specified diagonals.\n    diagflat : Create a 2-D array with the flattened input as a diagonal.\n    trace : Sum along diagonals.\n    triu : Upper triangle of an array.\n    tril : Lower triangle of an array.\n\n    Examples\n    --------\n    >>> x = np.arange(9).reshape((3,3))\n    >>> x\n    array([[0, 1, 2],\n           [3, 4, 5],\n           [6, 7, 8]])\n\n    >>> np.diag(x)\n    array([0, 4, 8])\n    >>> np.diag(x, k=1)\n    array([1, 5])\n    >>> np.diag(x, k=-1)\n    array([3, 7])\n\n    >>> np.diag(np.diag(x))\n    array([[0, 0, 0],\n           [0, 4, 0],\n           [0, 0, 8]])\n\n    '
    v = asanyarray(v)
    s = v.shape
    if len(s) == 1:
        n = s[0] + abs(k)
        res = zeros((n, n), v.dtype)
        if k >= 0:
            i = k
        else:
            i = -k * n
        res[:n - k].flat[i::n + 1] = v
        return res
    elif len(s) == 2:
        return diagonal(v, k)
    else:
        raise ValueError('Input must be 1- or 2-d.')

@array_function_dispatch(_diag_dispatcher)
def diagflat(v, k=0):
    if False:
        i = 10
        return i + 15
    '\n    Create a two-dimensional array with the flattened input as a diagonal.\n\n    Parameters\n    ----------\n    v : array_like\n        Input data, which is flattened and set as the `k`-th\n        diagonal of the output.\n    k : int, optional\n        Diagonal to set; 0, the default, corresponds to the "main" diagonal,\n        a positive (negative) `k` giving the number of the diagonal above\n        (below) the main.\n\n    Returns\n    -------\n    out : ndarray\n        The 2-D output array.\n\n    See Also\n    --------\n    diag : MATLAB work-alike for 1-D and 2-D arrays.\n    diagonal : Return specified diagonals.\n    trace : Sum along diagonals.\n\n    Examples\n    --------\n    >>> np.diagflat([[1,2], [3,4]])\n    array([[1, 0, 0, 0],\n           [0, 2, 0, 0],\n           [0, 0, 3, 0],\n           [0, 0, 0, 4]])\n\n    >>> np.diagflat([1,2], 1)\n    array([[0, 1, 0],\n           [0, 0, 2],\n           [0, 0, 0]])\n\n    '
    try:
        wrap = v.__array_wrap__
    except AttributeError:
        wrap = None
    v = asarray(v).ravel()
    s = len(v)
    n = s + abs(k)
    res = zeros((n, n), v.dtype)
    if k >= 0:
        i = arange(0, n - k)
        fi = i + k + i * n
    else:
        i = arange(0, n + k)
        fi = i + (i - k) * n
    res.flat[fi] = v
    if not wrap:
        return res
    return wrap(res)

@set_module('numpy')
def tri(N, M=None, k=0, dtype=float):
    if False:
        print('Hello World!')
    '\n    An array with ones at and below the given diagonal and zeros elsewhere.\n\n    Parameters\n    ----------\n    N : int\n        Number of rows in the array.\n    M : int, optional\n        Number of columns in the array.\n        By default, `M` is taken equal to `N`.\n    k : int, optional\n        The sub-diagonal at and below which the array is filled.\n        `k` = 0 is the main diagonal, while `k` < 0 is below it,\n        and `k` > 0 is above.  The default is 0.\n    dtype : dtype, optional\n        Data type of the returned array.  The default is float.\n\n    Returns\n    -------\n    tri : ndarray of shape (N, M)\n        Array with its lower triangle filled with ones and zero elsewhere;\n        in other words ``T[i,j] == 1`` for ``i <= j + k``, 0 otherwise.\n\n    Examples\n    --------\n    >>> np.tri(3, 5, 2, dtype=int)\n    array([[1, 1, 1, 0, 0],\n           [1, 1, 1, 1, 0],\n           [1, 1, 1, 1, 1]])\n\n    >>> np.tri(3, 5, -1)\n    array([[ 0.,  0.,  0.,  0.,  0.],\n           [ 1.,  0.,  0.,  0.,  0.],\n           [ 1.,  1.,  0.,  0.,  0.]])\n\n    '
    if M is None:
        M = N
    m = greater_equal.outer(arange(N, dtype=_min_int(0, N)), arange(-k, M - k, dtype=_min_int(-k, M - k)))
    m = m.astype(dtype, copy=False)
    return m

def _trilu_dispatcher(m, k=None):
    if False:
        return 10
    return (m,)

@array_function_dispatch(_trilu_dispatcher)
def tril(m, k=0):
    if False:
        i = 10
        return i + 15
    '\n    Lower triangle of an array.\n\n    Return a copy of an array with elements above the `k`-th diagonal zeroed.\n\n    Parameters\n    ----------\n    m : array_like, shape (M, N)\n        Input array.\n    k : int, optional\n        Diagonal above which to zero elements.  `k = 0` (the default) is the\n        main diagonal, `k < 0` is below it and `k > 0` is above.\n\n    Returns\n    -------\n    tril : ndarray, shape (M, N)\n        Lower triangle of `m`, of same shape and data-type as `m`.\n\n    See Also\n    --------\n    triu : same thing, only for the upper triangle\n\n    Examples\n    --------\n    >>> np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)\n    array([[ 0,  0,  0],\n           [ 4,  0,  0],\n           [ 7,  8,  0],\n           [10, 11, 12]])\n\n    '
    m = asanyarray(m)
    mask = tri(*m.shape[-2:], k=k, dtype=bool)
    return where(mask, m, zeros(1, m.dtype))

@array_function_dispatch(_trilu_dispatcher)
def triu(m, k=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Upper triangle of an array.\n\n    Return a copy of a matrix with the elements below the `k`-th diagonal\n    zeroed.\n\n    Please refer to the documentation for `tril` for further details.\n\n    See Also\n    --------\n    tril : lower triangle of an array\n\n    Examples\n    --------\n    >>> np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)\n    array([[ 1,  2,  3],\n           [ 4,  5,  6],\n           [ 0,  8,  9],\n           [ 0,  0, 12]])\n\n    '
    m = asanyarray(m)
    mask = tri(*m.shape[-2:], k=k - 1, dtype=bool)
    return where(mask, zeros(1, m.dtype), m)

def _vander_dispatcher(x, N=None, increasing=None):
    if False:
        i = 10
        return i + 15
    return (x,)

@array_function_dispatch(_vander_dispatcher)
def vander(x, N=None, increasing=False):
    if False:
        return 10
    '\n    Generate a Vandermonde matrix.\n\n    The columns of the output matrix are powers of the input vector. The\n    order of the powers is determined by the `increasing` boolean argument.\n    Specifically, when `increasing` is False, the `i`-th output column is\n    the input vector raised element-wise to the power of ``N - i - 1``. Such\n    a matrix with a geometric progression in each row is named for Alexandre-\n    Theophile Vandermonde.\n\n    Parameters\n    ----------\n    x : array_like\n        1-D input array.\n    N : int, optional\n        Number of columns in the output.  If `N` is not specified, a square\n        array is returned (``N = len(x)``).\n    increasing : bool, optional\n        Order of the powers of the columns.  If True, the powers increase\n        from left to right, if False (the default) they are reversed.\n\n        .. versionadded:: 1.9.0\n\n    Returns\n    -------\n    out : ndarray\n        Vandermonde matrix.  If `increasing` is False, the first column is\n        ``x^(N-1)``, the second ``x^(N-2)`` and so forth. If `increasing` is\n        True, the columns are ``x^0, x^1, ..., x^(N-1)``.\n\n    See Also\n    --------\n    polynomial.polynomial.polyvander\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3, 5])\n    >>> N = 3\n    >>> np.vander(x, N)\n    array([[ 1,  1,  1],\n           [ 4,  2,  1],\n           [ 9,  3,  1],\n           [25,  5,  1]])\n\n    >>> np.column_stack([x**(N-1-i) for i in range(N)])\n    array([[ 1,  1,  1],\n           [ 4,  2,  1],\n           [ 9,  3,  1],\n           [25,  5,  1]])\n\n    >>> x = np.array([1, 2, 3, 5])\n    >>> np.vander(x)\n    array([[  1,   1,   1,   1],\n           [  8,   4,   2,   1],\n           [ 27,   9,   3,   1],\n           [125,  25,   5,   1]])\n    >>> np.vander(x, increasing=True)\n    array([[  1,   1,   1,   1],\n           [  1,   2,   4,   8],\n           [  1,   3,   9,  27],\n           [  1,   5,  25, 125]])\n\n    The determinant of a square Vandermonde matrix is the product\n    of the differences between the values of the input vector:\n\n    >>> np.linalg.det(np.vander(x))\n    48.000000000000043\n    >>> (5-3)*(5-2)*(5-1)*(3-2)*(3-1)*(2-1)\n    48\n\n    '
    x = asarray(x)
    if x.ndim != 1:
        raise ValueError('x must be a one-dimensional array or sequence.')
    if N is None:
        N = len(x)
    v = empty((len(x), N), dtype=promote_types(x.dtype, int))
    tmp = v[:, ::-1] if not increasing else v
    if N > 0:
        tmp[:, 0] = 1
    if N > 1:
        tmp[:, 1:] = x[:, None]
        multiply.accumulate(tmp[:, 1:], out=tmp[:, 1:], axis=1)
    return v

def _histogram2d_dispatcher(x, y, bins=None, range=None, normed=None, weights=None, density=None):
    if False:
        print('Hello World!')
    return (x, y, bins, weights)

@array_function_dispatch(_histogram2d_dispatcher)
def histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None):
    if False:
        i = 10
        return i + 15
    "\n    Compute the bi-dimensional histogram of two data samples.\n\n    Parameters\n    ----------\n    x : array_like, shape (N,)\n        An array containing the x coordinates of the points to be\n        histogrammed.\n    y : array_like, shape (N,)\n        An array containing the y coordinates of the points to be\n        histogrammed.\n    bins : int or array_like or [int, int] or [array, array], optional\n        The bin specification:\n\n          * If int, the number of bins for the two dimensions (nx=ny=bins).\n          * If array_like, the bin edges for the two dimensions\n            (x_edges=y_edges=bins).\n          * If [int, int], the number of bins in each dimension\n            (nx, ny = bins).\n          * If [array, array], the bin edges in each dimension\n            (x_edges, y_edges = bins).\n          * A combination [int, array] or [array, int], where int\n            is the number of bins and array is the bin edges.\n\n    range : array_like, shape(2,2), optional\n        The leftmost and rightmost edges of the bins along each dimension\n        (if not specified explicitly in the `bins` parameters):\n        ``[[xmin, xmax], [ymin, ymax]]``. All values outside of this range\n        will be considered outliers and not tallied in the histogram.\n    density : bool, optional\n        If False, the default, returns the number of samples in each bin.\n        If True, returns the probability *density* function at the bin,\n        ``bin_count / sample_count / bin_area``.\n    normed : bool, optional\n        An alias for the density argument that behaves identically. To avoid\n        confusion with the broken normed argument to `histogram`, `density`\n        should be preferred.\n    weights : array_like, shape(N,), optional\n        An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.\n        Weights are normalized to 1 if `normed` is True. If `normed` is\n        False, the values of the returned histogram are equal to the sum of\n        the weights belonging to the samples falling into each bin.\n\n    Returns\n    -------\n    H : ndarray, shape(nx, ny)\n        The bi-dimensional histogram of samples `x` and `y`. Values in `x`\n        are histogrammed along the first dimension and values in `y` are\n        histogrammed along the second dimension.\n    xedges : ndarray, shape(nx+1,)\n        The bin edges along the first dimension.\n    yedges : ndarray, shape(ny+1,)\n        The bin edges along the second dimension.\n\n    See Also\n    --------\n    histogram : 1D histogram\n    histogramdd : Multidimensional histogram\n\n    Notes\n    -----\n    When `normed` is True, then the returned histogram is the sample\n    density, defined such that the sum over bins of the product\n    ``bin_value * bin_area`` is 1.\n\n    Please note that the histogram does not follow the Cartesian convention\n    where `x` values are on the abscissa and `y` values on the ordinate\n    axis.  Rather, `x` is histogrammed along the first dimension of the\n    array (vertical), and `y` along the second dimension of the array\n    (horizontal).  This ensures compatibility with `histogramdd`.\n\n    Examples\n    --------\n    >>> from matplotlib.image import NonUniformImage\n    >>> import matplotlib.pyplot as plt\n\n    Construct a 2-D histogram with variable bin width. First define the bin\n    edges:\n\n    >>> xedges = [0, 1, 3, 5]\n    >>> yedges = [0, 2, 3, 4, 6]\n\n    Next we create a histogram H with random bin content:\n\n    >>> x = np.random.normal(2, 1, 100)\n    >>> y = np.random.normal(1, 1, 100)\n    >>> H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))\n    >>> H = H.T  # Let each row list bins with common y range.\n\n    :func:`imshow <matplotlib.pyplot.imshow>` can only display square bins:\n\n    >>> fig = plt.figure(figsize=(7, 3))\n    >>> ax = fig.add_subplot(131, title='imshow: square bins')\n    >>> plt.imshow(H, interpolation='nearest', origin='low',\n    ...         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])\n\n    :func:`pcolormesh <matplotlib.pyplot.pcolormesh>` can display actual edges:\n\n    >>> ax = fig.add_subplot(132, title='pcolormesh: actual edges',\n    ...         aspect='equal')\n    >>> X, Y = np.meshgrid(xedges, yedges)\n    >>> ax.pcolormesh(X, Y, H)\n\n    :class:`NonUniformImage <matplotlib.image.NonUniformImage>` can be used to\n    display actual bin edges with interpolation:\n\n    >>> ax = fig.add_subplot(133, title='NonUniformImage: interpolated',\n    ...         aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])\n    >>> im = NonUniformImage(ax, interpolation='bilinear')\n    >>> xcenters = (xedges[:-1] + xedges[1:]) / 2\n    >>> ycenters = (yedges[:-1] + yedges[1:]) / 2\n    >>> im.set_data(xcenters, ycenters, H)\n    >>> ax.images.append(im)\n    >>> plt.show()\n\n    "
    from numpy import histogramdd
    try:
        N = len(bins)
    except TypeError:
        N = 1
    if N != 1 and N != 2:
        xedges = yedges = asarray(bins)
        bins = [xedges, yedges]
    (hist, edges) = histogramdd([x, y], bins, range, normed, weights, density)
    return (hist, edges[0], edges[1])

@set_module('numpy')
def mask_indices(n, mask_func, k=0):
    if False:
        while True:
            i = 10
    '\n    Return the indices to access (n, n) arrays, given a masking function.\n\n    Assume `mask_func` is a function that, for a square array a of size\n    ``(n, n)`` with a possible offset argument `k`, when called as\n    ``mask_func(a, k)`` returns a new array with zeros in certain locations\n    (functions like `triu` or `tril` do precisely this). Then this function\n    returns the indices where the non-zero values would be located.\n\n    Parameters\n    ----------\n    n : int\n        The returned indices will be valid to access arrays of shape (n, n).\n    mask_func : callable\n        A function whose call signature is similar to that of `triu`, `tril`.\n        That is, ``mask_func(x, k)`` returns a boolean array, shaped like `x`.\n        `k` is an optional argument to the function.\n    k : scalar\n        An optional argument which is passed through to `mask_func`. Functions\n        like `triu`, `tril` take a second argument that is interpreted as an\n        offset.\n\n    Returns\n    -------\n    indices : tuple of arrays.\n        The `n` arrays of indices corresponding to the locations where\n        ``mask_func(np.ones((n, n)), k)`` is True.\n\n    See Also\n    --------\n    triu, tril, triu_indices, tril_indices\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    Examples\n    --------\n    These are the indices that would allow you to access the upper triangular\n    part of any 3x3 array:\n\n    >>> iu = np.mask_indices(3, np.triu)\n\n    For example, if `a` is a 3x3 array:\n\n    >>> a = np.arange(9).reshape(3, 3)\n    >>> a\n    array([[0, 1, 2],\n           [3, 4, 5],\n           [6, 7, 8]])\n    >>> a[iu]\n    array([0, 1, 2, 4, 5, 8])\n\n    An offset can be passed also to the masking function.  This gets us the\n    indices starting on the first diagonal right of the main one:\n\n    >>> iu1 = np.mask_indices(3, np.triu, 1)\n\n    with which we now extract only three elements:\n\n    >>> a[iu1]\n    array([1, 2, 5])\n\n    '
    m = ones((n, n), int)
    a = mask_func(m, k)
    return nonzero(a != 0)

@set_module('numpy')
def tril_indices(n, k=0, m=None):
    if False:
        print('Hello World!')
    '\n    Return the indices for the lower-triangle of an (n, m) array.\n\n    Parameters\n    ----------\n    n : int\n        The row dimension of the arrays for which the returned\n        indices will be valid.\n    k : int, optional\n        Diagonal offset (see `tril` for details).\n    m : int, optional\n        .. versionadded:: 1.9.0\n\n        The column dimension of the arrays for which the returned\n        arrays will be valid.\n        By default `m` is taken equal to `n`.\n\n\n    Returns\n    -------\n    inds : tuple of arrays\n        The indices for the triangle. The returned tuple contains two arrays,\n        each with the indices along one dimension of the array.\n\n    See also\n    --------\n    triu_indices : similar function, for upper-triangular.\n    mask_indices : generic function accepting an arbitrary mask function.\n    tril, triu\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    Examples\n    --------\n    Compute two different sets of indices to access 4x4 arrays, one for the\n    lower triangular part starting at the main diagonal, and one starting two\n    diagonals further right:\n\n    >>> il1 = np.tril_indices(4)\n    >>> il2 = np.tril_indices(4, 2)\n\n    Here is how they can be used with a sample array:\n\n    >>> a = np.arange(16).reshape(4, 4)\n    >>> a\n    array([[ 0,  1,  2,  3],\n           [ 4,  5,  6,  7],\n           [ 8,  9, 10, 11],\n           [12, 13, 14, 15]])\n\n    Both for indexing:\n\n    >>> a[il1]\n    array([ 0,  4,  5,  8,  9, 10, 12, 13, 14, 15])\n\n    And for assigning values:\n\n    >>> a[il1] = -1\n    >>> a\n    array([[-1,  1,  2,  3],\n           [-1, -1,  6,  7],\n           [-1, -1, -1, 11],\n           [-1, -1, -1, -1]])\n\n    These cover almost the whole array (two diagonals right of the main one):\n\n    >>> a[il2] = -10\n    >>> a\n    array([[-10, -10, -10,   3],\n           [-10, -10, -10, -10],\n           [-10, -10, -10, -10],\n           [-10, -10, -10, -10]])\n\n    '
    return nonzero(tri(n, m, k=k, dtype=bool))

def _trilu_indices_form_dispatcher(arr, k=None):
    if False:
        print('Hello World!')
    return (arr,)

@array_function_dispatch(_trilu_indices_form_dispatcher)
def tril_indices_from(arr, k=0):
    if False:
        print('Hello World!')
    '\n    Return the indices for the lower-triangle of arr.\n\n    See `tril_indices` for full details.\n\n    Parameters\n    ----------\n    arr : array_like\n        The indices will be valid for square arrays whose dimensions are\n        the same as arr.\n    k : int, optional\n        Diagonal offset (see `tril` for details).\n\n    See Also\n    --------\n    tril_indices, tril\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    '
    if arr.ndim != 2:
        raise ValueError('input array must be 2-d')
    return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1])

@set_module('numpy')
def triu_indices(n, k=0, m=None):
    if False:
        print('Hello World!')
    '\n    Return the indices for the upper-triangle of an (n, m) array.\n\n    Parameters\n    ----------\n    n : int\n        The size of the arrays for which the returned indices will\n        be valid.\n    k : int, optional\n        Diagonal offset (see `triu` for details).\n    m : int, optional\n        .. versionadded:: 1.9.0\n\n        The column dimension of the arrays for which the returned\n        arrays will be valid.\n        By default `m` is taken equal to `n`.\n\n\n    Returns\n    -------\n    inds : tuple, shape(2) of ndarrays, shape(`n`)\n        The indices for the triangle. The returned tuple contains two arrays,\n        each with the indices along one dimension of the array.  Can be used\n        to slice a ndarray of shape(`n`, `n`).\n\n    See also\n    --------\n    tril_indices : similar function, for lower-triangular.\n    mask_indices : generic function accepting an arbitrary mask function.\n    triu, tril\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    Examples\n    --------\n    Compute two different sets of indices to access 4x4 arrays, one for the\n    upper triangular part starting at the main diagonal, and one starting two\n    diagonals further right:\n\n    >>> iu1 = np.triu_indices(4)\n    >>> iu2 = np.triu_indices(4, 2)\n\n    Here is how they can be used with a sample array:\n\n    >>> a = np.arange(16).reshape(4, 4)\n    >>> a\n    array([[ 0,  1,  2,  3],\n           [ 4,  5,  6,  7],\n           [ 8,  9, 10, 11],\n           [12, 13, 14, 15]])\n\n    Both for indexing:\n\n    >>> a[iu1]\n    array([ 0,  1,  2,  3,  5,  6,  7, 10, 11, 15])\n\n    And for assigning values:\n\n    >>> a[iu1] = -1\n    >>> a\n    array([[-1, -1, -1, -1],\n           [ 4, -1, -1, -1],\n           [ 8,  9, -1, -1],\n           [12, 13, 14, -1]])\n\n    These cover only a small part of the whole array (two diagonals right\n    of the main one):\n\n    >>> a[iu2] = -10\n    >>> a\n    array([[ -1,  -1, -10, -10],\n           [  4,  -1,  -1, -10],\n           [  8,   9,  -1,  -1],\n           [ 12,  13,  14,  -1]])\n\n    '
    return nonzero(~tri(n, m, k=k - 1, dtype=bool))

@array_function_dispatch(_trilu_indices_form_dispatcher)
def triu_indices_from(arr, k=0):
    if False:
        while True:
            i = 10
    '\n    Return the indices for the upper-triangle of arr.\n\n    See `triu_indices` for full details.\n\n    Parameters\n    ----------\n    arr : ndarray, shape(N, N)\n        The indices will be valid for square arrays.\n    k : int, optional\n        Diagonal offset (see `triu` for details).\n\n    Returns\n    -------\n    triu_indices_from : tuple, shape(2) of ndarray, shape(N)\n        Indices for the upper-triangle of `arr`.\n\n    See Also\n    --------\n    triu_indices, triu\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    '
    if arr.ndim != 2:
        raise ValueError('input array must be 2-d')
    return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1])