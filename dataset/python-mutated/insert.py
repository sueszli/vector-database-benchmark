import numpy
import cupy
from cupy import _core

def place(arr, mask, vals):
    if False:
        while True:
            i = 10
    'Change elements of an array based on conditional and input values.\n\n    This function uses the first N elements of `vals`, where N is the number\n    of true values in `mask`.\n\n    Args:\n        arr (cupy.ndarray): Array to put data into.\n        mask (array-like): Boolean mask array. Must have the same size as `a`.\n        vals (array-like): Values to put into `a`. Only the first\n            N elements are used, where N is the number of True values in\n            `mask`. If `vals` is smaller than N, it will be repeated, and if\n            elements of `a` are to be masked, this sequence must be non-empty.\n\n    Examples\n    --------\n    >>> arr = np.arange(6).reshape(2, 3)\n    >>> np.place(arr, arr>2, [44, 55])\n    >>> arr\n    array([[ 0,  1,  2],\n           [44, 55, 44]])\n\n    .. warning::\n\n        This function may synchronize the device.\n\n    .. seealso:: :func:`numpy.place`\n    '
    mask = cupy.asarray(mask)
    if arr.size != mask.size:
        raise ValueError('Mask and data must be the same size.')
    vals = cupy.asarray(vals)
    mask_indices = mask.ravel().nonzero()[0]
    if mask_indices.size == 0:
        return
    if vals.size == 0:
        raise ValueError('Cannot insert from an empty array.')
    arr.put(mask_indices, vals, mode='wrap')

def put(a, ind, v, mode='wrap'):
    if False:
        return 10
    "Replaces specified elements of an array with given values.\n\n    Args:\n        a (cupy.ndarray): Target array.\n        ind (array-like): Target indices, interpreted as integers.\n        v (array-like): Values to place in `a` at target indices.\n            If `v` is shorter than `ind` it will be repeated as necessary.\n        mode (str): How out-of-bounds indices will behave. Its value must be\n            either `'raise'`, `'wrap'` or `'clip'`. Otherwise,\n            :class:`TypeError` is raised.\n\n    .. note::\n        Default `mode` is set to `'wrap'` to avoid unintended performance drop.\n        If you need NumPy's behavior, please pass `mode='raise'` manually.\n\n    .. seealso:: :func:`numpy.put`\n    "
    a.put(ind, v, mode=mode)
_putmask_kernel = _core.ElementwiseKernel('Q mask, raw S values, uint64 len_vals', 'T out', '\n    if (mask) out = (T) values[i % len_vals];\n    ', 'cupy_putmask_kernel')

def putmask(a, mask, values):
    if False:
        print('Hello World!')
    '\n    Changes elements of an array inplace, based on a conditional mask and\n    input values.\n\n    Sets ``a.flat[n] = values[n]`` for each n where ``mask.flat[n]==True``.\n    If `values` is not the same size as `a` and `mask` then it will repeat.\n\n    Args:\n        a (cupy.ndarray): Target array.\n        mask (cupy.ndarray): Boolean mask array. It has to be\n            the same shape as `a`.\n        values (cupy.ndarray or scalar): Values to put into `a` where `mask`\n            is True. If `values` is smaller than `a`, then it will be\n            repeated.\n\n    Examples\n    --------\n    >>> x = cupy.arange(6).reshape(2, 3)\n    >>> cupy.putmask(x, x>2, x**2)\n    >>> x\n    array([[ 0,  1,  2],\n           [ 9, 16, 25]])\n\n    If `values` is smaller than `a` it is repeated:\n\n    >>> x = cupy.arange(6)\n    >>> cupy.putmask(x, x>2, cupy.array([-33, -44]))\n    >>> x\n    array([  0,   1,   2, -44, -33, -44])\n\n    .. seealso:: :func:`numpy.putmask`\n\n    '
    if not isinstance(a, cupy.ndarray):
        raise TypeError('`a` should be of type cupy.ndarray')
    if not isinstance(mask, cupy.ndarray):
        raise TypeError('`mask` should be of type cupy.ndarray')
    if not (cupy.isscalar(values) or isinstance(values, cupy.ndarray)):
        raise TypeError('`values` should be of type cupy.ndarray')
    if not a.shape == mask.shape:
        raise ValueError('mask and data must be the same size')
    mask = mask.astype(numpy.bool_)
    if cupy.isscalar(values):
        a[mask] = values
    elif not numpy.can_cast(values.dtype, a.dtype):
        raise TypeError("Cannot cast array data from {} to {} according to the rule 'safe'".format(values.dtype, a.dtype))
    elif a.shape == values.shape:
        a[mask] = values[mask]
    else:
        values = values.ravel()
        _putmask_kernel(mask, values, len(values), a)

def fill_diagonal(a, val, wrap=False):
    if False:
        print('Hello World!')
    'Fills the main diagonal of the given array of any dimensionality.\n\n    For an array `a` with ``a.ndim > 2``, the diagonal is the list of\n    locations with indices ``a[i, i, ..., i]`` all identical. This function\n    modifies the input array in-place, it does not return a value.\n\n    Args:\n        a (cupy.ndarray): The array, at least 2-D.\n        val (scalar): The value to be written on the diagonal.\n            Its type must be compatible with that of the array a.\n        wrap (bool): If specified, the diagonal is "wrapped" after N columns.\n            This affects only tall matrices.\n\n    Examples\n    --------\n    >>> a = cupy.zeros((3, 3), int)\n    >>> cupy.fill_diagonal(a, 5)\n    >>> a\n    array([[5, 0, 0],\n           [0, 5, 0],\n           [0, 0, 5]])\n\n    .. seealso:: :func:`numpy.fill_diagonal`\n    '
    if a.ndim < 2:
        raise ValueError('array must be at least 2-d')
    end = None
    if a.ndim == 2:
        step = a.shape[1] + 1
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        if not numpy.all(numpy.diff(a.shape) == 0):
            raise ValueError('All dimensions of input must be of equal length')
        step = 1 + numpy.cumprod(a.shape[:-1]).sum()
    a.flat[:end:step] = val

def diag_indices(n, ndim=2):
    if False:
        return 10
    'Return the indices to access the main diagonal of an array.\n\n    Returns a tuple of indices that can be used to access the main\n    diagonal of an array with ``ndim >= 2`` dimensions and shape\n    (n, n, ..., n).\n\n    Args:\n        n (int): The size, along each dimension of the arrays for which\n            the indices are to be returned.\n        ndim (int): The number of dimensions. default `2`.\n\n    Examples\n    --------\n    Create a set of indices to access the diagonal of a (4, 4) array:\n\n    >>> di = cupy.diag_indices(4)\n    >>> di\n    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))\n    >>> a = cupy.arange(16).reshape(4, 4)\n    >>> a\n    array([[ 0,  1,  2,  3],\n           [ 4,  5,  6,  7],\n           [ 8,  9, 10, 11],\n           [12, 13, 14, 15]])\n    >>> a[di] = 100\n    >>> a\n    array([[100,   1,   2,   3],\n           [  4, 100,   6,   7],\n           [  8,   9, 100,  11],\n           [ 12,  13,  14, 100]])\n\n    Create indices to manipulate a 3-D array:\n\n    >>> d3 = cupy.diag_indices(2, 3)\n    >>> d3\n    (array([0, 1]), array([0, 1]), array([0, 1]))\n\n    And use it to set the diagonal of an array of zeros to 1:\n\n    >>> a = cupy.zeros((2, 2, 2), dtype=int)\n    >>> a[d3] = 1\n    >>> a\n    array([[[1, 0],\n            [0, 0]],\n    <BLANKLINE>\n           [[0, 0],\n            [0, 1]]])\n\n    .. seealso:: :func:`numpy.diag_indices`\n\n    '
    idx = cupy.arange(n)
    return (idx,) * ndim

def diag_indices_from(arr):
    if False:
        i = 10
        return i + 15
    '\n    Return the indices to access the main diagonal of an n-dimensional array.\n    See `diag_indices` for full details.\n\n    Args:\n        arr (cupy.ndarray): At least 2-D.\n\n    .. seealso:: :func:`numpy.diag_indices_from`\n\n    '
    if not isinstance(arr, cupy.ndarray):
        raise TypeError('Argument must be cupy.ndarray')
    if not arr.ndim >= 2:
        raise ValueError('input array must be at least 2-d')
    if not cupy.all(cupy.diff(arr.shape) == 0):
        raise ValueError('All dimensions of input must be of equal length')
    return diag_indices(arr.shape[0], arr.ndim)