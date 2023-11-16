from numpy._core import asarray
from numpy._core.numeric import normalize_axis_tuple, normalize_axis_index
from numpy._utils import set_module
__all__ = ['byte_bounds', 'normalize_axis_tuple', 'normalize_axis_index']

@set_module('numpy.lib.array_utils')
def byte_bounds(a):
    if False:
        while True:
            i = 10
    "\n    Returns pointers to the end-points of an array.\n\n    Parameters\n    ----------\n    a : ndarray\n        Input array. It must conform to the Python-side of the array\n        interface.\n\n    Returns\n    -------\n    (low, high) : tuple of 2 integers\n        The first integer is the first byte of the array, the second\n        integer is just past the last byte of the array.  If `a` is not\n        contiguous it will not use every byte between the (`low`, `high`)\n        values.\n\n    Examples\n    --------\n    >>> I = np.eye(2, dtype='f'); I.dtype\n    dtype('float32')\n    >>> low, high = np.lib.array_utils.byte_bounds(I)\n    >>> high - low == I.size*I.itemsize\n    True\n    >>> I = np.eye(2); I.dtype\n    dtype('float64')\n    >>> low, high = np.lib.array_utils.byte_bounds(I)\n    >>> high - low == I.size*I.itemsize\n    True\n\n    "
    ai = a.__array_interface__
    a_data = ai['data'][0]
    astrides = ai['strides']
    ashape = ai['shape']
    bytes_a = asarray(a).dtype.itemsize
    a_low = a_high = a_data
    if astrides is None:
        a_high += a.size * bytes_a
    else:
        for (shape, stride) in zip(ashape, astrides):
            if stride < 0:
                a_low += (shape - 1) * stride
            else:
                a_high += (shape - 1) * stride
        a_high += bytes_a
    return (a_low, a_high)