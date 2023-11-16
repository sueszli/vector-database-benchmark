"""
Module of functions that are like ufuncs in acting on arrays and optionally
storing results in an output array.

"""
__all__ = ['fix', 'isneginf', 'isposinf']
import numpy._core.numeric as nx
from numpy._core.overrides import array_function_dispatch
import warnings
import functools

def _dispatcher(x, out=None):
    if False:
        print('Hello World!')
    return (x, out)

@array_function_dispatch(_dispatcher, verify=False, module='numpy')
def fix(x, out=None):
    if False:
        while True:
            i = 10
    '\n    Round to nearest integer towards zero.\n\n    Round an array of floats element-wise to nearest integer towards zero.\n    The rounded values are returned as floats.\n\n    Parameters\n    ----------\n    x : array_like\n        An array of floats to be rounded\n    out : ndarray, optional\n        A location into which the result is stored. If provided, it must have\n        a shape that the input broadcasts to. If not provided or None, a\n        freshly-allocated array is returned.\n\n    Returns\n    -------\n    out : ndarray of floats\n        A float array with the same dimensions as the input.\n        If second argument is not supplied then a float array is returned\n        with the rounded values.\n\n        If a second argument is supplied the result is stored there.\n        The return value `out` is then a reference to that array.\n\n    See Also\n    --------\n    rint, trunc, floor, ceil\n    around : Round to given number of decimals\n\n    Examples\n    --------\n    >>> np.fix(3.14)\n    3.0\n    >>> np.fix(3)\n    3.0\n    >>> np.fix([2.1, 2.9, -2.1, -2.9])\n    array([ 2.,  2., -2., -2.])\n\n    '
    res = nx.asanyarray(nx.ceil(x, out=out))
    res = nx.floor(x, out=res, where=nx.greater_equal(x, 0))
    if out is None and type(res) is nx.ndarray:
        res = res[()]
    return res

@array_function_dispatch(_dispatcher, verify=False, module='numpy')
def isposinf(x, out=None):
    if False:
        while True:
            i = 10
    '\n    Test element-wise for positive infinity, return result as bool array.\n\n    Parameters\n    ----------\n    x : array_like\n        The input array.\n    out : array_like, optional\n        A location into which the result is stored. If provided, it must have a\n        shape that the input broadcasts to. If not provided or None, a\n        freshly-allocated boolean array is returned.\n\n    Returns\n    -------\n    out : ndarray\n        A boolean array with the same dimensions as the input.\n        If second argument is not supplied then a boolean array is returned\n        with values True where the corresponding element of the input is\n        positive infinity and values False where the element of the input is\n        not positive infinity.\n\n        If a second argument is supplied the result is stored there. If the\n        type of that array is a numeric type the result is represented as zeros\n        and ones, if the type is boolean then as False and True.\n        The return value `out` is then a reference to that array.\n\n    See Also\n    --------\n    isinf, isneginf, isfinite, isnan\n\n    Notes\n    -----\n    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic\n    (IEEE 754).\n\n    Errors result if the second argument is also supplied when x is a scalar\n    input, if first and second arguments have different shapes, or if the\n    first argument has complex values\n\n    Examples\n    --------\n    >>> np.isposinf(np.inf)\n    True\n    >>> np.isposinf(-np.inf)\n    False\n    >>> np.isposinf([-np.inf, 0., np.inf])\n    array([False, False,  True])\n\n    >>> x = np.array([-np.inf, 0., np.inf])\n    >>> y = np.array([2, 2, 2])\n    >>> np.isposinf(x, y)\n    array([0, 0, 1])\n    >>> y\n    array([0, 0, 1])\n\n    '
    is_inf = nx.isinf(x)
    try:
        signbit = ~nx.signbit(x)
    except TypeError as e:
        dtype = nx.asanyarray(x).dtype
        raise TypeError(f'This operation is not supported for {dtype} values because it would be ambiguous.') from e
    else:
        return nx.logical_and(is_inf, signbit, out)

@array_function_dispatch(_dispatcher, verify=False, module='numpy')
def isneginf(x, out=None):
    if False:
        i = 10
        return i + 15
    '\n    Test element-wise for negative infinity, return result as bool array.\n\n    Parameters\n    ----------\n    x : array_like\n        The input array.\n    out : array_like, optional\n        A location into which the result is stored. If provided, it must have a\n        shape that the input broadcasts to. If not provided or None, a\n        freshly-allocated boolean array is returned.\n\n    Returns\n    -------\n    out : ndarray\n        A boolean array with the same dimensions as the input.\n        If second argument is not supplied then a numpy boolean array is\n        returned with values True where the corresponding element of the\n        input is negative infinity and values False where the element of\n        the input is not negative infinity.\n\n        If a second argument is supplied the result is stored there. If the\n        type of that array is a numeric type the result is represented as\n        zeros and ones, if the type is boolean then as False and True. The\n        return value `out` is then a reference to that array.\n\n    See Also\n    --------\n    isinf, isposinf, isnan, isfinite\n\n    Notes\n    -----\n    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic\n    (IEEE 754).\n\n    Errors result if the second argument is also supplied when x is a scalar\n    input, if first and second arguments have different shapes, or if the\n    first argument has complex values.\n\n    Examples\n    --------\n    >>> np.isneginf(-np.inf)\n    True\n    >>> np.isneginf(np.inf)\n    False\n    >>> np.isneginf([-np.inf, 0., np.inf])\n    array([ True, False, False])\n\n    >>> x = np.array([-np.inf, 0., np.inf])\n    >>> y = np.array([2, 2, 2])\n    >>> np.isneginf(x, y)\n    array([1, 0, 0])\n    >>> y\n    array([1, 0, 0])\n\n    '
    is_inf = nx.isinf(x)
    try:
        signbit = nx.signbit(x)
    except TypeError as e:
        dtype = nx.asanyarray(x).dtype
        raise TypeError(f'This operation is not supported for {dtype} values because it would be ambiguous.') from e
    else:
        return nx.logical_and(is_inf, signbit, out)