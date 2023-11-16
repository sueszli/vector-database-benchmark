from cupy import _core
import cupy

def _create_float_test_ufunc(name, doc):
    if False:
        return 10
    return _core.create_ufunc('cupy_' + name, ('e->?', 'f->?', 'd->?', 'F->?', 'D->?'), 'out0 = %s(in0)' % name, doc=doc)
isfinite = _create_float_test_ufunc('isfinite', 'Tests finiteness elementwise.\n\n    Each element of returned array is ``True`` only if the corresponding\n    element of the input is finite (i.e. not an infinity nor NaN).\n\n    .. seealso:: :data:`numpy.isfinite`\n\n    ')
isinf = _create_float_test_ufunc('isinf', 'Tests if each element is the positive or negative infinity.\n\n    .. seealso:: :data:`numpy.isinf`\n\n    ')
isnan = _create_float_test_ufunc('isnan', 'Tests if each element is a NaN.\n\n    .. seealso:: :data:`numpy.isnan`\n\n    ')

def isneginf(x, out=None):
    if False:
        return 10
    'Test element-wise for negative infinity, return result as bool array.\n\n    Parameters\n    ----------\n    x : cupy.ndarray\n        Input array.\n    out : cupy.ndarray, optional\n        A location into which the result is stored. If provided,\n        it should have a shape that input broadcasts to.\n        By default, None, a freshly- allocated boolean array,\n        is returned.\n\n    Returns\n    -------\n    y : cupy.ndarray\n        Boolean array of same shape as ``x``.\n\n    Examples\n    --------\n    >>> cupy.isneginf(0)\n    array(False)\n    >>> cupy.isneginf(-cupy.inf)\n    array(True)\n    >>> cupy.isneginf(cupy.array([-cupy.inf, -4, cupy.nan, 0, 4, cupy.inf]))\n    array([ True, False, False, False, False, False])\n\n    See Also\n    --------\n    numpy.isneginf\n\n    '
    is_inf = isinf(x)
    try:
        signbit = cupy.signbit(x)
    except TypeError as e:
        dtype = x.dtype
        raise TypeError(f'This operation is not supported for {dtype} values because it would be ambiguous.') from e
    return cupy.logical_and(is_inf, signbit, out=out)

def isposinf(x, out=None):
    if False:
        while True:
            i = 10
    'Test element-wise for positive infinity, return result as bool array.\n\n    Parameters\n    ----------\n    x : cupy.ndarray\n        Input array.\n    out : cupy.ndarray\n        A location into which the result is stored. If provided,\n        it should have a shape that input broadcasts to.\n        By default, None, a freshly- allocated boolean array,\n        is returned.\n\n    Returns\n    -------\n    y : cupy.ndarray\n        Boolean array of same shape as ``x``.\n\n    Examples\n    --------\n    >>> cupy.isposinf(0)\n    array(False)\n    >>> cupy.isposinf(cupy.inf)\n    array(True)\n    >>> cupy.isposinf(cupy.array([-cupy.inf, -4, cupy.nan, 0, 4, cupy.inf]))\n    array([False, False, False, False, False,  True])\n\n    See Also\n    --------\n    numpy.isposinf\n\n    '
    is_inf = isinf(x)
    try:
        signbit = ~cupy.signbit(x)
    except TypeError as e:
        dtype = x.dtype
        raise TypeError(f'This operation is not supported for {dtype} values because it would be ambiguous.') from e
    return cupy.logical_and(is_inf, signbit, out=out)