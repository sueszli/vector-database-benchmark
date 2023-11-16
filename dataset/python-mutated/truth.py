import warnings
import cupy
from cupy._core import _routines_logic as _logic
from cupy._core import _fusion_thread_local
from cupy._sorting import search as _search
from cupy import _util
_setxorkernel = cupy._core.ElementwiseKernel('raw T X, int64 len', 'bool z', 'z = (i == 0 || X[i] != X[i-1]) && (i == len - 1 || X[i] != X[i+1])', 'setxorkernel')

def all(a, axis=None, out=None, keepdims=False):
    if False:
        i = 10
        return i + 15
    'Tests whether all array elements along a given axis evaluate to True.\n\n    Parameters\n    ----------\n    a : cupy.ndarray\n        Input array.\n    axis : int or tuple of ints\n        Along which axis to compute all.\n        The flattened array is used by default.\n    out : cupy.ndarray\n        Output array.\n    keepdims : bool\n        If ``True``, the axis is remained as an axis of size one.\n\n    Returns\n    -------\n    y : cupy.ndarray\n        An array reduced of the input array along the axis.\n\n    See Also\n    --------\n    numpy.all\n\n    '
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError('cupy.all does not support `keepdims` in fusion yet.')
        return _fusion_thread_local.call_reduction(_logic.all, a, axis=axis, out=out)
    _util.check_array(a, arg_name='a')
    return a.all(axis=axis, out=out, keepdims=keepdims)

def any(a, axis=None, out=None, keepdims=False):
    if False:
        print('Hello World!')
    'Tests whether any array elements along a given axis evaluate to True.\n\n    Parameters\n    ----------\n    a : cupy.ndarray\n        Input array.\n    axis : int or tuple of ints\n        Along which axis to compute all.\n        The flattened array is used by default.\n    out : cupy.ndarray\n        Output array.\n    keepdims : bool\n        If ``True``, the axis is remained as an axis of size one.\n\n    Returns\n    -------\n    y : cupy.ndarray\n        An array reduced of the input array along the axis.\n\n    See Also\n    --------\n    numpy.any\n\n    '
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError('cupy.any does not support `keepdims` in fusion yet.')
        return _fusion_thread_local.call_reduction(_logic.any, a, axis=axis, out=out)
    _util.check_array(a, arg_name='a')
    return a.any(axis=axis, out=out, keepdims=keepdims)

def in1d(ar1, ar2, assume_unique=False, invert=False):
    if False:
        print('Hello World!')
    'Tests whether each element of a 1-D array is also present in a second\n    array.\n\n    Returns a boolean array the same length as ``ar1`` that is ``True``\n    where an element of ``ar1`` is in ``ar2`` and ``False`` otherwise.\n\n    Parameters\n    ----------\n    ar1 : cupy.ndarray\n        Input array.\n    ar2 : cupy.ndarray\n        The values against which to test each value of ``ar1``.\n    assume_unique : bool, optional\n        Ignored\n    invert : bool, optional\n        If ``True``, the values in the returned array\n        are inverted (that is, ``False`` where an element of ``ar1`` is in\n        ``ar2`` and ``True`` otherwise). Default is ``False``.\n\n    Returns\n    -------\n    y : cupy.ndarray, bool\n        The values ``ar1[in1d]`` are in ``ar2``.\n\n    '
    ar1 = ar1.ravel()
    ar2 = ar2.ravel()
    if ar1.size == 0 or ar2.size == 0:
        if invert:
            return cupy.ones(ar1.shape, dtype=cupy.bool_)
        else:
            return cupy.zeros(ar1.shape, dtype=cupy.bool_)
    ar2 = cupy.sort(ar2)
    return _search._exists_kernel(ar1, ar2, ar2.size, invert)

def intersect1d(arr1, arr2, assume_unique=False, return_indices=False):
    if False:
        print('Hello World!')
    'Find the intersection of two arrays.\n    Returns the sorted, unique values that are in both of the input arrays.\n\n    Parameters\n    ----------\n    arr1, arr2 : cupy.ndarray\n        Input arrays. Arrays will be flattened if they are not in 1D.\n    assume_unique : bool\n        By default, False. If set True, the input arrays will be\n        assumend to be unique, which speeds up the calculation. If set True,\n        but the arrays are not unique, incorrect results and out-of-bounds\n        indices could result.\n    return_indices : bool\n       By default, False. If True, the indices which correspond to the\n       intersection of the two arrays are returned.\n\n    Returns\n    -------\n    intersect1d : cupy.ndarray\n        Sorted 1D array of common and unique elements.\n    comm1 : cupy.ndarray\n        The indices of the first occurrences of the common values\n        in `arr1`. Only provided if `return_indices` is True.\n    comm2 : cupy.ndarray\n        The indices of the first occurrences of the common values\n        in `arr2`. Only provided if `return_indices` is True.\n\n    See Also\n    --------\n    numpy.intersect1d\n\n    '
    if not assume_unique:
        if return_indices:
            (arr1, ind1) = cupy.unique(arr1, return_index=True)
            (arr2, ind2) = cupy.unique(arr2, return_index=True)
        else:
            arr1 = cupy.unique(arr1)
            arr2 = cupy.unique(arr2)
    else:
        arr1 = arr1.ravel()
        arr2 = arr2.ravel()
    if not return_indices:
        mask = _search._exists_kernel(arr1, arr2, arr2.size, False)
        return arr1[mask]
    (mask, v1) = _search._exists_and_searchsorted_kernel(arr1, arr2, arr2.size, False)
    int1d = arr1[mask]
    arr1_indices = cupy.flatnonzero(mask)
    arr2_indices = v1[mask]
    if not assume_unique:
        arr1_indices = ind1[arr1_indices]
        arr2_indices = ind2[arr2_indices]
    return (int1d, arr1_indices, arr2_indices)

def isin(element, test_elements, assume_unique=False, invert=False):
    if False:
        i = 10
        return i + 15
    'Calculates element in ``test_elements``, broadcasting over ``element``\n    only. Returns a boolean array of the same shape as ``element`` that is\n    ``True`` where an element of ``element`` is in ``test_elements`` and\n    ``False`` otherwise.\n\n    Parameters\n    ----------\n    element : cupy.ndarray\n        Input array.\n    test_elements : cupy.ndarray\n        The values against which to test each\n        value of ``element``. This argument is flattened if it is an\n        array or array_like.\n    assume_unique : bool, optional\n        Ignored\n    invert : bool, optional\n        If ``True``, the values in the returned array\n        are inverted, as if calculating element not in ``test_elements``.\n        Default is ``False``.\n\n    Returns\n    -------\n    y : cupy.ndarray, bool\n        Has the same shape as ``element``. The values ``element[isin]``\n        are in ``test_elements``.\n\n    '
    return in1d(element, test_elements, assume_unique=assume_unique, invert=invert).reshape(element.shape)

def setdiff1d(ar1, ar2, assume_unique=False):
    if False:
        print('Hello World!')
    'Find the set difference of two arrays. It returns unique\n    values in `ar1` that are not in `ar2`.\n\n    Parameters\n    ----------\n    ar1 : cupy.ndarray\n        Input array\n    ar2 : cupy.ndarray\n        Input array for comparision\n    assume_unique : bool\n        By default, False, i.e. input arrays are not unique.\n        If True, input arrays are assumed to be unique. This can\n        speed up the calculation.\n\n    Returns\n    -------\n    setdiff1d : cupy.ndarray\n        Returns a 1D array of values in `ar1` that are not in `ar2`.\n        It always returns a sorted output for unsorted input only\n        if `assume_unique=False`.\n\n    See Also\n    --------\n    numpy.setdiff1d\n\n    '
    if assume_unique:
        ar1 = cupy.ravel(ar1)
    else:
        ar1 = cupy.unique(ar1)
        ar2 = cupy.unique(ar2)
    return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]

def setxor1d(ar1, ar2, assume_unique=False):
    if False:
        for i in range(10):
            print('nop')
    'Find the set exclusive-or of two arrays.\n\n    Parameters\n    ----------\n    ar1, ar2 : cupy.ndarray\n        Input arrays. They are flattend if they are not already 1-D.\n    assume_unique : bool\n        By default, False, i.e. input arrays are not unique.\n        If True, input arrays are assumed to be unique. This can\n        speed up the calculation.\n\n    Returns\n    -------\n    setxor1d : cupy.ndarray\n        Return the sorted, unique values that are in only one\n        (not both) of the input arrays.\n\n    See Also\n    --------\n    numpy.setxor1d\n\n    '
    if not assume_unique:
        ar1 = cupy.unique(ar1)
        ar2 = cupy.unique(ar2)
    aux = cupy.concatenate((ar1, ar2), axis=None)
    if aux.size == 0:
        return aux
    aux.sort()
    return aux[_setxorkernel(aux, aux.size, cupy.zeros(aux.size, dtype=cupy.bool_))]

def union1d(arr1, arr2):
    if False:
        return 10
    'Find the union of two arrays.\n\n    Returns the unique, sorted array of values that are in either of\n    the two input arrays.\n\n    Parameters\n    ----------\n    arr1, arr2 : cupy.ndarray\n        Input arrays. They are flattend if they are not already 1-D.\n\n    Returns\n    -------\n    union1d : cupy.ndarray\n        Sorted union of the input arrays.\n\n    See Also\n    --------\n    numpy.union1d\n\n    '
    return cupy.unique(cupy.concatenate((arr1, arr2), axis=None))

def alltrue(a, axis=None, out=None, keepdims=False):
    if False:
        while True:
            i = 10
    warnings.warn('Please use `all` instead.', DeprecationWarning)
    return all(a, axis, out, keepdims)

def sometrue(a, axis=None, out=None, keepdims=False):
    if False:
        while True:
            i = 10
    warnings.warn('Please use `any` instead.', DeprecationWarning)
    return any(a, axis, out, keepdims)