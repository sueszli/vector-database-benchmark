"""
Set operations for arrays based on sorting.

:Contains:
  unique,
  isin,
  ediff1d,
  intersect1d,
  setxor1d,
  in1d,
  union1d,
  setdiff1d

:Notes:

For floating point arrays, inaccurate results may appear due to usual round-off
and floating point comparison issues.

Speed could be gained in some operations by an implementation of
sort(), that can provide directly the permutation vectors, avoiding
thus calls to argsort().

To do: Optionally return indices analogously to unique for all functions.

:Author: Robert Cimrman

"""
from __future__ import division, absolute_import, print_function
import functools
import numpy as np
from numpy.core import overrides
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy')
__all__ = ['ediff1d', 'intersect1d', 'setxor1d', 'union1d', 'setdiff1d', 'unique', 'in1d', 'isin']

def _ediff1d_dispatcher(ary, to_end=None, to_begin=None):
    if False:
        while True:
            i = 10
    return (ary, to_end, to_begin)

@array_function_dispatch(_ediff1d_dispatcher)
def ediff1d(ary, to_end=None, to_begin=None):
    if False:
        return 10
    '\n    The differences between consecutive elements of an array.\n\n    Parameters\n    ----------\n    ary : array_like\n        If necessary, will be flattened before the differences are taken.\n    to_end : array_like, optional\n        Number(s) to append at the end of the returned differences.\n    to_begin : array_like, optional\n        Number(s) to prepend at the beginning of the returned differences.\n\n    Returns\n    -------\n    ediff1d : ndarray\n        The differences. Loosely, this is ``ary.flat[1:] - ary.flat[:-1]``.\n\n    See Also\n    --------\n    diff, gradient\n\n    Notes\n    -----\n    When applied to masked arrays, this function drops the mask information\n    if the `to_begin` and/or `to_end` parameters are used.\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 4, 7, 0])\n    >>> np.ediff1d(x)\n    array([ 1,  2,  3, -7])\n\n    >>> np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99]))\n    array([-99,   1,   2,   3,  -7,  88,  99])\n\n    The returned array is always 1D.\n\n    >>> y = [[1, 2, 4], [1, 6, 24]]\n    >>> np.ediff1d(y)\n    array([ 1,  2, -3,  5, 18])\n\n    '
    ary = np.asanyarray(ary).ravel()
    dtype_req = ary.dtype
    if to_begin is None and to_end is None:
        return ary[1:] - ary[:-1]
    if to_begin is None:
        l_begin = 0
    else:
        _to_begin = np.asanyarray(to_begin, dtype=dtype_req)
        if not np.all(_to_begin == to_begin):
            raise ValueError("cannot convert 'to_begin' to array with dtype '%r' as required for input ary" % dtype_req)
        to_begin = _to_begin.ravel()
        l_begin = len(to_begin)
    if to_end is None:
        l_end = 0
    else:
        _to_end = np.asanyarray(to_end, dtype=dtype_req)
        if not np.all(_to_end == to_end):
            raise ValueError("cannot convert 'to_end' to array with dtype '%r' as required for input ary" % dtype_req)
        to_end = _to_end.ravel()
        l_end = len(to_end)
    l_diff = max(len(ary) - 1, 0)
    result = np.empty(l_diff + l_begin + l_end, dtype=ary.dtype)
    result = ary.__array_wrap__(result)
    if l_begin > 0:
        result[:l_begin] = to_begin
    if l_end > 0:
        result[l_begin + l_diff:] = to_end
    np.subtract(ary[1:], ary[:-1], result[l_begin:l_begin + l_diff])
    return result

def _unpack_tuple(x):
    if False:
        return 10
    ' Unpacks one-element tuples for use as return values '
    if len(x) == 1:
        return x[0]
    else:
        return x

def _unique_dispatcher(ar, return_index=None, return_inverse=None, return_counts=None, axis=None):
    if False:
        i = 10
        return i + 15
    return (ar,)

@array_function_dispatch(_unique_dispatcher)
def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Find the unique elements of an array.\n\n    Returns the sorted unique elements of an array. There are three optional\n    outputs in addition to the unique elements:\n\n    * the indices of the input array that give the unique values\n    * the indices of the unique array that reconstruct the input array\n    * the number of times each unique value comes up in the input array\n\n    Parameters\n    ----------\n    ar : array_like\n        Input array. Unless `axis` is specified, this will be flattened if it\n        is not already 1-D.\n    return_index : bool, optional\n        If True, also return the indices of `ar` (along the specified axis,\n        if provided, or in the flattened array) that result in the unique array.\n    return_inverse : bool, optional\n        If True, also return the indices of the unique array (for the specified\n        axis, if provided) that can be used to reconstruct `ar`.\n    return_counts : bool, optional\n        If True, also return the number of times each unique item appears\n        in `ar`.\n\n        .. versionadded:: 1.9.0\n\n    axis : int or None, optional\n        The axis to operate on. If None, `ar` will be flattened. If an integer,\n        the subarrays indexed by the given axis will be flattened and treated\n        as the elements of a 1-D array with the dimension of the given axis,\n        see the notes for more details.  Object arrays or structured arrays\n        that contain objects are not supported if the `axis` kwarg is used. The\n        default is None.\n\n        .. versionadded:: 1.13.0\n\n    Returns\n    -------\n    unique : ndarray\n        The sorted unique values.\n    unique_indices : ndarray, optional\n        The indices of the first occurrences of the unique values in the\n        original array. Only provided if `return_index` is True.\n    unique_inverse : ndarray, optional\n        The indices to reconstruct the original array from the\n        unique array. Only provided if `return_inverse` is True.\n    unique_counts : ndarray, optional\n        The number of times each of the unique values comes up in the\n        original array. Only provided if `return_counts` is True.\n\n        .. versionadded:: 1.9.0\n\n    See Also\n    --------\n    numpy.lib.arraysetops : Module with a number of other functions for\n                            performing set operations on arrays.\n\n    Notes\n    -----\n    When an axis is specified the subarrays indexed by the axis are sorted.\n    This is done by making the specified axis the first dimension of the array\n    and then flattening the subarrays in C order. The flattened subarrays are\n    then viewed as a structured type with each element given a label, with the\n    effect that we end up with a 1-D array of structured types that can be\n    treated in the same way as any other 1-D array. The result is that the\n    flattened subarrays are sorted in lexicographic order starting with the\n    first element.\n\n    Examples\n    --------\n    >>> np.unique([1, 1, 2, 2, 3, 3])\n    array([1, 2, 3])\n    >>> a = np.array([[1, 1], [2, 3]])\n    >>> np.unique(a)\n    array([1, 2, 3])\n\n    Return the unique rows of a 2D array\n\n    >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])\n    >>> np.unique(a, axis=0)\n    array([[1, 0, 0], [2, 3, 4]])\n\n    Return the indices of the original array that give the unique values:\n\n    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])\n    >>> u, indices = np.unique(a, return_index=True)\n    >>> u\n    array(['a', 'b', 'c'],\n           dtype='|S1')\n    >>> indices\n    array([0, 1, 3])\n    >>> a[indices]\n    array(['a', 'b', 'c'],\n           dtype='|S1')\n\n    Reconstruct the input array from the unique values:\n\n    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])\n    >>> u, indices = np.unique(a, return_inverse=True)\n    >>> u\n    array([1, 2, 3, 4, 6])\n    >>> indices\n    array([0, 1, 4, 3, 1, 2, 1])\n    >>> u[indices]\n    array([1, 2, 6, 4, 2, 3, 2])\n\n    "
    ar = np.asanyarray(ar)
    if axis is None:
        ret = _unique1d(ar, return_index, return_inverse, return_counts)
        return _unpack_tuple(ret)
    try:
        ar = np.swapaxes(ar, axis, 0)
    except np.AxisError:
        raise np.AxisError(axis, ar.ndim)
    (orig_shape, orig_dtype) = (ar.shape, ar.dtype)
    ar = ar.reshape(orig_shape[0], -1)
    ar = np.ascontiguousarray(ar)
    dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]
    try:
        consolidated = ar.view(dtype)
    except TypeError:
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype))

    def reshape_uniq(uniq):
        if False:
            for i in range(10):
                print('nop')
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(-1, *orig_shape[1:])
        uniq = np.swapaxes(uniq, 0, axis)
        return uniq
    output = _unique1d(consolidated, return_index, return_inverse, return_counts)
    output = (reshape_uniq(output[0]),) + output[1:]
    return _unpack_tuple(output)

def _unique1d(ar, return_index=False, return_inverse=False, return_counts=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find the unique elements of an array, ignoring shape.\n    '
    ar = np.asanyarray(ar).flatten()
    optional_indices = return_index or return_inverse
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    ret = (aux[mask],)
    if return_index:
        ret += (perm[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        ret += (inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret

def _intersect1d_dispatcher(ar1, ar2, assume_unique=None, return_indices=None):
    if False:
        print('Hello World!')
    return (ar1, ar2)

@array_function_dispatch(_intersect1d_dispatcher)
def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    if False:
        print('Hello World!')
    '\n    Find the intersection of two arrays.\n\n    Return the sorted, unique values that are in both of the input arrays.\n\n    Parameters\n    ----------\n    ar1, ar2 : array_like\n        Input arrays. Will be flattened if not already 1D.\n    assume_unique : bool\n        If True, the input arrays are both assumed to be unique, which\n        can speed up the calculation.  Default is False.\n    return_indices : bool\n        If True, the indices which correspond to the intersection of the two\n        arrays are returned. The first instance of a value is used if there are\n        multiple. Default is False.\n\n        .. versionadded:: 1.15.0\n\n    Returns\n    -------\n    intersect1d : ndarray\n        Sorted 1D array of common and unique elements.\n    comm1 : ndarray\n        The indices of the first occurrences of the common values in `ar1`.\n        Only provided if `return_indices` is True.\n    comm2 : ndarray\n        The indices of the first occurrences of the common values in `ar2`.\n        Only provided if `return_indices` is True.\n\n\n    See Also\n    --------\n    numpy.lib.arraysetops : Module with a number of other functions for\n                            performing set operations on arrays.\n\n    Examples\n    --------\n    >>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])\n    array([1, 3])\n\n    To intersect more than two arrays, use functools.reduce:\n\n    >>> from functools import reduce\n    >>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))\n    array([3])\n\n    To return the indices of the values common to the input arrays\n    along with the intersected values:\n    >>> x = np.array([1, 1, 2, 3, 4])\n    >>> y = np.array([2, 1, 4, 6])\n    >>> xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)\n    >>> x_ind, y_ind\n    (array([0, 2, 4]), array([1, 0, 2]))\n    >>> xy, x[x_ind], y[y_ind]\n    (array([1, 2, 4]), array([1, 2, 4]), array([1, 2, 4]))\n\n    '
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)
    if not assume_unique:
        if return_indices:
            (ar1, ind1) = unique(ar1, return_index=True)
            (ar2, ind2) = unique(ar2, return_index=True)
        else:
            ar1 = unique(ar1)
            ar2 = unique(ar2)
    else:
        ar1 = ar1.ravel()
        ar2 = ar2.ravel()
    aux = np.concatenate((ar1, ar2))
    if return_indices:
        aux_sort_indices = np.argsort(aux, kind='mergesort')
        aux = aux[aux_sort_indices]
    else:
        aux.sort()
    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]
    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - ar1.size
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]
        return (int1d, ar1_indices, ar2_indices)
    else:
        return int1d

def _setxor1d_dispatcher(ar1, ar2, assume_unique=None):
    if False:
        return 10
    return (ar1, ar2)

@array_function_dispatch(_setxor1d_dispatcher)
def setxor1d(ar1, ar2, assume_unique=False):
    if False:
        return 10
    '\n    Find the set exclusive-or of two arrays.\n\n    Return the sorted, unique values that are in only one (not both) of the\n    input arrays.\n\n    Parameters\n    ----------\n    ar1, ar2 : array_like\n        Input arrays.\n    assume_unique : bool\n        If True, the input arrays are both assumed to be unique, which\n        can speed up the calculation.  Default is False.\n\n    Returns\n    -------\n    setxor1d : ndarray\n        Sorted 1D array of unique values that are in only one of the input\n        arrays.\n\n    Examples\n    --------\n    >>> a = np.array([1, 2, 3, 2, 4])\n    >>> b = np.array([2, 3, 5, 7, 5])\n    >>> np.setxor1d(a,b)\n    array([1, 4, 5, 7])\n\n    '
    if not assume_unique:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    aux = np.concatenate((ar1, ar2))
    if aux.size == 0:
        return aux
    aux.sort()
    flag = np.concatenate(([True], aux[1:] != aux[:-1], [True]))
    return aux[flag[1:] & flag[:-1]]

def _in1d_dispatcher(ar1, ar2, assume_unique=None, invert=None):
    if False:
        i = 10
        return i + 15
    return (ar1, ar2)

@array_function_dispatch(_in1d_dispatcher)
def in1d(ar1, ar2, assume_unique=False, invert=False):
    if False:
        print('Hello World!')
    '\n    Test whether each element of a 1-D array is also present in a second array.\n\n    Returns a boolean array the same length as `ar1` that is True\n    where an element of `ar1` is in `ar2` and False otherwise.\n\n    We recommend using :func:`isin` instead of `in1d` for new code.\n\n    Parameters\n    ----------\n    ar1 : (M,) array_like\n        Input array.\n    ar2 : array_like\n        The values against which to test each value of `ar1`.\n    assume_unique : bool, optional\n        If True, the input arrays are both assumed to be unique, which\n        can speed up the calculation.  Default is False.\n    invert : bool, optional\n        If True, the values in the returned array are inverted (that is,\n        False where an element of `ar1` is in `ar2` and True otherwise).\n        Default is False. ``np.in1d(a, b, invert=True)`` is equivalent\n        to (but is faster than) ``np.invert(in1d(a, b))``.\n\n        .. versionadded:: 1.8.0\n\n    Returns\n    -------\n    in1d : (M,) ndarray, bool\n        The values `ar1[in1d]` are in `ar2`.\n\n    See Also\n    --------\n    isin                  : Version of this function that preserves the\n                            shape of ar1.\n    numpy.lib.arraysetops : Module with a number of other functions for\n                            performing set operations on arrays.\n\n    Notes\n    -----\n    `in1d` can be considered as an element-wise function version of the\n    python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is roughly\n    equivalent to ``np.array([item in b for item in a])``.\n    However, this idea fails if `ar2` is a set, or similar (non-sequence)\n    container:  As ``ar2`` is converted to an array, in those cases\n    ``asarray(ar2)`` is an object array rather than the expected array of\n    contained values.\n\n    .. versionadded:: 1.4.0\n\n    Examples\n    --------\n    >>> test = np.array([0, 1, 2, 5, 0])\n    >>> states = [0, 2]\n    >>> mask = np.in1d(test, states)\n    >>> mask\n    array([ True, False,  True, False,  True])\n    >>> test[mask]\n    array([0, 2, 0])\n    >>> mask = np.in1d(test, states, invert=True)\n    >>> mask\n    array([False,  True, False,  True, False])\n    >>> test[mask]\n    array([1, 5])\n    '
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()
    contains_object = ar1.dtype.hasobject or ar2.dtype.hasobject
    if len(ar2) < 10 * len(ar1) ** 0.145 or contains_object:
        if invert:
            mask = np.ones(len(ar1), dtype=bool)
            for a in ar2:
                mask &= ar1 != a
        else:
            mask = np.zeros(len(ar1), dtype=bool)
            for a in ar2:
                mask |= ar1 == a
        return mask
    if not assume_unique:
        (ar1, rev_idx) = np.unique(ar1, return_inverse=True)
        ar2 = np.unique(ar2)
    ar = np.concatenate((ar1, ar2))
    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    if invert:
        bool_ar = sar[1:] != sar[:-1]
    else:
        bool_ar = sar[1:] == sar[:-1]
    flag = np.concatenate((bool_ar, [invert]))
    ret = np.empty(ar.shape, dtype=bool)
    ret[order] = flag
    if assume_unique:
        return ret[:len(ar1)]
    else:
        return ret[rev_idx]

def _isin_dispatcher(element, test_elements, assume_unique=None, invert=None):
    if False:
        i = 10
        return i + 15
    return (element, test_elements)

@array_function_dispatch(_isin_dispatcher)
def isin(element, test_elements, assume_unique=False, invert=False):
    if False:
        return 10
    "\n    Calculates `element in test_elements`, broadcasting over `element` only.\n    Returns a boolean array of the same shape as `element` that is True\n    where an element of `element` is in `test_elements` and False otherwise.\n\n    Parameters\n    ----------\n    element : array_like\n        Input array.\n    test_elements : array_like\n        The values against which to test each value of `element`.\n        This argument is flattened if it is an array or array_like.\n        See notes for behavior with non-array-like parameters.\n    assume_unique : bool, optional\n        If True, the input arrays are both assumed to be unique, which\n        can speed up the calculation.  Default is False.\n    invert : bool, optional\n        If True, the values in the returned array are inverted, as if\n        calculating `element not in test_elements`. Default is False.\n        ``np.isin(a, b, invert=True)`` is equivalent to (but faster\n        than) ``np.invert(np.isin(a, b))``.\n\n    Returns\n    -------\n    isin : ndarray, bool\n        Has the same shape as `element`. The values `element[isin]`\n        are in `test_elements`.\n\n    See Also\n    --------\n    in1d                  : Flattened version of this function.\n    numpy.lib.arraysetops : Module with a number of other functions for\n                            performing set operations on arrays.\n\n    Notes\n    -----\n\n    `isin` is an element-wise function version of the python keyword `in`.\n    ``isin(a, b)`` is roughly equivalent to\n    ``np.array([item in b for item in a])`` if `a` and `b` are 1-D sequences.\n\n    `element` and `test_elements` are converted to arrays if they are not\n    already. If `test_elements` is a set (or other non-sequence collection)\n    it will be converted to an object array with one element, rather than an\n    array of the values contained in `test_elements`. This is a consequence\n    of the `array` constructor's way of handling non-sequence collections.\n    Converting the set to a list usually gives the desired behavior.\n\n    .. versionadded:: 1.13.0\n\n    Examples\n    --------\n    >>> element = 2*np.arange(4).reshape((2, 2))\n    >>> element\n    array([[0, 2],\n           [4, 6]])\n    >>> test_elements = [1, 2, 4, 8]\n    >>> mask = np.isin(element, test_elements)\n    >>> mask\n    array([[ False,  True],\n           [ True,  False]])\n    >>> element[mask]\n    array([2, 4])\n\n    The indices of the matched values can be obtained with `nonzero`:\n\n    >>> np.nonzero(mask)\n    (array([0, 1]), array([1, 0]))\n\n    The test can also be inverted:\n\n    >>> mask = np.isin(element, test_elements, invert=True)\n    >>> mask\n    array([[ True, False],\n           [ False, True]])\n    >>> element[mask]\n    array([0, 6])\n\n    Because of how `array` handles sets, the following does not\n    work as expected:\n\n    >>> test_set = {1, 2, 4, 8}\n    >>> np.isin(element, test_set)\n    array([[ False, False],\n           [ False, False]])\n\n    Casting the set to a list gives the expected result:\n\n    >>> np.isin(element, list(test_set))\n    array([[ False,  True],\n           [ True,  False]])\n    "
    element = np.asarray(element)
    return in1d(element, test_elements, assume_unique=assume_unique, invert=invert).reshape(element.shape)

def _union1d_dispatcher(ar1, ar2):
    if False:
        for i in range(10):
            print('nop')
    return (ar1, ar2)

@array_function_dispatch(_union1d_dispatcher)
def union1d(ar1, ar2):
    if False:
        i = 10
        return i + 15
    '\n    Find the union of two arrays.\n\n    Return the unique, sorted array of values that are in either of the two\n    input arrays.\n\n    Parameters\n    ----------\n    ar1, ar2 : array_like\n        Input arrays. They are flattened if they are not already 1D.\n\n    Returns\n    -------\n    union1d : ndarray\n        Unique, sorted union of the input arrays.\n\n    See Also\n    --------\n    numpy.lib.arraysetops : Module with a number of other functions for\n                            performing set operations on arrays.\n\n    Examples\n    --------\n    >>> np.union1d([-1, 0, 1], [-2, 0, 2])\n    array([-2, -1,  0,  1,  2])\n\n    To find the union of more than two arrays, use functools.reduce:\n\n    >>> from functools import reduce\n    >>> reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))\n    array([1, 2, 3, 4, 6])\n    '
    return unique(np.concatenate((ar1, ar2), axis=None))

def _setdiff1d_dispatcher(ar1, ar2, assume_unique=None):
    if False:
        return 10
    return (ar1, ar2)

@array_function_dispatch(_setdiff1d_dispatcher)
def setdiff1d(ar1, ar2, assume_unique=False):
    if False:
        return 10
    '\n    Find the set difference of two arrays.\n\n    Return the unique values in `ar1` that are not in `ar2`.\n\n    Parameters\n    ----------\n    ar1 : array_like\n        Input array.\n    ar2 : array_like\n        Input comparison array.\n    assume_unique : bool\n        If True, the input arrays are both assumed to be unique, which\n        can speed up the calculation.  Default is False.\n\n    Returns\n    -------\n    setdiff1d : ndarray\n        1D array of values in `ar1` that are not in `ar2`. The result\n        is sorted when `assume_unique=False`, but otherwise only sorted\n        if the input is sorted.\n\n    See Also\n    --------\n    numpy.lib.arraysetops : Module with a number of other functions for\n                            performing set operations on arrays.\n\n    Examples\n    --------\n    >>> a = np.array([1, 2, 3, 2, 4, 1])\n    >>> b = np.array([3, 4, 5, 6])\n    >>> np.setdiff1d(a, b)\n    array([1, 2])\n\n    '
    if assume_unique:
        ar1 = np.asarray(ar1).ravel()
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]