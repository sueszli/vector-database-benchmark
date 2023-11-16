import builtins
import collections.abc
import functools
import re
import sys
import warnings
import numpy as np
import numpy._core.numeric as _nx
from numpy._core import transpose, overrides
from numpy._core.numeric import ones, zeros_like, arange, concatenate, array, asarray, asanyarray, empty, ndarray, take, dot, where, intp, integer, isscalar, absolute
from numpy._core.umath import pi, add, arctan2, frompyfunc, cos, less_equal, sqrt, sin, mod, exp, not_equal, subtract
from numpy._core.fromnumeric import ravel, nonzero, partition, mean, any, sum
from numpy._core.numerictypes import typecodes
from numpy.lib._twodim_base_impl import diag
from numpy._core.multiarray import _place, bincount, normalize_axis_index, _monotonicity, interp as compiled_interp, interp_complex as compiled_interp_complex
from numpy._utils import set_module
from numpy.lib._histograms_impl import histogram, histogramdd
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy')
__all__ = ['select', 'piecewise', 'trim_zeros', 'copy', 'iterable', 'percentile', 'diff', 'gradient', 'angle', 'unwrap', 'sort_complex', 'flip', 'rot90', 'extract', 'place', 'vectorize', 'asarray_chkfinite', 'average', 'bincount', 'digitize', 'cov', 'corrcoef', 'median', 'sinc', 'hamming', 'hanning', 'bartlett', 'blackman', 'kaiser', 'trapz', 'i0', 'meshgrid', 'delete', 'insert', 'append', 'interp', 'quantile']
_QuantileMethods = dict(inverted_cdf=dict(get_virtual_index=lambda n, quantiles: _inverted_cdf(n, quantiles), fix_gamma=lambda gamma, _: gamma), averaged_inverted_cdf=dict(get_virtual_index=lambda n, quantiles: n * quantiles - 1, fix_gamma=lambda gamma, _: _get_gamma_mask(shape=gamma.shape, default_value=1.0, conditioned_value=0.5, where=gamma == 0)), closest_observation=dict(get_virtual_index=lambda n, quantiles: _closest_observation(n, quantiles), fix_gamma=lambda gamma, _: gamma), interpolated_inverted_cdf=dict(get_virtual_index=lambda n, quantiles: _compute_virtual_index(n, quantiles, 0, 1), fix_gamma=lambda gamma, _: gamma), hazen=dict(get_virtual_index=lambda n, quantiles: _compute_virtual_index(n, quantiles, 0.5, 0.5), fix_gamma=lambda gamma, _: gamma), weibull=dict(get_virtual_index=lambda n, quantiles: _compute_virtual_index(n, quantiles, 0, 0), fix_gamma=lambda gamma, _: gamma), linear=dict(get_virtual_index=lambda n, quantiles: (n - 1) * quantiles, fix_gamma=lambda gamma, _: gamma), median_unbiased=dict(get_virtual_index=lambda n, quantiles: _compute_virtual_index(n, quantiles, 1 / 3.0, 1 / 3.0), fix_gamma=lambda gamma, _: gamma), normal_unbiased=dict(get_virtual_index=lambda n, quantiles: _compute_virtual_index(n, quantiles, 3 / 8.0, 3 / 8.0), fix_gamma=lambda gamma, _: gamma), lower=dict(get_virtual_index=lambda n, quantiles: np.floor((n - 1) * quantiles).astype(np.intp), fix_gamma=lambda gamma, _: gamma), higher=dict(get_virtual_index=lambda n, quantiles: np.ceil((n - 1) * quantiles).astype(np.intp), fix_gamma=lambda gamma, _: gamma), midpoint=dict(get_virtual_index=lambda n, quantiles: 0.5 * (np.floor((n - 1) * quantiles) + np.ceil((n - 1) * quantiles)), fix_gamma=lambda gamma, index: _get_gamma_mask(shape=gamma.shape, default_value=0.5, conditioned_value=0.0, where=index % 1 == 0)), nearest=dict(get_virtual_index=lambda n, quantiles: np.around((n - 1) * quantiles).astype(np.intp), fix_gamma=lambda gamma, _: gamma))

def _rot90_dispatcher(m, k=None, axes=None):
    if False:
        i = 10
        return i + 15
    return (m,)

@array_function_dispatch(_rot90_dispatcher)
def rot90(m, k=1, axes=(0, 1)):
    if False:
        while True:
            i = 10
    '\n    Rotate an array by 90 degrees in the plane specified by axes.\n\n    Rotation direction is from the first towards the second axis.\n    This means for a 2D array with the default `k` and `axes`, the\n    rotation will be counterclockwise.\n\n    Parameters\n    ----------\n    m : array_like\n        Array of two or more dimensions.\n    k : integer\n        Number of times the array is rotated by 90 degrees.\n    axes : (2,) array_like\n        The array is rotated in the plane defined by the axes.\n        Axes must be different.\n\n        .. versionadded:: 1.12.0\n\n    Returns\n    -------\n    y : ndarray\n        A rotated view of `m`.\n\n    See Also\n    --------\n    flip : Reverse the order of elements in an array along the given axis.\n    fliplr : Flip an array horizontally.\n    flipud : Flip an array vertically.\n\n    Notes\n    -----\n    ``rot90(m, k=1, axes=(1,0))``  is the reverse of\n    ``rot90(m, k=1, axes=(0,1))``\n\n    ``rot90(m, k=1, axes=(1,0))`` is equivalent to\n    ``rot90(m, k=-1, axes=(0,1))``\n\n    Examples\n    --------\n    >>> m = np.array([[1,2],[3,4]], int)\n    >>> m\n    array([[1, 2],\n           [3, 4]])\n    >>> np.rot90(m)\n    array([[2, 4],\n           [1, 3]])\n    >>> np.rot90(m, 2)\n    array([[4, 3],\n           [2, 1]])\n    >>> m = np.arange(8).reshape((2,2,2))\n    >>> np.rot90(m, 1, (1,2))\n    array([[[1, 3],\n            [0, 2]],\n           [[5, 7],\n            [4, 6]]])\n\n    '
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError('len(axes) must be 2.')
    m = asanyarray(m)
    if axes[0] == axes[1] or absolute(axes[0] - axes[1]) == m.ndim:
        raise ValueError('Axes must be different.')
    if axes[0] >= m.ndim or axes[0] < -m.ndim or axes[1] >= m.ndim or (axes[1] < -m.ndim):
        raise ValueError('Axes={} out of range for array of ndim={}.'.format(axes, m.ndim))
    k %= 4
    if k == 0:
        return m[:]
    if k == 2:
        return flip(flip(m, axes[0]), axes[1])
    axes_list = arange(0, m.ndim)
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]], axes_list[axes[0]])
    if k == 1:
        return transpose(flip(m, axes[1]), axes_list)
    else:
        return flip(transpose(m, axes_list), axes[1])

def _flip_dispatcher(m, axis=None):
    if False:
        i = 10
        return i + 15
    return (m,)

@array_function_dispatch(_flip_dispatcher)
def flip(m, axis=None):
    if False:
        while True:
            i = 10
    '\n    Reverse the order of elements in an array along the given axis.\n\n    The shape of the array is preserved, but the elements are reordered.\n\n    .. versionadded:: 1.12.0\n\n    Parameters\n    ----------\n    m : array_like\n        Input array.\n    axis : None or int or tuple of ints, optional\n         Axis or axes along which to flip over. The default,\n         axis=None, will flip over all of the axes of the input array.\n         If axis is negative it counts from the last to the first axis.\n\n         If axis is a tuple of ints, flipping is performed on all of the axes\n         specified in the tuple.\n\n         .. versionchanged:: 1.15.0\n            None and tuples of axes are supported\n\n    Returns\n    -------\n    out : array_like\n        A view of `m` with the entries of axis reversed.  Since a view is\n        returned, this operation is done in constant time.\n\n    See Also\n    --------\n    flipud : Flip an array vertically (axis=0).\n    fliplr : Flip an array horizontally (axis=1).\n\n    Notes\n    -----\n    flip(m, 0) is equivalent to flipud(m).\n\n    flip(m, 1) is equivalent to fliplr(m).\n\n    flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.\n\n    flip(m) corresponds to ``m[::-1,::-1,...,::-1]`` with ``::-1`` at all\n    positions.\n\n    flip(m, (0, 1)) corresponds to ``m[::-1,::-1,...]`` with ``::-1`` at\n    position 0 and position 1.\n\n    Examples\n    --------\n    >>> A = np.arange(8).reshape((2,2,2))\n    >>> A\n    array([[[0, 1],\n            [2, 3]],\n           [[4, 5],\n            [6, 7]]])\n    >>> np.flip(A, 0)\n    array([[[4, 5],\n            [6, 7]],\n           [[0, 1],\n            [2, 3]]])\n    >>> np.flip(A, 1)\n    array([[[2, 3],\n            [0, 1]],\n           [[6, 7],\n            [4, 5]]])\n    >>> np.flip(A)\n    array([[[7, 6],\n            [5, 4]],\n           [[3, 2],\n            [1, 0]]])\n    >>> np.flip(A, (0, 2))\n    array([[[5, 4],\n            [7, 6]],\n           [[1, 0],\n            [3, 2]]])\n    >>> A = np.random.randn(3,4,5)\n    >>> np.all(np.flip(A,2) == A[:,:,::-1,...])\n    True\n    '
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    if axis is None:
        indexer = (np.s_[::-1],) * m.ndim
    else:
        axis = _nx.normalize_axis_tuple(axis, m.ndim)
        indexer = [np.s_[:]] * m.ndim
        for ax in axis:
            indexer[ax] = np.s_[::-1]
        indexer = tuple(indexer)
    return m[indexer]

@set_module('numpy')
def iterable(y):
    if False:
        return 10
    '\n    Check whether or not an object can be iterated over.\n\n    Parameters\n    ----------\n    y : object\n      Input object.\n\n    Returns\n    -------\n    b : bool\n      Return ``True`` if the object has an iterator method or is a\n      sequence and ``False`` otherwise.\n\n\n    Examples\n    --------\n    >>> np.iterable([1, 2, 3])\n    True\n    >>> np.iterable(2)\n    False\n\n    Notes\n    -----\n    In most cases, the results of ``np.iterable(obj)`` are consistent with\n    ``isinstance(obj, collections.abc.Iterable)``. One notable exception is\n    the treatment of 0-dimensional arrays::\n\n        >>> from collections.abc import Iterable\n        >>> a = np.array(1.0)  # 0-dimensional numpy array\n        >>> isinstance(a, Iterable)\n        True\n        >>> np.iterable(a)\n        False\n\n    '
    try:
        iter(y)
    except TypeError:
        return False
    return True

def _average_dispatcher(a, axis=None, weights=None, returned=None, *, keepdims=None):
    if False:
        while True:
            i = 10
    return (a, weights)

@array_function_dispatch(_average_dispatcher)
def average(a, axis=None, weights=None, returned=False, *, keepdims=np._NoValue):
    if False:
        print('Hello World!')
    '\n    Compute the weighted average along the specified axis.\n\n    Parameters\n    ----------\n    a : array_like\n        Array containing data to be averaged. If `a` is not an array, a\n        conversion is attempted.\n    axis : None or int or tuple of ints, optional\n        Axis or axes along which to average `a`.  The default,\n        axis=None, will average over all of the elements of the input array.\n        If axis is negative it counts from the last to the first axis.\n\n        .. versionadded:: 1.7.0\n\n        If axis is a tuple of ints, averaging is performed on all of the axes\n        specified in the tuple instead of a single axis or all the axes as\n        before.\n    weights : array_like, optional\n        An array of weights associated with the values in `a`. Each value in\n        `a` contributes to the average according to its associated weight.\n        The weights array can either be 1-D (in which case its length must be\n        the size of `a` along the given axis) or of the same shape as `a`.\n        If `weights=None`, then all data in `a` are assumed to have a\n        weight equal to one.  The 1-D calculation is::\n\n            avg = sum(a * weights) / sum(weights)\n\n        The only constraint on `weights` is that `sum(weights)` must not be 0.\n    returned : bool, optional\n        Default is `False`. If `True`, the tuple (`average`, `sum_of_weights`)\n        is returned, otherwise only the average is returned.\n        If `weights=None`, `sum_of_weights` is equivalent to the number of\n        elements over which the average is taken.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left\n        in the result as dimensions with size one. With this option,\n        the result will broadcast correctly against the original `a`.\n        *Note:* `keepdims` will not work with instances of `numpy.matrix`\n        or other classes whose methods do not support `keepdims`.\n\n        .. versionadded:: 1.23.0\n\n    Returns\n    -------\n    retval, [sum_of_weights] : array_type or double\n        Return the average along the specified axis. When `returned` is `True`,\n        return a tuple with the average as the first element and the sum\n        of the weights as the second element. `sum_of_weights` is of the\n        same type as `retval`. The result dtype follows a general pattern.\n        If `weights` is None, the result dtype will be that of `a` , or ``float64``\n        if `a` is integral. Otherwise, if `weights` is not None and `a` is non-\n        integral, the result type will be the type of lowest precision capable of\n        representing values of both `a` and `weights`. If `a` happens to be\n        integral, the previous rules still applies but the result dtype will\n        at least be ``float64``.\n\n    Raises\n    ------\n    ZeroDivisionError\n        When all weights along axis are zero. See `numpy.ma.average` for a\n        version robust to this type of error.\n    TypeError\n        When the length of 1D `weights` is not the same as the shape of `a`\n        along axis.\n\n    See Also\n    --------\n    mean\n\n    ma.average : average for masked arrays -- useful if your data contains\n                 "missing" values\n    numpy.result_type : Returns the type that results from applying the\n                        numpy type promotion rules to the arguments.\n\n    Examples\n    --------\n    >>> data = np.arange(1, 5)\n    >>> data\n    array([1, 2, 3, 4])\n    >>> np.average(data)\n    2.5\n    >>> np.average(np.arange(1, 11), weights=np.arange(10, 0, -1))\n    4.0\n\n    >>> data = np.arange(6).reshape((3, 2))\n    >>> data\n    array([[0, 1],\n           [2, 3],\n           [4, 5]])\n    >>> np.average(data, axis=1, weights=[1./4, 3./4])\n    array([0.75, 2.75, 4.75])\n    >>> np.average(data, weights=[1./4, 3./4])\n    Traceback (most recent call last):\n        ...\n    TypeError: Axis must be specified when shapes of a and weights differ.\n\n    >>> a = np.ones(5, dtype=np.float64)\n    >>> w = np.ones(5, dtype=np.complex64)\n    >>> avg = np.average(a, weights=w)\n    >>> print(avg.dtype)\n    complex128\n\n    With ``keepdims=True``, the following result has shape (3, 1).\n\n    >>> np.average(data, axis=1, keepdims=True)\n    array([[0.5],\n           [2.5],\n           [4.5]])\n    '
    a = np.asanyarray(a)
    if keepdims is np._NoValue:
        keepdims_kw = {}
    else:
        keepdims_kw = {'keepdims': keepdims}
    if weights is None:
        avg = a.mean(axis, **keepdims_kw)
        avg_as_array = np.asanyarray(avg)
        scl = avg_as_array.dtype.type(a.size / avg_as_array.size)
    else:
        wgt = np.asanyarray(weights)
        if issubclass(a.dtype.type, (np.integer, np.bool_)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError('Axis must be specified when shapes of a and weights differ.')
            if wgt.ndim != 1:
                raise TypeError('1D weights expected when shapes of a and weights differ.')
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError('Length of weights not compatible with specified axis.')
            wgt = np.broadcast_to(wgt, (a.ndim - 1) * (1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)
        scl = wgt.sum(axis=axis, dtype=result_dtype, **keepdims_kw)
        if np.any(scl == 0.0):
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")
        avg = avg_as_array = np.multiply(a, wgt, dtype=result_dtype).sum(axis, **keepdims_kw) / scl
    if returned:
        if scl.shape != avg_as_array.shape:
            scl = np.broadcast_to(scl, avg_as_array.shape).copy()
        return (avg, scl)
    else:
        return avg

@set_module('numpy')
def asarray_chkfinite(a, dtype=None, order=None):
    if False:
        i = 10
        return i + 15
    "Convert the input to an array, checking for NaNs or Infs.\n\n    Parameters\n    ----------\n    a : array_like\n        Input data, in any form that can be converted to an array.  This\n        includes lists, lists of tuples, tuples, tuples of tuples, tuples\n        of lists and ndarrays.  Success requires no NaNs or Infs.\n    dtype : data-type, optional\n        By default, the data-type is inferred from the input data.\n    order : {'C', 'F', 'A', 'K'}, optional\n        Memory layout.  'A' and 'K' depend on the order of input array a.\n        'C' row-major (C-style),\n        'F' column-major (Fortran-style) memory representation.\n        'A' (any) means 'F' if `a` is Fortran contiguous, 'C' otherwise\n        'K' (keep) preserve input order\n        Defaults to 'C'.\n\n    Returns\n    -------\n    out : ndarray\n        Array interpretation of `a`.  No copy is performed if the input\n        is already an ndarray.  If `a` is a subclass of ndarray, a base\n        class ndarray is returned.\n\n    Raises\n    ------\n    ValueError\n        Raises ValueError if `a` contains NaN (Not a Number) or Inf (Infinity).\n\n    See Also\n    --------\n    asarray : Create and array.\n    asanyarray : Similar function which passes through subclasses.\n    ascontiguousarray : Convert input to a contiguous array.\n    asfortranarray : Convert input to an ndarray with column-major\n                     memory order.\n    fromiter : Create an array from an iterator.\n    fromfunction : Construct an array by executing a function on grid\n                   positions.\n\n    Examples\n    --------\n    Convert a list into an array.  If all elements are finite\n    ``asarray_chkfinite`` is identical to ``asarray``.\n\n    >>> a = [1, 2]\n    >>> np.asarray_chkfinite(a, dtype=float)\n    array([1., 2.])\n\n    Raises ValueError if array_like contains Nans or Infs.\n\n    >>> a = [1, 2, np.inf]\n    >>> try:\n    ...     np.asarray_chkfinite(a)\n    ... except ValueError:\n    ...     print('ValueError')\n    ...\n    ValueError\n\n    "
    a = asarray(a, dtype=dtype, order=order)
    if a.dtype.char in typecodes['AllFloat'] and (not np.isfinite(a).all()):
        raise ValueError('array must not contain infs or NaNs')
    return a

def _piecewise_dispatcher(x, condlist, funclist, *args, **kw):
    if False:
        return 10
    yield x
    if np.iterable(condlist):
        yield from condlist

@array_function_dispatch(_piecewise_dispatcher)
def piecewise(x, condlist, funclist, *args, **kw):
    if False:
        print('Hello World!')
    "\n    Evaluate a piecewise-defined function.\n\n    Given a set of conditions and corresponding functions, evaluate each\n    function on the input data wherever its condition is true.\n\n    Parameters\n    ----------\n    x : ndarray or scalar\n        The input domain.\n    condlist : list of bool arrays or bool scalars\n        Each boolean array corresponds to a function in `funclist`.  Wherever\n        `condlist[i]` is True, `funclist[i](x)` is used as the output value.\n\n        Each boolean array in `condlist` selects a piece of `x`,\n        and should therefore be of the same shape as `x`.\n\n        The length of `condlist` must correspond to that of `funclist`.\n        If one extra function is given, i.e. if\n        ``len(funclist) == len(condlist) + 1``, then that extra function\n        is the default value, used wherever all conditions are false.\n    funclist : list of callables, f(x,*args,**kw), or scalars\n        Each function is evaluated over `x` wherever its corresponding\n        condition is True.  It should take a 1d array as input and give an 1d\n        array or a scalar value as output.  If, instead of a callable,\n        a scalar is provided then a constant function (``lambda x: scalar``) is\n        assumed.\n    args : tuple, optional\n        Any further arguments given to `piecewise` are passed to the functions\n        upon execution, i.e., if called ``piecewise(..., ..., 1, 'a')``, then\n        each function is called as ``f(x, 1, 'a')``.\n    kw : dict, optional\n        Keyword arguments used in calling `piecewise` are passed to the\n        functions upon execution, i.e., if called\n        ``piecewise(..., ..., alpha=1)``, then each function is called as\n        ``f(x, alpha=1)``.\n\n    Returns\n    -------\n    out : ndarray\n        The output is the same shape and type as x and is found by\n        calling the functions in `funclist` on the appropriate portions of `x`,\n        as defined by the boolean arrays in `condlist`.  Portions not covered\n        by any condition have a default value of 0.\n\n\n    See Also\n    --------\n    choose, select, where\n\n    Notes\n    -----\n    This is similar to choose or select, except that functions are\n    evaluated on elements of `x` that satisfy the corresponding condition from\n    `condlist`.\n\n    The result is::\n\n            |--\n            |funclist[0](x[condlist[0]])\n      out = |funclist[1](x[condlist[1]])\n            |...\n            |funclist[n2](x[condlist[n2]])\n            |--\n\n    Examples\n    --------\n    Define the signum function, which is -1 for ``x < 0`` and +1 for ``x >= 0``.\n\n    >>> x = np.linspace(-2.5, 2.5, 6)\n    >>> np.piecewise(x, [x < 0, x >= 0], [-1, 1])\n    array([-1., -1., -1.,  1.,  1.,  1.])\n\n    Define the absolute value, which is ``-x`` for ``x <0`` and ``x`` for\n    ``x >= 0``.\n\n    >>> np.piecewise(x, [x < 0, x >= 0], [lambda x: -x, lambda x: x])\n    array([2.5,  1.5,  0.5,  0.5,  1.5,  2.5])\n\n    Apply the same function to a scalar value.\n\n    >>> y = -2\n    >>> np.piecewise(y, [y < 0, y >= 0], [lambda x: -x, lambda x: x])\n    array(2)\n\n    "
    x = asanyarray(x)
    n2 = len(funclist)
    if isscalar(condlist) or (not isinstance(condlist[0], (list, ndarray)) and x.ndim != 0):
        condlist = [condlist]
    condlist = asarray(condlist, dtype=bool)
    n = len(condlist)
    if n == n2 - 1:
        condelse = ~np.any(condlist, axis=0, keepdims=True)
        condlist = np.concatenate([condlist, condelse], axis=0)
        n += 1
    elif n != n2:
        raise ValueError('with {} condition(s), either {} or {} functions are expected'.format(n, n, n + 1))
    y = zeros_like(x)
    for (cond, func) in zip(condlist, funclist):
        if not isinstance(func, collections.abc.Callable):
            y[cond] = func
        else:
            vals = x[cond]
            if vals.size > 0:
                y[cond] = func(vals, *args, **kw)
    return y

def _select_dispatcher(condlist, choicelist, default=None):
    if False:
        for i in range(10):
            print('nop')
    yield from condlist
    yield from choicelist

@array_function_dispatch(_select_dispatcher)
def select(condlist, choicelist, default=0):
    if False:
        return 10
    '\n    Return an array drawn from elements in choicelist, depending on conditions.\n\n    Parameters\n    ----------\n    condlist : list of bool ndarrays\n        The list of conditions which determine from which array in `choicelist`\n        the output elements are taken. When multiple conditions are satisfied,\n        the first one encountered in `condlist` is used.\n    choicelist : list of ndarrays\n        The list of arrays from which the output elements are taken. It has\n        to be of the same length as `condlist`.\n    default : scalar, optional\n        The element inserted in `output` when all conditions evaluate to False.\n\n    Returns\n    -------\n    output : ndarray\n        The output at position m is the m-th element of the array in\n        `choicelist` where the m-th element of the corresponding array in\n        `condlist` is True.\n\n    See Also\n    --------\n    where : Return elements from one of two arrays depending on condition.\n    take, choose, compress, diag, diagonal\n\n    Examples\n    --------\n    >>> x = np.arange(6)\n    >>> condlist = [x<3, x>3]\n    >>> choicelist = [x, x**2]\n    >>> np.select(condlist, choicelist, 42)\n    array([ 0,  1,  2, 42, 16, 25])\n\n    >>> condlist = [x<=4, x>3]\n    >>> choicelist = [x, x**2]\n    >>> np.select(condlist, choicelist, 55)\n    array([ 0,  1,  2,  3,  4, 25])\n\n    '
    if len(condlist) != len(choicelist):
        raise ValueError('list of cases must be same length as list of conditions')
    if len(condlist) == 0:
        raise ValueError('select with an empty condition list is not possible')
    choicelist = [choice if type(choice) in (int, float, complex) else np.asarray(choice) for choice in choicelist]
    choicelist.append(default if type(default) in (int, float, complex) else np.asarray(default))
    try:
        dtype = np.result_type(*choicelist)
    except TypeError as e:
        msg = f'Choicelist and default value do not have a common dtype: {e}'
        raise TypeError(msg) from None
    condlist = np.broadcast_arrays(*condlist)
    choicelist = np.broadcast_arrays(*choicelist)
    for (i, cond) in enumerate(condlist):
        if cond.dtype.type is not np.bool_:
            raise TypeError('invalid entry {} in condlist: should be boolean ndarray'.format(i))
    if choicelist[0].ndim == 0:
        result_shape = condlist[0].shape
    else:
        result_shape = np.broadcast_arrays(condlist[0], choicelist[0])[0].shape
    result = np.full(result_shape, choicelist[-1], dtype)
    choicelist = choicelist[-2::-1]
    condlist = condlist[::-1]
    for (choice, cond) in zip(choicelist, condlist):
        np.copyto(result, choice, where=cond)
    return result

def _copy_dispatcher(a, order=None, subok=None):
    if False:
        i = 10
        return i + 15
    return (a,)

@array_function_dispatch(_copy_dispatcher)
def copy(a, order='K', subok=False):
    if False:
        print('Hello World!')
    '\n    Return an array copy of the given object.\n\n    Parameters\n    ----------\n    a : array_like\n        Input data.\n    order : {\'C\', \'F\', \'A\', \'K\'}, optional\n        Controls the memory layout of the copy. \'C\' means C-order,\n        \'F\' means F-order, \'A\' means \'F\' if `a` is Fortran contiguous,\n        \'C\' otherwise. \'K\' means match the layout of `a` as closely\n        as possible. (Note that this function and :meth:`ndarray.copy` are very\n        similar, but have different default values for their order=\n        arguments.)\n    subok : bool, optional\n        If True, then sub-classes will be passed-through, otherwise the\n        returned array will be forced to be a base-class array (defaults to False).\n\n        .. versionadded:: 1.19.0\n\n    Returns\n    -------\n    arr : ndarray\n        Array interpretation of `a`.\n\n    See Also\n    --------\n    ndarray.copy : Preferred method for creating an array copy\n\n    Notes\n    -----\n    This is equivalent to:\n\n    >>> np.array(a, copy=True)  #doctest: +SKIP\n\n    Examples\n    --------\n    Create an array x, with a reference y and a copy z:\n\n    >>> x = np.array([1, 2, 3])\n    >>> y = x\n    >>> z = np.copy(x)\n\n    Note that, when we modify x, y changes, but not z:\n\n    >>> x[0] = 10\n    >>> x[0] == y[0]\n    True\n    >>> x[0] == z[0]\n    False\n\n    Note that, np.copy clears previously set WRITEABLE=False flag.\n\n    >>> a = np.array([1, 2, 3])\n    >>> a.flags["WRITEABLE"] = False\n    >>> b = np.copy(a)\n    >>> b.flags["WRITEABLE"]\n    True\n    >>> b[0] = 3\n    >>> b\n    array([3, 2, 3])\n\n    Note that np.copy is a shallow copy and will not copy object\n    elements within arrays. This is mainly important for arrays\n    containing Python objects. The new array will contain the\n    same object which may lead to surprises if that object can\n    be modified (is mutable):\n\n    >>> a = np.array([1, \'m\', [2, 3, 4]], dtype=object)\n    >>> b = np.copy(a)\n    >>> b[2][0] = 10\n    >>> a\n    array([1, \'m\', list([10, 3, 4])], dtype=object)\n\n    To ensure all elements within an ``object`` array are copied,\n    use `copy.deepcopy`:\n\n    >>> import copy\n    >>> a = np.array([1, \'m\', [2, 3, 4]], dtype=object)\n    >>> c = copy.deepcopy(a)\n    >>> c[2][0] = 10\n    >>> c\n    array([1, \'m\', list([10, 3, 4])], dtype=object)\n    >>> a\n    array([1, \'m\', list([2, 3, 4])], dtype=object)\n\n    '
    return array(a, order=order, subok=subok, copy=True)

def _gradient_dispatcher(f, *varargs, axis=None, edge_order=None):
    if False:
        for i in range(10):
            print('nop')
    yield f
    yield from varargs

@array_function_dispatch(_gradient_dispatcher)
def gradient(f, *varargs, axis=None, edge_order=1):
    if False:
        print('Hello World!')
    '\n    Return the gradient of an N-dimensional array.\n\n    The gradient is computed using second order accurate central differences\n    in the interior points and either first or second order accurate one-sides\n    (forward or backwards) differences at the boundaries.\n    The returned gradient hence has the same shape as the input array.\n\n    Parameters\n    ----------\n    f : array_like\n        An N-dimensional array containing samples of a scalar function.\n    varargs : list of scalar or array, optional\n        Spacing between f values. Default unitary spacing for all dimensions.\n        Spacing can be specified using:\n\n        1. single scalar to specify a sample distance for all dimensions.\n        2. N scalars to specify a constant sample distance for each dimension.\n           i.e. `dx`, `dy`, `dz`, ...\n        3. N arrays to specify the coordinates of the values along each\n           dimension of F. The length of the array must match the size of\n           the corresponding dimension\n        4. Any combination of N scalars/arrays with the meaning of 2. and 3.\n\n        If `axis` is given, the number of varargs must equal the number of axes.\n        Default: 1.\n\n    edge_order : {1, 2}, optional\n        Gradient is calculated using N-th order accurate differences\n        at the boundaries. Default: 1.\n\n        .. versionadded:: 1.9.1\n\n    axis : None or int or tuple of ints, optional\n        Gradient is calculated only along the given axis or axes\n        The default (axis = None) is to calculate the gradient for all the axes\n        of the input array. axis may be negative, in which case it counts from\n        the last to the first axis.\n\n        .. versionadded:: 1.11.0\n\n    Returns\n    -------\n    gradient : ndarray or list of ndarray\n        A list of ndarrays (or a single ndarray if there is only one dimension)\n        corresponding to the derivatives of f with respect to each dimension.\n        Each derivative has the same shape as f.\n\n    Examples\n    --------\n    >>> f = np.array([1, 2, 4, 7, 11, 16], dtype=float)\n    >>> np.gradient(f)\n    array([1. , 1.5, 2.5, 3.5, 4.5, 5. ])\n    >>> np.gradient(f, 2)\n    array([0.5 ,  0.75,  1.25,  1.75,  2.25,  2.5 ])\n\n    Spacing can be also specified with an array that represents the coordinates\n    of the values F along the dimensions.\n    For instance a uniform spacing:\n\n    >>> x = np.arange(f.size)\n    >>> np.gradient(f, x)\n    array([1. ,  1.5,  2.5,  3.5,  4.5,  5. ])\n\n    Or a non uniform one:\n\n    >>> x = np.array([0., 1., 1.5, 3.5, 4., 6.], dtype=float)\n    >>> np.gradient(f, x)\n    array([1. ,  3. ,  3.5,  6.7,  6.9,  2.5])\n\n    For two dimensional arrays, the return will be two arrays ordered by\n    axis. In this example the first array stands for the gradient in\n    rows and the second one in columns direction:\n\n    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=float))\n    [array([[ 2.,  2., -1.],\n           [ 2.,  2., -1.]]), array([[1. , 2.5, 4. ],\n           [1. , 1. , 1. ]])]\n\n    In this example the spacing is also specified:\n    uniform for axis=0 and non uniform for axis=1\n\n    >>> dx = 2.\n    >>> y = [1., 1.5, 3.5]\n    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=float), dx, y)\n    [array([[ 1. ,  1. , -0.5],\n           [ 1. ,  1. , -0.5]]), array([[2. , 2. , 2. ],\n           [2. , 1.7, 0.5]])]\n\n    It is possible to specify how boundaries are treated using `edge_order`\n\n    >>> x = np.array([0, 1, 2, 3, 4])\n    >>> f = x**2\n    >>> np.gradient(f, edge_order=1)\n    array([1.,  2.,  4.,  6.,  7.])\n    >>> np.gradient(f, edge_order=2)\n    array([0., 2., 4., 6., 8.])\n\n    The `axis` keyword can be used to specify a subset of axes of which the\n    gradient is calculated\n\n    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=float), axis=0)\n    array([[ 2.,  2., -1.],\n           [ 2.,  2., -1.]])\n\n    Notes\n    -----\n    Assuming that :math:`f\\in C^{3}` (i.e., :math:`f` has at least 3 continuous\n    derivatives) and let :math:`h_{*}` be a non-homogeneous stepsize, we\n    minimize the "consistency error" :math:`\\eta_{i}` between the true gradient\n    and its estimate from a linear combination of the neighboring grid-points:\n\n    .. math::\n\n        \\eta_{i} = f_{i}^{\\left(1\\right)} -\n                    \\left[ \\alpha f\\left(x_{i}\\right) +\n                            \\beta f\\left(x_{i} + h_{d}\\right) +\n                            \\gamma f\\left(x_{i}-h_{s}\\right)\n                    \\right]\n\n    By substituting :math:`f(x_{i} + h_{d})` and :math:`f(x_{i} - h_{s})`\n    with their Taylor series expansion, this translates into solving\n    the following the linear system:\n\n    .. math::\n\n        \\left\\{\n            \\begin{array}{r}\n                \\alpha+\\beta+\\gamma=0 \\\\\n                \\beta h_{d}-\\gamma h_{s}=1 \\\\\n                \\beta h_{d}^{2}+\\gamma h_{s}^{2}=0\n            \\end{array}\n        \\right.\n\n    The resulting approximation of :math:`f_{i}^{(1)}` is the following:\n\n    .. math::\n\n        \\hat f_{i}^{(1)} =\n            \\frac{\n                h_{s}^{2}f\\left(x_{i} + h_{d}\\right)\n                + \\left(h_{d}^{2} - h_{s}^{2}\\right)f\\left(x_{i}\\right)\n                - h_{d}^{2}f\\left(x_{i}-h_{s}\\right)}\n                { h_{s}h_{d}\\left(h_{d} + h_{s}\\right)}\n            + \\mathcal{O}\\left(\\frac{h_{d}h_{s}^{2}\n                                + h_{s}h_{d}^{2}}{h_{d}\n                                + h_{s}}\\right)\n\n    It is worth noting that if :math:`h_{s}=h_{d}`\n    (i.e., data are evenly spaced)\n    we find the standard second order approximation:\n\n    .. math::\n\n        \\hat f_{i}^{(1)}=\n            \\frac{f\\left(x_{i+1}\\right) - f\\left(x_{i-1}\\right)}{2h}\n            + \\mathcal{O}\\left(h^{2}\\right)\n\n    With a similar procedure the forward/backward approximations used for\n    boundaries can be derived.\n\n    References\n    ----------\n    .. [1]  Quarteroni A., Sacco R., Saleri F. (2007) Numerical Mathematics\n            (Texts in Applied Mathematics). New York: Springer.\n    .. [2]  Durran D. R. (1999) Numerical Methods for Wave Equations\n            in Geophysical Fluid Dynamics. New York: Springer.\n    .. [3]  Fornberg B. (1988) Generation of Finite Difference Formulas on\n            Arbitrarily Spaced Grids,\n            Mathematics of Computation 51, no. 184 : 699-706.\n            `PDF <https://www.ams.org/journals/mcom/1988-51-184/\n            S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf>`_.\n    '
    f = np.asanyarray(f)
    N = f.ndim
    if axis is None:
        axes = tuple(range(N))
    else:
        axes = _nx.normalize_axis_tuple(axis, N)
    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        dx = [1.0] * len_axes
    elif n == 1 and np.ndim(varargs[0]) == 0:
        dx = varargs * len_axes
    elif n == len_axes:
        dx = list(varargs)
        for (i, distances) in enumerate(dx):
            distances = np.asanyarray(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError('distances must be either scalars or 1d')
            if len(distances) != f.shape[axes[i]]:
                raise ValueError('when 1d, distances must match the length of the corresponding dimension')
            if np.issubdtype(distances.dtype, np.integer):
                distances = distances.astype(np.float64)
            diffx = np.diff(distances)
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError('invalid number of arguments')
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")
    outvals = []
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N
    otype = f.dtype
    if otype.type is np.datetime64:
        otype = np.dtype(otype.name.replace('datetime', 'timedelta'))
        f = f.view(otype)
    elif otype.type is np.timedelta64:
        pass
    elif np.issubdtype(otype, np.inexact):
        pass
    else:
        if np.issubdtype(otype, np.integer):
            f = f.astype(np.float64)
        otype = np.float64
    for (axis, ax_dx) in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError('Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required.')
        out = np.empty_like(f, dtype=otype)
        uniform_spacing = np.ndim(ax_dx) == 0
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)
        if uniform_spacing:
            out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2.0 * ax_dx)
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            a = -dx2 / (dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            shape = np.ones(N, dtype=int)
            shape[axis] = -1
            a.shape = b.shape = c.shape = shape
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0
            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2.0 / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = -dx1 / (dx2 * (dx1 + dx2))
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2.0 / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = dx2 / (dx1 * (dx1 + dx2))
                b = -(dx2 + dx1) / (dx1 * dx2)
                c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
        outvals.append(out)
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)
    if len_axes == 1:
        return outvals[0]
    return tuple(outvals)

def _diff_dispatcher(a, n=None, axis=None, prepend=None, append=None):
    if False:
        for i in range(10):
            print('nop')
    return (a, prepend, append)

@array_function_dispatch(_diff_dispatcher)
def diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    if False:
        while True:
            i = 10
    "\n    Calculate the n-th discrete difference along the given axis.\n\n    The first difference is given by ``out[i] = a[i+1] - a[i]`` along\n    the given axis, higher differences are calculated by using `diff`\n    recursively.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array\n    n : int, optional\n        The number of times values are differenced. If zero, the input\n        is returned as-is.\n    axis : int, optional\n        The axis along which the difference is taken, default is the\n        last axis.\n    prepend, append : array_like, optional\n        Values to prepend or append to `a` along axis prior to\n        performing the difference.  Scalar values are expanded to\n        arrays with length 1 in the direction of axis and the shape\n        of the input array in along all other axes.  Otherwise the\n        dimension and shape must match `a` except along axis.\n\n        .. versionadded:: 1.16.0\n\n    Returns\n    -------\n    diff : ndarray\n        The n-th differences. The shape of the output is the same as `a`\n        except along `axis` where the dimension is smaller by `n`. The\n        type of the output is the same as the type of the difference\n        between any two elements of `a`. This is the same as the type of\n        `a` in most cases. A notable exception is `datetime64`, which\n        results in a `timedelta64` output array.\n\n    See Also\n    --------\n    gradient, ediff1d, cumsum\n\n    Notes\n    -----\n    Type is preserved for boolean arrays, so the result will contain\n    `False` when consecutive elements are the same and `True` when they\n    differ.\n\n    For unsigned integer arrays, the results will also be unsigned. This\n    should not be surprising, as the result is consistent with\n    calculating the difference directly:\n\n    >>> u8_arr = np.array([1, 0], dtype=np.uint8)\n    >>> np.diff(u8_arr)\n    array([255], dtype=uint8)\n    >>> u8_arr[1,...] - u8_arr[0,...]\n    255\n\n    If this is not desirable, then the array should be cast to a larger\n    integer type first:\n\n    >>> i16_arr = u8_arr.astype(np.int16)\n    >>> np.diff(i16_arr)\n    array([-1], dtype=int16)\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 4, 7, 0])\n    >>> np.diff(x)\n    array([ 1,  2,  3, -7])\n    >>> np.diff(x, n=2)\n    array([  1,   1, -10])\n\n    >>> x = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])\n    >>> np.diff(x)\n    array([[2, 3, 4],\n           [5, 1, 2]])\n    >>> np.diff(x, axis=0)\n    array([[-1,  2,  0, -2]])\n\n    >>> x = np.arange('1066-10-13', '1066-10-16', dtype=np.datetime64)\n    >>> np.diff(x)\n    array([1, 1], dtype='timedelta64[D]')\n\n    "
    if n == 0:
        return a
    if n < 0:
        raise ValueError('order must be non-negative but got ' + repr(n))
    a = asanyarray(a)
    nd = a.ndim
    if nd == 0:
        raise ValueError('diff requires input that is at least one dimensional')
    axis = normalize_axis_index(axis, nd)
    combined = []
    if prepend is not np._NoValue:
        prepend = np.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)
    combined.append(a)
    if append is not np._NoValue:
        append = np.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        combined.append(append)
    if len(combined) > 1:
        a = np.concatenate(combined, axis)
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    op = not_equal if a.dtype == np.bool_ else subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])
    return a

def _interp_dispatcher(x, xp, fp, left=None, right=None, period=None):
    if False:
        return 10
    return (x, xp, fp)

@array_function_dispatch(_interp_dispatcher)
def interp(x, xp, fp, left=None, right=None, period=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    One-dimensional linear interpolation for monotonically increasing sample points.\n\n    Returns the one-dimensional piecewise linear interpolant to a function\n    with given discrete data points (`xp`, `fp`), evaluated at `x`.\n\n    Parameters\n    ----------\n    x : array_like\n        The x-coordinates at which to evaluate the interpolated values.\n\n    xp : 1-D sequence of floats\n        The x-coordinates of the data points, must be increasing if argument\n        `period` is not specified. Otherwise, `xp` is internally sorted after\n        normalizing the periodic boundaries with ``xp = xp % period``.\n\n    fp : 1-D sequence of float or complex\n        The y-coordinates of the data points, same length as `xp`.\n\n    left : optional float or complex corresponding to fp\n        Value to return for `x < xp[0]`, default is `fp[0]`.\n\n    right : optional float or complex corresponding to fp\n        Value to return for `x > xp[-1]`, default is `fp[-1]`.\n\n    period : None or float, optional\n        A period for the x-coordinates. This parameter allows the proper\n        interpolation of angular x-coordinates. Parameters `left` and `right`\n        are ignored if `period` is specified.\n\n        .. versionadded:: 1.10.0\n\n    Returns\n    -------\n    y : float or complex (corresponding to fp) or ndarray\n        The interpolated values, same shape as `x`.\n\n    Raises\n    ------\n    ValueError\n        If `xp` and `fp` have different length\n        If `xp` or `fp` are not 1-D sequences\n        If `period == 0`\n\n    See Also\n    --------\n    scipy.interpolate\n\n    Warnings\n    --------\n    The x-coordinate sequence is expected to be increasing, but this is not\n    explicitly enforced.  However, if the sequence `xp` is non-increasing,\n    interpolation results are meaningless.\n\n    Note that, since NaN is unsortable, `xp` also cannot contain NaNs.\n\n    A simple check for `xp` being strictly increasing is::\n\n        np.all(np.diff(xp) > 0)\n\n    Examples\n    --------\n    >>> xp = [1, 2, 3]\n    >>> fp = [3, 2, 0]\n    >>> np.interp(2.5, xp, fp)\n    1.0\n    >>> np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)\n    array([3.  , 3.  , 2.5 , 0.56, 0.  ])\n    >>> UNDEF = -99.0\n    >>> np.interp(3.14, xp, fp, right=UNDEF)\n    -99.0\n\n    Plot an interpolant to the sine function:\n\n    >>> x = np.linspace(0, 2*np.pi, 10)\n    >>> y = np.sin(x)\n    >>> xvals = np.linspace(0, 2*np.pi, 50)\n    >>> yinterp = np.interp(xvals, x, y)\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(x, y, 'o')\n    [<matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.plot(xvals, yinterp, '-x')\n    [<matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.show()\n\n    Interpolation with periodic x-coordinates:\n\n    >>> x = [-180, -170, -185, 185, -10, -5, 0, 365]\n    >>> xp = [190, -190, 350, -350]\n    >>> fp = [5, 10, 3, 4]\n    >>> np.interp(x, xp, fp, period=360)\n    array([7.5 , 5.  , 8.75, 6.25, 3.  , 3.25, 3.5 , 3.75])\n\n    Complex interpolation:\n\n    >>> x = [1.5, 4.0]\n    >>> xp = [2,3,5]\n    >>> fp = [1.0j, 0, 2+3j]\n    >>> np.interp(x, xp, fp)\n    array([0.+1.j , 1.+1.5j])\n\n    "
    fp = np.asarray(fp)
    if np.iscomplexobj(fp):
        interp_func = compiled_interp_complex
        input_dtype = np.complex128
    else:
        interp_func = compiled_interp
        input_dtype = np.float64
    if period is not None:
        if period == 0:
            raise ValueError('period must be a non-zero value')
        period = abs(period)
        left = None
        right = None
        x = np.asarray(x, dtype=np.float64)
        xp = np.asarray(xp, dtype=np.float64)
        fp = np.asarray(fp, dtype=input_dtype)
        if xp.ndim != 1 or fp.ndim != 1:
            raise ValueError('Data points must be 1-D sequences')
        if xp.shape[0] != fp.shape[0]:
            raise ValueError('fp and xp are not of the same length')
        x = x % period
        xp = xp % period
        asort_xp = np.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = np.concatenate((xp[-1:] - period, xp, xp[0:1] + period))
        fp = np.concatenate((fp[-1:], fp, fp[0:1]))
    return interp_func(x, xp, fp, left, right)

def _angle_dispatcher(z, deg=None):
    if False:
        print('Hello World!')
    return (z,)

@array_function_dispatch(_angle_dispatcher)
def angle(z, deg=False):
    if False:
        print('Hello World!')
    '\n    Return the angle of the complex argument.\n\n    Parameters\n    ----------\n    z : array_like\n        A complex number or sequence of complex numbers.\n    deg : bool, optional\n        Return angle in degrees if True, radians if False (default).\n\n    Returns\n    -------\n    angle : ndarray or scalar\n        The counterclockwise angle from the positive real axis on the complex\n        plane in the range ``(-pi, pi]``, with dtype as numpy.float64.\n\n        .. versionchanged:: 1.16.0\n            This function works on subclasses of ndarray like `ma.array`.\n\n    See Also\n    --------\n    arctan2\n    absolute\n\n    Notes\n    -----\n    This function passes the imaginary and real parts of the argument to\n    `arctan2` to compute the result; consequently, it follows the convention\n    of `arctan2` when the magnitude of the argument is zero. See example.\n\n    Examples\n    --------\n    >>> np.angle([1.0, 1.0j, 1+1j])               # in radians\n    array([ 0.        ,  1.57079633,  0.78539816]) # may vary\n    >>> np.angle(1+1j, deg=True)                  # in degrees\n    45.0\n    >>> np.angle([0., -0., complex(0., -0.), complex(-0., -0.)])  # convention\n    array([ 0.        ,  3.14159265, -0.        , -3.14159265])\n\n    '
    z = asanyarray(z)
    if issubclass(z.dtype.type, _nx.complexfloating):
        zimag = z.imag
        zreal = z.real
    else:
        zimag = 0
        zreal = z
    a = arctan2(zimag, zreal)
    if deg:
        a *= 180 / pi
    return a

def _unwrap_dispatcher(p, discont=None, axis=None, *, period=None):
    if False:
        while True:
            i = 10
    return (p,)

@array_function_dispatch(_unwrap_dispatcher)
def unwrap(p, discont=None, axis=-1, *, period=2 * pi):
    if False:
        while True:
            i = 10
    '\n    Unwrap by taking the complement of large deltas with respect to the period.\n\n    This unwraps a signal `p` by changing elements which have an absolute\n    difference from their predecessor of more than ``max(discont, period/2)``\n    to their `period`-complementary values.\n\n    For the default case where `period` is :math:`2\\pi` and `discont` is\n    :math:`\\pi`, this unwraps a radian phase `p` such that adjacent differences\n    are never greater than :math:`\\pi` by adding :math:`2k\\pi` for some\n    integer :math:`k`.\n\n    Parameters\n    ----------\n    p : array_like\n        Input array.\n    discont : float, optional\n        Maximum discontinuity between values, default is ``period/2``.\n        Values below ``period/2`` are treated as if they were ``period/2``.\n        To have an effect different from the default, `discont` should be\n        larger than ``period/2``.\n    axis : int, optional\n        Axis along which unwrap will operate, default is the last axis.\n    period : float, optional\n        Size of the range over which the input wraps. By default, it is\n        ``2 pi``.\n\n        .. versionadded:: 1.21.0\n\n    Returns\n    -------\n    out : ndarray\n        Output array.\n\n    See Also\n    --------\n    rad2deg, deg2rad\n\n    Notes\n    -----\n    If the discontinuity in `p` is smaller than ``period/2``,\n    but larger than `discont`, no unwrapping is done because taking\n    the complement would only make the discontinuity larger.\n\n    Examples\n    --------\n    >>> phase = np.linspace(0, np.pi, num=5)\n    >>> phase[3:] += np.pi\n    >>> phase\n    array([ 0.        ,  0.78539816,  1.57079633,  5.49778714,  6.28318531]) # may vary\n    >>> np.unwrap(phase)\n    array([ 0.        ,  0.78539816,  1.57079633, -0.78539816,  0.        ]) # may vary\n    >>> np.unwrap([0, 1, 2, -1, 0], period=4)\n    array([0, 1, 2, 3, 4])\n    >>> np.unwrap([ 1, 2, 3, 4, 5, 6, 1, 2, 3], period=6)\n    array([1, 2, 3, 4, 5, 6, 7, 8, 9])\n    >>> np.unwrap([2, 3, 4, 5, 2, 3, 4, 5], period=4)\n    array([2, 3, 4, 5, 6, 7, 8, 9])\n    >>> phase_deg = np.mod(np.linspace(0 ,720, 19), 360) - 180\n    >>> np.unwrap(phase_deg, period=360)\n    array([-180., -140., -100.,  -60.,  -20.,   20.,   60.,  100.,  140.,\n            180.,  220.,  260.,  300.,  340.,  380.,  420.,  460.,  500.,\n            540.])\n    '
    p = asarray(p)
    nd = p.ndim
    dd = diff(p, axis=axis)
    if discont is None:
        discont = period / 2
    slice1 = [slice(None, None)] * nd
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    dtype = np.result_type(dd, period)
    if _nx.issubdtype(dtype, _nx.integer):
        (interval_high, rem) = divmod(period, 2)
        boundary_ambiguous = rem == 0
    else:
        interval_high = period / 2
        boundary_ambiguous = True
    interval_low = -interval_high
    ddmod = mod(dd - interval_low, period) + interval_low
    if boundary_ambiguous:
        _nx.copyto(ddmod, interval_high, where=(ddmod == interval_low) & (dd > 0))
    ph_correct = ddmod - dd
    _nx.copyto(ph_correct, 0, where=abs(dd) < discont)
    up = array(p, copy=True, dtype=dtype)
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)
    return up

def _sort_complex(a):
    if False:
        while True:
            i = 10
    return (a,)

@array_function_dispatch(_sort_complex)
def sort_complex(a):
    if False:
        print('Hello World!')
    '\n    Sort a complex array using the real part first, then the imaginary part.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array\n\n    Returns\n    -------\n    out : complex ndarray\n        Always returns a sorted complex array.\n\n    Examples\n    --------\n    >>> np.sort_complex([5, 3, 6, 2, 1])\n    array([1.+0.j, 2.+0.j, 3.+0.j, 5.+0.j, 6.+0.j])\n\n    >>> np.sort_complex([1 + 2j, 2 - 1j, 3 - 2j, 3 - 3j, 3 + 5j])\n    array([1.+2.j,  2.-1.j,  3.-3.j,  3.-2.j,  3.+5.j])\n\n    '
    b = array(a, copy=True)
    b.sort()
    if not issubclass(b.dtype.type, _nx.complexfloating):
        if b.dtype.char in 'bhBH':
            return b.astype('F')
        elif b.dtype.char == 'g':
            return b.astype('G')
        else:
            return b.astype('D')
    else:
        return b

def _trim_zeros(filt, trim=None):
    if False:
        for i in range(10):
            print('nop')
    return (filt,)

@array_function_dispatch(_trim_zeros)
def trim_zeros(filt, trim='fb'):
    if False:
        print('Hello World!')
    "\n    Trim the leading and/or trailing zeros from a 1-D array or sequence.\n\n    Parameters\n    ----------\n    filt : 1-D array or sequence\n        Input array.\n    trim : str, optional\n        A string with 'f' representing trim from front and 'b' to trim from\n        back. Default is 'fb', trim zeros from both front and back of the\n        array.\n\n    Returns\n    -------\n    trimmed : 1-D array or sequence\n        The result of trimming the input. The input data type is preserved.\n\n    Examples\n    --------\n    >>> a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))\n    >>> np.trim_zeros(a)\n    array([1, 2, 3, 0, 2, 1])\n\n    >>> np.trim_zeros(a, 'b')\n    array([0, 0, 0, ..., 0, 2, 1])\n\n    The input data type is preserved, list/tuple in means list/tuple out.\n\n    >>> np.trim_zeros([0, 1, 2, 0])\n    [1, 2]\n\n    "
    first = 0
    trim = trim.upper()
    if 'F' in trim:
        for i in filt:
            if i != 0.0:
                break
            else:
                first = first + 1
    last = len(filt)
    if 'B' in trim:
        for i in filt[::-1]:
            if i != 0.0:
                break
            else:
                last = last - 1
    return filt[first:last]

def _extract_dispatcher(condition, arr):
    if False:
        print('Hello World!')
    return (condition, arr)

@array_function_dispatch(_extract_dispatcher)
def extract(condition, arr):
    if False:
        return 10
    '\n    Return the elements of an array that satisfy some condition.\n\n    This is equivalent to ``np.compress(ravel(condition), ravel(arr))``.  If\n    `condition` is boolean ``np.extract`` is equivalent to ``arr[condition]``.\n\n    Note that `place` does the exact opposite of `extract`.\n\n    Parameters\n    ----------\n    condition : array_like\n        An array whose nonzero or True entries indicate the elements of `arr`\n        to extract.\n    arr : array_like\n        Input array of the same size as `condition`.\n\n    Returns\n    -------\n    extract : ndarray\n        Rank 1 array of values from `arr` where `condition` is True.\n\n    See Also\n    --------\n    take, put, copyto, compress, place\n\n    Examples\n    --------\n    >>> arr = np.arange(12).reshape((3, 4))\n    >>> arr\n    array([[ 0,  1,  2,  3],\n           [ 4,  5,  6,  7],\n           [ 8,  9, 10, 11]])\n    >>> condition = np.mod(arr, 3)==0\n    >>> condition\n    array([[ True, False, False,  True],\n           [False, False,  True, False],\n           [False,  True, False, False]])\n    >>> np.extract(condition, arr)\n    array([0, 3, 6, 9])\n\n\n    If `condition` is boolean:\n\n    >>> arr[condition]\n    array([0, 3, 6, 9])\n\n    '
    return _nx.take(ravel(arr), nonzero(ravel(condition))[0])

def _place_dispatcher(arr, mask, vals):
    if False:
        return 10
    return (arr, mask, vals)

@array_function_dispatch(_place_dispatcher)
def place(arr, mask, vals):
    if False:
        i = 10
        return i + 15
    '\n    Change elements of an array based on conditional and input values.\n\n    Similar to ``np.copyto(arr, vals, where=mask)``, the difference is that\n    `place` uses the first N elements of `vals`, where N is the number of\n    True values in `mask`, while `copyto` uses the elements where `mask`\n    is True.\n\n    Note that `extract` does the exact opposite of `place`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Array to put data into.\n    mask : array_like\n        Boolean mask array. Must have the same size as `a`.\n    vals : 1-D sequence\n        Values to put into `a`. Only the first N elements are used, where\n        N is the number of True values in `mask`. If `vals` is smaller\n        than N, it will be repeated, and if elements of `a` are to be masked,\n        this sequence must be non-empty.\n\n    See Also\n    --------\n    copyto, put, take, extract\n\n    Examples\n    --------\n    >>> arr = np.arange(6).reshape(2, 3)\n    >>> np.place(arr, arr>2, [44, 55])\n    >>> arr\n    array([[ 0,  1,  2],\n           [44, 55, 44]])\n\n    '
    return _place(arr, mask, vals)

def disp(mesg, device=None, linefeed=True):
    if False:
        print('Hello World!')
    '\n    Display a message on a device.\n\n    .. deprecated:: 2.0\n        Use your own printing function instead.\n\n    Parameters\n    ----------\n    mesg : str\n        Message to display.\n    device : object\n        Device to write message. If None, defaults to ``sys.stdout`` which is\n        very similar to ``print``. `device` needs to have ``write()`` and\n        ``flush()`` methods.\n    linefeed : bool, optional\n        Option whether to print a line feed or not. Defaults to True.\n\n    Raises\n    ------\n    AttributeError\n        If `device` does not have a ``write()`` or ``flush()`` method.\n\n    Examples\n    --------\n    Besides ``sys.stdout``, a file-like object can also be used as it has\n    both required methods:\n\n    >>> from io import StringIO\n    >>> buf = StringIO()\n    >>> np.disp(u\'"Display" in a file\', device=buf)\n    >>> buf.getvalue()\n    \'"Display" in a file\\n\'\n\n    '
    warnings.warn('`disp` is deprecated, use your own printing function instead. (deprecated in NumPy 2.0)', DeprecationWarning, stacklevel=2)
    if device is None:
        device = sys.stdout
    if linefeed:
        device.write('%s\n' % mesg)
    else:
        device.write('%s' % mesg)
    device.flush()
    return
_DIMENSION_NAME = '\\w+'
_CORE_DIMENSION_LIST = '(?:{0:}(?:,{0:})*)?'.format(_DIMENSION_NAME)
_ARGUMENT = '\\({}\\)'.format(_CORE_DIMENSION_LIST)
_ARGUMENT_LIST = '{0:}(?:,{0:})*'.format(_ARGUMENT)
_SIGNATURE = '^{0:}->{0:}$'.format(_ARGUMENT_LIST)

def _parse_gufunc_signature(signature):
    if False:
        return 10
    '\n    Parse string signatures for a generalized universal function.\n\n    Arguments\n    ---------\n    signature : string\n        Generalized universal function signature, e.g., ``(m,n),(n,p)->(m,p)``\n        for ``np.matmul``.\n\n    Returns\n    -------\n    Tuple of input and output core dimensions parsed from the signature, each\n    of the form List[Tuple[str, ...]].\n    '
    signature = re.sub('\\s+', '', signature)
    if not re.match(_SIGNATURE, signature):
        raise ValueError('not a valid gufunc signature: {}'.format(signature))
    return tuple(([tuple(re.findall(_DIMENSION_NAME, arg)) for arg in re.findall(_ARGUMENT, arg_list)] for arg_list in signature.split('->')))

def _update_dim_sizes(dim_sizes, arg, core_dims):
    if False:
        while True:
            i = 10
    '\n    Incrementally check and update core dimension sizes for a single argument.\n\n    Arguments\n    ---------\n    dim_sizes : Dict[str, int]\n        Sizes of existing core dimensions. Will be updated in-place.\n    arg : ndarray\n        Argument to examine.\n    core_dims : Tuple[str, ...]\n        Core dimensions for this argument.\n    '
    if not core_dims:
        return
    num_core_dims = len(core_dims)
    if arg.ndim < num_core_dims:
        raise ValueError('%d-dimensional argument does not have enough dimensions for all core dimensions %r' % (arg.ndim, core_dims))
    core_shape = arg.shape[-num_core_dims:]
    for (dim, size) in zip(core_dims, core_shape):
        if dim in dim_sizes:
            if size != dim_sizes[dim]:
                raise ValueError('inconsistent size for core dimension %r: %r vs %r' % (dim, size, dim_sizes[dim]))
        else:
            dim_sizes[dim] = size

def _parse_input_dimensions(args, input_core_dims):
    if False:
        return 10
    '\n    Parse broadcast and core dimensions for vectorize with a signature.\n\n    Arguments\n    ---------\n    args : Tuple[ndarray, ...]\n        Tuple of input arguments to examine.\n    input_core_dims : List[Tuple[str, ...]]\n        List of core dimensions corresponding to each input.\n\n    Returns\n    -------\n    broadcast_shape : Tuple[int, ...]\n        Common shape to broadcast all non-core dimensions to.\n    dim_sizes : Dict[str, int]\n        Common sizes for named core dimensions.\n    '
    broadcast_args = []
    dim_sizes = {}
    for (arg, core_dims) in zip(args, input_core_dims):
        _update_dim_sizes(dim_sizes, arg, core_dims)
        ndim = arg.ndim - len(core_dims)
        dummy_array = np.lib.stride_tricks.as_strided(0, arg.shape[:ndim])
        broadcast_args.append(dummy_array)
    broadcast_shape = np.lib._stride_tricks_impl._broadcast_shape(*broadcast_args)
    return (broadcast_shape, dim_sizes)

def _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims):
    if False:
        print('Hello World!')
    'Helper for calculating broadcast shapes with core dimensions.'
    return [broadcast_shape + tuple((dim_sizes[dim] for dim in core_dims)) for core_dims in list_of_core_dims]

def _create_arrays(broadcast_shape, dim_sizes, list_of_core_dims, dtypes, results=None):
    if False:
        return 10
    'Helper for creating output arrays in vectorize.'
    shapes = _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims)
    if dtypes is None:
        dtypes = [None] * len(shapes)
    if results is None:
        arrays = tuple((np.empty(shape=shape, dtype=dtype) for (shape, dtype) in zip(shapes, dtypes)))
    else:
        arrays = tuple((np.empty_like(result, shape=shape, dtype=dtype) for (result, shape, dtype) in zip(results, shapes, dtypes)))
    return arrays

@set_module('numpy')
class vectorize:
    """
    vectorize(pyfunc=np._NoValue, otypes=None, doc=None, excluded=None,
    cache=False, signature=None)

    Returns an object that acts like pyfunc, but takes arrays as input.

    Define a vectorized function which takes a nested sequence of objects or
    numpy arrays as inputs and returns a single numpy array or a tuple of numpy
    arrays. The vectorized function evaluates `pyfunc` over successive tuples
    of the input arrays like the python map function, except it uses the
    broadcasting rules of numpy.

    The data type of the output of `vectorized` is determined by calling
    the function with the first element of the input.  This can be avoided
    by specifying the `otypes` argument.

    Parameters
    ----------
    pyfunc : callable, optional
        A python function or method.
        Can be omitted to produce a decorator with keyword arguments.
    otypes : str or list of dtypes, optional
        The output data type. It must be specified as either a string of
        typecode characters or a list of data type specifiers. There should
        be one data type specifier for each output.
    doc : str, optional
        The docstring for the function. If None, the docstring will be the
        ``pyfunc.__doc__``.
    excluded : set, optional
        Set of strings or integers representing the positional or keyword
        arguments for which the function will not be vectorized.  These will be
        passed directly to `pyfunc` unmodified.

        .. versionadded:: 1.7.0

    cache : bool, optional
        If `True`, then cache the first function call that determines the number
        of outputs if `otypes` is not provided.

        .. versionadded:: 1.7.0

    signature : string, optional
        Generalized universal function signature, e.g., ``(m,n),(n)->(m)`` for
        vectorized matrix-vector multiplication. If provided, ``pyfunc`` will
        be called with (and expected to return) arrays with shapes given by the
        size of corresponding core dimensions. By default, ``pyfunc`` is
        assumed to take scalars as input and output.

        .. versionadded:: 1.12.0

    Returns
    -------
    out : callable
        A vectorized function if ``pyfunc`` was provided,
        a decorator otherwise.

    See Also
    --------
    frompyfunc : Takes an arbitrary Python function and returns a ufunc

    Notes
    -----
    The `vectorize` function is provided primarily for convenience, not for
    performance. The implementation is essentially a for loop.

    If `otypes` is not specified, then a call to the function with the
    first argument will be used to determine the number of outputs.  The
    results of this call will be cached if `cache` is `True` to prevent
    calling the function twice.  However, to implement the cache, the
    original function must be wrapped which will slow down subsequent
    calls, so only do this if your function is expensive.

    The new keyword argument interface and `excluded` argument support
    further degrades performance.

    References
    ----------
    .. [1] :doc:`/reference/c-api/generalized-ufuncs`

    Examples
    --------
    >>> def myfunc(a, b):
    ...     "Return a-b if a>b, otherwise return a+b"
    ...     if a > b:
    ...         return a - b
    ...     else:
    ...         return a + b

    >>> vfunc = np.vectorize(myfunc)
    >>> vfunc([1, 2, 3, 4], 2)
    array([3, 4, 1, 2])

    The docstring is taken from the input function to `vectorize` unless it
    is specified:

    >>> vfunc.__doc__
    'Return a-b if a>b, otherwise return a+b'
    >>> vfunc = np.vectorize(myfunc, doc='Vectorized `myfunc`')
    >>> vfunc.__doc__
    'Vectorized `myfunc`'

    The output type is determined by evaluating the first element of the input,
    unless it is specified:

    >>> out = vfunc([1, 2, 3, 4], 2)
    >>> type(out[0])
    <class 'numpy.int64'>
    >>> vfunc = np.vectorize(myfunc, otypes=[float])
    >>> out = vfunc([1, 2, 3, 4], 2)
    >>> type(out[0])
    <class 'numpy.float64'>

    The `excluded` argument can be used to prevent vectorizing over certain
    arguments.  This can be useful for array-like arguments of a fixed length
    such as the coefficients for a polynomial as in `polyval`:

    >>> def mypolyval(p, x):
    ...     _p = list(p)
    ...     res = _p.pop(0)
    ...     while _p:
    ...         res = res*x + _p.pop(0)
    ...     return res
    >>> vpolyval = np.vectorize(mypolyval, excluded=['p'])
    >>> vpolyval(p=[1, 2, 3], x=[0, 1])
    array([3, 6])

    Positional arguments may also be excluded by specifying their position:

    >>> vpolyval.excluded.add(0)
    >>> vpolyval([1, 2, 3], x=[0, 1])
    array([3, 6])

    The `signature` argument allows for vectorizing functions that act on
    non-scalar arrays of fixed length. For example, you can use it for a
    vectorized calculation of Pearson correlation coefficient and its p-value:

    >>> import scipy.stats
    >>> pearsonr = np.vectorize(scipy.stats.pearsonr,
    ...                 signature='(n),(n)->(),()')
    >>> pearsonr([[0, 1, 2, 3]], [[1, 2, 3, 4], [4, 3, 2, 1]])
    (array([ 1., -1.]), array([ 0.,  0.]))

    Or for a vectorized convolution:

    >>> convolve = np.vectorize(np.convolve, signature='(n),(m)->(k)')
    >>> convolve(np.eye(4), [1, 2, 1])
    array([[1., 2., 1., 0., 0., 0.],
           [0., 1., 2., 1., 0., 0.],
           [0., 0., 1., 2., 1., 0.],
           [0., 0., 0., 1., 2., 1.]])

    Decorator syntax is supported.  The decorator can be called as
    a function to provide keyword arguments:

    >>> @np.vectorize
    ... def identity(x):
    ...     return x
    ...
    >>> identity([0, 1, 2])
    array([0, 1, 2])
    >>> @np.vectorize(otypes=[float])
    ... def as_float(x):
    ...     return x
    ...
    >>> as_float([0, 1, 2])
    array([0., 1., 2.])
    """

    def __init__(self, pyfunc=np._NoValue, otypes=None, doc=None, excluded=None, cache=False, signature=None):
        if False:
            i = 10
            return i + 15
        if pyfunc != np._NoValue and (not callable(pyfunc)):
            part1 = 'When used as a decorator, '
            part2 = 'only accepts keyword arguments.'
            raise TypeError(part1 + part2)
        self.pyfunc = pyfunc
        self.cache = cache
        self.signature = signature
        if pyfunc != np._NoValue and hasattr(pyfunc, '__name__'):
            self.__name__ = pyfunc.__name__
        self._ufunc = {}
        self._doc = None
        self.__doc__ = doc
        if doc is None and hasattr(pyfunc, '__doc__'):
            self.__doc__ = pyfunc.__doc__
        else:
            self._doc = doc
        if isinstance(otypes, str):
            for char in otypes:
                if char not in typecodes['All']:
                    raise ValueError('Invalid otype specified: %s' % (char,))
        elif iterable(otypes):
            otypes = ''.join([_nx.dtype(x).char for x in otypes])
        elif otypes is not None:
            raise ValueError('Invalid otype specification')
        self.otypes = otypes
        if excluded is None:
            excluded = set()
        self.excluded = set(excluded)
        if signature is not None:
            self._in_and_out_core_dims = _parse_gufunc_signature(signature)
        else:
            self._in_and_out_core_dims = None

    def _init_stage_2(self, pyfunc, *args, **kwargs):
        if False:
            return 10
        self.__name__ = pyfunc.__name__
        self.pyfunc = pyfunc
        if self._doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = self._doc

    def _call_as_normal(self, *args, **kwargs):
        if False:
            return 10
        '\n        Return arrays with the results of `pyfunc` broadcast (vectorized) over\n        `args` and `kwargs` not in `excluded`.\n        '
        excluded = self.excluded
        if not kwargs and (not excluded):
            func = self.pyfunc
            vargs = args
        else:
            nargs = len(args)
            names = [_n for _n in kwargs if _n not in excluded]
            inds = [_i for _i in range(nargs) if _i not in excluded]
            the_args = list(args)

            def func(*vargs):
                if False:
                    i = 10
                    return i + 15
                for (_n, _i) in enumerate(inds):
                    the_args[_i] = vargs[_n]
                kwargs.update(zip(names, vargs[len(inds):]))
                return self.pyfunc(*the_args, **kwargs)
            vargs = [args[_i] for _i in inds]
            vargs.extend([kwargs[_n] for _n in names])
        return self._vectorize_call(func=func, args=vargs)

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self.pyfunc is np._NoValue:
            self._init_stage_2(*args, **kwargs)
            return self
        return self._call_as_normal(*args, **kwargs)

    def _get_ufunc_and_otypes(self, func, args):
        if False:
            print('Hello World!')
        'Return (ufunc, otypes).'
        if not args:
            raise ValueError('args can not be empty')
        if self.otypes is not None:
            otypes = self.otypes
            nin = len(args)
            nout = len(self.otypes)
            if func is not self.pyfunc or nin not in self._ufunc:
                ufunc = frompyfunc(func, nin, nout)
            else:
                ufunc = None
            if func is self.pyfunc:
                ufunc = self._ufunc.setdefault(nin, ufunc)
        else:
            args = [asarray(arg) for arg in args]
            if builtins.any((arg.size == 0 for arg in args)):
                raise ValueError('cannot call `vectorize` on size 0 inputs unless `otypes` is set')
            inputs = [arg.flat[0] for arg in args]
            outputs = func(*inputs)
            if self.cache:
                _cache = [outputs]

                def _func(*vargs):
                    if False:
                        while True:
                            i = 10
                    if _cache:
                        return _cache.pop()
                    else:
                        return func(*vargs)
            else:
                _func = func
            if isinstance(outputs, tuple):
                nout = len(outputs)
            else:
                nout = 1
                outputs = (outputs,)
            otypes = ''.join([asarray(outputs[_k]).dtype.char for _k in range(nout)])
            ufunc = frompyfunc(_func, len(args), nout)
        return (ufunc, otypes)

    def _vectorize_call(self, func, args):
        if False:
            i = 10
            return i + 15
        'Vectorized call to `func` over positional `args`.'
        if self.signature is not None:
            res = self._vectorize_call_with_signature(func, args)
        elif not args:
            res = func()
        else:
            (ufunc, otypes) = self._get_ufunc_and_otypes(func=func, args=args)
            inputs = [asanyarray(a, dtype=object) for a in args]
            outputs = ufunc(*inputs)
            if ufunc.nout == 1:
                res = asanyarray(outputs, dtype=otypes[0])
            else:
                res = tuple([asanyarray(x, dtype=t) for (x, t) in zip(outputs, otypes)])
        return res

    def _vectorize_call_with_signature(self, func, args):
        if False:
            for i in range(10):
                print('nop')
        'Vectorized call over positional arguments with a signature.'
        (input_core_dims, output_core_dims) = self._in_and_out_core_dims
        if len(args) != len(input_core_dims):
            raise TypeError('wrong number of positional arguments: expected %r, got %r' % (len(input_core_dims), len(args)))
        args = tuple((asanyarray(arg) for arg in args))
        (broadcast_shape, dim_sizes) = _parse_input_dimensions(args, input_core_dims)
        input_shapes = _calculate_shapes(broadcast_shape, dim_sizes, input_core_dims)
        args = [np.broadcast_to(arg, shape, subok=True) for (arg, shape) in zip(args, input_shapes)]
        outputs = None
        otypes = self.otypes
        nout = len(output_core_dims)
        for index in np.ndindex(*broadcast_shape):
            results = func(*(arg[index] for arg in args))
            n_results = len(results) if isinstance(results, tuple) else 1
            if nout != n_results:
                raise ValueError('wrong number of outputs from pyfunc: expected %r, got %r' % (nout, n_results))
            if nout == 1:
                results = (results,)
            if outputs is None:
                for (result, core_dims) in zip(results, output_core_dims):
                    _update_dim_sizes(dim_sizes, result, core_dims)
                outputs = _create_arrays(broadcast_shape, dim_sizes, output_core_dims, otypes, results)
            for (output, result) in zip(outputs, results):
                output[index] = result
        if outputs is None:
            if otypes is None:
                raise ValueError('cannot call `vectorize` on size 0 inputs unless `otypes` is set')
            if builtins.any((dim not in dim_sizes for dims in output_core_dims for dim in dims)):
                raise ValueError('cannot call `vectorize` with a signature including new output dimensions on size 0 inputs')
            outputs = _create_arrays(broadcast_shape, dim_sizes, output_core_dims, otypes)
        return outputs[0] if nout == 1 else outputs

def _cov_dispatcher(m, y=None, rowvar=None, bias=None, ddof=None, fweights=None, aweights=None, *, dtype=None):
    if False:
        return 10
    return (m, y, fweights, aweights)

@array_function_dispatch(_cov_dispatcher)
def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Estimate a covariance matrix, given data and weights.\n\n    Covariance indicates the level to which two variables vary together.\n    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,\n    then the covariance matrix element :math:`C_{ij}` is the covariance of\n    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance\n    of :math:`x_i`.\n\n    See the notes for an outline of the algorithm.\n\n    Parameters\n    ----------\n    m : array_like\n        A 1-D or 2-D array containing multiple variables and observations.\n        Each row of `m` represents a variable, and each column a single\n        observation of all those variables. Also see `rowvar` below.\n    y : array_like, optional\n        An additional set of variables and observations. `y` has the same form\n        as that of `m`.\n    rowvar : bool, optional\n        If `rowvar` is True (default), then each row represents a\n        variable, with observations in the columns. Otherwise, the relationship\n        is transposed: each column represents a variable, while the rows\n        contain observations.\n    bias : bool, optional\n        Default normalization (False) is by ``(N - 1)``, where ``N`` is the\n        number of observations given (unbiased estimate). If `bias` is True,\n        then normalization is by ``N``. These values can be overridden by using\n        the keyword ``ddof`` in numpy versions >= 1.5.\n    ddof : int, optional\n        If not ``None`` the default value implied by `bias` is overridden.\n        Note that ``ddof=1`` will return the unbiased estimate, even if both\n        `fweights` and `aweights` are specified, and ``ddof=0`` will return\n        the simple average. See the notes for the details. The default value\n        is ``None``.\n\n        .. versionadded:: 1.5\n    fweights : array_like, int, optional\n        1-D array of integer frequency weights; the number of times each\n        observation vector should be repeated.\n\n        .. versionadded:: 1.10\n    aweights : array_like, optional\n        1-D array of observation vector weights. These relative weights are\n        typically large for observations considered "important" and smaller for\n        observations considered less "important". If ``ddof=0`` the array of\n        weights can be used to assign probabilities to observation vectors.\n\n        .. versionadded:: 1.10\n    dtype : data-type, optional\n        Data-type of the result. By default, the return data-type will have\n        at least `numpy.float64` precision.\n\n        .. versionadded:: 1.20\n\n    Returns\n    -------\n    out : ndarray\n        The covariance matrix of the variables.\n\n    See Also\n    --------\n    corrcoef : Normalized covariance matrix\n\n    Notes\n    -----\n    Assume that the observations are in the columns of the observation\n    array `m` and let ``f = fweights`` and ``a = aweights`` for brevity. The\n    steps to compute the weighted covariance are as follows::\n\n        >>> m = np.arange(10, dtype=np.float64)\n        >>> f = np.arange(10) * 2\n        >>> a = np.arange(10) ** 2.\n        >>> ddof = 1\n        >>> w = f * a\n        >>> v1 = np.sum(w)\n        >>> v2 = np.sum(w * a)\n        >>> m -= np.sum(m * w, axis=None, keepdims=True) / v1\n        >>> cov = np.dot(m * w, m.T) * v1 / (v1**2 - ddof * v2)\n\n    Note that when ``a == 1``, the normalization factor\n    ``v1 / (v1**2 - ddof * v2)`` goes over to ``1 / (np.sum(f) - ddof)``\n    as it should.\n\n    Examples\n    --------\n    Consider two variables, :math:`x_0` and :math:`x_1`, which\n    correlate perfectly, but in opposite directions:\n\n    >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T\n    >>> x\n    array([[0, 1, 2],\n           [2, 1, 0]])\n\n    Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance\n    matrix shows this clearly:\n\n    >>> np.cov(x)\n    array([[ 1., -1.],\n           [-1.,  1.]])\n\n    Note that element :math:`C_{0,1}`, which shows the correlation between\n    :math:`x_0` and :math:`x_1`, is negative.\n\n    Further, note how `x` and `y` are combined:\n\n    >>> x = [-2.1, -1,  4.3]\n    >>> y = [3,  1.1,  0.12]\n    >>> X = np.stack((x, y), axis=0)\n    >>> np.cov(X)\n    array([[11.71      , -4.286     ], # may vary\n           [-4.286     ,  2.144133]])\n    >>> np.cov(x, y)\n    array([[11.71      , -4.286     ], # may vary\n           [-4.286     ,  2.144133]])\n    >>> np.cov(x)\n    array(11.71)\n\n    '
    if ddof is not None and ddof != int(ddof):
        raise ValueError('ddof must be integer')
    m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError('m has more than 2 dimensions')
    if y is not None:
        y = np.asarray(y)
        if y.ndim > 2:
            raise ValueError('y has more than 2 dimensions')
    if dtype is None:
        if y is None:
            dtype = np.result_type(m, np.float64)
        else:
            dtype = np.result_type(m, y, np.float64)
    X = array(m, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)
    if y is not None:
        y = array(y, copy=False, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = np.concatenate((X, y), axis=0)
    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError('fweights must be integer')
        if fweights.ndim > 1:
            raise RuntimeError('cannot handle multidimensional fweights')
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError('incompatible numbers of samples and fweights')
        if any(fweights < 0):
            raise ValueError('fweights cannot be negative')
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError('cannot handle multidimensional aweights')
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError('incompatible numbers of samples and aweights')
        if any(aweights < 0):
            raise ValueError('aweights cannot be negative')
        if w is None:
            w = aweights
        else:
            w *= aweights
    (avg, w_sum) = average(X, axis=1, weights=w, returned=True)
    w_sum = w_sum[0]
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * sum(w * aweights) / w_sum
    if fact <= 0:
        warnings.warn('Degrees of freedom <= 0 for slice', RuntimeWarning, stacklevel=2)
        fact = 0.0
    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X * w).T
    c = dot(X, X_T.conj())
    c *= np.true_divide(1, fact)
    return c.squeeze()

def _corrcoef_dispatcher(x, y=None, rowvar=None, bias=None, ddof=None, *, dtype=None):
    if False:
        return 10
    return (x, y)

@array_function_dispatch(_corrcoef_dispatcher)
def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, ddof=np._NoValue, *, dtype=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return Pearson product-moment correlation coefficients.\n\n    Please refer to the documentation for `cov` for more detail.  The\n    relationship between the correlation coefficient matrix, `R`, and the\n    covariance matrix, `C`, is\n\n    .. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} C_{jj} } }\n\n    The values of `R` are between -1 and 1, inclusive.\n\n    Parameters\n    ----------\n    x : array_like\n        A 1-D or 2-D array containing multiple variables and observations.\n        Each row of `x` represents a variable, and each column a single\n        observation of all those variables. Also see `rowvar` below.\n    y : array_like, optional\n        An additional set of variables and observations. `y` has the same\n        shape as `x`.\n    rowvar : bool, optional\n        If `rowvar` is True (default), then each row represents a\n        variable, with observations in the columns. Otherwise, the relationship\n        is transposed: each column represents a variable, while the rows\n        contain observations.\n    bias : _NoValue, optional\n        Has no effect, do not use.\n\n        .. deprecated:: 1.10.0\n    ddof : _NoValue, optional\n        Has no effect, do not use.\n\n        .. deprecated:: 1.10.0\n    dtype : data-type, optional\n        Data-type of the result. By default, the return data-type will have\n        at least `numpy.float64` precision.\n\n        .. versionadded:: 1.20\n\n    Returns\n    -------\n    R : ndarray\n        The correlation coefficient matrix of the variables.\n\n    See Also\n    --------\n    cov : Covariance matrix\n\n    Notes\n    -----\n    Due to floating point rounding the resulting array may not be Hermitian,\n    the diagonal elements may not be 1, and the elements may not satisfy the\n    inequality abs(a) <= 1. The real and imaginary parts are clipped to the\n    interval [-1,  1] in an attempt to improve on that situation but is not\n    much help in the complex case.\n\n    This function accepts but discards arguments `bias` and `ddof`.  This is\n    for backwards compatibility with previous versions of this function.  These\n    arguments had no effect on the return values of the function and can be\n    safely ignored in this and previous versions of numpy.\n\n    Examples\n    --------\n    In this example we generate two random arrays, ``xarr`` and ``yarr``, and\n    compute the row-wise and column-wise Pearson correlation coefficients,\n    ``R``. Since ``rowvar`` is  true by  default, we first find the row-wise\n    Pearson correlation coefficients between the variables of ``xarr``.\n\n    >>> import numpy as np\n    >>> rng = np.random.default_rng(seed=42)\n    >>> xarr = rng.random((3, 3))\n    >>> xarr\n    array([[0.77395605, 0.43887844, 0.85859792],\n           [0.69736803, 0.09417735, 0.97562235],\n           [0.7611397 , 0.78606431, 0.12811363]])\n    >>> R1 = np.corrcoef(xarr)\n    >>> R1\n    array([[ 1.        ,  0.99256089, -0.68080986],\n           [ 0.99256089,  1.        , -0.76492172],\n           [-0.68080986, -0.76492172,  1.        ]])\n\n    If we add another set of variables and observations ``yarr``, we can\n    compute the row-wise Pearson correlation coefficients between the\n    variables in ``xarr`` and ``yarr``.\n\n    >>> yarr = rng.random((3, 3))\n    >>> yarr\n    array([[0.45038594, 0.37079802, 0.92676499],\n           [0.64386512, 0.82276161, 0.4434142 ],\n           [0.22723872, 0.55458479, 0.06381726]])\n    >>> R2 = np.corrcoef(xarr, yarr)\n    >>> R2\n    array([[ 1.        ,  0.99256089, -0.68080986,  0.75008178, -0.934284  ,\n            -0.99004057],\n           [ 0.99256089,  1.        , -0.76492172,  0.82502011, -0.97074098,\n            -0.99981569],\n           [-0.68080986, -0.76492172,  1.        , -0.99507202,  0.89721355,\n             0.77714685],\n           [ 0.75008178,  0.82502011, -0.99507202,  1.        , -0.93657855,\n            -0.83571711],\n           [-0.934284  , -0.97074098,  0.89721355, -0.93657855,  1.        ,\n             0.97517215],\n           [-0.99004057, -0.99981569,  0.77714685, -0.83571711,  0.97517215,\n             1.        ]])\n\n    Finally if we use the option ``rowvar=False``, the columns are now\n    being treated as the variables and we will find the column-wise Pearson\n    correlation coefficients between variables in ``xarr`` and ``yarr``.\n\n    >>> R3 = np.corrcoef(xarr, yarr, rowvar=False)\n    >>> R3\n    array([[ 1.        ,  0.77598074, -0.47458546, -0.75078643, -0.9665554 ,\n             0.22423734],\n           [ 0.77598074,  1.        , -0.92346708, -0.99923895, -0.58826587,\n            -0.44069024],\n           [-0.47458546, -0.92346708,  1.        ,  0.93773029,  0.23297648,\n             0.75137473],\n           [-0.75078643, -0.99923895,  0.93773029,  1.        ,  0.55627469,\n             0.47536961],\n           [-0.9665554 , -0.58826587,  0.23297648,  0.55627469,  1.        ,\n            -0.46666491],\n           [ 0.22423734, -0.44069024,  0.75137473,  0.47536961, -0.46666491,\n             1.        ]])\n\n    '
    if bias is not np._NoValue or ddof is not np._NoValue:
        warnings.warn('bias and ddof have no effect and are deprecated', DeprecationWarning, stacklevel=2)
    c = cov(x, y, rowvar, dtype=dtype)
    try:
        d = diag(c)
    except ValueError:
        return c / c
    stddev = sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)
    return c

@set_module('numpy')
def blackman(M):
    if False:
        return 10
    '\n    Return the Blackman window.\n\n    The Blackman window is a taper formed by using the first three\n    terms of a summation of cosines. It was designed to have close to the\n    minimal leakage possible.  It is close to optimal, only slightly worse\n    than a Kaiser window.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n\n    Returns\n    -------\n    out : ndarray\n        The window, with the maximum value normalized to one (the value one\n        appears only if the number of samples is odd).\n\n    See Also\n    --------\n    bartlett, hamming, hanning, kaiser\n\n    Notes\n    -----\n    The Blackman window is defined as\n\n    .. math::  w(n) = 0.42 - 0.5 \\cos(2\\pi n/M) + 0.08 \\cos(4\\pi n/M)\n\n    Most references to the Blackman window come from the signal processing\n    literature, where it is used as one of many windowing functions for\n    smoothing values.  It is also known as an apodization (which means\n    "removing the foot", i.e. smoothing discontinuities at the beginning\n    and end of the sampled signal) or tapering function. It is known as a\n    "near optimal" tapering function, almost as good (by some measures)\n    as the kaiser window.\n\n    References\n    ----------\n    Blackman, R.B. and Tukey, J.W., (1958) The measurement of power spectra,\n    Dover Publications, New York.\n\n    Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.\n    Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.\n\n    Examples\n    --------\n    >>> import matplotlib.pyplot as plt\n    >>> np.blackman(12)\n    array([-1.38777878e-17,   3.26064346e-02,   1.59903635e-01, # may vary\n            4.14397981e-01,   7.36045180e-01,   9.67046769e-01,\n            9.67046769e-01,   7.36045180e-01,   4.14397981e-01,\n            1.59903635e-01,   3.26064346e-02,  -1.38777878e-17])\n\n    Plot the window and the frequency response.\n\n    .. plot::\n        :include-source:\n\n        import matplotlib.pyplot as plt\n        from numpy.fft import fft, fftshift\n        window = np.blackman(51)\n        plt.plot(window)\n        plt.title("Blackman window")\n        plt.ylabel("Amplitude")\n        plt.xlabel("Sample")\n        plt.show()  # doctest: +SKIP\n\n        plt.figure()\n        A = fft(window, 2048) / 25.5\n        mag = np.abs(fftshift(A))\n        freq = np.linspace(-0.5, 0.5, len(A))\n        with np.errstate(divide=\'ignore\', invalid=\'ignore\'):\n            response = 20 * np.log10(mag)\n        response = np.clip(response, -100, 100)\n        plt.plot(freq, response)\n        plt.title("Frequency response of Blackman window")\n        plt.ylabel("Magnitude [dB]")\n        plt.xlabel("Normalized frequency [cycles per sample]")\n        plt.axis(\'tight\')\n        plt.show()\n\n    '
    values = np.array([0.0, M])
    M = values[1]
    if M < 1:
        return array([], dtype=values.dtype)
    if M == 1:
        return ones(1, dtype=values.dtype)
    n = arange(1 - M, M, 2)
    return 0.42 + 0.5 * cos(pi * n / (M - 1)) + 0.08 * cos(2.0 * pi * n / (M - 1))

@set_module('numpy')
def bartlett(M):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the Bartlett window.\n\n    The Bartlett window is very similar to a triangular window, except\n    that the end points are at zero.  It is often used in signal\n    processing for tapering a signal, without generating too much\n    ripple in the frequency domain.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an\n        empty array is returned.\n\n    Returns\n    -------\n    out : array\n        The triangular window, with the maximum value normalized to one\n        (the value one appears only if the number of samples is odd), with\n        the first and last samples equal to zero.\n\n    See Also\n    --------\n    blackman, hamming, hanning, kaiser\n\n    Notes\n    -----\n    The Bartlett window is defined as\n\n    .. math:: w(n) = \\frac{2}{M-1} \\left(\n              \\frac{M-1}{2} - \\left|n - \\frac{M-1}{2}\\right|\n              \\right)\n\n    Most references to the Bartlett window come from the signal processing\n    literature, where it is used as one of many windowing functions for\n    smoothing values.  Note that convolution with this window produces linear\n    interpolation.  It is also known as an apodization (which means "removing\n    the foot", i.e. smoothing discontinuities at the beginning and end of the\n    sampled signal) or tapering function. The Fourier transform of the\n    Bartlett window is the product of two sinc functions. Note the excellent\n    discussion in Kanasewich [2]_.\n\n    References\n    ----------\n    .. [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",\n           Biometrika 37, 1-16, 1950.\n    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",\n           The University of Alberta Press, 1975, pp. 109-110.\n    .. [3] A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal\n           Processing", Prentice-Hall, 1999, pp. 468-471.\n    .. [4] Wikipedia, "Window function",\n           https://en.wikipedia.org/wiki/Window_function\n    .. [5] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,\n           "Numerical Recipes", Cambridge University Press, 1986, page 429.\n\n    Examples\n    --------\n    >>> import matplotlib.pyplot as plt\n    >>> np.bartlett(12)\n    array([ 0.        ,  0.18181818,  0.36363636,  0.54545455,  0.72727273, # may vary\n            0.90909091,  0.90909091,  0.72727273,  0.54545455,  0.36363636,\n            0.18181818,  0.        ])\n\n    Plot the window and its frequency response (requires SciPy and matplotlib).\n\n    .. plot::\n        :include-source:\n\n        import matplotlib.pyplot as plt\n        from numpy.fft import fft, fftshift\n        window = np.bartlett(51)\n        plt.plot(window)\n        plt.title("Bartlett window")\n        plt.ylabel("Amplitude")\n        plt.xlabel("Sample")\n        plt.show()\n        plt.figure()\n        A = fft(window, 2048) / 25.5\n        mag = np.abs(fftshift(A))\n        freq = np.linspace(-0.5, 0.5, len(A))\n        with np.errstate(divide=\'ignore\', invalid=\'ignore\'):\n            response = 20 * np.log10(mag)\n        response = np.clip(response, -100, 100)\n        plt.plot(freq, response)\n        plt.title("Frequency response of Bartlett window")\n        plt.ylabel("Magnitude [dB]")\n        plt.xlabel("Normalized frequency [cycles per sample]")\n        plt.axis(\'tight\')\n        plt.show()\n\n    '
    values = np.array([0.0, M])
    M = values[1]
    if M < 1:
        return array([], dtype=values.dtype)
    if M == 1:
        return ones(1, dtype=values.dtype)
    n = arange(1 - M, M, 2)
    return where(less_equal(n, 0), 1 + n / (M - 1), 1 - n / (M - 1))

@set_module('numpy')
def hanning(M):
    if False:
        print('Hello World!')
    '\n    Return the Hanning window.\n\n    The Hanning window is a taper formed by using a weighted cosine.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an\n        empty array is returned.\n\n    Returns\n    -------\n    out : ndarray, shape(M,)\n        The window, with the maximum value normalized to one (the value\n        one appears only if `M` is odd).\n\n    See Also\n    --------\n    bartlett, blackman, hamming, kaiser\n\n    Notes\n    -----\n    The Hanning window is defined as\n\n    .. math::  w(n) = 0.5 - 0.5\\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)\n               \\qquad 0 \\leq n \\leq M-1\n\n    The Hanning was named for Julius von Hann, an Austrian meteorologist.\n    It is also known as the Cosine Bell. Some authors prefer that it be\n    called a Hann window, to help avoid confusion with the very similar\n    Hamming window.\n\n    Most references to the Hanning window come from the signal processing\n    literature, where it is used as one of many windowing functions for\n    smoothing values.  It is also known as an apodization (which means\n    "removing the foot", i.e. smoothing discontinuities at the beginning\n    and end of the sampled signal) or tapering function.\n\n    References\n    ----------\n    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power\n           spectra, Dover Publications, New York.\n    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",\n           The University of Alberta Press, 1975, pp. 106-108.\n    .. [3] Wikipedia, "Window function",\n           https://en.wikipedia.org/wiki/Window_function\n    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,\n           "Numerical Recipes", Cambridge University Press, 1986, page 425.\n\n    Examples\n    --------\n    >>> np.hanning(12)\n    array([0.        , 0.07937323, 0.29229249, 0.57115742, 0.82743037,\n           0.97974649, 0.97974649, 0.82743037, 0.57115742, 0.29229249,\n           0.07937323, 0.        ])\n\n    Plot the window and its frequency response.\n\n    .. plot::\n        :include-source:\n\n        import matplotlib.pyplot as plt\n        from numpy.fft import fft, fftshift\n        window = np.hanning(51)\n        plt.plot(window)\n        plt.title("Hann window")\n        plt.ylabel("Amplitude")\n        plt.xlabel("Sample")\n        plt.show()\n\n        plt.figure()\n        A = fft(window, 2048) / 25.5\n        mag = np.abs(fftshift(A))\n        freq = np.linspace(-0.5, 0.5, len(A))\n        with np.errstate(divide=\'ignore\', invalid=\'ignore\'):\n            response = 20 * np.log10(mag)\n        response = np.clip(response, -100, 100)\n        plt.plot(freq, response)\n        plt.title("Frequency response of the Hann window")\n        plt.ylabel("Magnitude [dB]")\n        plt.xlabel("Normalized frequency [cycles per sample]")\n        plt.axis(\'tight\')\n        plt.show()\n\n    '
    values = np.array([0.0, M])
    M = values[1]
    if M < 1:
        return array([], dtype=values.dtype)
    if M == 1:
        return ones(1, dtype=values.dtype)
    n = arange(1 - M, M, 2)
    return 0.5 + 0.5 * cos(pi * n / (M - 1))

@set_module('numpy')
def hamming(M):
    if False:
        while True:
            i = 10
    '\n    Return the Hamming window.\n\n    The Hamming window is a taper formed by using a weighted cosine.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an\n        empty array is returned.\n\n    Returns\n    -------\n    out : ndarray\n        The window, with the maximum value normalized to one (the value\n        one appears only if the number of samples is odd).\n\n    See Also\n    --------\n    bartlett, blackman, hanning, kaiser\n\n    Notes\n    -----\n    The Hamming window is defined as\n\n    .. math::  w(n) = 0.54 - 0.46\\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)\n               \\qquad 0 \\leq n \\leq M-1\n\n    The Hamming was named for R. W. Hamming, an associate of J. W. Tukey\n    and is described in Blackman and Tukey. It was recommended for\n    smoothing the truncated autocovariance function in the time domain.\n    Most references to the Hamming window come from the signal processing\n    literature, where it is used as one of many windowing functions for\n    smoothing values.  It is also known as an apodization (which means\n    "removing the foot", i.e. smoothing discontinuities at the beginning\n    and end of the sampled signal) or tapering function.\n\n    References\n    ----------\n    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power\n           spectra, Dover Publications, New York.\n    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The\n           University of Alberta Press, 1975, pp. 109-110.\n    .. [3] Wikipedia, "Window function",\n           https://en.wikipedia.org/wiki/Window_function\n    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,\n           "Numerical Recipes", Cambridge University Press, 1986, page 425.\n\n    Examples\n    --------\n    >>> np.hamming(12)\n    array([ 0.08      ,  0.15302337,  0.34890909,  0.60546483,  0.84123594, # may vary\n            0.98136677,  0.98136677,  0.84123594,  0.60546483,  0.34890909,\n            0.15302337,  0.08      ])\n\n    Plot the window and the frequency response.\n\n    .. plot::\n        :include-source:\n\n        import matplotlib.pyplot as plt\n        from numpy.fft import fft, fftshift\n        window = np.hamming(51)\n        plt.plot(window)\n        plt.title("Hamming window")\n        plt.ylabel("Amplitude")\n        plt.xlabel("Sample")\n        plt.show()\n\n        plt.figure()\n        A = fft(window, 2048) / 25.5\n        mag = np.abs(fftshift(A))\n        freq = np.linspace(-0.5, 0.5, len(A))\n        response = 20 * np.log10(mag)\n        response = np.clip(response, -100, 100)\n        plt.plot(freq, response)\n        plt.title("Frequency response of Hamming window")\n        plt.ylabel("Magnitude [dB]")\n        plt.xlabel("Normalized frequency [cycles per sample]")\n        plt.axis(\'tight\')\n        plt.show()\n\n    '
    values = np.array([0.0, M])
    M = values[1]
    if M < 1:
        return array([], dtype=values.dtype)
    if M == 1:
        return ones(1, dtype=values.dtype)
    n = arange(1 - M, M, 2)
    return 0.54 + 0.46 * cos(pi * n / (M - 1))
_i0A = [-4.4153416464793395e-18, 3.3307945188222384e-17, -2.431279846547955e-16, 1.715391285555133e-15, -1.1685332877993451e-14, 7.676185498604936e-14, -4.856446783111929e-13, 2.95505266312964e-12, -1.726826291441556e-11, 9.675809035373237e-11, -5.189795601635263e-10, 2.6598237246823866e-09, -1.300025009986248e-08, 6.046995022541919e-08, -2.670793853940612e-07, 1.1173875391201037e-06, -4.4167383584587505e-06, 1.6448448070728896e-05, -5.754195010082104e-05, 0.00018850288509584165, -0.0005763755745385824, 0.0016394756169413357, -0.004324309995050576, 0.010546460394594998, -0.02373741480589947, 0.04930528423967071, -0.09490109704804764, 0.17162090152220877, -0.3046826723431984, 0.6767952744094761]
_i0B = [-7.233180487874754e-18, -4.830504485944182e-18, 4.46562142029676e-17, 3.461222867697461e-17, -2.8276239805165836e-16, -3.425485619677219e-16, 1.7725601330565263e-15, 3.8116806693526224e-15, -9.554846698828307e-15, -4.150569347287222e-14, 1.54008621752141e-14, 3.8527783827421426e-13, 7.180124451383666e-13, -1.7941785315068062e-12, -1.3215811840447713e-11, -3.1499165279632416e-11, 1.1889147107846439e-11, 4.94060238822497e-10, 3.3962320257083865e-09, 2.266668990498178e-08, 2.0489185894690638e-07, 2.8913705208347567e-06, 6.889758346916825e-05, 0.0033691164782556943, 0.8044904110141088]

def _chbevl(x, vals):
    if False:
        return 10
    b0 = vals[0]
    b1 = 0.0
    for i in range(1, len(vals)):
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + vals[i]
    return 0.5 * (b0 - b2)

def _i0_1(x):
    if False:
        i = 10
        return i + 15
    return exp(x) * _chbevl(x / 2.0 - 2, _i0A)

def _i0_2(x):
    if False:
        for i in range(10):
            print('nop')
    return exp(x) * _chbevl(32.0 / x - 2.0, _i0B) / sqrt(x)

def _i0_dispatcher(x):
    if False:
        return 10
    return (x,)

@array_function_dispatch(_i0_dispatcher)
def i0(x):
    if False:
        while True:
            i = 10
    '\n    Modified Bessel function of the first kind, order 0.\n\n    Usually denoted :math:`I_0`.\n\n    Parameters\n    ----------\n    x : array_like of float\n        Argument of the Bessel function.\n\n    Returns\n    -------\n    out : ndarray, shape = x.shape, dtype = float\n        The modified Bessel function evaluated at each of the elements of `x`.\n\n    See Also\n    --------\n    scipy.special.i0, scipy.special.iv, scipy.special.ive\n\n    Notes\n    -----\n    The scipy implementation is recommended over this function: it is a\n    proper ufunc written in C, and more than an order of magnitude faster.\n\n    We use the algorithm published by Clenshaw [1]_ and referenced by\n    Abramowitz and Stegun [2]_, for which the function domain is\n    partitioned into the two intervals [0,8] and (8,inf), and Chebyshev\n    polynomial expansions are employed in each interval. Relative error on\n    the domain [0,30] using IEEE arithmetic is documented [3]_ as having a\n    peak of 5.8e-16 with an rms of 1.4e-16 (n = 30000).\n\n    References\n    ----------\n    .. [1] C. W. Clenshaw, "Chebyshev series for mathematical functions", in\n           *National Physical Laboratory Mathematical Tables*, vol. 5, London:\n           Her Majesty\'s Stationery Office, 1962.\n    .. [2] M. Abramowitz and I. A. Stegun, *Handbook of Mathematical\n           Functions*, 10th printing, New York: Dover, 1964, pp. 379.\n           https://personal.math.ubc.ca/~cbm/aands/page_379.htm\n    .. [3] https://metacpan.org/pod/distribution/Math-Cephes/lib/Math/Cephes.pod#i0:-Modified-Bessel-function-of-order-zero\n\n    Examples\n    --------\n    >>> np.i0(0.)\n    array(1.0)\n    >>> np.i0([0, 1, 2, 3])\n    array([1.        , 1.26606588, 2.2795853 , 4.88079259])\n\n    '
    x = np.asanyarray(x)
    if x.dtype.kind == 'c':
        raise TypeError('i0 not supported for complex values')
    if x.dtype.kind != 'f':
        x = x.astype(float)
    x = np.abs(x)
    return piecewise(x, [x <= 8.0], [_i0_1, _i0_2])

@set_module('numpy')
def kaiser(M, beta):
    if False:
        i = 10
        return i + 15
    '\n    Return the Kaiser window.\n\n    The Kaiser window is a taper formed by using a Bessel function.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an\n        empty array is returned.\n    beta : float\n        Shape parameter for window.\n\n    Returns\n    -------\n    out : array\n        The window, with the maximum value normalized to one (the value\n        one appears only if the number of samples is odd).\n\n    See Also\n    --------\n    bartlett, blackman, hamming, hanning\n\n    Notes\n    -----\n    The Kaiser window is defined as\n\n    .. math::  w(n) = I_0\\left( \\beta \\sqrt{1-\\frac{4n^2}{(M-1)^2}}\n               \\right)/I_0(\\beta)\n\n    with\n\n    .. math:: \\quad -\\frac{M-1}{2} \\leq n \\leq \\frac{M-1}{2},\n\n    where :math:`I_0` is the modified zeroth-order Bessel function.\n\n    The Kaiser was named for Jim Kaiser, who discovered a simple\n    approximation to the DPSS window based on Bessel functions.  The Kaiser\n    window is a very good approximation to the Digital Prolate Spheroidal\n    Sequence, or Slepian window, which is the transform which maximizes the\n    energy in the main lobe of the window relative to total energy.\n\n    The Kaiser can approximate many other windows by varying the beta\n    parameter.\n\n    ====  =======================\n    beta  Window shape\n    ====  =======================\n    0     Rectangular\n    5     Similar to a Hamming\n    6     Similar to a Hanning\n    8.6   Similar to a Blackman\n    ====  =======================\n\n    A beta value of 14 is probably a good starting point. Note that as beta\n    gets large, the window narrows, and so the number of samples needs to be\n    large enough to sample the increasingly narrow spike, otherwise NaNs will\n    get returned.\n\n    Most references to the Kaiser window come from the signal processing\n    literature, where it is used as one of many windowing functions for\n    smoothing values.  It is also known as an apodization (which means\n    "removing the foot", i.e. smoothing discontinuities at the beginning\n    and end of the sampled signal) or tapering function.\n\n    References\n    ----------\n    .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by\n           digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.\n           John Wiley and Sons, New York, (1966).\n    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The\n           University of Alberta Press, 1975, pp. 177-178.\n    .. [3] Wikipedia, "Window function",\n           https://en.wikipedia.org/wiki/Window_function\n\n    Examples\n    --------\n    >>> import matplotlib.pyplot as plt\n    >>> np.kaiser(12, 14)\n     array([7.72686684e-06, 3.46009194e-03, 4.65200189e-02, # may vary\n            2.29737120e-01, 5.99885316e-01, 9.45674898e-01,\n            9.45674898e-01, 5.99885316e-01, 2.29737120e-01,\n            4.65200189e-02, 3.46009194e-03, 7.72686684e-06])\n\n\n    Plot the window and the frequency response.\n\n    .. plot::\n        :include-source:\n\n        import matplotlib.pyplot as plt\n        from numpy.fft import fft, fftshift\n        window = np.kaiser(51, 14)\n        plt.plot(window)\n        plt.title("Kaiser window")\n        plt.ylabel("Amplitude")\n        plt.xlabel("Sample")\n        plt.show()\n\n        plt.figure()\n        A = fft(window, 2048) / 25.5\n        mag = np.abs(fftshift(A))\n        freq = np.linspace(-0.5, 0.5, len(A))\n        response = 20 * np.log10(mag)\n        response = np.clip(response, -100, 100)\n        plt.plot(freq, response)\n        plt.title("Frequency response of Kaiser window")\n        plt.ylabel("Magnitude [dB]")\n        plt.xlabel("Normalized frequency [cycles per sample]")\n        plt.axis(\'tight\')\n        plt.show()\n\n    '
    values = np.array([0.0, M, beta])
    M = values[1]
    beta = values[2]
    if M == 1:
        return np.ones(1, dtype=values.dtype)
    n = arange(0, M)
    alpha = (M - 1) / 2.0
    return i0(beta * sqrt(1 - ((n - alpha) / alpha) ** 2.0)) / i0(beta)

def _sinc_dispatcher(x):
    if False:
        while True:
            i = 10
    return (x,)

@array_function_dispatch(_sinc_dispatcher)
def sinc(x):
    if False:
        print('Hello World!')
    '\n    Return the normalized sinc function.\n\n    The sinc function is equal to :math:`\\sin(\\pi x)/(\\pi x)` for any argument\n    :math:`x\\ne 0`. ``sinc(0)`` takes the limit value 1, making ``sinc`` not\n    only everywhere continuous but also infinitely differentiable.\n\n    .. note::\n\n        Note the normalization factor of ``pi`` used in the definition.\n        This is the most commonly used definition in signal processing.\n        Use ``sinc(x / np.pi)`` to obtain the unnormalized sinc function\n        :math:`\\sin(x)/x` that is more common in mathematics.\n\n    Parameters\n    ----------\n    x : ndarray\n        Array (possibly multi-dimensional) of values for which to calculate\n        ``sinc(x)``.\n\n    Returns\n    -------\n    out : ndarray\n        ``sinc(x)``, which has the same shape as the input.\n\n    Notes\n    -----\n    The name sinc is short for "sine cardinal" or "sinus cardinalis".\n\n    The sinc function is used in various signal processing applications,\n    including in anti-aliasing, in the construction of a Lanczos resampling\n    filter, and in interpolation.\n\n    For bandlimited interpolation of discrete-time signals, the ideal\n    interpolation kernel is proportional to the sinc function.\n\n    References\n    ----------\n    .. [1] Weisstein, Eric W. "Sinc Function." From MathWorld--A Wolfram Web\n           Resource. https://mathworld.wolfram.com/SincFunction.html\n    .. [2] Wikipedia, "Sinc function",\n           https://en.wikipedia.org/wiki/Sinc_function\n\n    Examples\n    --------\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.linspace(-4, 4, 41)\n    >>> np.sinc(x)\n     array([-3.89804309e-17,  -4.92362781e-02,  -8.40918587e-02, # may vary\n            -8.90384387e-02,  -5.84680802e-02,   3.89804309e-17,\n            6.68206631e-02,   1.16434881e-01,   1.26137788e-01,\n            8.50444803e-02,  -3.89804309e-17,  -1.03943254e-01,\n            -1.89206682e-01,  -2.16236208e-01,  -1.55914881e-01,\n            3.89804309e-17,   2.33872321e-01,   5.04551152e-01,\n            7.56826729e-01,   9.35489284e-01,   1.00000000e+00,\n            9.35489284e-01,   7.56826729e-01,   5.04551152e-01,\n            2.33872321e-01,   3.89804309e-17,  -1.55914881e-01,\n           -2.16236208e-01,  -1.89206682e-01,  -1.03943254e-01,\n           -3.89804309e-17,   8.50444803e-02,   1.26137788e-01,\n            1.16434881e-01,   6.68206631e-02,   3.89804309e-17,\n            -5.84680802e-02,  -8.90384387e-02,  -8.40918587e-02,\n            -4.92362781e-02,  -3.89804309e-17])\n\n    >>> plt.plot(x, np.sinc(x))\n    [<matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.title("Sinc Function")\n    Text(0.5, 1.0, \'Sinc Function\')\n    >>> plt.ylabel("Amplitude")\n    Text(0, 0.5, \'Amplitude\')\n    >>> plt.xlabel("X")\n    Text(0.5, 0, \'X\')\n    >>> plt.show()\n\n    '
    x = np.asanyarray(x)
    y = pi * where(x == 0, 1e-20, x)
    return sin(y) / y

def _ureduce(a, func, keepdims=False, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Internal Function.\n    Call `func` with `a` as first argument swapping the axes to use extended\n    axis on functions that don't support it natively.\n\n    Returns result and a.shape with axis dims set to 1.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array or object that can be converted to an array.\n    func : callable\n        Reduction function capable of receiving a single axis argument.\n        It is called with `a` as first argument followed by `kwargs`.\n    kwargs : keyword arguments\n        additional keyword arguments to pass to `func`.\n\n    Returns\n    -------\n    result : tuple\n        Result of func(a, **kwargs) and a.shape with axis dims set to 1\n        which can be used to reshape the result to the same shape a ufunc with\n        keepdims=True would produce.\n\n    "
    a = np.asanyarray(a)
    axis = kwargs.get('axis', None)
    out = kwargs.get('out', None)
    if keepdims is np._NoValue:
        keepdims = False
    nd = a.ndim
    if axis is not None:
        axis = _nx.normalize_axis_tuple(axis, nd)
        if keepdims:
            if out is not None:
                index_out = tuple((0 if i in axis else slice(None) for i in range(nd)))
                kwargs['out'] = out[(Ellipsis,) + index_out]
        if len(axis) == 1:
            kwargs['axis'] = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            for (i, s) in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
            a = a.reshape(a.shape[:nkeep] + (-1,))
            kwargs['axis'] = -1
    elif keepdims:
        if out is not None:
            index_out = (0,) * nd
            kwargs['out'] = out[(Ellipsis,) + index_out]
    r = func(a, **kwargs)
    if out is not None:
        return out
    if keepdims:
        if axis is None:
            index_r = (np.newaxis,) * nd
        else:
            index_r = tuple((np.newaxis if i in axis else slice(None) for i in range(nd)))
        r = r[(Ellipsis,) + index_r]
    return r

def _median_dispatcher(a, axis=None, out=None, overwrite_input=None, keepdims=None):
    if False:
        while True:
            i = 10
    return (a, out)

@array_function_dispatch(_median_dispatcher)
def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    if False:
        i = 10
        return i + 15
    '\n    Compute the median along the specified axis.\n\n    Returns the median of the array elements.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array or object that can be converted to an array.\n    axis : {int, sequence of int, None}, optional\n        Axis or axes along which the medians are computed. The default\n        is to compute the median along a flattened version of the array.\n        A sequence of axes is supported since version 1.9.0.\n    out : ndarray, optional\n        Alternative output array in which to place the result. It must\n        have the same shape and buffer length as the expected output,\n        but the type (of the output) will be cast if necessary.\n    overwrite_input : bool, optional\n       If True, then allow use of memory of input array `a` for\n       calculations. The input array will be modified by the call to\n       `median`. This will save memory when you do not need to preserve\n       the contents of the input array. Treat the input as undefined,\n       but it will probably be fully or partially sorted. Default is\n       False. If `overwrite_input` is ``True`` and `a` is not already an\n       `ndarray`, an error will be raised.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left\n        in the result as dimensions with size one. With this option,\n        the result will broadcast correctly against the original `arr`.\n\n        .. versionadded:: 1.9.0\n\n    Returns\n    -------\n    median : ndarray\n        A new array holding the result. If the input contains integers\n        or floats smaller than ``float64``, then the output data-type is\n        ``np.float64``.  Otherwise, the data-type of the output is the\n        same as that of the input. If `out` is specified, that array is\n        returned instead.\n\n    See Also\n    --------\n    mean, percentile\n\n    Notes\n    -----\n    Given a vector ``V`` of length ``N``, the median of ``V`` is the\n    middle value of a sorted copy of ``V``, ``V_sorted`` - i\n    e., ``V_sorted[(N-1)/2]``, when ``N`` is odd, and the average of the\n    two middle values of ``V_sorted`` when ``N`` is even.\n\n    Examples\n    --------\n    >>> a = np.array([[10, 7, 4], [3, 2, 1]])\n    >>> a\n    array([[10,  7,  4],\n           [ 3,  2,  1]])\n    >>> np.median(a)\n    3.5\n    >>> np.median(a, axis=0)\n    array([6.5, 4.5, 2.5])\n    >>> np.median(a, axis=1)\n    array([7.,  2.])\n    >>> m = np.median(a, axis=0)\n    >>> out = np.zeros_like(m)\n    >>> np.median(a, axis=0, out=m)\n    array([6.5,  4.5,  2.5])\n    >>> m\n    array([6.5,  4.5,  2.5])\n    >>> b = a.copy()\n    >>> np.median(b, axis=1, overwrite_input=True)\n    array([7.,  2.])\n    >>> assert not np.all(a==b)\n    >>> b = a.copy()\n    >>> np.median(b, axis=None, overwrite_input=True)\n    3.5\n    >>> assert not np.all(a==b)\n\n    '
    return _ureduce(a, func=_median, keepdims=keepdims, axis=axis, out=out, overwrite_input=overwrite_input)

def _median(a, axis=None, out=None, overwrite_input=False):
    if False:
        print('Hello World!')
    a = np.asanyarray(a)
    if axis is None:
        sz = a.size
    else:
        sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        kth = [szh - 1, szh]
    else:
        kth = [(sz - 1) // 2]
    supports_nans = np.issubdtype(a.dtype, np.inexact) or a.dtype.kind in 'Mm'
    if supports_nans:
        kth.append(-1)
    if overwrite_input:
        if axis is None:
            part = a.ravel()
            part.partition(kth)
        else:
            a.partition(kth, axis=axis)
            part = a
    else:
        part = partition(a, kth, axis=axis)
    if part.shape == ():
        return part.item()
    if axis is None:
        axis = 0
    indexer = [slice(None)] * part.ndim
    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        indexer[axis] = slice(index, index + 1)
    else:
        indexer[axis] = slice(index - 1, index + 1)
    indexer = tuple(indexer)
    rout = mean(part[indexer], axis=axis, out=out)
    if supports_nans and sz > 0:
        rout = np.lib._utils_impl._median_nancheck(part, rout, axis)
    return rout

def _percentile_dispatcher(a, q, axis=None, out=None, overwrite_input=None, method=None, keepdims=None, *, interpolation=None):
    if False:
        while True:
            i = 10
    return (a, q, out)

@array_function_dispatch(_percentile_dispatcher)
def percentile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *, interpolation=None):
    if False:
        return 10
    '\n    Compute the q-th percentile of the data along the specified axis.\n\n    Returns the q-th percentile(s) of the array elements.\n\n    Parameters\n    ----------\n    a : array_like of real numbers\n        Input array or object that can be converted to an array.\n    q : array_like of float\n        Percentage or sequence of percentages for the percentiles to compute.\n        Values must be between 0 and 100 inclusive.\n    axis : {int, tuple of int, None}, optional\n        Axis or axes along which the percentiles are computed. The\n        default is to compute the percentile(s) along a flattened\n        version of the array.\n\n        .. versionchanged:: 1.9.0\n            A tuple of axes is supported\n    out : ndarray, optional\n        Alternative output array in which to place the result. It must\n        have the same shape and buffer length as the expected output,\n        but the type (of the output) will be cast if necessary.\n    overwrite_input : bool, optional\n        If True, then allow the input array `a` to be modified by intermediate\n        calculations, to save memory. In this case, the contents of the input\n        `a` after this function completes is undefined.\n    method : str, optional\n        This parameter specifies the method to use for estimating the\n        percentile.  There are many different methods, some unique to NumPy.\n        See the notes for explanation.  The options sorted by their R type\n        as summarized in the H&F paper [1]_ are:\n\n        1. \'inverted_cdf\'\n        2. \'averaged_inverted_cdf\'\n        3. \'closest_observation\'\n        4. \'interpolated_inverted_cdf\'\n        5. \'hazen\'\n        6. \'weibull\'\n        7. \'linear\'  (default)\n        8. \'median_unbiased\'\n        9. \'normal_unbiased\'\n\n        The first three methods are discontinuous.  NumPy further defines the\n        following discontinuous variations of the default \'linear\' (7.) option:\n\n        * \'lower\'\n        * \'higher\',\n        * \'midpoint\'\n        * \'nearest\'\n\n        .. versionchanged:: 1.22.0\n            This argument was previously called "interpolation" and only\n            offered the "linear" default and last four options.\n\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left in\n        the result as dimensions with size one. With this option, the\n        result will broadcast correctly against the original array `a`.\n\n        .. versionadded:: 1.9.0\n\n    interpolation : str, optional\n        Deprecated name for the method keyword argument.\n\n        .. deprecated:: 1.22.0\n\n    Returns\n    -------\n    percentile : scalar or ndarray\n        If `q` is a single percentile and `axis=None`, then the result\n        is a scalar. If multiple percentiles are given, first axis of\n        the result corresponds to the percentiles. The other axes are\n        the axes that remain after the reduction of `a`. If the input\n        contains integers or floats smaller than ``float64``, the output\n        data-type is ``float64``. Otherwise, the output data-type is the\n        same as that of the input. If `out` is specified, that array is\n        returned instead.\n\n    See Also\n    --------\n    mean\n    median : equivalent to ``percentile(..., 50)``\n    nanpercentile\n    quantile : equivalent to percentile, except q in the range [0, 1].\n\n    Notes\n    -----\n    Given a vector ``V`` of length ``n``, the q-th percentile of ``V`` is\n    the value ``q/100`` of the way from the minimum to the maximum in a\n    sorted copy of ``V``. The values and distances of the two nearest\n    neighbors as well as the `method` parameter will determine the\n    percentile if the normalized ranking does not match the location of\n    ``q`` exactly. This function is the same as the median if ``q=50``, the\n    same as the minimum if ``q=0`` and the same as the maximum if\n    ``q=100``.\n\n    The optional `method` parameter specifies the method to use when the\n    desired percentile lies between two indexes ``i`` and ``j = i + 1``.\n    In that case, we first determine ``i + g``, a virtual index that lies\n    between ``i`` and ``j``, where  ``i`` is the floor and ``g`` is the\n    fractional part of the index. The final result is, then, an interpolation\n    of ``a[i]`` and ``a[j]`` based on ``g``. During the computation of ``g``,\n    ``i`` and ``j`` are modified using correction constants ``alpha`` and\n    ``beta`` whose choices depend on the ``method`` used. Finally, note that\n    since Python uses 0-based indexing, the code subtracts another 1 from the\n    index internally.\n\n    The following formula determines the virtual index ``i + g``, the location\n    of the percentile in the sorted sample:\n\n    .. math::\n        i + g = (q / 100) * ( n - alpha - beta + 1 ) + alpha\n\n    The different methods then work as follows\n\n    inverted_cdf:\n        method 1 of H&F [1]_.\n        This method gives discontinuous results:\n\n        * if g > 0 ; then take j\n        * if g = 0 ; then take i\n\n    averaged_inverted_cdf:\n        method 2 of H&F [1]_.\n        This method gives discontinuous results:\n\n        * if g > 0 ; then take j\n        * if g = 0 ; then average between bounds\n\n    closest_observation:\n        method 3 of H&F [1]_.\n        This method gives discontinuous results:\n\n        * if g > 0 ; then take j\n        * if g = 0 and index is odd ; then take j\n        * if g = 0 and index is even ; then take i\n\n    interpolated_inverted_cdf:\n        method 4 of H&F [1]_.\n        This method gives continuous results using:\n\n        * alpha = 0\n        * beta = 1\n\n    hazen:\n        method 5 of H&F [1]_.\n        This method gives continuous results using:\n\n        * alpha = 1/2\n        * beta = 1/2\n\n    weibull:\n        method 6 of H&F [1]_.\n        This method gives continuous results using:\n\n        * alpha = 0\n        * beta = 0\n\n    linear:\n        method 7 of H&F [1]_.\n        This method gives continuous results using:\n\n        * alpha = 1\n        * beta = 1\n\n    median_unbiased:\n        method 8 of H&F [1]_.\n        This method is probably the best method if the sample\n        distribution function is unknown (see reference).\n        This method gives continuous results using:\n\n        * alpha = 1/3\n        * beta = 1/3\n\n    normal_unbiased:\n        method 9 of H&F [1]_.\n        This method is probably the best method if the sample\n        distribution function is known to be normal.\n        This method gives continuous results using:\n\n        * alpha = 3/8\n        * beta = 3/8\n\n    lower:\n        NumPy method kept for backwards compatibility.\n        Takes ``i`` as the interpolation point.\n\n    higher:\n        NumPy method kept for backwards compatibility.\n        Takes ``j`` as the interpolation point.\n\n    nearest:\n        NumPy method kept for backwards compatibility.\n        Takes ``i`` or ``j``, whichever is nearest.\n\n    midpoint:\n        NumPy method kept for backwards compatibility.\n        Uses ``(i + j) / 2``.\n\n    Examples\n    --------\n    >>> a = np.array([[10, 7, 4], [3, 2, 1]])\n    >>> a\n    array([[10,  7,  4],\n           [ 3,  2,  1]])\n    >>> np.percentile(a, 50)\n    3.5\n    >>> np.percentile(a, 50, axis=0)\n    array([6.5, 4.5, 2.5])\n    >>> np.percentile(a, 50, axis=1)\n    array([7.,  2.])\n    >>> np.percentile(a, 50, axis=1, keepdims=True)\n    array([[7.],\n           [2.]])\n\n    >>> m = np.percentile(a, 50, axis=0)\n    >>> out = np.zeros_like(m)\n    >>> np.percentile(a, 50, axis=0, out=out)\n    array([6.5, 4.5, 2.5])\n    >>> m\n    array([6.5, 4.5, 2.5])\n\n    >>> b = a.copy()\n    >>> np.percentile(b, 50, axis=1, overwrite_input=True)\n    array([7.,  2.])\n    >>> assert not np.all(a == b)\n\n    The different methods can be visualized graphically:\n\n    .. plot::\n\n        import matplotlib.pyplot as plt\n\n        a = np.arange(4)\n        p = np.linspace(0, 100, 6001)\n        ax = plt.gca()\n        lines = [\n            (\'linear\', \'-\', \'C0\'),\n            (\'inverted_cdf\', \':\', \'C1\'),\n            # Almost the same as `inverted_cdf`:\n            (\'averaged_inverted_cdf\', \'-.\', \'C1\'),\n            (\'closest_observation\', \':\', \'C2\'),\n            (\'interpolated_inverted_cdf\', \'--\', \'C1\'),\n            (\'hazen\', \'--\', \'C3\'),\n            (\'weibull\', \'-.\', \'C4\'),\n            (\'median_unbiased\', \'--\', \'C5\'),\n            (\'normal_unbiased\', \'-.\', \'C6\'),\n            ]\n        for method, style, color in lines:\n            ax.plot(\n                p, np.percentile(a, p, method=method),\n                label=method, linestyle=style, color=color)\n        ax.set(\n            title=\'Percentiles for different methods and data: \' + str(a),\n            xlabel=\'Percentile\',\n            ylabel=\'Estimated percentile value\',\n            yticks=a)\n        ax.legend(bbox_to_anchor=(1.03, 1))\n        plt.tight_layout()\n        plt.show()\n\n    References\n    ----------\n    .. [1] R. J. Hyndman and Y. Fan,\n       "Sample quantiles in statistical packages,"\n       The American Statistician, 50(4), pp. 361-365, 1996\n\n    '
    if interpolation is not None:
        method = _check_interpolation_as_method(method, interpolation, 'percentile')
    a = np.asanyarray(a)
    if a.dtype.kind == 'c':
        raise TypeError('a must be an array of real numbers')
    q = np.true_divide(q, a.dtype.type(100) if a.dtype.kind == 'f' else 100)
    q = asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError('Percentiles must be in the range [0, 100]')
    return _quantile_unchecked(a, q, axis, out, overwrite_input, method, keepdims)

def _quantile_dispatcher(a, q, axis=None, out=None, overwrite_input=None, method=None, keepdims=None, *, interpolation=None):
    if False:
        i = 10
        return i + 15
    return (a, q, out)

@array_function_dispatch(_quantile_dispatcher)
def quantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *, interpolation=None):
    if False:
        print('Hello World!')
    '\n    Compute the q-th quantile of the data along the specified axis.\n\n    .. versionadded:: 1.15.0\n\n    Parameters\n    ----------\n    a : array_like of real numbers\n        Input array or object that can be converted to an array.\n    q : array_like of float\n        Probability or sequence of probabilities for the quantiles to compute.\n        Values must be between 0 and 1 inclusive.\n    axis : {int, tuple of int, None}, optional\n        Axis or axes along which the quantiles are computed. The default is\n        to compute the quantile(s) along a flattened version of the array.\n    out : ndarray, optional\n        Alternative output array in which to place the result. It must have\n        the same shape and buffer length as the expected output, but the\n        type (of the output) will be cast if necessary.\n    overwrite_input : bool, optional\n        If True, then allow the input array `a` to be modified by\n        intermediate calculations, to save memory. In this case, the\n        contents of the input `a` after this function completes is\n        undefined.\n    method : str, optional\n        This parameter specifies the method to use for estimating the\n        quantile.  There are many different methods, some unique to NumPy.\n        See the notes for explanation.  The options sorted by their R type\n        as summarized in the H&F paper [1]_ are:\n\n        1. \'inverted_cdf\'\n        2. \'averaged_inverted_cdf\'\n        3. \'closest_observation\'\n        4. \'interpolated_inverted_cdf\'\n        5. \'hazen\'\n        6. \'weibull\'\n        7. \'linear\'  (default)\n        8. \'median_unbiased\'\n        9. \'normal_unbiased\'\n\n        The first three methods are discontinuous.  NumPy further defines the\n        following discontinuous variations of the default \'linear\' (7.) option:\n\n        * \'lower\'\n        * \'higher\',\n        * \'midpoint\'\n        * \'nearest\'\n\n        .. versionchanged:: 1.22.0\n            This argument was previously called "interpolation" and only\n            offered the "linear" default and last four options.\n\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left in\n        the result as dimensions with size one. With this option, the\n        result will broadcast correctly against the original array `a`.\n\n    interpolation : str, optional\n        Deprecated name for the method keyword argument.\n\n        .. deprecated:: 1.22.0\n\n    Returns\n    -------\n    quantile : scalar or ndarray\n        If `q` is a single probability and `axis=None`, then the result\n        is a scalar. If multiple probability levels are given, first axis\n        of the result corresponds to the quantiles. The other axes are\n        the axes that remain after the reduction of `a`. If the input\n        contains integers or floats smaller than ``float64``, the output\n        data-type is ``float64``. Otherwise, the output data-type is the\n        same as that of the input. If `out` is specified, that array is\n        returned instead.\n\n    See Also\n    --------\n    mean\n    percentile : equivalent to quantile, but with q in the range [0, 100].\n    median : equivalent to ``quantile(..., 0.5)``\n    nanquantile\n\n    Notes\n    -----\n    Given a vector ``V`` of length ``n``, the q-th quantile of ``V`` is\n    the value ``q`` of the way from the minimum to the maximum in a\n    sorted copy of ``V``. The values and distances of the two nearest\n    neighbors as well as the `method` parameter will determine the\n    quantile if the normalized ranking does not match the location of\n    ``q`` exactly. This function is the same as the median if ``q=0.5``, the\n    same as the minimum if ``q=0.0`` and the same as the maximum if\n    ``q=1.0``.\n\n    The optional `method` parameter specifies the method to use when the\n    desired quantile lies between two indexes ``i`` and ``j = i + 1``.\n    In that case, we first determine ``i + g``, a virtual index that lies\n    between ``i`` and ``j``, where  ``i`` is the floor and ``g`` is the\n    fractional part of the index. The final result is, then, an interpolation\n    of ``a[i]`` and ``a[j]`` based on ``g``. During the computation of ``g``,\n    ``i`` and ``j`` are modified using correction constants ``alpha`` and\n    ``beta`` whose choices depend on the ``method`` used. Finally, note that\n    since Python uses 0-based indexing, the code subtracts another 1 from the\n    index internally.\n\n    The following formula determines the virtual index ``i + g``, the location\n    of the quantile in the sorted sample:\n\n    .. math::\n        i + g = q * ( n - alpha - beta + 1 ) + alpha\n\n    The different methods then work as follows\n\n    inverted_cdf:\n        method 1 of H&F [1]_.\n        This method gives discontinuous results:\n\n        * if g > 0 ; then take j\n        * if g = 0 ; then take i\n\n    averaged_inverted_cdf:\n        method 2 of H&F [1]_.\n        This method gives discontinuous results:\n\n        * if g > 0 ; then take j\n        * if g = 0 ; then average between bounds\n\n    closest_observation:\n        method 3 of H&F [1]_.\n        This method gives discontinuous results:\n\n        * if g > 0 ; then take j\n        * if g = 0 and index is odd ; then take j\n        * if g = 0 and index is even ; then take i\n\n    interpolated_inverted_cdf:\n        method 4 of H&F [1]_.\n        This method gives continuous results using:\n\n        * alpha = 0\n        * beta = 1\n\n    hazen:\n        method 5 of H&F [1]_.\n        This method gives continuous results using:\n\n        * alpha = 1/2\n        * beta = 1/2\n\n    weibull:\n        method 6 of H&F [1]_.\n        This method gives continuous results using:\n\n        * alpha = 0\n        * beta = 0\n\n    linear:\n        method 7 of H&F [1]_.\n        This method gives continuous results using:\n\n        * alpha = 1\n        * beta = 1\n\n    median_unbiased:\n        method 8 of H&F [1]_.\n        This method is probably the best method if the sample\n        distribution function is unknown (see reference).\n        This method gives continuous results using:\n\n        * alpha = 1/3\n        * beta = 1/3\n\n    normal_unbiased:\n        method 9 of H&F [1]_.\n        This method is probably the best method if the sample\n        distribution function is known to be normal.\n        This method gives continuous results using:\n\n        * alpha = 3/8\n        * beta = 3/8\n\n    lower:\n        NumPy method kept for backwards compatibility.\n        Takes ``i`` as the interpolation point.\n\n    higher:\n        NumPy method kept for backwards compatibility.\n        Takes ``j`` as the interpolation point.\n\n    nearest:\n        NumPy method kept for backwards compatibility.\n        Takes ``i`` or ``j``, whichever is nearest.\n\n    midpoint:\n        NumPy method kept for backwards compatibility.\n        Uses ``(i + j) / 2``.\n\n    Examples\n    --------\n    >>> a = np.array([[10, 7, 4], [3, 2, 1]])\n    >>> a\n    array([[10,  7,  4],\n           [ 3,  2,  1]])\n    >>> np.quantile(a, 0.5)\n    3.5\n    >>> np.quantile(a, 0.5, axis=0)\n    array([6.5, 4.5, 2.5])\n    >>> np.quantile(a, 0.5, axis=1)\n    array([7.,  2.])\n    >>> np.quantile(a, 0.5, axis=1, keepdims=True)\n    array([[7.],\n           [2.]])\n    >>> m = np.quantile(a, 0.5, axis=0)\n    >>> out = np.zeros_like(m)\n    >>> np.quantile(a, 0.5, axis=0, out=out)\n    array([6.5, 4.5, 2.5])\n    >>> m\n    array([6.5, 4.5, 2.5])\n    >>> b = a.copy()\n    >>> np.quantile(b, 0.5, axis=1, overwrite_input=True)\n    array([7.,  2.])\n    >>> assert not np.all(a == b)\n\n    See also `numpy.percentile` for a visualization of most methods.\n\n    References\n    ----------\n    .. [1] R. J. Hyndman and Y. Fan,\n       "Sample quantiles in statistical packages,"\n       The American Statistician, 50(4), pp. 361-365, 1996\n\n    '
    if interpolation is not None:
        method = _check_interpolation_as_method(method, interpolation, 'quantile')
    a = np.asanyarray(a)
    if a.dtype.kind == 'c':
        raise TypeError('a must be an array of real numbers')
    if isinstance(q, (int, float)) and a.dtype.kind == 'f':
        q = np.asanyarray(q, dtype=a.dtype)
    else:
        q = np.asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError('Quantiles must be in the range [0, 1]')
    return _quantile_unchecked(a, q, axis, out, overwrite_input, method, keepdims)

def _quantile_unchecked(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False):
    if False:
        for i in range(10):
            print('nop')
    'Assumes that q is in [0, 1], and is an ndarray'
    return _ureduce(a, func=_quantile_ureduce_func, q=q, keepdims=keepdims, axis=axis, out=out, overwrite_input=overwrite_input, method=method)

def _quantile_is_valid(q):
    if False:
        for i in range(10):
            print('nop')
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not 0.0 <= q[i] <= 1.0:
                return False
    elif not (q.min() >= 0 and q.max() <= 1):
        return False
    return True

def _check_interpolation_as_method(method, interpolation, fname):
    if False:
        for i in range(10):
            print('nop')
    warnings.warn(f"the `interpolation=` argument to {fname} was renamed to `method=`, which has additional options.\nUsers of the modes 'nearest', 'lower', 'higher', or 'midpoint' are encouraged to review the method they used. (Deprecated NumPy 1.22)", DeprecationWarning, stacklevel=4)
    if method != 'linear':
        raise TypeError('You shall not pass both `method` and `interpolation`!\n(`interpolation` is Deprecated in favor of `method`)')
    return interpolation

def _compute_virtual_index(n, quantiles, alpha: float, beta: float):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the floating point indexes of an array for the linear\n    interpolation of quantiles.\n    n : array_like\n        The sample sizes.\n    quantiles : array_like\n        The quantiles values.\n    alpha : float\n        A constant used to correct the index computed.\n    beta : float\n        A constant used to correct the index computed.\n\n    alpha and beta values depend on the chosen method\n    (see quantile documentation)\n\n    Reference:\n    Hyndman&Fan paper "Sample Quantiles in Statistical Packages",\n    DOI: 10.1080/00031305.1996.10473566\n    '
    return n * quantiles + (alpha + quantiles * (1 - alpha - beta)) - 1

def _get_gamma(virtual_indexes, previous_indexes, method):
    if False:
        print('Hello World!')
    "\n    Compute gamma (a.k.a 'm' or 'weight') for the linear interpolation\n    of quantiles.\n\n    virtual_indexes : array_like\n        The indexes where the percentile is supposed to be found in the sorted\n        sample.\n    previous_indexes : array_like\n        The floor values of virtual_indexes.\n    interpolation : dict\n        The interpolation method chosen, which may have a specific rule\n        modifying gamma.\n\n    gamma is usually the fractional part of virtual_indexes but can be modified\n    by the interpolation method.\n    "
    gamma = np.asanyarray(virtual_indexes - previous_indexes)
    gamma = method['fix_gamma'](gamma, virtual_indexes)
    return np.asanyarray(gamma, dtype=virtual_indexes.dtype)

def _lerp(a, b, t, out=None):
    if False:
        while True:
            i = 10
    '\n    Compute the linear interpolation weighted by gamma on each point of\n    two same shape array.\n\n    a : array_like\n        Left bound.\n    b : array_like\n        Right bound.\n    t : array_like\n        The interpolation weight.\n    out : array_like\n        Output array.\n    '
    diff_b_a = subtract(b, a)
    lerp_interpolation = asanyarray(add(a, diff_b_a * t, out=out))
    subtract(b, diff_b_a * (1 - t), out=lerp_interpolation, where=t >= 0.5)
    if lerp_interpolation.ndim == 0 and out is None:
        lerp_interpolation = lerp_interpolation[()]
    return lerp_interpolation

def _get_gamma_mask(shape, default_value, conditioned_value, where):
    if False:
        print('Hello World!')
    out = np.full(shape, default_value)
    np.copyto(out, conditioned_value, where=where, casting='unsafe')
    return out

def _discret_interpolation_to_boundaries(index, gamma_condition_fun):
    if False:
        while True:
            i = 10
    previous = np.floor(index)
    next = previous + 1
    gamma = index - previous
    res = _get_gamma_mask(shape=index.shape, default_value=next, conditioned_value=previous, where=gamma_condition_fun(gamma, index)).astype(np.intp)
    res[res < 0] = 0
    return res

def _closest_observation(n, quantiles):
    if False:
        i = 10
        return i + 15
    gamma_fun = lambda gamma, index: (gamma == 0) & (np.floor(index) % 2 == 0)
    return _discret_interpolation_to_boundaries(n * quantiles - 1 - 0.5, gamma_fun)

def _inverted_cdf(n, quantiles):
    if False:
        i = 10
        return i + 15
    gamma_fun = lambda gamma, _: gamma == 0
    return _discret_interpolation_to_boundaries(n * quantiles - 1, gamma_fun)

def _quantile_ureduce_func(a: np.array, q: np.array, axis: int=None, out=None, overwrite_input: bool=False, method='linear') -> np.array:
    if False:
        return 10
    if q.ndim > 2:
        raise ValueError('q must be a scalar or 1d')
    if overwrite_input:
        if axis is None:
            axis = 0
            arr = a.ravel()
        else:
            arr = a
    elif axis is None:
        axis = 0
        arr = a.flatten()
    else:
        arr = a.copy()
    result = _quantile(arr, quantiles=q, axis=axis, method=method, out=out)
    return result

def _get_indexes(arr, virtual_indexes, valid_values_count):
    if False:
        i = 10
        return i + 15
    '\n    Get the valid indexes of arr neighbouring virtual_indexes.\n    Note\n    This is a companion function to linear interpolation of\n    Quantiles\n\n    Returns\n    -------\n    (previous_indexes, next_indexes): Tuple\n        A Tuple of virtual_indexes neighbouring indexes\n    '
    previous_indexes = np.asanyarray(np.floor(virtual_indexes))
    next_indexes = np.asanyarray(previous_indexes + 1)
    indexes_above_bounds = virtual_indexes >= valid_values_count - 1
    if indexes_above_bounds.any():
        previous_indexes[indexes_above_bounds] = -1
        next_indexes[indexes_above_bounds] = -1
    indexes_below_bounds = virtual_indexes < 0
    if indexes_below_bounds.any():
        previous_indexes[indexes_below_bounds] = 0
        next_indexes[indexes_below_bounds] = 0
    if np.issubdtype(arr.dtype, np.inexact):
        virtual_indexes_nans = np.isnan(virtual_indexes)
        if virtual_indexes_nans.any():
            previous_indexes[virtual_indexes_nans] = -1
            next_indexes[virtual_indexes_nans] = -1
    previous_indexes = previous_indexes.astype(np.intp)
    next_indexes = next_indexes.astype(np.intp)
    return (previous_indexes, next_indexes)

def _quantile(arr: np.array, quantiles: np.array, axis: int=-1, method='linear', out=None):
    if False:
        return 10
    '\n    Private function that doesn\'t support extended axis or keepdims.\n    These methods are extended to this function using _ureduce\n    See nanpercentile for parameter usage\n    It computes the quantiles of the array for the given axis.\n    A linear interpolation is performed based on the `interpolation`.\n\n    By default, the method is "linear" where alpha == beta == 1 which\n    performs the 7th method of Hyndman&Fan.\n    With "median_unbiased" we get alpha == beta == 1/3\n    thus the 8th method of Hyndman&Fan.\n    '
    arr = np.asanyarray(arr)
    values_count = arr.shape[axis]
    if axis != 0:
        arr = np.moveaxis(arr, axis, destination=0)
    try:
        method = _QuantileMethods[method]
    except KeyError:
        raise ValueError(f'{method!r} is not a valid method. Use one of: {_QuantileMethods.keys()}') from None
    virtual_indexes = method['get_virtual_index'](values_count, quantiles)
    virtual_indexes = np.asanyarray(virtual_indexes)
    supports_nans = np.issubdtype(arr.dtype, np.inexact) or arr.dtype.kind in 'Mm'
    if np.issubdtype(virtual_indexes.dtype, np.integer):
        if supports_nans:
            arr.partition(concatenate((virtual_indexes.ravel(), [-1])), axis=0)
            slices_having_nans = np.isnan(arr[-1, ...])
        else:
            arr.partition(virtual_indexes.ravel(), axis=0)
            slices_having_nans = np.array(False, dtype=bool)
        result = take(arr, virtual_indexes, axis=0, out=out)
    else:
        (previous_indexes, next_indexes) = _get_indexes(arr, virtual_indexes, values_count)
        arr.partition(np.unique(np.concatenate(([0, -1], previous_indexes.ravel(), next_indexes.ravel()))), axis=0)
        if supports_nans:
            slices_having_nans = np.isnan(arr[-1, ...])
        else:
            slices_having_nans = None
        previous = arr[previous_indexes]
        next = arr[next_indexes]
        gamma = _get_gamma(virtual_indexes, previous_indexes, method)
        result_shape = virtual_indexes.shape + (1,) * (arr.ndim - 1)
        gamma = gamma.reshape(result_shape)
        result = _lerp(previous, next, gamma, out=out)
    if np.any(slices_having_nans):
        if result.ndim == 0 and out is None:
            result = arr[-1]
        else:
            np.copyto(result, arr[-1, ...], where=slices_having_nans)
    return result

def _trapz_dispatcher(y, x=None, dx=None, axis=None):
    if False:
        return 10
    return (y, x)

@array_function_dispatch(_trapz_dispatcher)
def trapz(y, x=None, dx=1.0, axis=-1):
    if False:
        i = 10
        return i + 15
    '\n    Integrate along the given axis using the composite trapezoidal rule.\n\n    .. deprecated:: 2.0\n        Use `scipy.integrate.trapezoid` instead.\n\n    If `x` is provided, the integration happens in sequence along its\n    elements - they are not sorted.\n\n    Integrate `y` (`x`) along each 1d slice on the given axis, compute\n    :math:`\\int y(x) dx`.\n    When `x` is specified, this integrates along the parametric curve,\n    computing :math:`\\int_t y(t) dt =\n    \\int_t y(t) \\left.\\frac{dx}{dt}\\right|_{x=x(t)} dt`.\n\n    Parameters\n    ----------\n    y : array_like\n        Input array to integrate.\n    x : array_like, optional\n        The sample points corresponding to the `y` values. If `x` is None,\n        the sample points are assumed to be evenly spaced `dx` apart. The\n        default is None.\n    dx : scalar, optional\n        The spacing between sample points when `x` is None. The default is 1.\n    axis : int, optional\n        The axis along which to integrate.\n\n    Returns\n    -------\n    trapz : float or ndarray\n        Definite integral of `y` = n-dimensional array as approximated along\n        a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,\n        then the result is a float. If `n` is greater than 1, then the result\n        is an `n`-1 dimensional array.\n\n    See Also\n    --------\n    sum, cumsum\n\n    Notes\n    -----\n    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points\n    will be taken from `y` array, by default x-axis distances between\n    points will be 1.0, alternatively they can be provided with `x` array\n    or with `dx` scalar.  Return value will be equal to combined area under\n    the red lines.\n\n\n    References\n    ----------\n    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule\n\n    .. [2] Illustration image:\n           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png\n\n    Examples\n    --------\n    Use the trapezoidal rule on evenly spaced points:\n\n    >>> np.trapz([1, 2, 3])\n    4.0\n\n    The spacing between sample points can be selected by either the\n    ``x`` or ``dx`` arguments:\n\n    >>> np.trapz([1, 2, 3], x=[4, 6, 8])\n    8.0\n    >>> np.trapz([1, 2, 3], dx=2)\n    8.0\n\n    Using a decreasing ``x`` corresponds to integrating in reverse:\n\n    >>> np.trapz([1, 2, 3], x=[8, 6, 4])\n    -8.0\n\n    More generally ``x`` is used to integrate along a parametric curve. We can\n    estimate the integral :math:`\\int_0^1 x^2 = 1/3` using:\n\n    >>> x = np.linspace(0, 1, num=50)\n    >>> y = x**2\n    >>> np.trapz(y, x)\n    0.33340274885464394\n\n    Or estimate the area of a circle, noting we repeat the sample which closes\n    the curve:\n\n    >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)\n    >>> np.trapz(np.cos(theta), x=np.sin(theta))\n    3.141571941375841\n\n    ``np.trapz`` can be applied along a specified axis to do multiple\n    computations in one call:\n\n    >>> a = np.arange(6).reshape(2, 3)\n    >>> a\n    array([[0, 1, 2],\n           [3, 4, 5]])\n    >>> np.trapz(a, axis=0)\n    array([1.5, 2.5, 3.5])\n    >>> np.trapz(a, axis=1)\n    array([2.,  8.])\n    '
    warnings.warn('`trapz` is deprecated. Use `scipy.integrate.trapezoid` instead.', DeprecationWarning, stacklevel=2)
    y = asanyarray(y)
    if x is None:
        d = dx
    else:
        x = asanyarray(x)
        if x.ndim == 1:
            d = diff(x)
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    except ValueError:
        d = np.asarray(d)
        y = np.asarray(y)
        ret = add.reduce(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
    return ret
assert not hasattr(trapz, '__code__')

def _fake_trapz(y, x=None, dx=1.0, axis=-1):
    if False:
        i = 10
        return i + 15
    return trapz(y, x=x, dx=dx, axis=axis)
trapz.__code__ = _fake_trapz.__code__
trapz.__globals__ = _fake_trapz.__globals__
trapz.__defaults__ = _fake_trapz.__defaults__
trapz.__closure__ = _fake_trapz.__closure__
trapz.__kwdefaults__ = _fake_trapz.__kwdefaults__

def _meshgrid_dispatcher(*xi, copy=None, sparse=None, indexing=None):
    if False:
        print('Hello World!')
    return xi

@array_function_dispatch(_meshgrid_dispatcher)
def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    if False:
        return 10
    '\n    Return a list of coordinate matrices from coordinate vectors.\n\n    Make N-D coordinate arrays for vectorized evaluations of\n    N-D scalar/vector fields over N-D grids, given\n    one-dimensional coordinate arrays x1, x2,..., xn.\n\n    .. versionchanged:: 1.9\n       1-D and 0-D cases are allowed.\n\n    Parameters\n    ----------\n    x1, x2,..., xn : array_like\n        1-D arrays representing the coordinates of a grid.\n    indexing : {\'xy\', \'ij\'}, optional\n        Cartesian (\'xy\', default) or matrix (\'ij\') indexing of output.\n        See Notes for more details.\n\n        .. versionadded:: 1.7.0\n    sparse : bool, optional\n        If True the shape of the returned coordinate array for dimension *i*\n        is reduced from ``(N1, ..., Ni, ... Nn)`` to\n        ``(1, ..., 1, Ni, 1, ..., 1)``.  These sparse coordinate grids are\n        intended to be use with :ref:`basics.broadcasting`.  When all\n        coordinates are used in an expression, broadcasting still leads to a\n        fully-dimensonal result array.\n\n        Default is False.\n\n        .. versionadded:: 1.7.0\n    copy : bool, optional\n        If False, a view into the original arrays are returned in order to\n        conserve memory.  Default is True.  Please note that\n        ``sparse=False, copy=False`` will likely return non-contiguous\n        arrays.  Furthermore, more than one element of a broadcast array\n        may refer to a single memory location.  If you need to write to the\n        arrays, make copies first.\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    X1, X2,..., XN : list of ndarrays\n        For vectors `x1`, `x2`,..., `xn` with lengths ``Ni=len(xi)``,\n        returns ``(N1, N2, N3,..., Nn)`` shaped arrays if indexing=\'ij\'\n        or ``(N2, N1, N3,..., Nn)`` shaped arrays if indexing=\'xy\'\n        with the elements of `xi` repeated to fill the matrix along\n        the first dimension for `x1`, the second for `x2` and so on.\n\n    Notes\n    -----\n    This function supports both indexing conventions through the indexing\n    keyword argument.  Giving the string \'ij\' returns a meshgrid with\n    matrix indexing, while \'xy\' returns a meshgrid with Cartesian indexing.\n    In the 2-D case with inputs of length M and N, the outputs are of shape\n    (N, M) for \'xy\' indexing and (M, N) for \'ij\' indexing.  In the 3-D case\n    with inputs of length M, N and P, outputs are of shape (N, M, P) for\n    \'xy\' indexing and (M, N, P) for \'ij\' indexing.  The difference is\n    illustrated by the following code snippet::\n\n        xv, yv = np.meshgrid(x, y, indexing=\'ij\')\n        for i in range(nx):\n            for j in range(ny):\n                # treat xv[i,j], yv[i,j]\n\n        xv, yv = np.meshgrid(x, y, indexing=\'xy\')\n        for i in range(nx):\n            for j in range(ny):\n                # treat xv[j,i], yv[j,i]\n\n    In the 1-D and 0-D case, the indexing and sparse keywords have no effect.\n\n    See Also\n    --------\n    mgrid : Construct a multi-dimensional "meshgrid" using indexing notation.\n    ogrid : Construct an open multi-dimensional "meshgrid" using indexing\n            notation.\n    :ref:`how-to-index`\n\n    Examples\n    --------\n    >>> nx, ny = (3, 2)\n    >>> x = np.linspace(0, 1, nx)\n    >>> y = np.linspace(0, 1, ny)\n    >>> xv, yv = np.meshgrid(x, y)\n    >>> xv\n    array([[0. , 0.5, 1. ],\n           [0. , 0.5, 1. ]])\n    >>> yv\n    array([[0.,  0.,  0.],\n           [1.,  1.,  1.]])\n\n    The result of `meshgrid` is a coordinate grid:\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(xv, yv, marker=\'o\', color=\'k\', linestyle=\'none\')\n    >>> plt.show()\n\n    You can create sparse output arrays to save memory and computation time.\n\n    >>> xv, yv = np.meshgrid(x, y, sparse=True)\n    >>> xv\n    array([[0. ,  0.5,  1. ]])\n    >>> yv\n    array([[0.],\n           [1.]])\n\n    `meshgrid` is very useful to evaluate functions on a grid. If the\n    function depends on all coordinates, both dense and sparse outputs can be\n    used.\n\n    >>> x = np.linspace(-5, 5, 101)\n    >>> y = np.linspace(-5, 5, 101)\n    >>> # full coordinate arrays\n    >>> xx, yy = np.meshgrid(x, y)\n    >>> zz = np.sqrt(xx**2 + yy**2)\n    >>> xx.shape, yy.shape, zz.shape\n    ((101, 101), (101, 101), (101, 101))\n    >>> # sparse coordinate arrays\n    >>> xs, ys = np.meshgrid(x, y, sparse=True)\n    >>> zs = np.sqrt(xs**2 + ys**2)\n    >>> xs.shape, ys.shape, zs.shape\n    ((1, 101), (101, 1), (101, 101))\n    >>> np.array_equal(zz, zs)\n    True\n\n    >>> h = plt.contourf(x, y, zs)\n    >>> plt.axis(\'scaled\')\n    >>> plt.colorbar()\n    >>> plt.show()\n    '
    ndim = len(xi)
    if indexing not in ['xy', 'ij']:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")
    s0 = (1,) * ndim
    output = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:]) for (i, x) in enumerate(xi)]
    if indexing == 'xy' and ndim > 1:
        output[0].shape = (1, -1) + s0[2:]
        output[1].shape = (-1, 1) + s0[2:]
    if not sparse:
        output = np.broadcast_arrays(*output, subok=True)
    if copy:
        output = [x.copy() for x in output]
    return output

def _delete_dispatcher(arr, obj, axis=None):
    if False:
        for i in range(10):
            print('nop')
    return (arr, obj)

@array_function_dispatch(_delete_dispatcher)
def delete(arr, obj, axis=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a new array with sub-arrays along an axis deleted. For a one\n    dimensional array, this returns those entries not returned by\n    `arr[obj]`.\n\n    Parameters\n    ----------\n    arr : array_like\n        Input array.\n    obj : slice, int or array of ints\n        Indicate indices of sub-arrays to remove along the specified axis.\n\n        .. versionchanged:: 1.19.0\n            Boolean indices are now treated as a mask of elements to remove,\n            rather than being cast to the integers 0 and 1.\n\n    axis : int, optional\n        The axis along which to delete the subarray defined by `obj`.\n        If `axis` is None, `obj` is applied to the flattened array.\n\n    Returns\n    -------\n    out : ndarray\n        A copy of `arr` with the elements specified by `obj` removed. Note\n        that `delete` does not occur in-place. If `axis` is None, `out` is\n        a flattened array.\n\n    See Also\n    --------\n    insert : Insert elements into an array.\n    append : Append elements at the end of an array.\n\n    Notes\n    -----\n    Often it is preferable to use a boolean mask. For example:\n\n    >>> arr = np.arange(12) + 1\n    >>> mask = np.ones(len(arr), dtype=bool)\n    >>> mask[[0,2,4]] = False\n    >>> result = arr[mask,...]\n\n    Is equivalent to ``np.delete(arr, [0,2,4], axis=0)``, but allows further\n    use of `mask`.\n\n    Examples\n    --------\n    >>> arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])\n    >>> arr\n    array([[ 1,  2,  3,  4],\n           [ 5,  6,  7,  8],\n           [ 9, 10, 11, 12]])\n    >>> np.delete(arr, 1, 0)\n    array([[ 1,  2,  3,  4],\n           [ 9, 10, 11, 12]])\n\n    >>> np.delete(arr, np.s_[::2], 1)\n    array([[ 2,  4],\n           [ 6,  8],\n           [10, 12]])\n    >>> np.delete(arr, [1,3,5], None)\n    array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])\n\n    '
    wrap = None
    if type(arr) is not ndarray:
        try:
            wrap = arr.__array_wrap__
        except AttributeError:
            pass
    arr = asarray(arr)
    ndim = arr.ndim
    arrorder = 'F' if arr.flags.fnc else 'C'
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        ndim = arr.ndim
        axis = ndim - 1
    else:
        axis = normalize_axis_index(axis, ndim)
    slobj = [slice(None)] * ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)
    if isinstance(obj, slice):
        (start, stop, step) = obj.indices(N)
        xr = range(start, stop, step)
        numtodel = len(xr)
        if numtodel <= 0:
            if wrap:
                return wrap(arr.copy(order=arrorder))
            else:
                return arr.copy(order=arrorder)
        if step < 0:
            step = -step
            start = xr[-1]
            stop = xr[0] + 1
        newshape[axis] -= numtodel
        new = empty(newshape, arr.dtype, arrorder)
        if start == 0:
            pass
        else:
            slobj[axis] = slice(None, start)
            new[tuple(slobj)] = arr[tuple(slobj)]
        if stop == N:
            pass
        else:
            slobj[axis] = slice(stop - numtodel, None)
            slobj2 = [slice(None)] * ndim
            slobj2[axis] = slice(stop, None)
            new[tuple(slobj)] = arr[tuple(slobj2)]
        if step == 1:
            pass
        else:
            keep = ones(stop - start, dtype=bool)
            keep[:stop - start:step] = False
            slobj[axis] = slice(start, stop - numtodel)
            slobj2 = [slice(None)] * ndim
            slobj2[axis] = slice(start, stop)
            arr = arr[tuple(slobj2)]
            slobj2[axis] = keep
            new[tuple(slobj)] = arr[tuple(slobj2)]
        if wrap:
            return wrap(new)
        else:
            return new
    if isinstance(obj, (int, integer)) and (not isinstance(obj, bool)):
        single_value = True
    else:
        single_value = False
        _obj = obj
        obj = np.asarray(obj)
        if obj.size == 0 and (not isinstance(_obj, np.ndarray)):
            obj = obj.astype(intp)
        elif obj.size == 1 and obj.dtype.kind in 'ui':
            obj = obj.item()
            single_value = True
    if single_value:
        if obj < -N or obj >= N:
            raise IndexError('index %i is out of bounds for axis %i with size %i' % (obj, axis, N))
        if obj < 0:
            obj += N
        newshape[axis] -= 1
        new = empty(newshape, arr.dtype, arrorder)
        slobj[axis] = slice(None, obj)
        new[tuple(slobj)] = arr[tuple(slobj)]
        slobj[axis] = slice(obj, None)
        slobj2 = [slice(None)] * ndim
        slobj2[axis] = slice(obj + 1, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]
    else:
        if obj.dtype == bool:
            if obj.shape != (N,):
                raise ValueError('boolean array argument obj to delete must be one dimensional and match the axis length of {}'.format(N))
            keep = ~obj
        else:
            keep = ones(N, dtype=bool)
            keep[obj,] = False
        slobj[axis] = keep
        new = arr[tuple(slobj)]
    if wrap:
        return wrap(new)
    else:
        return new

def _insert_dispatcher(arr, obj, values, axis=None):
    if False:
        for i in range(10):
            print('nop')
    return (arr, obj, values)

@array_function_dispatch(_insert_dispatcher)
def insert(arr, obj, values, axis=None):
    if False:
        return 10
    '\n    Insert values along the given axis before the given indices.\n\n    Parameters\n    ----------\n    arr : array_like\n        Input array.\n    obj : int, slice or sequence of ints\n        Object that defines the index or indices before which `values` is\n        inserted.\n\n        .. versionadded:: 1.8.0\n\n        Support for multiple insertions when `obj` is a single scalar or a\n        sequence with one element (similar to calling insert multiple\n        times).\n    values : array_like\n        Values to insert into `arr`. If the type of `values` is different\n        from that of `arr`, `values` is converted to the type of `arr`.\n        `values` should be shaped so that ``arr[...,obj,...] = values``\n        is legal.\n    axis : int, optional\n        Axis along which to insert `values`.  If `axis` is None then `arr`\n        is flattened first.\n\n    Returns\n    -------\n    out : ndarray\n        A copy of `arr` with `values` inserted.  Note that `insert`\n        does not occur in-place: a new array is returned. If\n        `axis` is None, `out` is a flattened array.\n\n    See Also\n    --------\n    append : Append elements at the end of an array.\n    concatenate : Join a sequence of arrays along an existing axis.\n    delete : Delete elements from an array.\n\n    Notes\n    -----\n    Note that for higher dimensional inserts ``obj=0`` behaves very different\n    from ``obj=[0]`` just like ``arr[:,0,:] = values`` is different from\n    ``arr[:,[0],:] = values``.\n\n    Examples\n    --------\n    >>> a = np.array([[1, 1], [2, 2], [3, 3]])\n    >>> a\n    array([[1, 1],\n           [2, 2],\n           [3, 3]])\n    >>> np.insert(a, 1, 5)\n    array([1, 5, 1, ..., 2, 3, 3])\n    >>> np.insert(a, 1, 5, axis=1)\n    array([[1, 5, 1],\n           [2, 5, 2],\n           [3, 5, 3]])\n\n    Difference between sequence and scalars:\n\n    >>> np.insert(a, [1], [[1],[2],[3]], axis=1)\n    array([[1, 1, 1],\n           [2, 2, 2],\n           [3, 3, 3]])\n    >>> np.array_equal(np.insert(a, 1, [1, 2, 3], axis=1),\n    ...                np.insert(a, [1], [[1],[2],[3]], axis=1))\n    True\n\n    >>> b = a.flatten()\n    >>> b\n    array([1, 1, 2, 2, 3, 3])\n    >>> np.insert(b, [2, 2], [5, 6])\n    array([1, 1, 5, ..., 2, 3, 3])\n\n    >>> np.insert(b, slice(2, 4), [5, 6])\n    array([1, 1, 5, ..., 2, 3, 3])\n\n    >>> np.insert(b, [2, 2], [7.13, False]) # type casting\n    array([1, 1, 7, ..., 2, 3, 3])\n\n    >>> x = np.arange(8).reshape(2, 4)\n    >>> idx = (1, 3)\n    >>> np.insert(x, idx, 999, axis=1)\n    array([[  0, 999,   1,   2, 999,   3],\n           [  4, 999,   5,   6, 999,   7]])\n\n    '
    wrap = None
    if type(arr) is not ndarray:
        try:
            wrap = arr.__array_wrap__
        except AttributeError:
            pass
    arr = asarray(arr)
    ndim = arr.ndim
    arrorder = 'F' if arr.flags.fnc else 'C'
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        ndim = arr.ndim
        axis = ndim - 1
    else:
        axis = normalize_axis_index(axis, ndim)
    slobj = [slice(None)] * ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)
    if isinstance(obj, slice):
        indices = arange(*obj.indices(N), dtype=intp)
    else:
        indices = np.array(obj)
        if indices.dtype == bool:
            warnings.warn('in the future insert will treat boolean arrays and array-likes as a boolean index instead of casting it to integer', FutureWarning, stacklevel=2)
            indices = indices.astype(intp)
        elif indices.ndim > 1:
            raise ValueError('index array argument obj to insert must be one dimensional or scalar')
    if indices.size == 1:
        index = indices.item()
        if index < -N or index > N:
            raise IndexError(f'index {obj} is out of bounds for axis {axis} with size {N}')
        if index < 0:
            index += N
        values = array(values, copy=False, ndmin=arr.ndim, dtype=arr.dtype)
        if indices.ndim == 0:
            values = np.moveaxis(values, 0, axis)
        numnew = values.shape[axis]
        newshape[axis] += numnew
        new = empty(newshape, arr.dtype, arrorder)
        slobj[axis] = slice(None, index)
        new[tuple(slobj)] = arr[tuple(slobj)]
        slobj[axis] = slice(index, index + numnew)
        new[tuple(slobj)] = values
        slobj[axis] = slice(index + numnew, None)
        slobj2 = [slice(None)] * ndim
        slobj2[axis] = slice(index, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]
        if wrap:
            return wrap(new)
        return new
    elif indices.size == 0 and (not isinstance(obj, np.ndarray)):
        indices = indices.astype(intp)
    indices[indices < 0] += N
    numnew = len(indices)
    order = indices.argsort(kind='mergesort')
    indices[order] += np.arange(numnew)
    newshape[axis] += numnew
    old_mask = ones(newshape[axis], dtype=bool)
    old_mask[indices] = False
    new = empty(newshape, arr.dtype, arrorder)
    slobj2 = [slice(None)] * ndim
    slobj[axis] = indices
    slobj2[axis] = old_mask
    new[tuple(slobj)] = values
    new[tuple(slobj2)] = arr
    if wrap:
        return wrap(new)
    return new

def _append_dispatcher(arr, values, axis=None):
    if False:
        for i in range(10):
            print('nop')
    return (arr, values)

@array_function_dispatch(_append_dispatcher)
def append(arr, values, axis=None):
    if False:
        while True:
            i = 10
    '\n    Append values to the end of an array.\n\n    Parameters\n    ----------\n    arr : array_like\n        Values are appended to a copy of this array.\n    values : array_like\n        These values are appended to a copy of `arr`.  It must be of the\n        correct shape (the same shape as `arr`, excluding `axis`).  If\n        `axis` is not specified, `values` can be any shape and will be\n        flattened before use.\n    axis : int, optional\n        The axis along which `values` are appended.  If `axis` is not\n        given, both `arr` and `values` are flattened before use.\n\n    Returns\n    -------\n    append : ndarray\n        A copy of `arr` with `values` appended to `axis`.  Note that\n        `append` does not occur in-place: a new array is allocated and\n        filled.  If `axis` is None, `out` is a flattened array.\n\n    See Also\n    --------\n    insert : Insert elements into an array.\n    delete : Delete elements from an array.\n\n    Examples\n    --------\n    >>> np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])\n    array([1, 2, 3, ..., 7, 8, 9])\n\n    When `axis` is specified, `values` must have the correct shape.\n\n    >>> np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)\n    array([[1, 2, 3],\n           [4, 5, 6],\n           [7, 8, 9]])\n    >>> np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)\n    Traceback (most recent call last):\n        ...\n    ValueError: all the input arrays must have same number of dimensions, but\n    the array at index 0 has 2 dimension(s) and the array at index 1 has 1\n    dimension(s)\n\n    '
    arr = asanyarray(arr)
    if axis is None:
        if arr.ndim != 1:
            arr = arr.ravel()
        values = ravel(values)
        axis = arr.ndim - 1
    return concatenate((arr, values), axis=axis)

def _digitize_dispatcher(x, bins, right=None):
    if False:
        while True:
            i = 10
    return (x, bins)

@array_function_dispatch(_digitize_dispatcher)
def digitize(x, bins, right=False):
    if False:
        i = 10
        return i + 15
    '\n    Return the indices of the bins to which each value in input array belongs.\n\n    =========  =============  ============================\n    `right`    order of bins  returned index `i` satisfies\n    =========  =============  ============================\n    ``False``  increasing     ``bins[i-1] <= x < bins[i]``\n    ``True``   increasing     ``bins[i-1] < x <= bins[i]``\n    ``False``  decreasing     ``bins[i-1] > x >= bins[i]``\n    ``True``   decreasing     ``bins[i-1] >= x > bins[i]``\n    =========  =============  ============================\n\n    If values in `x` are beyond the bounds of `bins`, 0 or ``len(bins)`` is\n    returned as appropriate.\n\n    Parameters\n    ----------\n    x : array_like\n        Input array to be binned. Prior to NumPy 1.10.0, this array had to\n        be 1-dimensional, but can now have any shape.\n    bins : array_like\n        Array of bins. It has to be 1-dimensional and monotonic.\n    right : bool, optional\n        Indicating whether the intervals include the right or the left bin\n        edge. Default behavior is (right==False) indicating that the interval\n        does not include the right edge. The left bin end is open in this\n        case, i.e., bins[i-1] <= x < bins[i] is the default behavior for\n        monotonically increasing bins.\n\n    Returns\n    -------\n    indices : ndarray of ints\n        Output array of indices, of same shape as `x`.\n\n    Raises\n    ------\n    ValueError\n        If `bins` is not monotonic.\n    TypeError\n        If the type of the input is complex.\n\n    See Also\n    --------\n    bincount, histogram, unique, searchsorted\n\n    Notes\n    -----\n    If values in `x` are such that they fall outside the bin range,\n    attempting to index `bins` with the indices that `digitize` returns\n    will result in an IndexError.\n\n    .. versionadded:: 1.10.0\n\n    `numpy.digitize` is  implemented in terms of `numpy.searchsorted`.\n    This means that a binary search is used to bin the values, which scales\n    much better for larger number of bins than the previous linear search.\n    It also removes the requirement for the input array to be 1-dimensional.\n\n    For monotonically *increasing* `bins`, the following are equivalent::\n\n        np.digitize(x, bins, right=True)\n        np.searchsorted(bins, x, side=\'left\')\n\n    Note that as the order of the arguments are reversed, the side must be too.\n    The `searchsorted` call is marginally faster, as it does not do any\n    monotonicity checks. Perhaps more importantly, it supports all dtypes.\n\n    Examples\n    --------\n    >>> x = np.array([0.2, 6.4, 3.0, 1.6])\n    >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])\n    >>> inds = np.digitize(x, bins)\n    >>> inds\n    array([1, 4, 3, 2])\n    >>> for n in range(x.size):\n    ...   print(bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]])\n    ...\n    0.0 <= 0.2 < 1.0\n    4.0 <= 6.4 < 10.0\n    2.5 <= 3.0 < 4.0\n    1.0 <= 1.6 < 2.5\n\n    >>> x = np.array([1.2, 10.0, 12.4, 15.5, 20.])\n    >>> bins = np.array([0, 5, 10, 15, 20])\n    >>> np.digitize(x,bins,right=True)\n    array([1, 2, 3, 4, 4])\n    >>> np.digitize(x,bins,right=False)\n    array([1, 3, 3, 4, 5])\n    '
    x = _nx.asarray(x)
    bins = _nx.asarray(bins)
    if np.issubdtype(x.dtype, _nx.complexfloating):
        raise TypeError('x may not be complex')
    mono = _monotonicity(bins)
    if mono == 0:
        raise ValueError('bins must be monotonically increasing or decreasing')
    side = 'left' if right else 'right'
    if mono == -1:
        return len(bins) - _nx.searchsorted(bins[::-1], x, side=side)
    else:
        return _nx.searchsorted(bins, x, side=side)