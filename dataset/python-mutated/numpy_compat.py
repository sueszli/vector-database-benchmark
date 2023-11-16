from __future__ import annotations
import warnings
import numpy as np
from packaging.version import parse as parse_version
from dask.utils import derived_from
_np_version = parse_version(np.__version__)
_numpy_122 = _np_version >= parse_version('1.22.0')
_numpy_123 = _np_version >= parse_version('1.23.0')
_numpy_124 = _np_version >= parse_version('1.24.0')
_numpy_125 = _np_version.release >= (1, 25, 0)
_numpy_200 = _np_version.release >= (2, 0, 0)
if _numpy_200:
    from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple
else:
    from numpy.core.numeric import normalize_axis_index
    from numpy.core.numeric import normalize_axis_tuple
try:
    with warnings.catch_warnings():
        if not np.allclose(np.divide(0.4, 1, casting='unsafe'), np.divide(0.4, 1, casting='unsafe', dtype=float)) or not np.allclose(np.divide(1, 0.5, dtype='i8'), 2) or (not np.allclose(np.divide(0.4, 1), 0.4)):
            raise TypeError('Divide not working with dtype: https://github.com/numpy/numpy/issues/3484')
        divide = np.divide
        ma_divide = np.ma.divide
except TypeError:

    def divide(x1, x2, out=None, dtype=None):
        if False:
            for i in range(10):
                print('nop')
        "Implementation of numpy.divide that works with dtype kwarg.\n\n        Temporary compatibility fix for a bug in numpy's version. See\n        https://github.com/numpy/numpy/issues/3484 for the relevant issue."
        x = np.divide(x1, x2, out)
        if dtype is not None:
            x = x.astype(dtype)
        return x
    ma_divide = np.ma.core._DomainedBinaryOperation(divide, np.ma.core._DomainSafeDivide(), 0, 1)

class _Recurser:
    """
    Utility class for recursing over nested iterables
    """

    def __init__(self, recurse_if):
        if False:
            i = 10
            return i + 15
        self.recurse_if = recurse_if

    def map_reduce(self, x, f_map=lambda x, **kwargs: x, f_reduce=lambda x, **kwargs: x, f_kwargs=lambda **kwargs: kwargs, **kwargs):
        if False:
            return 10
        '\n        Iterate over the nested list, applying:\n        * ``f_map`` (T -> U) to items\n        * ``f_reduce`` (Iterable[U] -> U) to mapped items\n\n        For instance, ``map_reduce([[1, 2], 3, 4])`` is::\n\n            f_reduce([\n              f_reduce([\n                f_map(1),\n                f_map(2)\n              ]),\n              f_map(3),\n              f_map(4)\n            ]])\n\n\n        State can be passed down through the calls with `f_kwargs`,\n        to iterables of mapped items. When kwargs are passed, as in\n        ``map_reduce([[1, 2], 3, 4], **kw)``, this becomes::\n\n            kw1 = f_kwargs(**kw)\n            kw2 = f_kwargs(**kw1)\n            f_reduce([\n              f_reduce([\n                f_map(1), **kw2)\n                f_map(2,  **kw2)\n              ],      **kw1),\n              f_map(3, **kw1),\n              f_map(4, **kw1)\n            ]],     **kw)\n        '

        def f(x, **kwargs):
            if False:
                return 10
            if not self.recurse_if(x):
                return f_map(x, **kwargs)
            else:
                next_kwargs = f_kwargs(**kwargs)
                return f_reduce((f(xi, **next_kwargs) for xi in x), **kwargs)
        return f(x, **kwargs)

    def walk(self, x, index=()):
        if False:
            return 10
        '\n        Iterate over x, yielding (index, value, entering), where\n\n        * ``index``: a tuple of indices up to this point\n        * ``value``: equal to ``x[index[0]][...][index[-1]]``. On the first iteration, is\n                     ``x`` itself\n        * ``entering``: bool. The result of ``recurse_if(value)``\n        '
        do_recurse = self.recurse_if(x)
        yield (index, x, do_recurse)
        if not do_recurse:
            return
        for (i, xi) in enumerate(x):
            yield from self.walk(xi, index + (i,))

@derived_from(np)
def moveaxis(a, source, destination):
    if False:
        for i in range(10):
            print('nop')
    source = normalize_axis_tuple(source, a.ndim, 'source')
    destination = normalize_axis_tuple(destination, a.ndim, 'destination')
    if len(source) != len(destination):
        raise ValueError('`source` and `destination` arguments must have the same number of elements')
    order = [n for n in range(a.ndim) if n not in source]
    for (dest, src) in sorted(zip(destination, source)):
        order.insert(dest, src)
    result = a.transpose(order)
    return result

def rollaxis(a, axis, start=0):
    if False:
        i = 10
        return i + 15
    n = a.ndim
    axis = normalize_axis_index(axis, n)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not 0 <= start < n + 1:
        raise ValueError(msg % ('start', -n, 'start', n + 1, start))
    if axis < start:
        start -= 1
    if axis == start:
        return a[...]
    axes = list(range(0, n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.transpose(axes)

def percentile(a, q, method='linear'):
    if False:
        while True:
            i = 10
    if _numpy_122:
        return np.percentile(a, q, method=method)
    else:
        return np.percentile(a, q, interpolation=method)
ComplexWarning = np.exceptions.ComplexWarning if _numpy_200 else np.ComplexWarning
AxisError = np.exceptions.AxisError if _numpy_200 else np.AxisError