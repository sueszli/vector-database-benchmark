""" Implementation of reduction operations, to be wrapped into arrays, dtypes etc
in the 'public' layer.

Anything here only deals with torch objects, e.g. "dtype" is a torch.dtype instance etc
"""
from __future__ import annotations
import functools
from typing import Optional
import torch
from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, AxisLike, DTypeLike, KeepDims, NotImplementedType, OutArray

def _deco_axis_expand(func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generically handle axis arguments in reductions.\n    axis is *always* the 2nd arg in the function so no need to have a look at its signature\n    '

    @functools.wraps(func)
    def wrapped(a, axis=None, *args, **kwds):
        if False:
            i = 10
            return i + 15
        if axis is not None:
            axis = _util.normalize_axis_tuple(axis, a.ndim)
        if axis == ():
            newshape = _util.expand_shape(a.shape, axis=0)
            a = a.reshape(newshape)
            axis = (0,)
        return func(a, axis, *args, **kwds)
    return wrapped

def _atleast_float(dtype, other_dtype):
    if False:
        i = 10
        return i + 15
    'Return a dtype that is real or complex floating-point.\n\n    For inputs that are boolean or integer dtypes, this returns the default\n    float dtype; inputs that are complex get converted to the default complex\n    dtype; real floating-point dtypes (`float*`) get passed through unchanged\n    '
    if dtype is None:
        dtype = other_dtype
    if not (dtype.is_floating_point or dtype.is_complex):
        return _dtypes_impl.default_dtypes().float_dtype
    return dtype

@_deco_axis_expand
def count_nonzero(a: ArrayLike, axis: AxisLike=None, *, keepdims: KeepDims=False):
    if False:
        while True:
            i = 10
    return a.count_nonzero(axis)

@_deco_axis_expand
def argmax(a: ArrayLike, axis: AxisLike=None, out: Optional[OutArray]=None, *, keepdims: KeepDims=False):
    if False:
        while True:
            i = 10
    if a.is_complex():
        raise NotImplementedError(f'argmax with dtype={a.dtype}.')
    axis = _util.allow_only_single_axis(axis)
    if a.dtype == torch.bool:
        a = a.to(torch.uint8)
    return torch.argmax(a, axis)

@_deco_axis_expand
def argmin(a: ArrayLike, axis: AxisLike=None, out: Optional[OutArray]=None, *, keepdims: KeepDims=False):
    if False:
        i = 10
        return i + 15
    if a.is_complex():
        raise NotImplementedError(f'argmin with dtype={a.dtype}.')
    axis = _util.allow_only_single_axis(axis)
    if a.dtype == torch.bool:
        a = a.to(torch.uint8)
    return torch.argmin(a, axis)

@_deco_axis_expand
def any(a: ArrayLike, axis: AxisLike=None, out: Optional[OutArray]=None, keepdims: KeepDims=False, *, where: NotImplementedType=None):
    if False:
        while True:
            i = 10
    axis = _util.allow_only_single_axis(axis)
    axis_kw = {} if axis is None else {'dim': axis}
    return torch.any(a, **axis_kw)

@_deco_axis_expand
def all(a: ArrayLike, axis: AxisLike=None, out: Optional[OutArray]=None, keepdims: KeepDims=False, *, where: NotImplementedType=None):
    if False:
        return 10
    axis = _util.allow_only_single_axis(axis)
    axis_kw = {} if axis is None else {'dim': axis}
    return torch.all(a, **axis_kw)

@_deco_axis_expand
def amax(a: ArrayLike, axis: AxisLike=None, out: Optional[OutArray]=None, keepdims: KeepDims=False, initial: NotImplementedType=None, where: NotImplementedType=None):
    if False:
        for i in range(10):
            print('nop')
    if a.is_complex():
        raise NotImplementedError(f'amax with dtype={a.dtype}')
    return a.amax(axis)
max = amax

@_deco_axis_expand
def amin(a: ArrayLike, axis: AxisLike=None, out: Optional[OutArray]=None, keepdims: KeepDims=False, initial: NotImplementedType=None, where: NotImplementedType=None):
    if False:
        print('Hello World!')
    if a.is_complex():
        raise NotImplementedError(f'amin with dtype={a.dtype}')
    return a.amin(axis)
min = amin

@_deco_axis_expand
def ptp(a: ArrayLike, axis: AxisLike=None, out: Optional[OutArray]=None, keepdims: KeepDims=False):
    if False:
        while True:
            i = 10
    return a.amax(axis) - a.amin(axis)

@_deco_axis_expand
def sum(a: ArrayLike, axis: AxisLike=None, dtype: Optional[DTypeLike]=None, out: Optional[OutArray]=None, keepdims: KeepDims=False, initial: NotImplementedType=None, where: NotImplementedType=None):
    if False:
        print('Hello World!')
    assert dtype is None or isinstance(dtype, torch.dtype)
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype
    axis_kw = {} if axis is None else {'dim': axis}
    return a.sum(dtype=dtype, **axis_kw)

@_deco_axis_expand
def prod(a: ArrayLike, axis: AxisLike=None, dtype: Optional[DTypeLike]=None, out: Optional[OutArray]=None, keepdims: KeepDims=False, initial: NotImplementedType=None, where: NotImplementedType=None):
    if False:
        while True:
            i = 10
    axis = _util.allow_only_single_axis(axis)
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype
    axis_kw = {} if axis is None else {'dim': axis}
    return a.prod(dtype=dtype, **axis_kw)
product = prod

@_deco_axis_expand
def mean(a: ArrayLike, axis: AxisLike=None, dtype: Optional[DTypeLike]=None, out: Optional[OutArray]=None, keepdims: KeepDims=False, *, where: NotImplementedType=None):
    if False:
        for i in range(10):
            print('nop')
    dtype = _atleast_float(dtype, a.dtype)
    axis_kw = {} if axis is None else {'dim': axis}
    result = a.mean(dtype=dtype, **axis_kw)
    return result

@_deco_axis_expand
def std(a: ArrayLike, axis: AxisLike=None, dtype: Optional[DTypeLike]=None, out: Optional[OutArray]=None, ddof=0, keepdims: KeepDims=False, *, where: NotImplementedType=None):
    if False:
        while True:
            i = 10
    in_dtype = dtype
    dtype = _atleast_float(dtype, a.dtype)
    tensor = _util.cast_if_needed(a, dtype)
    result = tensor.std(dim=axis, correction=ddof)
    return _util.cast_if_needed(result, in_dtype)

@_deco_axis_expand
def var(a: ArrayLike, axis: AxisLike=None, dtype: Optional[DTypeLike]=None, out: Optional[OutArray]=None, ddof=0, keepdims: KeepDims=False, *, where: NotImplementedType=None):
    if False:
        i = 10
        return i + 15
    in_dtype = dtype
    dtype = _atleast_float(dtype, a.dtype)
    tensor = _util.cast_if_needed(a, dtype)
    result = tensor.var(dim=axis, correction=ddof)
    return _util.cast_if_needed(result, in_dtype)

def cumsum(a: ArrayLike, axis: AxisLike=None, dtype: Optional[DTypeLike]=None, out: Optional[OutArray]=None):
    if False:
        print('Hello World!')
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype
    if dtype is None:
        dtype = a.dtype
    ((a,), axis) = _util.axis_none_flatten(a, axis=axis)
    axis = _util.normalize_axis_index(axis, a.ndim)
    return a.cumsum(axis=axis, dtype=dtype)

def cumprod(a: ArrayLike, axis: AxisLike=None, dtype: Optional[DTypeLike]=None, out: Optional[OutArray]=None):
    if False:
        while True:
            i = 10
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype
    if dtype is None:
        dtype = a.dtype
    ((a,), axis) = _util.axis_none_flatten(a, axis=axis)
    axis = _util.normalize_axis_index(axis, a.ndim)
    return a.cumprod(axis=axis, dtype=dtype)
cumproduct = cumprod

def average(a: ArrayLike, axis=None, weights: ArrayLike=None, returned=False, *, keepdims=False):
    if False:
        return 10
    if weights is None:
        result = mean(a, axis=axis)
        wsum = torch.as_tensor(a.numel() / result.numel(), dtype=result.dtype)
    else:
        if not a.dtype.is_floating_point:
            a = a.double()
        if a.shape != weights.shape:
            if axis is None:
                raise TypeError('Axis must be specified when shapes of a and weights differ.')
            if weights.ndim != 1:
                raise TypeError('1D weights expected when shapes of a and weights differ.')
            if weights.shape[0] != a.shape[axis]:
                raise ValueError('Length of weights not compatible with specified axis.')
            weights = torch.broadcast_to(weights, (a.ndim - 1) * (1,) + weights.shape)
            weights = weights.swapaxes(-1, axis)
        result_dtype = _dtypes_impl.result_type_impl(a, weights)
        numerator = sum(a * weights, axis, dtype=result_dtype)
        wsum = sum(weights, axis, dtype=result_dtype)
        result = numerator / wsum
    if keepdims:
        result = _util.apply_keepdims(result, axis, a.ndim)
    if returned:
        if wsum.shape != result.shape:
            wsum = torch.broadcast_to(wsum, result.shape).clone()
        return (result, wsum)
    else:
        return result

def quantile(a: ArrayLike, q: ArrayLike, axis: AxisLike=None, out: Optional[OutArray]=None, overwrite_input=False, method='linear', keepdims: KeepDims=False, *, interpolation: NotImplementedType=None):
    if False:
        print('Hello World!')
    if overwrite_input:
        pass
    if not a.dtype.is_floating_point:
        dtype = _dtypes_impl.default_dtypes().float_dtype
        a = a.to(dtype)
    if a.dtype == torch.float16:
        a = a.to(torch.float32)
    if axis is None:
        a = a.flatten()
        q = q.flatten()
        axis = (0,)
    else:
        axis = _util.normalize_axis_tuple(axis, a.ndim)
    axis = _util.allow_only_single_axis(axis)
    q = _util.cast_if_needed(q, a.dtype)
    return torch.quantile(a, q, axis=axis, interpolation=method)

def percentile(a: ArrayLike, q: ArrayLike, axis: AxisLike=None, out: Optional[OutArray]=None, overwrite_input=False, method='linear', keepdims: KeepDims=False, *, interpolation: NotImplementedType=None):
    if False:
        return 10
    return quantile(a, q / 100.0, axis=axis, overwrite_input=overwrite_input, method=method, keepdims=keepdims, interpolation=interpolation)

def median(a: ArrayLike, axis=None, out: Optional[OutArray]=None, overwrite_input=False, keepdims: KeepDims=False):
    if False:
        print('Hello World!')
    return quantile(a, torch.as_tensor(0.5), axis=axis, overwrite_input=overwrite_input, out=out, keepdims=keepdims)