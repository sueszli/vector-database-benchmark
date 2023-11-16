"""A thin pytorch / numpy compat layer.

Things imported from here have numpy-compatible signatures but operate on
pytorch tensors.
"""
from __future__ import annotations
import builtins
import itertools
import operator
from typing import Optional, Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, ArrayLikeOrScalar, CastingModes, DTypeLike, NDArray, NotImplementedType, OutArray

def copy(a: ArrayLike, order: NotImplementedType='K', subok: NotImplementedType=False):
    if False:
        return 10
    return a.clone()

def copyto(dst: NDArray, src: ArrayLike, casting: Optional[CastingModes]='same_kind', where: NotImplementedType=None):
    if False:
        return 10
    (src,) = _util.typecast_tensors((src,), dst.dtype, casting=casting)
    dst.copy_(src)

def atleast_1d(*arys: ArrayLike):
    if False:
        i = 10
        return i + 15
    res = torch.atleast_1d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res

def atleast_2d(*arys: ArrayLike):
    if False:
        for i in range(10):
            print('nop')
    res = torch.atleast_2d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res

def atleast_3d(*arys: ArrayLike):
    if False:
        while True:
            i = 10
    res = torch.atleast_3d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res

def _concat_check(tup, dtype, out):
    if False:
        for i in range(10):
            print('nop')
    if tup == ():
        raise ValueError('need at least one array to concatenate')
    'Check inputs in concatenate et al.'
    if out is not None and dtype is not None:
        raise TypeError('concatenate() only takes `out` or `dtype` as an argument, but both were provided.')

def _concat_cast_helper(tensors, out=None, dtype=None, casting='same_kind'):
    if False:
        print('Hello World!')
    'Figure out dtypes, cast if necessary.'
    if out is not None or dtype is not None:
        out_dtype = out.dtype.torch_dtype if dtype is None else dtype
    else:
        out_dtype = _dtypes_impl.result_type_impl(*tensors)
    tensors = _util.typecast_tensors(tensors, out_dtype, casting)
    return tensors

def _concatenate(tensors, axis=0, out=None, dtype=None, casting: Optional[CastingModes]='same_kind'):
    if False:
        while True:
            i = 10
    (tensors, axis) = _util.axis_none_flatten(*tensors, axis=axis)
    tensors = _concat_cast_helper(tensors, out, dtype, casting)
    return torch.cat(tensors, axis)

def concatenate(ar_tuple: Sequence[ArrayLike], axis=0, out: Optional[OutArray]=None, dtype: Optional[DTypeLike]=None, casting: Optional[CastingModes]='same_kind'):
    if False:
        while True:
            i = 10
    _concat_check(ar_tuple, dtype, out=out)
    result = _concatenate(ar_tuple, axis=axis, out=out, dtype=dtype, casting=casting)
    return result

def vstack(tup: Sequence[ArrayLike], *, dtype: Optional[DTypeLike]=None, casting: Optional[CastingModes]='same_kind'):
    if False:
        print('Hello World!')
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.vstack(tensors)
row_stack = vstack

def hstack(tup: Sequence[ArrayLike], *, dtype: Optional[DTypeLike]=None, casting: Optional[CastingModes]='same_kind'):
    if False:
        i = 10
        return i + 15
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.hstack(tensors)

def dstack(tup: Sequence[ArrayLike], *, dtype: Optional[DTypeLike]=None, casting: Optional[CastingModes]='same_kind'):
    if False:
        while True:
            i = 10
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.dstack(tensors)

def column_stack(tup: Sequence[ArrayLike], *, dtype: Optional[DTypeLike]=None, casting: Optional[CastingModes]='same_kind'):
    if False:
        for i in range(10):
            print('nop')
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.column_stack(tensors)

def stack(arrays: Sequence[ArrayLike], axis=0, out: Optional[OutArray]=None, *, dtype: Optional[DTypeLike]=None, casting: Optional[CastingModes]='same_kind'):
    if False:
        print('Hello World!')
    _concat_check(arrays, dtype, out=out)
    tensors = _concat_cast_helper(arrays, dtype=dtype, casting=casting)
    result_ndim = tensors[0].ndim + 1
    axis = _util.normalize_axis_index(axis, result_ndim)
    return torch.stack(tensors, axis=axis)

def append(arr: ArrayLike, values: ArrayLike, axis=None):
    if False:
        while True:
            i = 10
    if axis is None:
        if arr.ndim != 1:
            arr = arr.flatten()
        values = values.flatten()
        axis = arr.ndim - 1
    return _concatenate((arr, values), axis=axis)

def _split_helper(tensor, indices_or_sections, axis, strict=False):
    if False:
        return 10
    if isinstance(indices_or_sections, int):
        return _split_helper_int(tensor, indices_or_sections, axis, strict)
    elif isinstance(indices_or_sections, (list, tuple)):
        return _split_helper_list(tensor, list(indices_or_sections), axis)
    else:
        raise TypeError('split_helper: ', type(indices_or_sections))

def _split_helper_int(tensor, indices_or_sections, axis, strict=False):
    if False:
        print('Hello World!')
    if not isinstance(indices_or_sections, int):
        raise NotImplementedError('split: indices_or_sections')
    axis = _util.normalize_axis_index(axis, tensor.ndim)
    (l, n) = (tensor.shape[axis], indices_or_sections)
    if n <= 0:
        raise ValueError()
    if l % n == 0:
        (num, sz) = (n, l // n)
        lst = [sz] * num
    else:
        if strict:
            raise ValueError('array split does not result in an equal division')
        (num, sz) = (l % n, l // n + 1)
        lst = [sz] * num
    lst += [sz - 1] * (n - num)
    return torch.split(tensor, lst, axis)

def _split_helper_list(tensor, indices_or_sections, axis):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(indices_or_sections, list):
        raise NotImplementedError('split: indices_or_sections: list')
    lst = [x for x in indices_or_sections if x <= tensor.shape[axis]]
    num_extra = len(indices_or_sections) - len(lst)
    lst.append(tensor.shape[axis])
    lst = [lst[0]] + [a - b for (a, b) in zip(lst[1:], lst[:-1])]
    lst += [0] * num_extra
    return torch.split(tensor, lst, axis)

def array_split(ary: ArrayLike, indices_or_sections, axis=0):
    if False:
        return 10
    return _split_helper(ary, indices_or_sections, axis)

def split(ary: ArrayLike, indices_or_sections, axis=0):
    if False:
        return 10
    return _split_helper(ary, indices_or_sections, axis, strict=True)

def hsplit(ary: ArrayLike, indices_or_sections):
    if False:
        print('Hello World!')
    if ary.ndim == 0:
        raise ValueError('hsplit only works on arrays of 1 or more dimensions')
    axis = 1 if ary.ndim > 1 else 0
    return _split_helper(ary, indices_or_sections, axis, strict=True)

def vsplit(ary: ArrayLike, indices_or_sections):
    if False:
        while True:
            i = 10
    if ary.ndim < 2:
        raise ValueError('vsplit only works on arrays of 2 or more dimensions')
    return _split_helper(ary, indices_or_sections, 0, strict=True)

def dsplit(ary: ArrayLike, indices_or_sections):
    if False:
        for i in range(10):
            print('nop')
    if ary.ndim < 3:
        raise ValueError('dsplit only works on arrays of 3 or more dimensions')
    return _split_helper(ary, indices_or_sections, 2, strict=True)

def kron(a: ArrayLike, b: ArrayLike):
    if False:
        while True:
            i = 10
    return torch.kron(a, b)

def vander(x: ArrayLike, N=None, increasing=False):
    if False:
        return 10
    return torch.vander(x, N, increasing)

def linspace(start: ArrayLike, stop: ArrayLike, num=50, endpoint=True, retstep=False, dtype: Optional[DTypeLike]=None, axis=0):
    if False:
        while True:
            i = 10
    if axis != 0 or retstep or (not endpoint):
        raise NotImplementedError
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.linspace(start, stop, num, dtype=dtype)

def geomspace(start: ArrayLike, stop: ArrayLike, num=50, endpoint=True, dtype: Optional[DTypeLike]=None, axis=0):
    if False:
        return 10
    if axis != 0 or not endpoint:
        raise NotImplementedError
    base = torch.pow(stop / start, 1.0 / (num - 1))
    logbase = torch.log(base)
    return torch.logspace(torch.log(start) / logbase, torch.log(stop) / logbase, num, base=base)

def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype: Optional[DTypeLike]=None, axis=0):
    if False:
        print('Hello World!')
    if axis != 0 or not endpoint:
        raise NotImplementedError
    return torch.logspace(start, stop, num, base=base, dtype=dtype)

def arange(start: Optional[ArrayLikeOrScalar]=None, stop: Optional[ArrayLikeOrScalar]=None, step: Optional[ArrayLikeOrScalar]=1, dtype: Optional[DTypeLike]=None, *, like: NotImplementedType=None):
    if False:
        while True:
            i = 10
    if step == 0:
        raise ZeroDivisionError
    if stop is None and start is None:
        raise TypeError
    if stop is None:
        (start, stop) = (0, start)
    if start is None:
        start = 0
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype if any((_dtypes_impl.is_float_or_fp_tensor(x) for x in (start, stop, step))) else _dtypes_impl.default_dtypes().int_dtype
    work_dtype = torch.float64 if dtype.is_complex else dtype
    if any((_dtypes_impl.is_complex_or_complex_tensor(x) for x in (start, stop, step))):
        raise NotImplementedError
    if step > 0 and start > stop or (step < 0 and start < stop):
        return torch.empty(0, dtype=dtype)
    result = torch.arange(start, stop, step, dtype=work_dtype)
    result = _util.cast_if_needed(result, dtype)
    return result

def empty(shape, dtype: Optional[DTypeLike]=None, order: NotImplementedType='C', *, like: NotImplementedType=None):
    if False:
        return 10
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.empty(shape, dtype=dtype)

def empty_like(prototype: ArrayLike, dtype: Optional[DTypeLike]=None, order: NotImplementedType='K', subok: NotImplementedType=False, shape=None):
    if False:
        for i in range(10):
            print('nop')
    result = torch.empty_like(prototype, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result

def full(shape, fill_value: ArrayLike, dtype: Optional[DTypeLike]=None, order: NotImplementedType='C', *, like: NotImplementedType=None):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(shape, int):
        shape = (shape,)
    if dtype is None:
        dtype = fill_value.dtype
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    return torch.full(shape, fill_value, dtype=dtype)

def full_like(a: ArrayLike, fill_value, dtype: Optional[DTypeLike]=None, order: NotImplementedType='K', subok: NotImplementedType=False, shape=None):
    if False:
        for i in range(10):
            print('nop')
    result = torch.full_like(a, fill_value, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result

def ones(shape, dtype: Optional[DTypeLike]=None, order: NotImplementedType='C', *, like: NotImplementedType=None):
    if False:
        print('Hello World!')
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.ones(shape, dtype=dtype)

def ones_like(a: ArrayLike, dtype: Optional[DTypeLike]=None, order: NotImplementedType='K', subok: NotImplementedType=False, shape=None):
    if False:
        for i in range(10):
            print('nop')
    result = torch.ones_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result

def zeros(shape, dtype: Optional[DTypeLike]=None, order: NotImplementedType='C', *, like: NotImplementedType=None):
    if False:
        print('Hello World!')
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.zeros(shape, dtype=dtype)

def zeros_like(a: ArrayLike, dtype: Optional[DTypeLike]=None, order: NotImplementedType='K', subok: NotImplementedType=False, shape=None):
    if False:
        for i in range(10):
            print('nop')
    result = torch.zeros_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result

def _xy_helper_corrcoef(x_tensor, y_tensor=None, rowvar=True):
    if False:
        return 10
    'Prepare inputs for cov and corrcoef.'
    if y_tensor is not None:
        ndim_extra = 2 - x_tensor.ndim
        if ndim_extra > 0:
            x_tensor = x_tensor.view((1,) * ndim_extra + x_tensor.shape)
        if not rowvar and x_tensor.shape[0] != 1:
            x_tensor = x_tensor.mT
        x_tensor = x_tensor.clone()
        ndim_extra = 2 - y_tensor.ndim
        if ndim_extra > 0:
            y_tensor = y_tensor.view((1,) * ndim_extra + y_tensor.shape)
        if not rowvar and y_tensor.shape[0] != 1:
            y_tensor = y_tensor.mT
        y_tensor = y_tensor.clone()
        x_tensor = _concatenate((x_tensor, y_tensor), axis=0)
    return x_tensor

def corrcoef(x: ArrayLike, y: Optional[ArrayLike]=None, rowvar=True, bias=None, ddof=None, *, dtype: Optional[DTypeLike]=None):
    if False:
        while True:
            i = 10
    if bias is not None or ddof is not None:
        raise NotImplementedError
    xy_tensor = _xy_helper_corrcoef(x, y, rowvar)
    is_half = xy_tensor.dtype == torch.float16 and xy_tensor.is_cpu
    if is_half:
        dtype = torch.float32
    xy_tensor = _util.cast_if_needed(xy_tensor, dtype)
    result = torch.corrcoef(xy_tensor)
    if is_half:
        result = result.to(torch.float16)
    return result

def cov(m: ArrayLike, y: Optional[ArrayLike]=None, rowvar=True, bias=False, ddof=None, fweights: Optional[ArrayLike]=None, aweights: Optional[ArrayLike]=None, *, dtype: Optional[DTypeLike]=None):
    if False:
        print('Hello World!')
    m = _xy_helper_corrcoef(m, y, rowvar)
    if ddof is None:
        ddof = 1 if bias == 0 else 0
    is_half = m.dtype == torch.float16 and m.is_cpu
    if is_half:
        dtype = torch.float32
    m = _util.cast_if_needed(m, dtype)
    result = torch.cov(m, correction=ddof, aweights=aweights, fweights=fweights)
    if is_half:
        result = result.to(torch.float16)
    return result

def _conv_corr_impl(a, v, mode):
    if False:
        while True:
            i = 10
    dt = _dtypes_impl.result_type_impl(a, v)
    a = _util.cast_if_needed(a, dt)
    v = _util.cast_if_needed(v, dt)
    padding = v.shape[0] - 1 if mode == 'full' else mode
    if padding == 'same' and v.shape[0] % 2 == 0:
        raise NotImplementedError("mode='same' and even-length weights")
    aa = a[None, :]
    vv = v[None, None, :]
    result = torch.nn.functional.conv1d(aa, vv, padding=padding)
    return result[0, :]

def convolve(a: ArrayLike, v: ArrayLike, mode='full'):
    if False:
        return 10
    if a.shape[0] < v.shape[0]:
        (a, v) = (v, a)
    v = torch.flip(v, (0,))
    return _conv_corr_impl(a, v, mode)

def correlate(a: ArrayLike, v: ArrayLike, mode='valid'):
    if False:
        print('Hello World!')
    v = torch.conj_physical(v)
    return _conv_corr_impl(a, v, mode)

def bincount(x: ArrayLike, /, weights: Optional[ArrayLike]=None, minlength=0):
    if False:
        for i in range(10):
            print('nop')
    if x.numel() == 0:
        x = x.new_empty(0, dtype=int)
    int_dtype = _dtypes_impl.default_dtypes().int_dtype
    (x,) = _util.typecast_tensors((x,), int_dtype, casting='safe')
    return torch.bincount(x, weights, minlength)

def where(condition: ArrayLike, x: Optional[ArrayLikeOrScalar]=None, y: Optional[ArrayLikeOrScalar]=None, /):
    if False:
        return 10
    if (x is None) != (y is None):
        raise ValueError('either both or neither of x and y should be given')
    if condition.dtype != torch.bool:
        condition = condition.to(torch.bool)
    if x is None and y is None:
        result = torch.where(condition)
    else:
        result = torch.where(condition, x, y)
    return result

def ndim(a: ArrayLike):
    if False:
        for i in range(10):
            print('nop')
    return a.ndim

def shape(a: ArrayLike):
    if False:
        i = 10
        return i + 15
    return tuple(a.shape)

def size(a: ArrayLike, axis=None):
    if False:
        while True:
            i = 10
    if axis is None:
        return a.numel()
    else:
        return a.shape[axis]

def expand_dims(a: ArrayLike, axis):
    if False:
        while True:
            i = 10
    shape = _util.expand_shape(a.shape, axis)
    return a.view(shape)

def flip(m: ArrayLike, axis=None):
    if False:
        for i in range(10):
            print('nop')
    if axis is None:
        axis = tuple(range(m.ndim))
    else:
        axis = _util.normalize_axis_tuple(axis, m.ndim)
    return torch.flip(m, axis)

def flipud(m: ArrayLike):
    if False:
        while True:
            i = 10
    return torch.flipud(m)

def fliplr(m: ArrayLike):
    if False:
        for i in range(10):
            print('nop')
    return torch.fliplr(m)

def rot90(m: ArrayLike, k=1, axes=(0, 1)):
    if False:
        i = 10
        return i + 15
    axes = _util.normalize_axis_tuple(axes, m.ndim)
    return torch.rot90(m, k, axes)

def broadcast_to(array: ArrayLike, shape, subok: NotImplementedType=False):
    if False:
        while True:
            i = 10
    return torch.broadcast_to(array, size=shape)
from torch import broadcast_shapes

def broadcast_arrays(*args: ArrayLike, subok: NotImplementedType=False):
    if False:
        return 10
    return torch.broadcast_tensors(*args)

def meshgrid(*xi: ArrayLike, copy=True, sparse=False, indexing='xy'):
    if False:
        print('Hello World!')
    ndim = len(xi)
    if indexing not in ['xy', 'ij']:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")
    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1:]) for (i, x) in enumerate(xi)]
    if indexing == 'xy' and ndim > 1:
        output[0] = output[0].reshape((1, -1) + s0[2:])
        output[1] = output[1].reshape((-1, 1) + s0[2:])
    if not sparse:
        output = torch.broadcast_tensors(*output)
    if copy:
        output = [x.clone() for x in output]
    return list(output)

def indices(dimensions, dtype: Optional[DTypeLike]=int, sparse=False):
    if False:
        while True:
            i = 10
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    if sparse:
        res = tuple()
    else:
        res = torch.empty((N,) + dimensions, dtype=dtype)
    for (i, dim) in enumerate(dimensions):
        idx = torch.arange(dim, dtype=dtype).reshape(shape[:i] + (dim,) + shape[i + 1:])
        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx
    return res

def tril(m: ArrayLike, k=0):
    if False:
        for i in range(10):
            print('nop')
    return torch.tril(m, k)

def triu(m: ArrayLike, k=0):
    if False:
        return 10
    return torch.triu(m, k)

def tril_indices(n, k=0, m=None):
    if False:
        print('Hello World!')
    if m is None:
        m = n
    return torch.tril_indices(n, m, offset=k)

def triu_indices(n, k=0, m=None):
    if False:
        for i in range(10):
            print('nop')
    if m is None:
        m = n
    return torch.triu_indices(n, m, offset=k)

def tril_indices_from(arr: ArrayLike, k=0):
    if False:
        print('Hello World!')
    if arr.ndim != 2:
        raise ValueError('input array must be 2-d')
    return torch.tril_indices(arr.shape[0], arr.shape[1], offset=k)

def triu_indices_from(arr: ArrayLike, k=0):
    if False:
        print('Hello World!')
    if arr.ndim != 2:
        raise ValueError('input array must be 2-d')
    return torch.triu_indices(arr.shape[0], arr.shape[1], offset=k)

def tri(N, M=None, k=0, dtype: Optional[DTypeLike]=None, *, like: NotImplementedType=None):
    if False:
        i = 10
        return i + 15
    if M is None:
        M = N
    tensor = torch.ones((N, M), dtype=dtype)
    return torch.tril(tensor, diagonal=k)

def isclose(a: ArrayLike, b: ArrayLike, rtol=1e-05, atol=1e-08, equal_nan=False):
    if False:
        i = 10
        return i + 15
    dtype = _dtypes_impl.result_type_impl(a, b)
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    return torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def allclose(a: ArrayLike, b: ArrayLike, rtol=1e-05, atol=1e-08, equal_nan=False):
    if False:
        print('Hello World!')
    dtype = _dtypes_impl.result_type_impl(a, b)
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def _tensor_equal(a1, a2, equal_nan=False):
    if False:
        for i in range(10):
            print('nop')
    if a1.shape != a2.shape:
        return False
    cond = a1 == a2
    if equal_nan:
        cond = cond | torch.isnan(a1) & torch.isnan(a2)
    return cond.all().item()

def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan=False):
    if False:
        print('Hello World!')
    return _tensor_equal(a1, a2, equal_nan=equal_nan)

def array_equiv(a1: ArrayLike, a2: ArrayLike):
    if False:
        for i in range(10):
            print('nop')
    try:
        (a1_t, a2_t) = torch.broadcast_tensors(a1, a2)
    except RuntimeError:
        return False
    return _tensor_equal(a1_t, a2_t)

def nan_to_num(x: ArrayLike, copy: NotImplementedType=True, nan=0.0, posinf=None, neginf=None):
    if False:
        while True:
            i = 10
    if x.is_complex():
        re = torch.nan_to_num(x.real, nan=nan, posinf=posinf, neginf=neginf)
        im = torch.nan_to_num(x.imag, nan=nan, posinf=posinf, neginf=neginf)
        return re + 1j * im
    else:
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

def take(a: ArrayLike, indices: ArrayLike, axis=None, out: Optional[OutArray]=None, mode: NotImplementedType='raise'):
    if False:
        i = 10
        return i + 15
    ((a,), axis) = _util.axis_none_flatten(a, axis=axis)
    axis = _util.normalize_axis_index(axis, a.ndim)
    idx = (slice(None),) * axis + (indices, ...)
    result = a[idx]
    return result

def take_along_axis(arr: ArrayLike, indices: ArrayLike, axis):
    if False:
        return 10
    ((arr,), axis) = _util.axis_none_flatten(arr, axis=axis)
    axis = _util.normalize_axis_index(axis, arr.ndim)
    return torch.take_along_dim(arr, indices, axis)

def put(a: NDArray, ind: ArrayLike, v: ArrayLike, mode: NotImplementedType='raise'):
    if False:
        print('Hello World!')
    v = v.type(a.dtype)
    if ind.numel() > v.numel():
        ratio = (ind.numel() + v.numel() - 1) // v.numel()
        v = v.unsqueeze(0).expand((ratio,) + v.shape)
    if ind.numel() < v.numel():
        v = v.flatten()
        v = v[:ind.numel()]
    a.put_(ind, v)
    return None

def put_along_axis(arr: ArrayLike, indices: ArrayLike, values: ArrayLike, axis):
    if False:
        for i in range(10):
            print('nop')
    ((arr,), axis) = _util.axis_none_flatten(arr, axis=axis)
    axis = _util.normalize_axis_index(axis, arr.ndim)
    (indices, values) = torch.broadcast_tensors(indices, values)
    values = _util.cast_if_needed(values, arr.dtype)
    result = torch.scatter(arr, axis, indices, values)
    arr.copy_(result.reshape(arr.shape))
    return None

def choose(a: ArrayLike, choices: Sequence[ArrayLike], out: Optional[OutArray]=None, mode: NotImplementedType='raise'):
    if False:
        i = 10
        return i + 15
    choices = torch.stack(torch.broadcast_tensors(*choices))
    idx_list = [torch.arange(dim).view((1,) * i + (dim,) + (1,) * (choices.ndim - i - 1)) for (i, dim) in enumerate(choices.shape)]
    idx_list[0] = a
    return choices[idx_list].squeeze(0)

def unique(ar: ArrayLike, return_index: NotImplementedType=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan: NotImplementedType=True):
    if False:
        while True:
            i = 10
    ((ar,), axis) = _util.axis_none_flatten(ar, axis=axis)
    axis = _util.normalize_axis_index(axis, ar.ndim)
    result = torch.unique(ar, return_inverse=return_inverse, return_counts=return_counts, dim=axis)
    return result

def nonzero(a: ArrayLike):
    if False:
        print('Hello World!')
    return torch.nonzero(a, as_tuple=True)

def argwhere(a: ArrayLike):
    if False:
        while True:
            i = 10
    return torch.argwhere(a)

def flatnonzero(a: ArrayLike):
    if False:
        return 10
    return torch.flatten(a).nonzero(as_tuple=True)[0]

def clip(a: ArrayLike, min: Optional[ArrayLike]=None, max: Optional[ArrayLike]=None, out: Optional[OutArray]=None):
    if False:
        while True:
            i = 10
    return torch.clamp(a, min, max)

def repeat(a: ArrayLike, repeats: ArrayLikeOrScalar, axis=None):
    if False:
        return 10
    return torch.repeat_interleave(a, repeats, axis)

def tile(A: ArrayLike, reps):
    if False:
        print('Hello World!')
    if isinstance(reps, int):
        reps = (reps,)
    return torch.tile(A, reps)

def resize(a: ArrayLike, new_shape=None):
    if False:
        i = 10
        return i + 15
    if new_shape is None:
        return a
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    a = a.flatten()
    new_size = 1
    for dim_length in new_shape:
        new_size *= dim_length
        if dim_length < 0:
            raise ValueError('all elements of `new_shape` must be non-negative')
    if a.numel() == 0 or new_size == 0:
        return torch.zeros(new_shape, dtype=a.dtype)
    repeats = -(-new_size // a.numel())
    a = concatenate((a,) * repeats)[:new_size]
    return reshape(a, new_shape)

def diagonal(a: ArrayLike, offset=0, axis1=0, axis2=1):
    if False:
        while True:
            i = 10
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    return torch.diagonal(a, offset, axis1, axis2)

def trace(a: ArrayLike, offset=0, axis1=0, axis2=1, dtype: Optional[DTypeLike]=None, out: Optional[OutArray]=None):
    if False:
        i = 10
        return i + 15
    result = torch.diagonal(a, offset, dim1=axis1, dim2=axis2).sum(-1, dtype=dtype)
    return result

def eye(N, M=None, k=0, dtype: Optional[DTypeLike]=None, order: NotImplementedType='C', *, like: NotImplementedType=None):
    if False:
        return 10
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    if M is None:
        M = N
    z = torch.zeros(N, M, dtype=dtype)
    z.diagonal(k).fill_(1)
    return z

def identity(n, dtype: Optional[DTypeLike]=None, *, like: NotImplementedType=None):
    if False:
        while True:
            i = 10
    return torch.eye(n, dtype=dtype)

def diag(v: ArrayLike, k=0):
    if False:
        i = 10
        return i + 15
    return torch.diag(v, k)

def diagflat(v: ArrayLike, k=0):
    if False:
        for i in range(10):
            print('nop')
    return torch.diagflat(v, k)

def diag_indices(n, ndim=2):
    if False:
        print('Hello World!')
    idx = torch.arange(n)
    return (idx,) * ndim

def diag_indices_from(arr: ArrayLike):
    if False:
        i = 10
        return i + 15
    if not arr.ndim >= 2:
        raise ValueError('input array must be at least 2-d')
    s = arr.shape
    if s[1:] != s[:-1]:
        raise ValueError('All dimensions of input must be of equal length')
    return diag_indices(s[0], arr.ndim)

def fill_diagonal(a: ArrayLike, val: ArrayLike, wrap=False):
    if False:
        while True:
            i = 10
    if a.ndim < 2:
        raise ValueError('array must be at least 2-d')
    if val.numel() == 0 and (not wrap):
        a.fill_diagonal_(val)
        return a
    if val.ndim == 0:
        val = val.unsqueeze(0)
    if a.ndim == 2:
        tall = a.shape[0] > a.shape[1]
        if not wrap or not tall:
            diag = a.diagonal()
            diag.copy_(val[:diag.numel()])
        else:
            (max_, min_) = a.shape
            idx = torch.arange(max_ - max_ // (min_ + 1))
            mod = idx % min_
            div = idx // min_
            a[div * (min_ + 1) + mod, mod] = val[:idx.numel()]
    else:
        idx = diag_indices_from(a)
        a[idx] = val[:a.shape[0]]
    return a

def vdot(a: ArrayLike, b: ArrayLike, /):
    if False:
        i = 10
        return i + 15
    (t_a, t_b) = torch.atleast_1d(a, b)
    if t_a.ndim > 1:
        t_a = t_a.flatten()
    if t_b.ndim > 1:
        t_b = t_b.flatten()
    dtype = _dtypes_impl.result_type_impl(t_a, t_b)
    is_half = dtype == torch.float16 and (t_a.is_cpu or t_b.is_cpu)
    is_bool = dtype == torch.bool
    if is_half:
        dtype = torch.float32
    elif is_bool:
        dtype = torch.uint8
    t_a = _util.cast_if_needed(t_a, dtype)
    t_b = _util.cast_if_needed(t_b, dtype)
    result = torch.vdot(t_a, t_b)
    if is_half:
        result = result.to(torch.float16)
    elif is_bool:
        result = result.to(torch.bool)
    return result

def tensordot(a: ArrayLike, b: ArrayLike, axes=2):
    if False:
        return 10
    if isinstance(axes, (list, tuple)):
        axes = [[ax] if isinstance(ax, int) else ax for ax in axes]
    target_dtype = _dtypes_impl.result_type_impl(a, b)
    a = _util.cast_if_needed(a, target_dtype)
    b = _util.cast_if_needed(b, target_dtype)
    return torch.tensordot(a, b, dims=axes)

def dot(a: ArrayLike, b: ArrayLike, out: Optional[OutArray]=None):
    if False:
        i = 10
        return i + 15
    dtype = _dtypes_impl.result_type_impl(a, b)
    is_bool = dtype == torch.bool
    if is_bool:
        dtype = torch.uint8
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    if a.ndim == 0 or b.ndim == 0:
        result = a * b
    else:
        result = torch.matmul(a, b)
    if is_bool:
        result = result.to(torch.bool)
    return result

def inner(a: ArrayLike, b: ArrayLike, /):
    if False:
        for i in range(10):
            print('nop')
    dtype = _dtypes_impl.result_type_impl(a, b)
    is_half = dtype == torch.float16 and (a.is_cpu or b.is_cpu)
    is_bool = dtype == torch.bool
    if is_half:
        dtype = torch.float32
    elif is_bool:
        dtype = torch.uint8
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    result = torch.inner(a, b)
    if is_half:
        result = result.to(torch.float16)
    elif is_bool:
        result = result.to(torch.bool)
    return result

def outer(a: ArrayLike, b: ArrayLike, out: Optional[OutArray]=None):
    if False:
        for i in range(10):
            print('nop')
    return torch.outer(a, b)

def cross(a: ArrayLike, b: ArrayLike, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if False:
        while True:
            i = 10
    if axis is not None:
        (axisa, axisb, axisc) = (axis,) * 3
    axisa = _util.normalize_axis_index(axisa, a.ndim)
    axisb = _util.normalize_axis_index(axisb, b.ndim)
    a = torch.moveaxis(a, axisa, -1)
    b = torch.moveaxis(b, axisb, -1)
    msg = 'incompatible dimensions for cross product\n(dimension must be 2 or 3)'
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        raise ValueError(msg)
    shape = broadcast_shapes(a[..., 0].shape, b[..., 0].shape)
    if a.shape[-1] == 3 or b.shape[-1] == 3:
        shape += (3,)
        axisc = _util.normalize_axis_index(axisc, len(shape))
    dtype = _dtypes_impl.result_type_impl(a, b)
    cp = torch.empty(shape, dtype=dtype)
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    a0 = a[..., 0]
    a1 = a[..., 1]
    if a.shape[-1] == 3:
        a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    if b.shape[-1] == 3:
        b2 = b[..., 2]
    if cp.ndim != 0 and cp.shape[-1] == 3:
        cp0 = cp[..., 0]
        cp1 = cp[..., 1]
        cp2 = cp[..., 2]
    if a.shape[-1] == 2:
        if b.shape[-1] == 2:
            cp[...] = a0 * b1 - a1 * b0
            return cp
        else:
            assert b.shape[-1] == 3
            cp0[...] = a1 * b2
            cp1[...] = -a0 * b2
            cp2[...] = a0 * b1 - a1 * b0
    else:
        assert a.shape[-1] == 3
        if b.shape[-1] == 3:
            cp0[...] = a1 * b2 - a2 * b1
            cp1[...] = a2 * b0 - a0 * b2
            cp2[...] = a0 * b1 - a1 * b0
        else:
            assert b.shape[-1] == 2
            cp0[...] = -a2 * b1
            cp1[...] = a2 * b0
            cp2[...] = a0 * b1 - a1 * b0
    return torch.moveaxis(cp, -1, axisc)

def einsum(*operands, out=None, dtype=None, order='K', casting='safe', optimize=False):
    if False:
        i = 10
        return i + 15
    from ._ndarray import ndarray
    from ._normalizations import maybe_copy_to, normalize_array_like, normalize_casting, normalize_dtype, wrap_tensors
    dtype = normalize_dtype(dtype)
    casting = normalize_casting(casting)
    if out is not None and (not isinstance(out, ndarray)):
        raise TypeError("'out' must be an array")
    if order != 'K':
        raise NotImplementedError("'order' parameter is not supported.")
    sublist_format = not isinstance(operands[0], str)
    if sublist_format:
        array_operands = operands[:-1][::2]
    else:
        (subscripts, array_operands) = (operands[0], operands[1:])
    tensors = [normalize_array_like(op) for op in array_operands]
    target_dtype = _dtypes_impl.result_type_impl(*tensors) if dtype is None else dtype
    is_half = target_dtype == torch.float16 and all((t.is_cpu for t in tensors))
    if is_half:
        target_dtype = torch.float32
    is_short_int = target_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32]
    if is_short_int:
        target_dtype = torch.int64
    tensors = _util.typecast_tensors(tensors, target_dtype, casting)
    from torch.backends import opt_einsum
    try:
        if opt_einsum.is_available():
            old_strategy = torch.backends.opt_einsum.strategy
            old_enabled = torch.backends.opt_einsum.enabled
            if optimize is True:
                optimize = 'auto'
            elif optimize is False:
                torch.backends.opt_einsum.enabled = False
            torch.backends.opt_einsum.strategy = optimize
        if sublist_format:
            sublists = operands[1::2]
            has_sublistout = len(operands) % 2 == 1
            if has_sublistout:
                sublistout = operands[-1]
            operands = list(itertools.chain(*zip(tensors, sublists)))
            if has_sublistout:
                operands.append(sublistout)
            result = torch.einsum(*operands)
        else:
            result = torch.einsum(subscripts, *tensors)
    finally:
        if opt_einsum.is_available():
            torch.backends.opt_einsum.strategy = old_strategy
            torch.backends.opt_einsum.enabled = old_enabled
    result = maybe_copy_to(out, result)
    return wrap_tensors(result)

def _sort_helper(tensor, axis, kind, order):
    if False:
        for i in range(10):
            print('nop')
    if tensor.dtype.is_complex:
        raise NotImplementedError(f'sorting {tensor.dtype} is not supported')
    ((tensor,), axis) = _util.axis_none_flatten(tensor, axis=axis)
    axis = _util.normalize_axis_index(axis, tensor.ndim)
    stable = kind == 'stable'
    return (tensor, axis, stable)

def sort(a: ArrayLike, axis=-1, kind=None, order: NotImplementedType=None):
    if False:
        return 10
    (a, axis, stable) = _sort_helper(a, axis, kind, order)
    result = torch.sort(a, dim=axis, stable=stable)
    return result.values

def argsort(a: ArrayLike, axis=-1, kind=None, order: NotImplementedType=None):
    if False:
        for i in range(10):
            print('nop')
    (a, axis, stable) = _sort_helper(a, axis, kind, order)
    return torch.argsort(a, dim=axis, stable=stable)

def searchsorted(a: ArrayLike, v: ArrayLike, side='left', sorter: Optional[ArrayLike]=None):
    if False:
        while True:
            i = 10
    if a.dtype.is_complex:
        raise NotImplementedError(f'searchsorted with dtype={a.dtype}')
    return torch.searchsorted(a, v, side=side, sorter=sorter)

def moveaxis(a: ArrayLike, source, destination):
    if False:
        while True:
            i = 10
    source = _util.normalize_axis_tuple(source, a.ndim, 'source')
    destination = _util.normalize_axis_tuple(destination, a.ndim, 'destination')
    return torch.moveaxis(a, source, destination)

def swapaxes(a: ArrayLike, axis1, axis2):
    if False:
        i = 10
        return i + 15
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    return torch.swapaxes(a, axis1, axis2)

def rollaxis(a: ArrayLike, axis, start=0):
    if False:
        print('Hello World!')
    n = a.ndim
    axis = _util.normalize_axis_index(axis, n)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not 0 <= start < n + 1:
        raise _util.AxisError(msg % ('start', -n, 'start', n + 1, start))
    if axis < start:
        start -= 1
    if axis == start:
        return a
    axes = list(range(0, n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.view(axes)

def roll(a: ArrayLike, shift, axis=None):
    if False:
        print('Hello World!')
    if axis is not None:
        axis = _util.normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        if not isinstance(shift, tuple):
            shift = (shift,) * len(axis)
    return torch.roll(a, shift, axis)

def squeeze(a: ArrayLike, axis=None):
    if False:
        i = 10
        return i + 15
    if axis == ():
        result = a
    elif axis is None:
        result = a.squeeze()
    elif isinstance(axis, tuple):
        result = a
        for ax in axis:
            result = a.squeeze(ax)
    else:
        result = a.squeeze(axis)
    return result

def reshape(a: ArrayLike, newshape, order: NotImplementedType='C'):
    if False:
        i = 10
        return i + 15
    newshape = newshape[0] if len(newshape) == 1 else newshape
    return a.reshape(newshape)

def transpose(a: ArrayLike, axes=None):
    if False:
        i = 10
        return i + 15
    if axes in [(), None, (None,)]:
        axes = tuple(reversed(range(a.ndim)))
    elif len(axes) == 1:
        axes = axes[0]
    return a.permute(axes)

def ravel(a: ArrayLike, order: NotImplementedType='C'):
    if False:
        while True:
            i = 10
    return torch.flatten(a)

def diff(a: ArrayLike, n=1, axis=-1, prepend: Optional[ArrayLike]=None, append: Optional[ArrayLike]=None):
    if False:
        while True:
            i = 10
    axis = _util.normalize_axis_index(axis, a.ndim)
    if n < 0:
        raise ValueError(f'order must be non-negative but got {n}')
    if n == 0:
        return a
    if prepend is not None:
        shape = list(a.shape)
        shape[axis] = prepend.shape[axis] if prepend.ndim > 0 else 1
        prepend = torch.broadcast_to(prepend, shape)
    if append is not None:
        shape = list(a.shape)
        shape[axis] = append.shape[axis] if append.ndim > 0 else 1
        append = torch.broadcast_to(append, shape)
    return torch.diff(a, n, axis=axis, prepend=prepend, append=append)

def angle(z: ArrayLike, deg=False):
    if False:
        for i in range(10):
            print('nop')
    result = torch.angle(z)
    if deg:
        result = result * (180 / torch.pi)
    return result

def sinc(x: ArrayLike):
    if False:
        return 10
    return torch.sinc(x)

def gradient(f: ArrayLike, *varargs, axis=None, edge_order=1):
    if False:
        for i in range(10):
            print('nop')
    N = f.ndim
    varargs = _util.ndarrays_to_tensors(varargs)
    if axis is None:
        axes = tuple(range(N))
    else:
        axes = _util.normalize_axis_tuple(axis, N)
    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        dx = [1.0] * len_axes
    elif n == 1 and (_dtypes_impl.is_scalar(varargs[0]) or varargs[0].ndim == 0):
        dx = varargs * len_axes
    elif n == len_axes:
        dx = list(varargs)
        for (i, distances) in enumerate(dx):
            distances = torch.as_tensor(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError('distances must be either scalars or 1d')
            if len(distances) != f.shape[axes[i]]:
                raise ValueError('when 1d, distances must match the length of the corresponding dimension')
            if not (distances.dtype.is_floating_point or distances.dtype.is_complex):
                distances = distances.double()
            diffx = torch.diff(distances)
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
    if _dtypes_impl.python_type_for_torch(otype) in (int, bool):
        f = f.double()
        otype = torch.float64
    for (axis, ax_dx) in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError('Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required.')
        out = torch.empty_like(f, dtype=otype)
        uniform_spacing = _dtypes_impl.is_scalar(ax_dx) or ax_dx.ndim == 0
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
            shape = [1] * N
            shape[axis] = -1
            a = a.reshape(shape)
            b = b.reshape(shape)
            c = c.reshape(shape)
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
    else:
        return outvals

def round(a: ArrayLike, decimals=0, out: Optional[OutArray]=None):
    if False:
        print('Hello World!')
    if a.is_floating_point():
        result = torch.round(a, decimals=decimals)
    elif a.is_complex():
        result = torch.complex(torch.round(a.real, decimals=decimals), torch.round(a.imag, decimals=decimals))
    else:
        result = a
    return result
around = round
round_ = round

def real_if_close(a: ArrayLike, tol=100):
    if False:
        print('Hello World!')
    if not torch.is_complex(a):
        return a
    if tol > 1:
        tol = tol * torch.finfo(a.dtype).eps
    mask = torch.abs(a.imag) < tol
    return a.real if mask.all() else a

def real(a: ArrayLike):
    if False:
        for i in range(10):
            print('nop')
    return torch.real(a)

def imag(a: ArrayLike):
    if False:
        print('Hello World!')
    if a.is_complex():
        return a.imag
    return torch.zeros_like(a)

def iscomplex(x: ArrayLike):
    if False:
        print('Hello World!')
    if torch.is_complex(x):
        return x.imag != 0
    return torch.zeros_like(x, dtype=torch.bool)

def isreal(x: ArrayLike):
    if False:
        print('Hello World!')
    if torch.is_complex(x):
        return x.imag == 0
    return torch.ones_like(x, dtype=torch.bool)

def iscomplexobj(x: ArrayLike):
    if False:
        return 10
    return torch.is_complex(x)

def isrealobj(x: ArrayLike):
    if False:
        i = 10
        return i + 15
    return not torch.is_complex(x)

def isneginf(x: ArrayLike, out: Optional[OutArray]=None):
    if False:
        print('Hello World!')
    return torch.isneginf(x)

def isposinf(x: ArrayLike, out: Optional[OutArray]=None):
    if False:
        for i in range(10):
            print('nop')
    return torch.isposinf(x)

def i0(x: ArrayLike):
    if False:
        i = 10
        return i + 15
    return torch.special.i0(x)

def isscalar(a):
    if False:
        while True:
            i = 10
    from ._normalizations import normalize_array_like
    try:
        t = normalize_array_like(a)
        return t.numel() == 1
    except Exception:
        return False

def hamming(M):
    if False:
        while True:
            i = 10
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.hamming_window(M, periodic=False, dtype=dtype)

def hanning(M):
    if False:
        for i in range(10):
            print('nop')
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.hann_window(M, periodic=False, dtype=dtype)

def kaiser(M, beta):
    if False:
        i = 10
        return i + 15
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.kaiser_window(M, beta=beta, periodic=False, dtype=dtype)

def blackman(M):
    if False:
        print('Hello World!')
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.blackman_window(M, periodic=False, dtype=dtype)

def bartlett(M):
    if False:
        return 10
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.bartlett_window(M, periodic=False, dtype=dtype)
array_type = [[torch.float16, torch.float32, torch.float64], [None, torch.complex64, torch.complex128]]
array_precision = {torch.float16: 0, torch.float32: 1, torch.float64: 2, torch.complex64: 1, torch.complex128: 2}

def common_type(*tensors: ArrayLike):
    if False:
        i = 10
        return i + 15
    is_complex = False
    precision = 0
    for a in tensors:
        t = a.dtype
        if iscomplexobj(a):
            is_complex = True
        if not (t.is_floating_point or t.is_complex):
            p = 2
        else:
            p = array_precision.get(t, None)
            if p is None:
                raise TypeError("can't get common type for non-numeric array")
        precision = builtins.max(precision, p)
    if is_complex:
        return array_type[1][precision]
    else:
        return array_type[0][precision]

def histogram(a: ArrayLike, bins: ArrayLike=10, range=None, normed=None, weights: Optional[ArrayLike]=None, density=None):
    if False:
        print('Hello World!')
    if normed is not None:
        raise ValueError('normed argument is deprecated, use density= instead')
    is_a_int = not (a.dtype.is_floating_point or a.dtype.is_complex)
    is_w_int = weights is None or not weights.dtype.is_floating_point
    if is_a_int:
        a = a.double()
    if weights is not None:
        weights = _util.cast_if_needed(weights, a.dtype)
    if isinstance(bins, torch.Tensor):
        if bins.ndim == 0:
            bins = operator.index(bins)
        else:
            bins = _util.cast_if_needed(bins, a.dtype)
    if range is None:
        (h, b) = torch.histogram(a, bins, weight=weights, density=bool(density))
    else:
        (h, b) = torch.histogram(a, bins, range=range, weight=weights, density=bool(density))
    if not density and is_w_int:
        h = h.long()
    if is_a_int:
        b = b.long()
    return (h, b)

def histogram2d(x, y, bins=10, range: Optional[ArrayLike]=None, normed=None, weights: Optional[ArrayLike]=None, density=None):
    if False:
        print('Hello World!')
    if len(x) != len(y):
        raise ValueError('x and y must have the same length.')
    try:
        N = len(bins)
    except TypeError:
        N = 1
    if N != 1 and N != 2:
        bins = [bins, bins]
    (h, e) = histogramdd((x, y), bins, range, normed, weights, density)
    return (h, e[0], e[1])

def histogramdd(sample, bins=10, range: Optional[ArrayLike]=None, normed=None, weights: Optional[ArrayLike]=None, density=None):
    if False:
        for i in range(10):
            print('nop')
    if normed is not None:
        raise ValueError('normed argument is deprecated, use density= instead')
    from ._normalizations import normalize_array_like, normalize_seq_array_like
    if isinstance(sample, (list, tuple)):
        sample = normalize_array_like(sample).T
    else:
        sample = normalize_array_like(sample)
    sample = torch.atleast_2d(sample)
    if not (sample.dtype.is_floating_point or sample.dtype.is_complex):
        sample = sample.double()
    bins_is_array = not (isinstance(bins, int) or builtins.all((isinstance(b, int) for b in bins)))
    if bins_is_array:
        bins = normalize_seq_array_like(bins)
        bins_dtypes = [b.dtype for b in bins]
        bins = [_util.cast_if_needed(b, sample.dtype) for b in bins]
    if range is not None:
        range = range.flatten().tolist()
    if weights is not None:
        mm = sample.aminmax(dim=0)
        range = torch.cat(mm).reshape(2, -1).T.flatten()
        range = tuple(range.tolist())
        weights = _util.cast_if_needed(weights, sample.dtype)
        w_kwd = {'weight': weights}
    else:
        w_kwd = {}
    (h, b) = torch.histogramdd(sample, bins, range, density=bool(density), **w_kwd)
    if bins_is_array:
        b = [_util.cast_if_needed(bb, dtyp) for (bb, dtyp) in zip(b, bins_dtypes)]
    return (h, b)

def min_scalar_type(a: ArrayLike, /):
    if False:
        i = 10
        return i + 15
    from ._dtypes import DType
    if a.numel() > 1:
        return DType(a.dtype)
    if a.dtype == torch.bool:
        dtype = torch.bool
    elif a.dtype.is_complex:
        fi = torch.finfo(torch.float32)
        fits_in_single = a.dtype == torch.complex64 or (fi.min <= a.real <= fi.max and fi.min <= a.imag <= fi.max)
        dtype = torch.complex64 if fits_in_single else torch.complex128
    elif a.dtype.is_floating_point:
        for dt in [torch.float16, torch.float32, torch.float64]:
            fi = torch.finfo(dt)
            if fi.min <= a <= fi.max:
                dtype = dt
                break
    else:
        for dt in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            ii = torch.iinfo(dt)
            if ii.min <= a <= ii.max:
                dtype = dt
                break
    return DType(dtype)

def pad(array: ArrayLike, pad_width: ArrayLike, mode='constant', **kwargs):
    if False:
        i = 10
        return i + 15
    if mode != 'constant':
        raise NotImplementedError
    value = kwargs.get('constant_values', 0)
    typ = _dtypes_impl.python_type_for_torch(array.dtype)
    value = typ(value)
    pad_width = torch.broadcast_to(pad_width, (array.ndim, 2))
    pad_width = torch.flip(pad_width, (0,)).flatten()
    return torch.nn.functional.pad(array, tuple(pad_width), value=value)