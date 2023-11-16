""" "Normalize" arguments: convert array_likes to tensors, dtypes to torch dtypes and so on.
"""
from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
ArrayLike = typing.TypeVar('ArrayLike')
Scalar = typing.Union[int, float, complex, bool]
ArrayLikeOrScalar = typing.Union[ArrayLike, Scalar]
DTypeLike = typing.TypeVar('DTypeLike')
AxisLike = typing.TypeVar('AxisLike')
NDArray = typing.TypeVar('NDarray')
CastingModes = typing.TypeVar('CastingModes')
KeepDims = typing.TypeVar('KeepDims')
OutArray = typing.TypeVar('OutArray')
try:
    from typing import NotImplementedType
except ImportError:
    NotImplementedType = typing.TypeVar('NotImplementedType')

def normalize_array_like(x, parm=None):
    if False:
        for i in range(10):
            print('nop')
    from ._ndarray import asarray
    return asarray(x).tensor

def normalize_array_like_or_scalar(x, parm=None):
    if False:
        i = 10
        return i + 15
    if _dtypes_impl.is_scalar_or_symbolic(x):
        return x
    return normalize_array_like(x, parm)

def normalize_optional_array_like_or_scalar(x, parm=None):
    if False:
        i = 10
        return i + 15
    if x is None:
        return None
    return normalize_array_like_or_scalar(x, parm)

def normalize_optional_array_like(x, parm=None):
    if False:
        print('Hello World!')
    return None if x is None else normalize_array_like(x, parm)

def normalize_seq_array_like(x, parm=None):
    if False:
        return 10
    return tuple((normalize_array_like(value) for value in x))

def normalize_dtype(dtype, parm=None):
    if False:
        print('Hello World!')
    torch_dtype = None
    if dtype is not None:
        dtype = _dtypes.dtype(dtype)
        torch_dtype = dtype.torch_dtype
    return torch_dtype

def normalize_not_implemented(arg, parm):
    if False:
        while True:
            i = 10
    if arg != parm.default:
        raise NotImplementedError(f"'{parm.name}' parameter is not supported.")

def normalize_axis_like(arg, parm=None):
    if False:
        i = 10
        return i + 15
    from ._ndarray import ndarray
    if isinstance(arg, ndarray):
        arg = operator.index(arg)
    return arg

def normalize_ndarray(arg, parm=None):
    if False:
        for i in range(10):
            print('nop')
    if arg is None:
        return arg
    from ._ndarray import ndarray
    if not isinstance(arg, ndarray):
        raise TypeError(f"'{parm.name}' must be an array")
    return arg.tensor

def normalize_outarray(arg, parm=None):
    if False:
        while True:
            i = 10
    if arg is None:
        return arg
    from ._ndarray import ndarray
    if not isinstance(arg, ndarray):
        raise TypeError(f"'{parm.name}' must be an array")
    return arg

def normalize_casting(arg, parm=None):
    if False:
        return 10
    if arg not in ['no', 'equiv', 'safe', 'same_kind', 'unsafe']:
        raise ValueError(f"casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe' (got '{arg}')")
    return arg
normalizers = {'ArrayLike': normalize_array_like, 'ArrayLikeOrScalar': normalize_array_like_or_scalar, 'Optional[ArrayLike]': normalize_optional_array_like, 'Sequence[ArrayLike]': normalize_seq_array_like, 'Optional[ArrayLikeOrScalar]': normalize_optional_array_like_or_scalar, 'Optional[NDArray]': normalize_ndarray, 'Optional[OutArray]': normalize_outarray, 'NDArray': normalize_ndarray, 'Optional[DTypeLike]': normalize_dtype, 'AxisLike': normalize_axis_like, 'NotImplementedType': normalize_not_implemented, 'Optional[CastingModes]': normalize_casting}

def maybe_normalize(arg, parm):
    if False:
        while True:
            i = 10
    'Normalize arg if a normalizer is registered.'
    normalizer = normalizers.get(parm.annotation, None)
    return normalizer(arg, parm) if normalizer else arg

def maybe_copy_to(out, result, promote_scalar_result=False):
    if False:
        while True:
            i = 10
    if out is None:
        return result
    elif isinstance(result, torch.Tensor):
        if result.shape != out.shape:
            can_fit = result.numel() == 1 and out.ndim == 0
            if promote_scalar_result and can_fit:
                result = result.squeeze()
            else:
                raise ValueError(f'Bad size of the out array: out.shape = {out.shape} while result.shape = {result.shape}.')
        out.tensor.copy_(result)
        return out
    elif isinstance(result, (tuple, list)):
        return type(result)((maybe_copy_to(o, r, promote_scalar_result) for (o, r) in zip(out, result)))
    else:
        raise AssertionError()

def wrap_tensors(result):
    if False:
        while True:
            i = 10
    from ._ndarray import ndarray
    if isinstance(result, torch.Tensor):
        return ndarray(result)
    elif isinstance(result, (tuple, list)):
        result = type(result)((wrap_tensors(x) for x in result))
    return result

def array_or_scalar(values, py_type=float, return_scalar=False):
    if False:
        for i in range(10):
            print('nop')
    if return_scalar:
        return py_type(values.item())
    else:
        from ._ndarray import ndarray
        return ndarray(values)

def normalizer(_func=None, *, promote_scalar_result=False):
    if False:
        while True:
            i = 10

    def normalizer_inner(func):
        if False:
            i = 10
            return i + 15

        @functools.wraps(func)
        def wrapped(*args, **kwds):
            if False:
                return 10
            sig = inspect.signature(func)
            params = sig.parameters
            first_param = next(iter(params.values()))
            if first_param.kind == inspect.Parameter.VAR_POSITIONAL:
                args = [maybe_normalize(arg, first_param) for arg in args]
            else:
                args = tuple((maybe_normalize(arg, parm) for (arg, parm) in zip(args, params.values()))) + args[len(params.values()):]
            kwds = {name: maybe_normalize(arg, params[name]) if name in params else arg for (name, arg) in kwds.items()}
            result = func(*args, **kwds)
            bound_args = None
            if 'keepdims' in params and params['keepdims'].annotation == 'KeepDims':
                bound_args = sig.bind(*args, **kwds).arguments
                if bound_args.get('keepdims', False):
                    tensor = args[0]
                    axis = bound_args.get('axis')
                    result = _util.apply_keepdims(result, axis, tensor.ndim)
            if 'out' in params:
                if bound_args is None:
                    bound_args = sig.bind(*args, **kwds).arguments
                out = bound_args.get('out')
                result = maybe_copy_to(out, result, promote_scalar_result)
            result = wrap_tensors(result)
            return result
        return wrapped
    if _func is None:
        return normalizer_inner
    else:
        return normalizer_inner(_func)