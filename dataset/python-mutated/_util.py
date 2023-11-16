"""Assorted utilities, which do not need anything other then torch and stdlib.
"""
import operator
import torch
from . import _dtypes_impl

def is_sequence(seq):
    if False:
        return 10
    if isinstance(seq, str):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True

class AxisError(ValueError, IndexError):
    pass

class UFuncTypeError(TypeError, RuntimeError):
    pass

def cast_if_needed(tensor, dtype):
    if False:
        print('Hello World!')
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor

def cast_int_to_float(x):
    if False:
        while True:
            i = 10
    if _dtypes_impl._category(x.dtype) < 2:
        x = x.to(_dtypes_impl.default_dtypes().float_dtype)
    return x

def normalize_axis_index(ax, ndim, argname=None):
    if False:
        i = 10
        return i + 15
    if not -ndim <= ax < ndim:
        raise AxisError(f'axis {ax} is out of bounds for array of dimension {ndim}')
    if ax < 0:
        ax += ndim
    return ax

def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    if False:
        print('Hello World!')
    '\n    Normalizes an axis argument into a tuple of non-negative integer axes.\n\n    This handles shorthands such as ``1`` and converts them to ``(1,)``,\n    as well as performing the handling of negative indices covered by\n    `normalize_axis_index`.\n\n    By default, this forbids axes from being specified multiple times.\n    Used internally by multi-axis-checking logic.\n\n    Parameters\n    ----------\n    axis : int, iterable of int\n        The un-normalized index or indices of the axis.\n    ndim : int\n        The number of dimensions of the array that `axis` should be normalized\n        against.\n    argname : str, optional\n        A prefix to put before the error message, typically the name of the\n        argument.\n    allow_duplicate : bool, optional\n        If False, the default, disallow an axis from being specified twice.\n\n    Returns\n    -------\n    normalized_axes : tuple of int\n        The normalized axis index, such that `0 <= normalized_axis < ndim`\n    '
    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    axis = tuple([normalize_axis_index(ax, ndim, argname) for ax in axis])
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError(f'repeated axis in `{argname}` argument')
        else:
            raise ValueError('repeated axis')
    return axis

def allow_only_single_axis(axis):
    if False:
        for i in range(10):
            print('nop')
    if axis is None:
        return axis
    if len(axis) != 1:
        raise NotImplementedError('does not handle tuple axis')
    return axis[0]

def expand_shape(arr_shape, axis):
    if False:
        for i in range(10):
            print('nop')
    if type(axis) not in (list, tuple):
        axis = (axis,)
    out_ndim = len(axis) + len(arr_shape)
    axis = normalize_axis_tuple(axis, out_ndim)
    shape_it = iter(arr_shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
    return shape

def apply_keepdims(tensor, axis, ndim):
    if False:
        while True:
            i = 10
    if axis is None:
        shape = (1,) * ndim
        tensor = tensor.expand(shape).contiguous()
    else:
        shape = expand_shape(tensor.shape, axis)
        tensor = tensor.reshape(shape)
    return tensor

def axis_none_flatten(*tensors, axis=None):
    if False:
        print('Hello World!')
    'Flatten the arrays if axis is None.'
    if axis is None:
        tensors = tuple((ar.flatten() for ar in tensors))
        return (tensors, 0)
    else:
        return (tensors, axis)

def typecast_tensor(t, target_dtype, casting):
    if False:
        return 10
    'Dtype-cast tensor to target_dtype.\n\n    Parameters\n    ----------\n    t : torch.Tensor\n        The tensor to cast\n    target_dtype : torch dtype object\n        The array dtype to cast all tensors to\n    casting : str\n        The casting mode, see `np.can_cast`\n\n     Returns\n     -------\n    `torch.Tensor` of the `target_dtype` dtype\n\n     Raises\n     ------\n     ValueError\n        if the argument cannot be cast according to the `casting` rule\n\n    '
    can_cast = _dtypes_impl.can_cast_impl
    if not can_cast(t.dtype, target_dtype, casting=casting):
        raise TypeError(f"Cannot cast array data from {t.dtype} to {target_dtype} according to the rule '{casting}'")
    return cast_if_needed(t, target_dtype)

def typecast_tensors(tensors, target_dtype, casting):
    if False:
        return 10
    return tuple((typecast_tensor(t, target_dtype, casting) for t in tensors))

def _try_convert_to_tensor(obj):
    if False:
        return 10
    try:
        tensor = torch.as_tensor(obj)
    except Exception as e:
        mesg = f'failed to convert {obj} to ndarray. \nInternal error is: {str(e)}.'
        raise NotImplementedError(mesg)
    return tensor

def _coerce_to_tensor(obj, dtype=None, copy=False, ndmin=0):
    if False:
        while True:
            i = 10
    'The core logic of the array(...) function.\n\n    Parameters\n    ----------\n    obj : tensor_like\n        The thing to coerce\n    dtype : torch.dtype object or None\n        Coerce to this torch dtype\n    copy : bool\n        Copy or not\n    ndmin : int\n        The results as least this many dimensions\n    is_weak : bool\n        Whether obj is a weakly typed python scalar.\n\n    Returns\n    -------\n    tensor : torch.Tensor\n        a tensor object with requested dtype, ndim and copy semantics.\n\n    Notes\n    -----\n    This is almost a "tensor_like" coersion function. Does not handle wrapper\n    ndarrays (those should be handled in the ndarray-aware layer prior to\n    invoking this function).\n    '
    if isinstance(obj, torch.Tensor):
        tensor = obj
    else:
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(_dtypes_impl.get_default_dtype_for(torch.float32))
        try:
            tensor = _try_convert_to_tensor(obj)
        finally:
            torch.set_default_dtype(default_dtype)
    tensor = cast_if_needed(tensor, dtype)
    ndim_extra = ndmin - tensor.ndim
    if ndim_extra > 0:
        tensor = tensor.view((1,) * ndim_extra + tensor.shape)
    if copy:
        tensor = tensor.clone()
    return tensor

def ndarrays_to_tensors(*inputs):
    if False:
        for i in range(10):
            print('nop')
    'Convert all ndarrays from `inputs` to tensors. (other things are intact)'
    from ._ndarray import ndarray
    if len(inputs) == 0:
        return ValueError()
    elif len(inputs) == 1:
        input_ = inputs[0]
        if isinstance(input_, ndarray):
            return input_.tensor
        elif isinstance(input_, tuple):
            result = []
            for sub_input in input_:
                sub_result = ndarrays_to_tensors(sub_input)
                result.append(sub_result)
            return tuple(result)
        else:
            return input_
    else:
        assert isinstance(inputs, tuple)
        return ndarrays_to_tensors(inputs)