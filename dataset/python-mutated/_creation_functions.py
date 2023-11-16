from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
if TYPE_CHECKING:
    from ._typing import Array, Device, Dtype, NestedSequence, SupportsBufferProtocol
    from collections.abc import Sequence
from ._dtypes import _all_dtypes
import cupy as np
from cupy.cuda import Device as _Device
from cupy_backends.cuda.api import runtime

def _check_valid_dtype(dtype):
    if False:
        print('Hello World!')
    for d in (None,) + _all_dtypes:
        if dtype is d:
            return
    raise ValueError('dtype must be one of the supported dtypes')

def asarray(obj: Union[Array, bool, int, float, NestedSequence[bool | int | float], SupportsBufferProtocol], /, *, dtype: Optional[Dtype]=None, device: Optional[Device]=None, copy: Optional[bool]=None) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.asarray <numpy.asarray>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    _check_valid_dtype(dtype)
    if device is not None and (not isinstance(device, _Device)):
        raise ValueError(f'Unsupported device {device!r}')
    if device is None:
        device = _Device()
    if copy is False:
        raise NotImplementedError('copy=False is not yet implemented')
    if isinstance(obj, Array):
        if dtype is not None and obj.dtype != dtype:
            copy = True
        if copy is True:
            prev_device = runtime.getDevice()
            try:
                runtime.setDevice(device.id)
                obj = Array._new(np.array(obj._array, copy=True, dtype=dtype))
            finally:
                runtime.setDevice(prev_device)
        return obj
    if dtype is None and isinstance(obj, int) and (obj > 2 ** 64 or obj < -2 ** 63):
        raise OverflowError('Integer out of bounds for array dtypes')
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        res = np.asarray(obj, dtype=dtype)
    finally:
        runtime.setDevice(prev_device)
    return Array._new(res)

def arange(start: Union[int, float], /, stop: Optional[Union[int, float]]=None, step: Union[int, float]=1, *, dtype: Optional[Dtype]=None, device: Optional[Device]=None) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.arange <numpy.arange>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    _check_valid_dtype(dtype)
    if device is not None and (not isinstance(device, _Device)):
        raise ValueError(f'Unsupported device {device!r}')
    if device is None:
        device = _Device()
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        return Array._new(np.arange(start, stop=stop, step=step, dtype=dtype))
    finally:
        runtime.setDevice(prev_device)

def empty(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[Dtype]=None, device: Optional[Device]=None) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.empty <numpy.empty>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    _check_valid_dtype(dtype)
    if device is not None and (not isinstance(device, _Device)):
        raise ValueError(f'Unsupported device {device!r}')
    if device is None:
        device = _Device()
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        return Array._new(np.empty(shape, dtype=dtype))
    finally:
        runtime.setDevice(prev_device)

def empty_like(x: Array, /, *, dtype: Optional[Dtype]=None, device: Optional[Device]=None) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.empty_like <numpy.empty_like>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    _check_valid_dtype(dtype)
    if device is not None and (not isinstance(device, _Device)):
        raise ValueError(f'Unsupported device {device!r}')
    if device is None:
        device = _Device()
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        return Array._new(np.empty_like(x._array, dtype=dtype))
    finally:
        runtime.setDevice(prev_device)

def eye(n_rows: int, n_cols: Optional[int]=None, /, *, k: int=0, dtype: Optional[Dtype]=None, device: Optional[Device]=None) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.eye <numpy.eye>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    _check_valid_dtype(dtype)
    if device is not None and (not isinstance(device, _Device)):
        raise ValueError(f'Unsupported device {device!r}')
    if device is None:
        device = _Device()
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        return Array._new(np.eye(n_rows, M=n_cols, k=k, dtype=dtype))
    finally:
        runtime.setDevice(prev_device)

def from_dlpack(x: object, /) -> Array:
    if False:
        i = 10
        return i + 15
    '\n    Array API compatible wrapper for :py:func:`np.from_dlpack <numpy.from_dlpack>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    return Array._new(np.from_dlpack(x))

def full(shape: Union[int, Tuple[int, ...]], fill_value: Union[int, float], *, dtype: Optional[Dtype]=None, device: Optional[Device]=None) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.full <numpy.full>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    _check_valid_dtype(dtype)
    if device is not None and (not isinstance(device, _Device)):
        raise ValueError(f'Unsupported device {device!r}')
    if device is None:
        device = _Device()
    if isinstance(fill_value, Array) and fill_value.ndim == 0:
        fill_value = fill_value._array
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        res = np.full(shape, fill_value, dtype=dtype)
    finally:
        runtime.setDevice(prev_device)
    if res.dtype not in _all_dtypes:
        raise TypeError('Invalid input to full')
    return Array._new(res)

def full_like(x: Array, /, fill_value: Union[int, float], *, dtype: Optional[Dtype]=None, device: Optional[Device]=None) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.full_like <numpy.full_like>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    _check_valid_dtype(dtype)
    if device is not None and (not isinstance(device, _Device)):
        raise ValueError(f'Unsupported device {device!r}')
    if device is None:
        device = _Device()
    if isinstance(fill_value, Array) and fill_value.ndim == 0:
        fill_value = fill_value._array
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        res = np.full_like(x._array, fill_value, dtype=dtype)
    finally:
        runtime.setDevice(prev_device)
    if res.dtype not in _all_dtypes:
        raise TypeError('Invalid input to full_like')
    return Array._new(res)

def linspace(start: Union[int, float], stop: Union[int, float], /, num: int, *, dtype: Optional[Dtype]=None, device: Optional[Device]=None, endpoint: bool=True) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.linspace <numpy.linspace>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    _check_valid_dtype(dtype)
    if device is None:
        device = _Device()
    elif not isinstance(device, _Device):
        raise ValueError(f'Unsupported device {device!r}')
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        return Array._new(np.linspace(start, stop, num, dtype=dtype, endpoint=endpoint))
    finally:
        runtime.setDevice(prev_device)

def meshgrid(*arrays: Array, indexing: str='xy') -> List[Array]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.meshgrid <numpy.meshgrid>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    if len({a.dtype for a in arrays}) > 1:
        raise ValueError('meshgrid inputs must all have the same dtype')
    return [Array._new(array) for array in np.meshgrid(*[a._array for a in arrays], indexing=indexing)]

def ones(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[Dtype]=None, device: Optional[Device]=None) -> Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Array API compatible wrapper for :py:func:`np.ones <numpy.ones>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    _check_valid_dtype(dtype)
    if device is not None and (not isinstance(device, _Device)):
        raise ValueError(f'Unsupported device {device!r}')
    if device is None:
        device = _Device()
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        return Array._new(np.ones(shape, dtype=dtype))
    finally:
        runtime.setDevice(prev_device)

def ones_like(x: Array, /, *, dtype: Optional[Dtype]=None, device: Optional[Device]=None) -> Array:
    if False:
        print('Hello World!')
    '\n    Array API compatible wrapper for :py:func:`np.ones_like <numpy.ones_like>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    _check_valid_dtype(dtype)
    if device is not None and (not isinstance(device, _Device)):
        raise ValueError(f'Unsupported device {device!r}')
    if device is None:
        device = _Device()
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        return Array._new(np.ones_like(x._array, dtype=dtype))
    finally:
        runtime.setDevice(prev_device)

def tril(x: Array, /, *, k: int=0) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.tril <numpy.tril>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    if x.ndim < 2:
        raise ValueError('x must be at least 2-dimensional for tril')
    return Array._new(np.tril(x._array, k=k))

def triu(x: Array, /, *, k: int=0) -> Array:
    if False:
        return 10
    '\n    Array API compatible wrapper for :py:func:`np.triu <numpy.triu>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    if x.ndim < 2:
        raise ValueError('x must be at least 2-dimensional for triu')
    return Array._new(np.triu(x._array, k=k))

def zeros(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[Dtype]=None, device: Optional[Device]=None) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.zeros <numpy.zeros>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    _check_valid_dtype(dtype)
    if device is not None and (not isinstance(device, _Device)):
        raise ValueError(f'Unsupported device {device!r}')
    if device is None:
        device = _Device()
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        return Array._new(np.zeros(shape, dtype=dtype))
    finally:
        runtime.setDevice(prev_device)

def zeros_like(x: Array, /, *, dtype: Optional[Dtype]=None, device: Optional[Device]=None) -> Array:
    if False:
        while True:
            i = 10
    '\n    Array API compatible wrapper for :py:func:`np.zeros_like <numpy.zeros_like>`.\n\n    See its docstring for more information.\n    '
    from ._array_object import Array
    _check_valid_dtype(dtype)
    if device is not None and (not isinstance(device, _Device)):
        raise ValueError(f'Unsupported device {device!r}')
    if device is None:
        device = _Device()
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        return Array._new(np.zeros_like(x._array, dtype=dtype))
    finally:
        runtime.setDevice(prev_device)