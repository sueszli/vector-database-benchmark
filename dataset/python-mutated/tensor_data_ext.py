from __future__ import annotations
import collections
from math import prod
from typing import TYPE_CHECKING, Any, Final, Protocol, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from rerun.error_utils import _send_warning_or_raise
from .._unions import build_dense_union

class TorchTensorLike(Protocol):
    """Describes what is need from a Torch Tensor to be loggable to Rerun."""

    def numpy(self, force: bool) -> npt.NDArray[Any]:
        if False:
            i = 10
            return i + 15
        ...
if TYPE_CHECKING:
    from . import TensorBufferLike, TensorDataArrayLike, TensorDataLike, TensorDimension, TensorDimensionLike
    TensorLike = Union[TensorDataLike, TorchTensorLike]
    'Type helper for a tensor-like object that can be logged to Rerun.'

def _to_numpy(tensor: TensorLike) -> npt.NDArray[Any]:
    if False:
        i = 10
        return i + 15
    if isinstance(tensor, np.ndarray):
        return tensor
    try:
        return tensor.numpy(force=True)
    except AttributeError:
        return np.array(tensor, copy=False)

class TensorDataExt:
    """Extension for [TensorData][rerun.datatypes.TensorData]."""

    def __init__(self: Any, *, shape: Sequence[TensorDimensionLike] | None=None, buffer: TensorBufferLike | None=None, array: TensorLike | None=None, dim_names: Sequence[str | None] | None=None) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Construct a `TensorData` object.\n\n        The `TensorData` object is internally represented by three fields: `shape` and `buffer`.\n\n        This constructor provides additional arguments 'array', and 'dim_names'. When passing in a\n        multi-dimensional array such as a `np.ndarray`, the `shape` and `buffer` fields will be\n        populated automagically.\n\n        Parameters\n        ----------\n        self: TensorData\n            The TensorData object to construct.\n        shape: Sequence[TensorDimensionLike] | None\n            The shape of the tensor. If None, and an array is proviced, the shape will be inferred\n            from the shape of the array.\n        buffer: TensorBufferLike | None\n            The buffer of the tensor. If None, and an array is provided, the buffer will be generated\n            from the array.\n        array: Tensor | None\n            A numpy array (or The array of the tensor. If None, the array will be inferred from the buffer.\n        dim_names: Sequence[str] | None\n            The names of the tensor dimensions when generating the shape from an array.\n        "
        if array is None and buffer is None:
            raise ValueError("Must provide one of 'array' or 'buffer'")
        if array is not None and buffer is not None:
            raise ValueError("Can only provide one of 'array' or 'buffer'")
        if buffer is not None and shape is None:
            raise ValueError("If 'buffer' is provided, 'shape' is also required")
        if shape is not None and dim_names is not None:
            raise ValueError("Can only provide one of 'shape' or 'names'")
        from . import TensorBuffer, TensorDimension
        from .tensor_data import _tensor_data__buffer__special_field_converter_override
        if shape is not None:
            resolved_shape = list(shape)
        else:
            resolved_shape = None
        if array is not None:
            array = _to_numpy(array)
            if resolved_shape:
                shape_tuple = tuple((d.size for d in resolved_shape))
                if shape_tuple != array.shape:
                    _send_warning_or_raise(f'Provided array ({array.shape}) does not match shape argument ({shape_tuple}). ' + 'Ignoring shape argument.', 2)
                resolved_shape = None
            if resolved_shape is None:
                if dim_names:
                    if len(array.shape) != len(dim_names):
                        _send_warning_or_raise(f'len(array.shape) = {len(array.shape)} != ' + f'len(dim_names) = {len(dim_names)}. Dropping tensor dimension names.', 2)
                    resolved_shape = [TensorDimension(size, name) for (size, name) in zip(array.shape, dim_names)]
                else:
                    resolved_shape = [TensorDimension(size) for size in array.shape]
        if resolved_shape is not None:
            self.shape = resolved_shape
        else:
            raise ValueError('No shape provided.')
        if buffer is not None:
            self.buffer = _tensor_data__buffer__special_field_converter_override(buffer)
        elif array is not None:
            self.buffer = TensorBuffer(array.flatten())
        if self.buffer.kind != 'jpeg' and self.buffer.kind != 'nv12':
            expected_buffer_size = prod((d.size for d in self.shape))
            if len(self.buffer.inner) != expected_buffer_size:
                raise ValueError(f'Shape and buffer size do not match. {len(self.buffer.inner)} {self.shape}->{expected_buffer_size}')

    @staticmethod
    def native_to_pa_array_override(data: TensorDataArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            i = 10
            return i + 15
        from . import TensorData
        if isinstance(data, collections.abc.Sequence):
            if len(data) > 0:
                if isinstance(data[0], TensorData):
                    if len(data) > 1:
                        raise ValueError('Tensors do not support batches')
                    data = data[0]
        if not isinstance(data, TensorData):
            array = _to_numpy(data)
            data = TensorData(array=array)
        shape = _build_shape_array(data.shape).cast(data_type.field('shape').type)
        buffer = _build_buffer_array(data.buffer)
        return pa.StructArray.from_arrays([shape, buffer], fields=[data_type.field('shape'), data_type.field('buffer')]).cast(data_type)

def _build_shape_array(dims: list[TensorDimension]) -> pa.Array:
    if False:
        for i in range(10):
            print('nop')
    from . import TensorDimensionType
    data_type = TensorDimensionType().storage_type
    array = np.asarray([d.size for d in dims], dtype=np.uint64).flatten()
    names = pa.array([d.name for d in dims], mask=[d is None for d in dims], type=data_type.field('name').type)
    return pa.ListArray.from_arrays(offsets=[0, len(array)], values=pa.StructArray.from_arrays([array, names], fields=[data_type.field('size'), data_type.field('name')]))
DTYPE_MAP: Final[dict[npt.DTypeLike, str]] = {np.uint8: 'U8', np.uint16: 'U16', np.uint32: 'U32', np.uint64: 'U64', np.int8: 'I8', np.int16: 'I16', np.int32: 'I32', np.int64: 'I64', np.float16: 'F16', np.float32: 'F32', np.float64: 'F64'}

def _build_buffer_array(buffer: TensorBufferLike) -> pa.Array:
    if False:
        while True:
            i = 10
    from . import TensorBuffer, TensorBufferType
    data_type = TensorBufferType().storage_type
    kind = None
    if isinstance(buffer, TensorBuffer):
        kind = buffer.kind
        buffer = buffer.inner
    buffer = buffer.flatten()
    data_inner = pa.ListArray.from_arrays(pa.array([0, len(buffer)]), buffer)
    if kind == 'jpeg':
        discriminant = 'JPEG'
    elif kind == 'nv12':
        discriminant = 'NV12'
    else:
        assert buffer.dtype.type in DTYPE_MAP, f'Failed to find {buffer.dtype.type} in f{DTYPE_MAP}'
        discriminant = DTYPE_MAP[buffer.dtype.type]
    return build_dense_union(data_type, discriminant=discriminant, child=data_inner)