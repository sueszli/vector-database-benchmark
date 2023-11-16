from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType
from .tensor_buffer_ext import TensorBufferExt
__all__ = ['TensorBuffer', 'TensorBufferArrayLike', 'TensorBufferBatch', 'TensorBufferLike', 'TensorBufferType']

@define
class TensorBuffer(TensorBufferExt):
    """
    **Datatype**: The underlying storage for a `Tensor`.

    Tensor elements are stored in a contiguous buffer of a single type.
    """
    inner: Union[npt.NDArray[np.float16], npt.NDArray[np.float32], npt.NDArray[np.float64], npt.NDArray[np.int16], npt.NDArray[np.int32], npt.NDArray[np.int64], npt.NDArray[np.int8], npt.NDArray[np.uint16], npt.NDArray[np.uint32], npt.NDArray[np.uint64], npt.NDArray[np.uint8]] = field(converter=TensorBufferExt.inner__field_converter_override)
    '\n    U8 (npt.NDArray[np.uint8]):\n\n    U16 (npt.NDArray[np.uint16]):\n\n    U32 (npt.NDArray[np.uint32]):\n\n    U64 (npt.NDArray[np.uint64]):\n\n    I8 (npt.NDArray[np.int8]):\n\n    I16 (npt.NDArray[np.int16]):\n\n    I32 (npt.NDArray[np.int32]):\n\n    I64 (npt.NDArray[np.int64]):\n\n    F16 (npt.NDArray[np.float16]):\n\n    F32 (npt.NDArray[np.float32]):\n\n    F64 (npt.NDArray[np.float64]):\n\n    JPEG (npt.NDArray[np.uint8]):\n\n    NV12 (npt.NDArray[np.uint8]):\n    '
    kind: Literal['u8', 'u16', 'u32', 'u64', 'i8', 'i16', 'i32', 'i64', 'f16', 'f32', 'f64', 'jpeg', 'nv12'] = field(default='u8')
if TYPE_CHECKING:
    TensorBufferLike = Union[TensorBuffer, npt.NDArray[np.float16], npt.NDArray[np.float32], npt.NDArray[np.float64], npt.NDArray[np.int16], npt.NDArray[np.int32], npt.NDArray[np.int64], npt.NDArray[np.int8], npt.NDArray[np.uint16], npt.NDArray[np.uint32], npt.NDArray[np.uint64], npt.NDArray[np.uint8]]
    TensorBufferArrayLike = Union[TensorBuffer, npt.NDArray[np.float16], npt.NDArray[np.float32], npt.NDArray[np.float64], npt.NDArray[np.int16], npt.NDArray[np.int32], npt.NDArray[np.int64], npt.NDArray[np.int8], npt.NDArray[np.uint16], npt.NDArray[np.uint32], npt.NDArray[np.uint64], npt.NDArray[np.uint8], Sequence[TensorBufferLike]]
else:
    TensorBufferLike = Any
    TensorBufferArrayLike = Any

class TensorBufferType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.datatypes.TensorBuffer'

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pa.ExtensionType.__init__(self, pa.dense_union([pa.field('_null_markers', pa.null(), nullable=True, metadata={}), pa.field('U8', pa.list_(pa.field('item', pa.uint8(), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('U16', pa.list_(pa.field('item', pa.uint16(), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('U32', pa.list_(pa.field('item', pa.uint32(), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('U64', pa.list_(pa.field('item', pa.uint64(), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('I8', pa.list_(pa.field('item', pa.int8(), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('I16', pa.list_(pa.field('item', pa.int16(), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('I32', pa.list_(pa.field('item', pa.int32(), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('I64', pa.list_(pa.field('item', pa.int64(), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('F16', pa.list_(pa.field('item', pa.float16(), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('F32', pa.list_(pa.field('item', pa.float32(), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('F64', pa.list_(pa.field('item', pa.float64(), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('JPEG', pa.list_(pa.field('item', pa.uint8(), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('NV12', pa.list_(pa.field('item', pa.uint8(), nullable=False, metadata={})), nullable=False, metadata={})]), self._TYPE_NAME)

class TensorBufferBatch(BaseBatch[TensorBufferArrayLike]):
    _ARROW_TYPE = TensorBufferType()

    @staticmethod
    def _native_to_pa_array(data: TensorBufferArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            print('Hello World!')
        raise NotImplementedError