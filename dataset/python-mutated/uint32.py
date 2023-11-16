from __future__ import annotations
from typing import Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType
__all__ = ['UInt32', 'UInt32ArrayLike', 'UInt32Batch', 'UInt32Like', 'UInt32Type']

@define(init=False)
class UInt32:
    """**Datatype**: A 32bit unsigned integer."""

    def __init__(self: Any, value: UInt32Like):
        if False:
            i = 10
            return i + 15
        'Create a new instance of the UInt32 datatype.'
        self.__attrs_init__(value=value)
    value: int = field(converter=int)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            for i in range(10):
                print('nop')
        return np.asarray(self.value, dtype=dtype)

    def __int__(self) -> int:
        if False:
            i = 10
            return i + 15
        return int(self.value)
UInt32Like = UInt32
UInt32ArrayLike = Union[UInt32, Sequence[UInt32Like]]

class UInt32Type(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.datatypes.UInt32'

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        pa.ExtensionType.__init__(self, pa.uint32(), self._TYPE_NAME)

class UInt32Batch(BaseBatch[UInt32ArrayLike]):
    _ARROW_TYPE = UInt32Type()

    @staticmethod
    def _native_to_pa_array(data: UInt32ArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError