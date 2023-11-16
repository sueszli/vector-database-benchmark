from __future__ import annotations
from typing import Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType
__all__ = ['Float32', 'Float32ArrayLike', 'Float32Batch', 'Float32Like', 'Float32Type']

@define(init=False)
class Float32:
    """**Datatype**: A single-precision 32-bit IEEE 754 floating point number."""

    def __init__(self: Any, value: Float32Like):
        if False:
            print('Hello World!')
        'Create a new instance of the Float32 datatype.'
        self.__attrs_init__(value=value)
    value: float = field(converter=float)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            print('Hello World!')
        return np.asarray(self.value, dtype=dtype)

    def __float__(self) -> float:
        if False:
            while True:
                i = 10
        return float(self.value)
Float32Like = Float32
Float32ArrayLike = Union[Float32, Sequence[Float32Like]]

class Float32Type(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.datatypes.Float32'

    def __init__(self) -> None:
        if False:
            return 10
        pa.ExtensionType.__init__(self, pa.float32(), self._TYPE_NAME)

class Float32Batch(BaseBatch[Float32ArrayLike]):
    _ARROW_TYPE = Float32Type()

    @staticmethod
    def _native_to_pa_array(data: Float32ArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            return 10
        raise NotImplementedError