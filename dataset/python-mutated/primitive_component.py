from __future__ import annotations
from typing import Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from rerun._baseclasses import BaseBatch, BaseExtensionType
__all__ = ['PrimitiveComponent', 'PrimitiveComponentArrayLike', 'PrimitiveComponentBatch', 'PrimitiveComponentLike', 'PrimitiveComponentType']

@define(init=False)
class PrimitiveComponent:

    def __init__(self: Any, value: PrimitiveComponentLike):
        if False:
            while True:
                i = 10
        'Create a new instance of the PrimitiveComponent datatype.'
        self.__attrs_init__(value=value)
    value: int = field(converter=int)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            return 10
        return np.asarray(self.value, dtype=dtype)

    def __int__(self) -> int:
        if False:
            return 10
        return int(self.value)
PrimitiveComponentLike = PrimitiveComponent
PrimitiveComponentArrayLike = Union[PrimitiveComponent, Sequence[PrimitiveComponentLike]]

class PrimitiveComponentType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.testing.datatypes.PrimitiveComponent'

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pa.ExtensionType.__init__(self, pa.uint32(), self._TYPE_NAME)

class PrimitiveComponentBatch(BaseBatch[PrimitiveComponentArrayLike]):
    _ARROW_TYPE = PrimitiveComponentType()

    @staticmethod
    def _native_to_pa_array(data: PrimitiveComponentArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            while True:
                i = 10
        raise NotImplementedError