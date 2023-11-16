from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType
from .class_id_ext import ClassIdExt
__all__ = ['ClassId', 'ClassIdArrayLike', 'ClassIdBatch', 'ClassIdLike', 'ClassIdType']

@define(init=False)
class ClassId(ClassIdExt):
    """**Datatype**: A 16-bit ID representing a type of semantic class."""

    def __init__(self: Any, id: ClassIdLike):
        if False:
            i = 10
            return i + 15
        'Create a new instance of the ClassId datatype.'
        self.__attrs_init__(id=id)
    id: int = field(converter=int)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            return 10
        return np.asarray(self.id, dtype=dtype)

    def __int__(self) -> int:
        if False:
            i = 10
            return i + 15
        return int(self.id)
if TYPE_CHECKING:
    ClassIdLike = Union[ClassId, int]
else:
    ClassIdLike = Any
ClassIdArrayLike = Union[ClassId, Sequence[ClassIdLike], int, npt.ArrayLike]

class ClassIdType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.datatypes.ClassId'

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        pa.ExtensionType.__init__(self, pa.uint16(), self._TYPE_NAME)

class ClassIdBatch(BaseBatch[ClassIdArrayLike]):
    _ARROW_TYPE = ClassIdType()

    @staticmethod
    def _native_to_pa_array(data: ClassIdArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            return 10
        return ClassIdExt.native_to_pa_array_override(data, data_type)