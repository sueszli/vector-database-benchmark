from __future__ import annotations
from typing import Any, Sequence, Union
import pyarrow as pa
from attrs import define, field
from rerun._baseclasses import BaseBatch, BaseExtensionType
__all__ = ['StringComponent', 'StringComponentArrayLike', 'StringComponentBatch', 'StringComponentLike', 'StringComponentType']

@define(init=False)
class StringComponent:

    def __init__(self: Any, value: StringComponentLike):
        if False:
            return 10
        'Create a new instance of the StringComponent datatype.'
        self.__attrs_init__(value=value)
    value: str = field(converter=str)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return str(self.value)
StringComponentLike = StringComponent
StringComponentArrayLike = Union[StringComponent, Sequence[StringComponentLike]]

class StringComponentType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.testing.datatypes.StringComponent'

    def __init__(self) -> None:
        if False:
            return 10
        pa.ExtensionType.__init__(self, pa.utf8(), self._TYPE_NAME)

class StringComponentBatch(BaseBatch[StringComponentArrayLike]):
    _ARROW_TYPE = StringComponentType()

    @staticmethod
    def _native_to_pa_array(data: StringComponentArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            return 10
        raise NotImplementedError