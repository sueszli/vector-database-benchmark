from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union
import pyarrow as pa
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType
from .utf8_ext import Utf8Ext
__all__ = ['Utf8', 'Utf8ArrayLike', 'Utf8Batch', 'Utf8Like', 'Utf8Type']

@define(init=False)
class Utf8(Utf8Ext):
    """**Datatype**: A string of text, encoded as UTF-8."""

    def __init__(self: Any, value: Utf8Like):
        if False:
            print('Hello World!')
        'Create a new instance of the Utf8 datatype.'
        self.__attrs_init__(value=value)
    value: str = field(converter=str)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return str(self.value)
if TYPE_CHECKING:
    Utf8Like = Union[Utf8, str]
else:
    Utf8Like = Any
Utf8ArrayLike = Union[Utf8, Sequence[Utf8Like], str, Sequence[str]]

class Utf8Type(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.datatypes.Utf8'

    def __init__(self) -> None:
        if False:
            return 10
        pa.ExtensionType.__init__(self, pa.utf8(), self._TYPE_NAME)

class Utf8Batch(BaseBatch[Utf8ArrayLike]):
    _ARROW_TYPE = Utf8Type()

    @staticmethod
    def _native_to_pa_array(data: Utf8ArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            for i in range(10):
                print('nop')
        return Utf8Ext.native_to_pa_array_override(data, data_type)