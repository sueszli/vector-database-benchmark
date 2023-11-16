from __future__ import annotations
from typing import Any, Sequence, Union
import pyarrow as pa
from attrs import define, field
from rerun._baseclasses import BaseBatch, BaseExtensionType, ComponentBatchMixin
__all__ = ['AffixFuzzer13', 'AffixFuzzer13ArrayLike', 'AffixFuzzer13Batch', 'AffixFuzzer13Like', 'AffixFuzzer13Type']

@define(init=False)
class AffixFuzzer13:

    def __init__(self: Any, many_strings_optional: list[str] | None=None):
        if False:
            i = 10
            return i + 15
        'Create a new instance of the AffixFuzzer13 component.'
        self.__attrs_init__(many_strings_optional=many_strings_optional)
    many_strings_optional: list[str] | None = field(default=None)
AffixFuzzer13Like = AffixFuzzer13
AffixFuzzer13ArrayLike = Union[AffixFuzzer13, Sequence[AffixFuzzer13Like]]

class AffixFuzzer13Type(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.testing.components.AffixFuzzer13'

    def __init__(self) -> None:
        if False:
            return 10
        pa.ExtensionType.__init__(self, pa.list_(pa.field('item', pa.utf8(), nullable=False, metadata={})), self._TYPE_NAME)

class AffixFuzzer13Batch(BaseBatch[AffixFuzzer13ArrayLike], ComponentBatchMixin):
    _ARROW_TYPE = AffixFuzzer13Type()

    @staticmethod
    def _native_to_pa_array(data: AffixFuzzer13ArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            while True:
                i = 10
        raise NotImplementedError