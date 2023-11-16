from __future__ import annotations
from typing import Any, Sequence, Union
import pyarrow as pa
from attrs import define, field
from rerun._baseclasses import BaseBatch, BaseExtensionType, ComponentBatchMixin
__all__ = ['AffixFuzzer9', 'AffixFuzzer9ArrayLike', 'AffixFuzzer9Batch', 'AffixFuzzer9Like', 'AffixFuzzer9Type']

@define(init=False)
class AffixFuzzer9:

    def __init__(self: Any, single_string_required: AffixFuzzer9Like):
        if False:
            return 10
        'Create a new instance of the AffixFuzzer9 component.'
        self.__attrs_init__(single_string_required=single_string_required)
    single_string_required: str = field(converter=str)

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return str(self.single_string_required)
AffixFuzzer9Like = AffixFuzzer9
AffixFuzzer9ArrayLike = Union[AffixFuzzer9, Sequence[AffixFuzzer9Like]]

class AffixFuzzer9Type(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.testing.components.AffixFuzzer9'

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        pa.ExtensionType.__init__(self, pa.utf8(), self._TYPE_NAME)

class AffixFuzzer9Batch(BaseBatch[AffixFuzzer9ArrayLike], ComponentBatchMixin):
    _ARROW_TYPE = AffixFuzzer9Type()

    @staticmethod
    def _native_to_pa_array(data: AffixFuzzer9ArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            while True:
                i = 10
        raise NotImplementedError