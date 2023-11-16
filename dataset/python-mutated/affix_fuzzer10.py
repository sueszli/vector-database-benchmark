from __future__ import annotations
from typing import Any, Sequence, Union
import pyarrow as pa
from attrs import define, field
from rerun._baseclasses import BaseBatch, BaseExtensionType, ComponentBatchMixin
from rerun._converters import str_or_none
__all__ = ['AffixFuzzer10', 'AffixFuzzer10ArrayLike', 'AffixFuzzer10Batch', 'AffixFuzzer10Like', 'AffixFuzzer10Type']

@define(init=False)
class AffixFuzzer10:

    def __init__(self: Any, single_string_optional: str | None=None):
        if False:
            print('Hello World!')
        'Create a new instance of the AffixFuzzer10 component.'
        self.__attrs_init__(single_string_optional=single_string_optional)
    single_string_optional: str | None = field(default=None, converter=str_or_none)
AffixFuzzer10Like = AffixFuzzer10
AffixFuzzer10ArrayLike = Union[AffixFuzzer10, Sequence[AffixFuzzer10Like]]

class AffixFuzzer10Type(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.testing.components.AffixFuzzer10'

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        pa.ExtensionType.__init__(self, pa.utf8(), self._TYPE_NAME)

class AffixFuzzer10Batch(BaseBatch[AffixFuzzer10ArrayLike], ComponentBatchMixin):
    _ARROW_TYPE = AffixFuzzer10Type()

    @staticmethod
    def _native_to_pa_array(data: AffixFuzzer10ArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            print('Hello World!')
        raise NotImplementedError