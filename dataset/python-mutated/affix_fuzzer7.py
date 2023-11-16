from __future__ import annotations
from typing import Any, Sequence, Union
import pyarrow as pa
from attrs import define, field
from rerun._baseclasses import BaseBatch, BaseExtensionType, ComponentBatchMixin
from .. import datatypes
__all__ = ['AffixFuzzer7', 'AffixFuzzer7ArrayLike', 'AffixFuzzer7Batch', 'AffixFuzzer7Like', 'AffixFuzzer7Type']

@define(init=False)
class AffixFuzzer7:

    def __init__(self: Any, many_optional: datatypes.AffixFuzzer1ArrayLike | None=None):
        if False:
            print('Hello World!')
        'Create a new instance of the AffixFuzzer7 component.'
        self.__attrs_init__(many_optional=many_optional)
    many_optional: list[datatypes.AffixFuzzer1] | None = field(default=None)
AffixFuzzer7Like = AffixFuzzer7
AffixFuzzer7ArrayLike = Union[AffixFuzzer7, Sequence[AffixFuzzer7Like]]

class AffixFuzzer7Type(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.testing.components.AffixFuzzer7'

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        pa.ExtensionType.__init__(self, pa.list_(pa.field('item', pa.struct([pa.field('single_float_optional', pa.float32(), nullable=True, metadata={}), pa.field('single_string_required', pa.utf8(), nullable=False, metadata={}), pa.field('single_string_optional', pa.utf8(), nullable=True, metadata={}), pa.field('many_floats_optional', pa.list_(pa.field('item', pa.float32(), nullable=False, metadata={})), nullable=True, metadata={}), pa.field('many_strings_required', pa.list_(pa.field('item', pa.utf8(), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('many_strings_optional', pa.list_(pa.field('item', pa.utf8(), nullable=False, metadata={})), nullable=True, metadata={}), pa.field('flattened_scalar', pa.float32(), nullable=False, metadata={}), pa.field('almost_flattened_scalar', pa.struct([pa.field('value', pa.float32(), nullable=False, metadata={})]), nullable=False, metadata={}), pa.field('from_parent', pa.bool_(), nullable=True, metadata={})]), nullable=False, metadata={})), self._TYPE_NAME)

class AffixFuzzer7Batch(BaseBatch[AffixFuzzer7ArrayLike], ComponentBatchMixin):
    _ARROW_TYPE = AffixFuzzer7Type()

    @staticmethod
    def _native_to_pa_array(data: AffixFuzzer7ArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError