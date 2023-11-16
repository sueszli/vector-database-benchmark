from __future__ import annotations
from typing import Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from rerun._baseclasses import BaseBatch, BaseExtensionType, ComponentBatchMixin
from rerun._converters import float_or_none
__all__ = ['AffixFuzzer8', 'AffixFuzzer8ArrayLike', 'AffixFuzzer8Batch', 'AffixFuzzer8Like', 'AffixFuzzer8Type']

@define(init=False)
class AffixFuzzer8:

    def __init__(self: Any, single_float_optional: float | None=None):
        if False:
            return 10
        'Create a new instance of the AffixFuzzer8 component.'
        self.__attrs_init__(single_float_optional=single_float_optional)
    single_float_optional: float | None = field(default=None, converter=float_or_none)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            i = 10
            return i + 15
        return np.asarray(self.single_float_optional, dtype=dtype)
AffixFuzzer8Like = AffixFuzzer8
AffixFuzzer8ArrayLike = Union[AffixFuzzer8, Sequence[AffixFuzzer8Like]]

class AffixFuzzer8Type(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.testing.components.AffixFuzzer8'

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        pa.ExtensionType.__init__(self, pa.float32(), self._TYPE_NAME)

class AffixFuzzer8Batch(BaseBatch[AffixFuzzer8ArrayLike], ComponentBatchMixin):
    _ARROW_TYPE = AffixFuzzer8Type()

    @staticmethod
    def _native_to_pa_array(data: AffixFuzzer8ArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError