from __future__ import annotations
from typing import Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from rerun._baseclasses import BaseBatch, BaseExtensionType, ComponentBatchMixin
from rerun._converters import to_np_float32
__all__ = ['AffixFuzzer11', 'AffixFuzzer11ArrayLike', 'AffixFuzzer11Batch', 'AffixFuzzer11Like', 'AffixFuzzer11Type']

@define(init=False)
class AffixFuzzer11:

    def __init__(self: Any, many_floats_optional: npt.ArrayLike | None=None):
        if False:
            while True:
                i = 10
        'Create a new instance of the AffixFuzzer11 component.'
        self.__attrs_init__(many_floats_optional=many_floats_optional)
    many_floats_optional: npt.NDArray[np.float32] | None = field(default=None, converter=to_np_float32)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            for i in range(10):
                print('nop')
        return np.asarray(self.many_floats_optional, dtype=dtype)
AffixFuzzer11Like = AffixFuzzer11
AffixFuzzer11ArrayLike = Union[AffixFuzzer11, Sequence[AffixFuzzer11Like]]

class AffixFuzzer11Type(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.testing.components.AffixFuzzer11'

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        pa.ExtensionType.__init__(self, pa.list_(pa.field('item', pa.float32(), nullable=False, metadata={})), self._TYPE_NAME)

class AffixFuzzer11Batch(BaseBatch[AffixFuzzer11ArrayLike], ComponentBatchMixin):
    _ARROW_TYPE = AffixFuzzer11Type()

    @staticmethod
    def _native_to_pa_array(data: AffixFuzzer11ArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            return 10
        raise NotImplementedError