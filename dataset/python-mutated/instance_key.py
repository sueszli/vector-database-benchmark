from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType, ComponentBatchMixin
from .instance_key_ext import InstanceKeyExt
__all__ = ['InstanceKey', 'InstanceKeyArrayLike', 'InstanceKeyBatch', 'InstanceKeyLike', 'InstanceKeyType']

@define(init=False)
class InstanceKey(InstanceKeyExt):
    """**Component**: A unique numeric identifier for each individual instance within a batch."""

    def __init__(self: Any, value: InstanceKeyLike):
        if False:
            print('Hello World!')
        'Create a new instance of the InstanceKey component.'
        self.__attrs_init__(value=value)
    value: int = field(converter=int)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            i = 10
            return i + 15
        return np.asarray(self.value, dtype=dtype)

    def __int__(self) -> int:
        if False:
            return 10
        return int(self.value)
if TYPE_CHECKING:
    InstanceKeyLike = Union[InstanceKey, int]
else:
    InstanceKeyLike = Any
InstanceKeyArrayLike = Union[InstanceKey, Sequence[InstanceKeyLike], int, npt.NDArray[np.uint64]]

class InstanceKeyType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.components.InstanceKey'

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        pa.ExtensionType.__init__(self, pa.uint64(), self._TYPE_NAME)

class InstanceKeyBatch(BaseBatch[InstanceKeyArrayLike], ComponentBatchMixin):
    _ARROW_TYPE = InstanceKeyType()

    @staticmethod
    def _native_to_pa_array(data: InstanceKeyArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            for i in range(10):
                print('nop')
        return InstanceKeyExt.native_to_pa_array_override(data, data_type)