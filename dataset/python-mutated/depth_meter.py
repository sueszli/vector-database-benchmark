from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType, ComponentBatchMixin
from .depth_meter_ext import DepthMeterExt
__all__ = ['DepthMeter', 'DepthMeterArrayLike', 'DepthMeterBatch', 'DepthMeterLike', 'DepthMeterType']

@define(init=False)
class DepthMeter(DepthMeterExt):
    """**Component**: A component indicating how long a meter is, expressed in native units."""

    def __init__(self: Any, value: DepthMeterLike):
        if False:
            for i in range(10):
                print('nop')
        'Create a new instance of the DepthMeter component.'
        self.__attrs_init__(value=value)
    value: float = field(converter=float)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            while True:
                i = 10
        return np.asarray(self.value, dtype=dtype)

    def __float__(self) -> float:
        if False:
            return 10
        return float(self.value)
if TYPE_CHECKING:
    DepthMeterLike = Union[DepthMeter, float]
else:
    DepthMeterLike = Any
DepthMeterArrayLike = Union[DepthMeter, Sequence[DepthMeterLike], float, npt.NDArray[np.float32]]

class DepthMeterType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.components.DepthMeter'

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        pa.ExtensionType.__init__(self, pa.float32(), self._TYPE_NAME)

class DepthMeterBatch(BaseBatch[DepthMeterArrayLike], ComponentBatchMixin):
    _ARROW_TYPE = DepthMeterType()

    @staticmethod
    def _native_to_pa_array(data: DepthMeterArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            return 10
        return DepthMeterExt.native_to_pa_array_override(data, data_type)