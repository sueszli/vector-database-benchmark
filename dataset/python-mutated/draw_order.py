from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType, ComponentBatchMixin
from .draw_order_ext import DrawOrderExt
__all__ = ['DrawOrder', 'DrawOrderArrayLike', 'DrawOrderBatch', 'DrawOrderLike', 'DrawOrderType']

@define(init=False)
class DrawOrder(DrawOrderExt):
    """
    **Component**: Draw order used for the display order of 2D elements.

    Higher values are drawn on top of lower values.
    An entity can have only a single draw order component.
    Within an entity draw order is governed by the order of the components.

    Draw order for entities with the same draw order is generally undefined.
    """

    def __init__(self: Any, value: DrawOrderLike):
        if False:
            for i in range(10):
                print('nop')
        'Create a new instance of the DrawOrder component.'
        self.__attrs_init__(value=value)
    value: float = field(converter=float)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            return 10
        return np.asarray(self.value, dtype=dtype)

    def __float__(self) -> float:
        if False:
            while True:
                i = 10
        return float(self.value)
if TYPE_CHECKING:
    DrawOrderLike = Union[DrawOrder, float]
else:
    DrawOrderLike = Any
DrawOrderArrayLike = Union[DrawOrder, Sequence[DrawOrderLike], float, npt.NDArray[np.float32]]

class DrawOrderType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.components.DrawOrder'

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        pa.ExtensionType.__init__(self, pa.float32(), self._TYPE_NAME)

class DrawOrderBatch(BaseBatch[DrawOrderArrayLike], ComponentBatchMixin):
    _ARROW_TYPE = DrawOrderType()

    @staticmethod
    def _native_to_pa_array(data: DrawOrderArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            return 10
        return DrawOrderExt.native_to_pa_array_override(data, data_type)