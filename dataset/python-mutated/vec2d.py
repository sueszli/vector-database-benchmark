from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType
from .._converters import to_np_float32
from .vec2d_ext import Vec2DExt
__all__ = ['Vec2D', 'Vec2DArrayLike', 'Vec2DBatch', 'Vec2DLike', 'Vec2DType']

@define(init=False)
class Vec2D(Vec2DExt):
    """**Datatype**: A vector in 2D space."""

    def __init__(self: Any, xy: Vec2DLike):
        if False:
            while True:
                i = 10
        'Create a new instance of the Vec2D datatype.'
        self.__attrs_init__(xy=xy)
    xy: npt.NDArray[np.float32] = field(converter=to_np_float32)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            for i in range(10):
                print('nop')
        return np.asarray(self.xy, dtype=dtype)
if TYPE_CHECKING:
    Vec2DLike = Union[Vec2D, npt.NDArray[Any], npt.ArrayLike, Sequence[float]]
else:
    Vec2DLike = Any
Vec2DArrayLike = Union[Vec2D, Sequence[Vec2DLike], npt.NDArray[Any], npt.ArrayLike, Sequence[Sequence[float]], Sequence[float]]

class Vec2DType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.datatypes.Vec2D'

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        pa.ExtensionType.__init__(self, pa.list_(pa.field('item', pa.float32(), nullable=False, metadata={}), 2), self._TYPE_NAME)

class Vec2DBatch(BaseBatch[Vec2DArrayLike]):
    _ARROW_TYPE = Vec2DType()

    @staticmethod
    def _native_to_pa_array(data: Vec2DArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            i = 10
            return i + 15
        return Vec2DExt.native_to_pa_array_override(data, data_type)