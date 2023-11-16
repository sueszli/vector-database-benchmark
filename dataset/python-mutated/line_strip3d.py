from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from .. import datatypes
from .._baseclasses import BaseBatch, BaseExtensionType, ComponentBatchMixin
from .line_strip3d_ext import LineStrip3DExt
__all__ = ['LineStrip3D', 'LineStrip3DArrayLike', 'LineStrip3DBatch', 'LineStrip3DLike', 'LineStrip3DType']

@define(init=False)
class LineStrip3D(LineStrip3DExt):
    """
    **Component**: A line strip in 3D space.

    A line strip is a list of points connected by line segments. It can be used to draw
    approximations of smooth curves.

    The points will be connected in order, like so:
    ```text
           2------3     5
          /        \\   /
    0----1          \\ /
                     4
    ```
    """

    def __init__(self: Any, points: LineStrip3DLike):
        if False:
            return 10
        'Create a new instance of the LineStrip3D component.'
        self.__attrs_init__(points=points)
    points: list[datatypes.Vec3D] = field()
if TYPE_CHECKING:
    LineStrip3DLike = Union[LineStrip3D, datatypes.Vec3DArrayLike, npt.NDArray[np.float32]]
else:
    LineStrip3DLike = Any
LineStrip3DArrayLike = Union[LineStrip3D, Sequence[LineStrip3DLike], npt.NDArray[np.float32]]

class LineStrip3DType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.components.LineStrip3D'

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pa.ExtensionType.__init__(self, pa.list_(pa.field('item', pa.list_(pa.field('item', pa.float32(), nullable=False, metadata={}), 3), nullable=False, metadata={})), self._TYPE_NAME)

class LineStrip3DBatch(BaseBatch[LineStrip3DArrayLike], ComponentBatchMixin):
    _ARROW_TYPE = LineStrip3DType()

    @staticmethod
    def _native_to_pa_array(data: LineStrip3DArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            print('Hello World!')
        return LineStrip3DExt.native_to_pa_array_override(data, data_type)