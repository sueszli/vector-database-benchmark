from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType, ComponentBatchMixin
from .view_coordinates_ext import ViewCoordinatesExt
__all__ = ['ViewCoordinates', 'ViewCoordinatesArrayLike', 'ViewCoordinatesBatch', 'ViewCoordinatesLike', 'ViewCoordinatesType']

@define(init=False)
class ViewCoordinates(ViewCoordinatesExt):
    """
    **Component**: How we interpret the coordinate system of an entity/space.

    For instance: What is "up"? What does the Z axis mean? Is this right-handed or left-handed?

    The three coordinates are always ordered as [x, y, z].

    For example [Right, Down, Forward] means that the X axis points to the right, the Y axis points
    down, and the Z axis points forward.

    The following constants are used to represent the different directions:
     * Up = 1
     * Down = 2
     * Right = 3
     * Left = 4
     * Forward = 5
     * Back = 6
    """

    def __init__(self: Any, coordinates: ViewCoordinatesLike):
        if False:
            print('Hello World!')
        '\n        Create a new instance of the ViewCoordinates component.\n\n        Parameters\n        ----------\n        coordinates:\n            The directions of the [x, y, z] axes.\n        '
        self.__attrs_init__(coordinates=coordinates)
    coordinates: npt.NDArray[np.uint8] = field(converter=ViewCoordinatesExt.coordinates__field_converter_override)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            return 10
        return np.asarray(self.coordinates, dtype=dtype)
if TYPE_CHECKING:
    ViewCoordinatesLike = Union[ViewCoordinates, npt.ArrayLike]
else:
    ViewCoordinatesLike = Any
ViewCoordinatesArrayLike = Union[ViewCoordinates, Sequence[ViewCoordinatesLike], npt.ArrayLike]

class ViewCoordinatesType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.components.ViewCoordinates'

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        pa.ExtensionType.__init__(self, pa.list_(pa.field('item', pa.uint8(), nullable=False, metadata={}), 3), self._TYPE_NAME)

class ViewCoordinatesBatch(BaseBatch[ViewCoordinatesArrayLike], ComponentBatchMixin):
    _ARROW_TYPE = ViewCoordinatesType()

    @staticmethod
    def _native_to_pa_array(data: ViewCoordinatesArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            for i in range(10):
                print('nop')
        return ViewCoordinatesExt.native_to_pa_array_override(data, data_type)
ViewCoordinatesExt.deferred_patch_class(ViewCoordinates)