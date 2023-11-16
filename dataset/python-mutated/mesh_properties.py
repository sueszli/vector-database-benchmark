from __future__ import annotations
from typing import Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType
from .._converters import to_np_uint32
from .mesh_properties_ext import MeshPropertiesExt
__all__ = ['MeshProperties', 'MeshPropertiesArrayLike', 'MeshPropertiesBatch', 'MeshPropertiesLike', 'MeshPropertiesType']

@define(init=False)
class MeshProperties(MeshPropertiesExt):
    """**Datatype**: Optional triangle indices for a mesh."""

    def __init__(self: Any, indices: npt.ArrayLike | None=None):
        if False:
            while True:
                i = 10
        "\n        Create a new instance of the MeshProperties datatype.\n\n        Parameters\n        ----------\n        indices:\n            A flattened array of vertex indices that describe the mesh's triangles.\n\n            Its length must be divisible by 3.\n        "
        self.__attrs_init__(indices=indices)
    indices: npt.NDArray[np.uint32] | None = field(default=None, converter=to_np_uint32)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            for i in range(10):
                print('nop')
        return np.asarray(self.indices, dtype=dtype)
MeshPropertiesLike = MeshProperties
MeshPropertiesArrayLike = Union[MeshProperties, Sequence[MeshPropertiesLike]]

class MeshPropertiesType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.datatypes.MeshProperties'

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        pa.ExtensionType.__init__(self, pa.struct([pa.field('indices', pa.list_(pa.field('item', pa.uint32(), nullable=False, metadata={})), nullable=True, metadata={})]), self._TYPE_NAME)

class MeshPropertiesBatch(BaseBatch[MeshPropertiesArrayLike]):
    _ARROW_TYPE = MeshPropertiesType()

    @staticmethod
    def _native_to_pa_array(data: MeshPropertiesArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            i = 10
            return i + 15
        return MeshPropertiesExt.native_to_pa_array_override(data, data_type)