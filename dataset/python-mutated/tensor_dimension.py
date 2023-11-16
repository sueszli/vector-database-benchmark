from __future__ import annotations
from typing import Any, Sequence, Union
import pyarrow as pa
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType
from .._converters import str_or_none
__all__ = ['TensorDimension', 'TensorDimensionArrayLike', 'TensorDimensionBatch', 'TensorDimensionLike', 'TensorDimensionType']

@define(init=False)
class TensorDimension:
    """**Datatype**: A single dimension within a multi-dimensional tensor."""

    def __init__(self: Any, size: int, name: str | None=None):
        if False:
            while True:
                i = 10
        '\n        Create a new instance of the TensorDimension datatype.\n\n        Parameters\n        ----------\n        size:\n            The length of this dimension.\n        name:\n            The name of this dimension, e.g. "width", "height", "channel", "batch\', ….\n        '
        self.__attrs_init__(size=size, name=name)
    size: int = field(converter=int)
    name: str | None = field(default=None, converter=str_or_none)
TensorDimensionLike = TensorDimension
TensorDimensionArrayLike = Union[TensorDimension, Sequence[TensorDimensionLike]]

class TensorDimensionType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.datatypes.TensorDimension'

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        pa.ExtensionType.__init__(self, pa.struct([pa.field('size', pa.uint64(), nullable=False, metadata={}), pa.field('name', pa.utf8(), nullable=True, metadata={})]), self._TYPE_NAME)

class TensorDimensionBatch(BaseBatch[TensorDimensionArrayLike]):
    _ARROW_TYPE = TensorDimensionType()

    @staticmethod
    def _native_to_pa_array(data: TensorDimensionArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            while True:
                i = 10
        raise NotImplementedError