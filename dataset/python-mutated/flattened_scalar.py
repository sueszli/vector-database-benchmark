from __future__ import annotations
from typing import Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from rerun._baseclasses import BaseBatch, BaseExtensionType
__all__ = ['FlattenedScalar', 'FlattenedScalarArrayLike', 'FlattenedScalarBatch', 'FlattenedScalarLike', 'FlattenedScalarType']

@define(init=False)
class FlattenedScalar:

    def __init__(self: Any, value: FlattenedScalarLike):
        if False:
            while True:
                i = 10
        'Create a new instance of the FlattenedScalar datatype.'
        self.__attrs_init__(value=value)
    value: float = field(converter=float)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            i = 10
            return i + 15
        return np.asarray(self.value, dtype=dtype)

    def __float__(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        return float(self.value)
FlattenedScalarLike = FlattenedScalar
FlattenedScalarArrayLike = Union[FlattenedScalar, Sequence[FlattenedScalarLike]]

class FlattenedScalarType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.testing.datatypes.FlattenedScalar'

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        pa.ExtensionType.__init__(self, pa.struct([pa.field('value', pa.float32(), nullable=False, metadata={})]), self._TYPE_NAME)

class FlattenedScalarBatch(BaseBatch[FlattenedScalarArrayLike]):
    _ARROW_TYPE = FlattenedScalarType()

    @staticmethod
    def _native_to_pa_array(data: FlattenedScalarArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            return 10
        raise NotImplementedError