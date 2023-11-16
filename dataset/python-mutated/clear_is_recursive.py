from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType, ComponentBatchMixin
from .clear_is_recursive_ext import ClearIsRecursiveExt
__all__ = ['ClearIsRecursive', 'ClearIsRecursiveArrayLike', 'ClearIsRecursiveBatch', 'ClearIsRecursiveLike', 'ClearIsRecursiveType']

@define(init=False)
class ClearIsRecursive(ClearIsRecursiveExt):
    """**Component**: Configures how a clear operation should behave - recursive or not."""

    def __init__(self: Any, recursive: ClearIsRecursiveLike):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new instance of the ClearIsRecursive component.\n\n        Parameters\n        ----------\n        recursive:\n            If true, also clears all recursive children entities.\n        '
        self.__attrs_init__(recursive=recursive)
    recursive: bool = field(converter=bool)
if TYPE_CHECKING:
    ClearIsRecursiveLike = Union[ClearIsRecursive, bool]
else:
    ClearIsRecursiveLike = Any
ClearIsRecursiveArrayLike = Union[ClearIsRecursive, Sequence[ClearIsRecursiveLike], bool, npt.NDArray[np.bool_]]

class ClearIsRecursiveType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.components.ClearIsRecursive'

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        pa.ExtensionType.__init__(self, pa.bool_(), self._TYPE_NAME)

class ClearIsRecursiveBatch(BaseBatch[ClearIsRecursiveArrayLike], ComponentBatchMixin):
    _ARROW_TYPE = ClearIsRecursiveType()

    @staticmethod
    def _native_to_pa_array(data: ClearIsRecursiveArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            return 10
        return ClearIsRecursiveExt.native_to_pa_array_override(data, data_type)