from __future__ import annotations
from typing import Any, Sequence, Union
import numpy as np
import numpy.typing as npt
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType
from .._converters import to_np_uint8
__all__ = ['SpaceViewMaximized', 'SpaceViewMaximizedArrayLike', 'SpaceViewMaximizedBatch', 'SpaceViewMaximizedLike', 'SpaceViewMaximizedType']

@define(init=False)
class SpaceViewMaximized:
    """
    **Blueprint**: Whether a space view is maximized.

    Unstable. Used for the ongoing blueprint experimentations.
    """

    def __init__(self: Any, id: npt.ArrayLike | None=None):
        if False:
            print('Hello World!')
        'Create a new instance of the SpaceViewMaximized blueprint.'
        self.__attrs_init__(id=id)
    id: npt.NDArray[np.uint8] | None = field(default=None, converter=to_np_uint8)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            i = 10
            return i + 15
        return np.asarray(self.id, dtype=dtype)
SpaceViewMaximizedLike = SpaceViewMaximized
SpaceViewMaximizedArrayLike = Union[SpaceViewMaximized, Sequence[SpaceViewMaximizedLike]]

class SpaceViewMaximizedType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.blueprint.SpaceViewMaximized'

class SpaceViewMaximizedBatch(BaseBatch[SpaceViewMaximizedArrayLike]):
    _ARROW_TYPE = SpaceViewMaximizedType()