from __future__ import annotations
from typing import Any, Sequence, Union
import numpy as np
import numpy.typing as npt
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType
from .._converters import to_np_uint8
__all__ = ['EntityPropertiesComponent', 'EntityPropertiesComponentArrayLike', 'EntityPropertiesComponentBatch', 'EntityPropertiesComponentLike', 'EntityPropertiesComponentType']

@define(init=False)
class EntityPropertiesComponent:
    """
    **Blueprint**: The configurable set of overridable properties.

    Unstable. Used for the ongoing blueprint experimentations.
    """

    def __init__(self: Any, props: EntityPropertiesComponentLike):
        if False:
            return 10
        'Create a new instance of the EntityPropertiesComponent blueprint.'
        self.__attrs_init__(props=props)
    props: npt.NDArray[np.uint8] = field(converter=to_np_uint8)

    def __array__(self, dtype: npt.DTypeLike=None) -> npt.NDArray[Any]:
        if False:
            while True:
                i = 10
        return np.asarray(self.props, dtype=dtype)
EntityPropertiesComponentLike = EntityPropertiesComponent
EntityPropertiesComponentArrayLike = Union[EntityPropertiesComponent, Sequence[EntityPropertiesComponentLike]]

class EntityPropertiesComponentType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.blueprint.EntityPropertiesComponent'

class EntityPropertiesComponentBatch(BaseBatch[EntityPropertiesComponentArrayLike]):
    _ARROW_TYPE = EntityPropertiesComponentType()