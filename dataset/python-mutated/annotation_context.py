from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union
import pyarrow as pa
from attrs import define, field
from .. import datatypes
from .._baseclasses import BaseBatch, BaseExtensionType, ComponentBatchMixin
from .annotation_context_ext import AnnotationContextExt
__all__ = ['AnnotationContext', 'AnnotationContextArrayLike', 'AnnotationContextBatch', 'AnnotationContextLike', 'AnnotationContextType']

@define(init=False)
class AnnotationContext(AnnotationContextExt):
    """
    **Component**: The `AnnotationContext` provides additional information on how to display entities.

    Entities can use `ClassId`s and `KeypointId`s to provide annotations, and
    the labels and colors will be looked up in the appropriate
    `AnnotationContext`. We use the *first* annotation context we find in the
    path-hierarchy when searching up through the ancestors of a given entity
    path.
    """

    def __init__(self: Any, class_map: AnnotationContextLike):
        if False:
            return 10
        '\n        Create a new instance of the AnnotationContext component.\n\n        Parameters\n        ----------\n        class_map:\n            List of class descriptions, mapping class indices to class names, colors etc.\n        '
        self.__attrs_init__(class_map=class_map)
    class_map: list[datatypes.ClassDescriptionMapElem] = field(converter=AnnotationContextExt.class_map__field_converter_override)
if TYPE_CHECKING:
    AnnotationContextLike = Union[AnnotationContext, datatypes.ClassDescriptionArrayLike, Sequence[datatypes.ClassDescriptionMapElemLike]]
else:
    AnnotationContextLike = Any
AnnotationContextArrayLike = Union[AnnotationContext, Sequence[AnnotationContextLike]]

class AnnotationContextType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.components.AnnotationContext'

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        pa.ExtensionType.__init__(self, pa.list_(pa.field('item', pa.struct([pa.field('class_id', pa.uint16(), nullable=False, metadata={}), pa.field('class_description', pa.struct([pa.field('info', pa.struct([pa.field('id', pa.uint16(), nullable=False, metadata={}), pa.field('label', pa.utf8(), nullable=True, metadata={}), pa.field('color', pa.uint32(), nullable=True, metadata={})]), nullable=False, metadata={}), pa.field('keypoint_annotations', pa.list_(pa.field('item', pa.struct([pa.field('id', pa.uint16(), nullable=False, metadata={}), pa.field('label', pa.utf8(), nullable=True, metadata={}), pa.field('color', pa.uint32(), nullable=True, metadata={})]), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('keypoint_connections', pa.list_(pa.field('item', pa.struct([pa.field('keypoint0', pa.uint16(), nullable=False, metadata={}), pa.field('keypoint1', pa.uint16(), nullable=False, metadata={})]), nullable=False, metadata={})), nullable=False, metadata={})]), nullable=False, metadata={})]), nullable=False, metadata={})), self._TYPE_NAME)

class AnnotationContextBatch(BaseBatch[AnnotationContextArrayLike], ComponentBatchMixin):
    _ARROW_TYPE = AnnotationContextType()

    @staticmethod
    def _native_to_pa_array(data: AnnotationContextArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            i = 10
            return i + 15
        return AnnotationContextExt.native_to_pa_array_override(data, data_type)