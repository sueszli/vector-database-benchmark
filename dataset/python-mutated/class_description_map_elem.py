from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union
import pyarrow as pa
from attrs import define, field
from .. import datatypes
from .._baseclasses import BaseBatch, BaseExtensionType
from .class_description_map_elem_ext import ClassDescriptionMapElemExt
__all__ = ['ClassDescriptionMapElem', 'ClassDescriptionMapElemArrayLike', 'ClassDescriptionMapElemBatch', 'ClassDescriptionMapElemLike', 'ClassDescriptionMapElemType']

def _class_description_map_elem__class_id__special_field_converter_override(x: datatypes.ClassIdLike) -> datatypes.ClassId:
    if False:
        return 10
    if isinstance(x, datatypes.ClassId):
        return x
    else:
        return datatypes.ClassId(x)

@define(init=False)
class ClassDescriptionMapElem(ClassDescriptionMapElemExt):
    """
    **Datatype**: A helper type for mapping class IDs to class descriptions.

    This is internal to the `AnnotationContext` structure.
    """

    def __init__(self: Any, class_id: datatypes.ClassIdLike, class_description: datatypes.ClassDescriptionLike):
        if False:
            return 10
        '\n        Create a new instance of the ClassDescriptionMapElem datatype.\n\n        Parameters\n        ----------\n        class_id:\n            The key: the class ID.\n        class_description:\n            The value: class name, color, etc.\n        '
        self.__attrs_init__(class_id=class_id, class_description=class_description)
    class_id: datatypes.ClassId = field(converter=_class_description_map_elem__class_id__special_field_converter_override)
    class_description: datatypes.ClassDescription = field()
if TYPE_CHECKING:
    ClassDescriptionMapElemLike = Union[ClassDescriptionMapElem, datatypes.ClassDescriptionLike]
else:
    ClassDescriptionMapElemLike = Any
ClassDescriptionMapElemArrayLike = Union[ClassDescriptionMapElem, Sequence[ClassDescriptionMapElemLike]]

class ClassDescriptionMapElemType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.datatypes.ClassDescriptionMapElem'

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pa.ExtensionType.__init__(self, pa.struct([pa.field('class_id', pa.uint16(), nullable=False, metadata={}), pa.field('class_description', pa.struct([pa.field('info', pa.struct([pa.field('id', pa.uint16(), nullable=False, metadata={}), pa.field('label', pa.utf8(), nullable=True, metadata={}), pa.field('color', pa.uint32(), nullable=True, metadata={})]), nullable=False, metadata={}), pa.field('keypoint_annotations', pa.list_(pa.field('item', pa.struct([pa.field('id', pa.uint16(), nullable=False, metadata={}), pa.field('label', pa.utf8(), nullable=True, metadata={}), pa.field('color', pa.uint32(), nullable=True, metadata={})]), nullable=False, metadata={})), nullable=False, metadata={}), pa.field('keypoint_connections', pa.list_(pa.field('item', pa.struct([pa.field('keypoint0', pa.uint16(), nullable=False, metadata={}), pa.field('keypoint1', pa.uint16(), nullable=False, metadata={})]), nullable=False, metadata={})), nullable=False, metadata={})]), nullable=False, metadata={})]), self._TYPE_NAME)

class ClassDescriptionMapElemBatch(BaseBatch[ClassDescriptionMapElemArrayLike]):
    _ARROW_TYPE = ClassDescriptionMapElemType()

    @staticmethod
    def _native_to_pa_array(data: ClassDescriptionMapElemArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            for i in range(10):
                print('nop')
        return ClassDescriptionMapElemExt.native_to_pa_array_override(data, data_type)