from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
import pyarrow as pa
if TYPE_CHECKING:
    from ..datatypes import ClassDescriptionMapElem, ClassDescriptionMapElemLike
    from . import AnnotationContextArrayLike
from ..datatypes.class_description_map_elem_ext import _class_description_map_elem_converter

class AnnotationContextExt:
    """Extension for [AnnotationContext][rerun.components.AnnotationContext]."""

    @staticmethod
    def class_map__field_converter_override(data: Sequence[ClassDescriptionMapElemLike]) -> list[ClassDescriptionMapElem]:
        if False:
            print('Hello World!')
        return [_class_description_map_elem_converter(item) for item in data]

    @staticmethod
    def native_to_pa_array_override(data: AnnotationContextArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            i = 10
            return i + 15
        from ..datatypes import ClassDescription, ClassDescriptionMapElemBatch
        from . import AnnotationContext
        if isinstance(data, ClassDescription):
            data = [data]
        if not isinstance(data, AnnotationContext):
            data = AnnotationContext(class_map=data)
        internal_array = ClassDescriptionMapElemBatch(data.class_map).as_arrow_array().storage
        return pa.ListArray.from_arrays(offsets=[0, len(internal_array)], values=internal_array).cast(data_type)