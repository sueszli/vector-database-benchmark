from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any, Generator, Generic, List, Optional, TypeVar

from msgspec import Meta
from typing_extensions import Annotated

from litestar.dto import AbstractDTO, DTOField, Mark
from litestar.dto.data_structures import DTOFieldDefinition
from litestar.exceptions import MissingDependencyException
from litestar.types import Empty

try:
    import piccolo  # noqa: F401
except ImportError as e:
    raise MissingDependencyException("piccolo") from e

from piccolo.columns import Column, column_types
from piccolo.table import Table

from litestar.typing import FieldDefinition

T = TypeVar("T", bound=Table)

__all__ = ("PiccoloDTO",)


def _parse_piccolo_type(column: Column, extra: dict[str, Any]) -> FieldDefinition:
    if isinstance(column, (column_types.Decimal, column_types.Numeric)):
        column_type: Any = Decimal
        meta = Meta(extra=extra)
    elif isinstance(column, (column_types.Email, column_types.Varchar)):
        column_type = str
        meta = Meta(max_length=column.length, extra=extra)
    elif isinstance(column, column_types.Array):
        column_type = List[column.base_column.value_type]  # type: ignore
        meta = Meta(extra=extra)
    elif isinstance(column, (column_types.JSON, column_types.JSONB)):
        column_type = str
        meta = Meta(extra={**extra, "format": "json"})
    elif isinstance(column, column_types.Text):
        column_type = str
        meta = Meta(extra={**extra, "format": "text-area"})
    else:
        column_type = column.value_type
        meta = Meta(extra=extra)

    if not column._meta.required:
        column_type = Optional[column_type]

    return FieldDefinition.from_annotation(Annotated[column_type, meta])


def _create_column_extra(column: Column) -> dict[str, Any]:
    extra: dict[str, Any] = {}

    if column._meta.help_text:
        extra["description"] = column._meta.help_text

    if column._meta.get_choices_dict():
        extra["enum"] = column._meta.get_choices_dict()

    return extra


class PiccoloDTO(AbstractDTO[T], Generic[T]):
    @classmethod
    def generate_field_definitions(cls, model_type: type[Table]) -> Generator[DTOFieldDefinition, None, None]:
        for column in model_type._meta.columns:
            yield replace(
                DTOFieldDefinition.from_field_definition(
                    field_definition=_parse_piccolo_type(column, _create_column_extra(column)),
                    dto_field=DTOField(mark=Mark.READ_ONLY if column._meta.primary_key else None),
                    model_name=model_type.__name__,
                    default_factory=Empty,
                ),
                default=Empty if column._meta.required else None,
                name=column._meta.name,
            )

    @classmethod
    def detect_nested_field(cls, field_definition: FieldDefinition) -> bool:
        return field_definition.is_subclass_of(Table)
