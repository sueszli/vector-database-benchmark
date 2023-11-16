from __future__ import annotations
from typing import Any, List, Mapping, TypeVar, cast
from datetime import date, datetime
from typing_extensions import Literal, get_args, override, get_type_hints
import pydantic
from ._utils import is_list, is_mapping, is_list_type, is_union_type, extract_type_arg, is_required_type, is_annotated_type, strip_annotated_type
from .._compat import model_dump, is_typeddict
_T = TypeVar('_T')
PropertyFormat = Literal['iso8601', 'custom']

class PropertyInfo:
    """Metadata class to be used in Annotated types to provide information about a given type.

    For example:

    class MyParams(TypedDict):
        account_holder_name: Annotated[str, PropertyInfo(alias='accountHolderName')]

    This means that {'account_holder_name': 'Robert'} will be transformed to {'accountHolderName': 'Robert'} before being sent to the API.
    """
    alias: str | None
    format: PropertyFormat | None
    format_template: str | None

    def __init__(self, *, alias: str | None=None, format: PropertyFormat | None=None, format_template: str | None=None) -> None:
        if False:
            i = 10
            return i + 15
        self.alias = alias
        self.format = format
        self.format_template = format_template

    @override
    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f"{self.__class__.__name__}(alias='{self.alias}', format={self.format}, format_template='{self.format_template}')"

def maybe_transform(data: Mapping[str, object] | List[Any] | None, expected_type: object) -> Any | None:
    if False:
        for i in range(10):
            print('nop')
    'Wrapper over `transform()` that allows `None` to be passed.\n\n    See `transform()` for more details.\n    '
    if data is None:
        return None
    return transform(data, expected_type)

def transform(data: _T, expected_type: object) -> _T:
    if False:
        i = 10
        return i + 15
    "Transform dictionaries based off of type information from the given type, for example:\n\n    ```py\n    class Params(TypedDict, total=False):\n        card_id: Required[Annotated[str, PropertyInfo(alias='cardID')]]\n\n    transformed = transform({'card_id': '<my card ID>'}, Params)\n    # {'cardID': '<my card ID>'}\n    ```\n\n    Any keys / data that does not have type information given will be included as is.\n\n    It should be noted that the transformations that this function does are not represented in the type system.\n    "
    transformed = _transform_recursive(data, annotation=cast(type, expected_type))
    return cast(_T, transformed)

def _get_annotated_type(type_: type) -> type | None:
    if False:
        i = 10
        return i + 15
    'If the given type is an `Annotated` type then it is returned, if not `None` is returned.\n\n    This also unwraps the type when applicable, e.g. `Required[Annotated[T, ...]]`\n    '
    if is_required_type(type_):
        type_ = get_args(type_)[0]
    if is_annotated_type(type_):
        return type_
    return None

def _maybe_transform_key(key: str, type_: type) -> str:
    if False:
        while True:
            i = 10
    'Transform the given `data` based on the annotations provided in `type_`.\n\n    Note: this function only looks at `Annotated` types that contain `PropertInfo` metadata.\n    '
    annotated_type = _get_annotated_type(type_)
    if annotated_type is None:
        return key
    annotations = get_args(annotated_type)[1:]
    for annotation in annotations:
        if isinstance(annotation, PropertyInfo) and annotation.alias is not None:
            return annotation.alias
    return key

def _transform_recursive(data: object, *, annotation: type, inner_type: type | None=None) -> object:
    if False:
        i = 10
        return i + 15
    'Transform the given data against the expected type.\n\n    Args:\n        annotation: The direct type annotation given to the particular piece of data.\n            This may or may not be wrapped in metadata types, e.g. `Required[T]`, `Annotated[T, ...]` etc\n\n        inner_type: If applicable, this is the "inside" type. This is useful in certain cases where the outside type\n            is a container type such as `List[T]`. In that case `inner_type` should be set to `T` so that each entry in\n            the list can be transformed using the metadata from the container type.\n\n            Defaults to the same value as the `annotation` argument.\n    '
    if inner_type is None:
        inner_type = annotation
    stripped_type = strip_annotated_type(inner_type)
    if is_typeddict(stripped_type) and is_mapping(data):
        return _transform_typeddict(data, stripped_type)
    if is_list_type(stripped_type) and is_list(data):
        inner_type = extract_type_arg(stripped_type, 0)
        return [_transform_recursive(d, annotation=annotation, inner_type=inner_type) for d in data]
    if is_union_type(stripped_type):
        for subtype in get_args(stripped_type):
            data = _transform_recursive(data, annotation=annotation, inner_type=subtype)
        return data
    if isinstance(data, pydantic.BaseModel):
        return model_dump(data, exclude_unset=True)
    return _transform_value(data, annotation)

def _transform_value(data: object, type_: type) -> object:
    if False:
        for i in range(10):
            print('nop')
    annotated_type = _get_annotated_type(type_)
    if annotated_type is None:
        return data
    annotations = get_args(annotated_type)[1:]
    for annotation in annotations:
        if isinstance(annotation, PropertyInfo) and annotation.format is not None:
            return _format_data(data, annotation.format, annotation.format_template)
    return data

def _format_data(data: object, format_: PropertyFormat, format_template: str | None) -> object:
    if False:
        return 10
    if isinstance(data, (date, datetime)):
        if format_ == 'iso8601':
            return data.isoformat()
        if format_ == 'custom' and format_template is not None:
            return data.strftime(format_template)
    return data

def _transform_typeddict(data: Mapping[str, object], expected_type: type) -> Mapping[str, object]:
    if False:
        for i in range(10):
            print('nop')
    result: dict[str, object] = {}
    annotations = get_type_hints(expected_type, include_extras=True)
    for (key, value) in data.items():
        type_ = annotations.get(key)
        if type_ is None:
            result[key] = value
        else:
            result[_maybe_transform_key(key, type_)] = _transform_recursive(value, annotation=type_)
    return result