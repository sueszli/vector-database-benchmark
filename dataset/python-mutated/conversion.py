from __future__ import annotations
import copy
import dataclasses
from typing import TYPE_CHECKING, Any, Type, Union, cast
from strawberry.enum import EnumDefinition
from strawberry.type import StrawberryList, StrawberryOptional, has_object_definition
from strawberry.union import StrawberryUnion
if TYPE_CHECKING:
    from strawberry.field import StrawberryField
    from strawberry.type import StrawberryType

def _convert_from_pydantic_to_strawberry_type(type_: Union[StrawberryType, type], data_from_model=None, extra=None):
    if False:
        print('Hello World!')
    data = data_from_model if data_from_model is not None else extra
    if isinstance(type_, StrawberryOptional):
        if data is None:
            return data
        return _convert_from_pydantic_to_strawberry_type(type_.of_type, data_from_model=data, extra=extra)
    if isinstance(type_, StrawberryUnion):
        for option_type in type_.types:
            if hasattr(option_type, '_pydantic_type'):
                source_type = option_type._pydantic_type
            else:
                source_type = cast(type, option_type)
            if isinstance(data, source_type):
                return _convert_from_pydantic_to_strawberry_type(option_type, data_from_model=data, extra=extra)
    if isinstance(type_, EnumDefinition):
        return data
    if isinstance(type_, StrawberryList):
        items = []
        for (index, item) in enumerate(data):
            items.append(_convert_from_pydantic_to_strawberry_type(type_.of_type, data_from_model=item, extra=extra[index] if extra else None))
        return items
    if has_object_definition(type_):
        if hasattr(type(data), '_strawberry_type'):
            type_ = type(data)._strawberry_type
        if hasattr(type_, 'from_pydantic'):
            return type_.from_pydantic(data_from_model, extra)
        return convert_pydantic_model_to_strawberry_class(type_, model_instance=data_from_model, extra=extra)
    return data

def convert_pydantic_model_to_strawberry_class(cls, *, model_instance=None, extra=None) -> Any:
    if False:
        while True:
            i = 10
    extra = extra or {}
    kwargs = {}
    for field_ in cls.__strawberry_definition__.fields:
        field = cast('StrawberryField', field_)
        python_name = field.python_name
        data_from_extra = extra.get(python_name, None)
        data_from_model = getattr(model_instance, python_name, None) if model_instance else None
        if field.init:
            kwargs[python_name] = _convert_from_pydantic_to_strawberry_type(field.type, data_from_model, extra=data_from_extra)
    return cls(**kwargs)

def convert_strawberry_class_to_pydantic_model(obj: Type) -> Any:
    if False:
        while True:
            i = 10
    if hasattr(obj, 'to_pydantic'):
        return obj.to_pydantic()
    elif dataclasses.is_dataclass(obj):
        result = []
        for f in dataclasses.fields(obj):
            value = convert_strawberry_class_to_pydantic_model(getattr(obj, f.name))
            result.append((f.name, value))
        return dict(result)
    elif isinstance(obj, (list, tuple)):
        return type(obj)((convert_strawberry_class_to_pydantic_model(v) for v in obj))
    elif isinstance(obj, dict):
        return type(obj)(((convert_strawberry_class_to_pydantic_model(k), convert_strawberry_class_to_pydantic_model(v)) for (k, v) in obj.items()))
    else:
        return copy.deepcopy(obj)