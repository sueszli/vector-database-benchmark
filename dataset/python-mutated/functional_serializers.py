"""This module contains related classes and functions for serialization."""
from __future__ import annotations
import dataclasses
from functools import partialmethod
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
from pydantic_core import PydanticUndefined, core_schema
from pydantic_core import core_schema as _core_schema
from typing_extensions import Annotated, Literal, TypeAlias
from . import PydanticUndefinedAnnotation
from ._internal import _decorators, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler

@dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class PlainSerializer:
    """Plain serializers use a function to modify the output of serialization.

    Attributes:
        func: The serializer function.
        return_type: The return type for the function. If omitted it will be inferred from the type annotation.
        when_used: Determines when this serializer should be used. Accepts a string with values `'always'`,
            `'unless-none'`, `'json'`, and `'json-unless-none'`. Defaults to 'always'.
    """
    func: core_schema.SerializerFunction
    return_type: Any = PydanticUndefined
    when_used: Literal['always', 'unless-none', 'json', 'json-unless-none'] = 'always'

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        if False:
            return 10
        'Gets the Pydantic core schema.\n\n        Args:\n            source_type: The source type.\n            handler: The `GetCoreSchemaHandler` instance.\n\n        Returns:\n            The Pydantic core schema.\n        '
        schema = handler(source_type)
        try:
            return_type = _decorators.get_function_return_type(self.func, self.return_type, handler._get_types_namespace())
        except NameError as e:
            raise PydanticUndefinedAnnotation.from_name_error(e) from e
        return_schema = None if return_type is PydanticUndefined else handler.generate_schema(return_type)
        schema['serialization'] = core_schema.plain_serializer_function_ser_schema(function=self.func, info_arg=_decorators.inspect_annotated_serializer(self.func, 'plain'), return_schema=return_schema, when_used=self.when_used)
        return schema

@dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class WrapSerializer:
    """Wrap serializers receive the raw inputs along with a handler function that applies the standard serialization
    logic, and can modify the resulting value before returning it as the final output of serialization.

    Attributes:
        func: The serializer function to be wrapped.
        return_type: The return type for the function. If omitted it will be inferred from the type annotation.
        when_used: Determines when this serializer should be used. Accepts a string with values `'always'`,
            `'unless-none'`, `'json'`, and `'json-unless-none'`. Defaults to 'always'.
    """
    func: core_schema.WrapSerializerFunction
    return_type: Any = PydanticUndefined
    when_used: Literal['always', 'unless-none', 'json', 'json-unless-none'] = 'always'

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        if False:
            print('Hello World!')
        'This method is used to get the Pydantic core schema of the class.\n\n        Args:\n            source_type: Source type.\n            handler: Core schema handler.\n\n        Returns:\n            The generated core schema of the class.\n        '
        schema = handler(source_type)
        try:
            return_type = _decorators.get_function_return_type(self.func, self.return_type, handler._get_types_namespace())
        except NameError as e:
            raise PydanticUndefinedAnnotation.from_name_error(e) from e
        return_schema = None if return_type is PydanticUndefined else handler.generate_schema(return_type)
        schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(function=self.func, info_arg=_decorators.inspect_annotated_serializer(self.func, 'wrap'), return_schema=return_schema, when_used=self.when_used)
        return schema
if TYPE_CHECKING:
    _PartialClsOrStaticMethod: TypeAlias = Union[classmethod[Any, Any, Any], staticmethod[Any, Any], partialmethod[Any]]
    _PlainSerializationFunction = Union[_core_schema.SerializerFunction, _PartialClsOrStaticMethod]
    _WrapSerializationFunction = Union[_core_schema.WrapSerializerFunction, _PartialClsOrStaticMethod]
    _PlainSerializeMethodType = TypeVar('_PlainSerializeMethodType', bound=_PlainSerializationFunction)
    _WrapSerializeMethodType = TypeVar('_WrapSerializeMethodType', bound=_WrapSerializationFunction)

@overload
def field_serializer(__field: str, *fields: str, return_type: Any=..., when_used: Literal['always', 'unless-none', 'json', 'json-unless-none']=..., check_fields: bool | None=...) -> Callable[[_PlainSerializeMethodType], _PlainSerializeMethodType]:
    if False:
        while True:
            i = 10
    ...

@overload
def field_serializer(__field: str, *fields: str, mode: Literal['plain'], return_type: Any=..., when_used: Literal['always', 'unless-none', 'json', 'json-unless-none']=..., check_fields: bool | None=...) -> Callable[[_PlainSerializeMethodType], _PlainSerializeMethodType]:
    if False:
        print('Hello World!')
    ...

@overload
def field_serializer(__field: str, *fields: str, mode: Literal['wrap'], return_type: Any=..., when_used: Literal['always', 'unless-none', 'json', 'json-unless-none']=..., check_fields: bool | None=...) -> Callable[[_WrapSerializeMethodType], _WrapSerializeMethodType]:
    if False:
        i = 10
        return i + 15
    ...

def field_serializer(*fields: str, mode: Literal['plain', 'wrap']='plain', return_type: Any=PydanticUndefined, when_used: Literal['always', 'unless-none', 'json', 'json-unless-none']='always', check_fields: bool | None=None) -> Callable[[Any], Any]:
    if False:
        print('Hello World!')
    'Decorator that enables custom field serialization.\n\n    See [Custom serializers](../concepts/serialization.md#custom-serializers) for more information.\n\n    Four signatures are supported:\n\n    - `(self, value: Any, info: FieldSerializationInfo)`\n    - `(self, value: Any, nxt: SerializerFunctionWrapHandler, info: FieldSerializationInfo)`\n    - `(value: Any, info: SerializationInfo)`\n    - `(value: Any, nxt: SerializerFunctionWrapHandler, info: SerializationInfo)`\n\n    Args:\n        fields: Which field(s) the method should be called on.\n        mode: The serialization mode.\n\n            - `plain` means the function will be called instead of the default serialization logic,\n            - `wrap` means the function will be called with an argument to optionally call the\n               default serialization logic.\n        return_type: Optional return type for the function, if omitted it will be inferred from the type annotation.\n        when_used: Determines the serializer will be used for serialization.\n        check_fields: Whether to check that the fields actually exist on the model.\n\n    Returns:\n        The decorator function.\n    '

    def dec(f: Callable[..., Any] | staticmethod[Any, Any] | classmethod[Any, Any, Any]) -> _decorators.PydanticDescriptorProxy[Any]:
        if False:
            return 10
        dec_info = _decorators.FieldSerializerDecoratorInfo(fields=fields, mode=mode, return_type=return_type, when_used=when_used, check_fields=check_fields)
        return _decorators.PydanticDescriptorProxy(f, dec_info)
    return dec
FuncType = TypeVar('FuncType', bound=Callable[..., Any])

@overload
def model_serializer(__f: FuncType) -> FuncType:
    if False:
        print('Hello World!')
    ...

@overload
def model_serializer(*, mode: Literal['plain', 'wrap']=..., when_used: Literal['always', 'unless-none', 'json', 'json-unless-none']='always', return_type: Any=...) -> Callable[[FuncType], FuncType]:
    if False:
        for i in range(10):
            print('nop')
    ...

def model_serializer(__f: Callable[..., Any] | None=None, *, mode: Literal['plain', 'wrap']='plain', when_used: Literal['always', 'unless-none', 'json', 'json-unless-none']='always', return_type: Any=PydanticUndefined) -> Callable[[Any], Any]:
    if False:
        while True:
            i = 10
    "Decorator that enables custom model serialization.\n\n    See [Custom serializers](../concepts/serialization.md#custom-serializers) for more information.\n\n    Args:\n        __f: The function to be decorated.\n        mode: The serialization mode.\n\n            - `'plain'` means the function will be called instead of the default serialization logic\n            - `'wrap'` means the function will be called with an argument to optionally call the default\n                serialization logic.\n        when_used: Determines when this serializer should be used.\n        return_type: The return type for the function. If omitted it will be inferred from the type annotation.\n\n    Returns:\n        The decorator function.\n    "

    def dec(f: Callable[..., Any]) -> _decorators.PydanticDescriptorProxy[Any]:
        if False:
            print('Hello World!')
        dec_info = _decorators.ModelSerializerDecoratorInfo(mode=mode, return_type=return_type, when_used=when_used)
        return _decorators.PydanticDescriptorProxy(f, dec_info)
    if __f is None:
        return dec
    else:
        return dec(__f)
AnyType = TypeVar('AnyType')
if TYPE_CHECKING:
    SerializeAsAny = Annotated[AnyType, ...]
    'Force serialization to ignore whatever is defined in the schema and instead ask the object\n    itself how it should be serialized.\n    In particular, this means that when model subclasses are serialized, fields present in the subclass\n    but not in the original schema will be included.\n    '
else:

    @dataclasses.dataclass(**_internal_dataclass.slots_true)
    class SerializeAsAny:

        def __class_getitem__(cls, item: Any) -> Any:
            if False:
                for i in range(10):
                    print('nop')
            return Annotated[item, SerializeAsAny()]

        def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
            if False:
                i = 10
                return i + 15
            schema = handler(source_type)
            schema_to_update = schema
            while schema_to_update['type'] == 'definitions':
                schema_to_update = schema_to_update.copy()
                schema_to_update = schema_to_update['schema']
            schema_to_update['serialization'] = core_schema.wrap_serializer_function_ser_schema(lambda x, h: h(x), schema=core_schema.any_schema())
            return schema
        __hash__ = object.__hash__