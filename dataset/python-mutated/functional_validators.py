"""This module contains related classes and functions for validation."""
from __future__ import annotations as _annotations
import dataclasses
import sys
from functools import partialmethod
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, cast, overload
from pydantic_core import core_schema
from pydantic_core import core_schema as _core_schema
from typing_extensions import Annotated, Literal, TypeAlias
from . import GetCoreSchemaHandler as _GetCoreSchemaHandler
from ._internal import _core_metadata, _decorators, _generics, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler
from .errors import PydanticUserError
if sys.version_info < (3, 11):
    from typing_extensions import Protocol
else:
    from typing import Protocol
_inspect_validator = _decorators.inspect_validator

@dataclasses.dataclass(frozen=True, **_internal_dataclass.slots_true)
class AfterValidator:
    """Usage docs: https://docs.pydantic.dev/2.6/concepts/validators/#annotated-validators

    A metadata class that indicates that a validation should be applied **after** the inner validation logic.

    Attributes:
        func: The validator function.

    Example:
        ```py
        from typing_extensions import Annotated

        from pydantic import AfterValidator, BaseModel, ValidationError

        MyInt = Annotated[int, AfterValidator(lambda v: v + 1)]

        class Model(BaseModel):
            a: MyInt

        print(Model(a=1).a)
        #> 2

        try:
            Model(a='a')
        except ValidationError as e:
            print(e.json(indent=2))
            '''
            [
              {
                "type": "int_parsing",
                "loc": [
                  "a"
                ],
                "msg": "Input should be a valid integer, unable to parse string as an integer",
                "input": "a",
                "url": "https://errors.pydantic.dev/2/v/int_parsing"
              }
            ]
            '''
        ```
    """
    func: core_schema.NoInfoValidatorFunction | core_schema.WithInfoValidatorFunction

    def __get_pydantic_core_schema__(self, source_type: Any, handler: _GetCoreSchemaHandler) -> core_schema.CoreSchema:
        if False:
            return 10
        schema = handler(source_type)
        info_arg = _inspect_validator(self.func, 'after')
        if info_arg:
            func = cast(core_schema.WithInfoValidatorFunction, self.func)
            return core_schema.with_info_after_validator_function(func, schema=schema, field_name=handler.field_name)
        else:
            func = cast(core_schema.NoInfoValidatorFunction, self.func)
            return core_schema.no_info_after_validator_function(func, schema=schema)

@dataclasses.dataclass(frozen=True, **_internal_dataclass.slots_true)
class BeforeValidator:
    """Usage docs: https://docs.pydantic.dev/2.6/concepts/validators/#annotated-validators

    A metadata class that indicates that a validation should be applied **before** the inner validation logic.

    Attributes:
        func: The validator function.

    Example:
        ```py
        from typing_extensions import Annotated

        from pydantic import BaseModel, BeforeValidator

        MyInt = Annotated[int, BeforeValidator(lambda v: v + 1)]

        class Model(BaseModel):
            a: MyInt

        print(Model(a=1).a)
        #> 2

        try:
            Model(a='a')
        except TypeError as e:
            print(e)
            #> can only concatenate str (not "int") to str
        ```
    """
    func: core_schema.NoInfoValidatorFunction | core_schema.WithInfoValidatorFunction

    def __get_pydantic_core_schema__(self, source_type: Any, handler: _GetCoreSchemaHandler) -> core_schema.CoreSchema:
        if False:
            while True:
                i = 10
        schema = handler(source_type)
        info_arg = _inspect_validator(self.func, 'before')
        if info_arg:
            func = cast(core_schema.WithInfoValidatorFunction, self.func)
            return core_schema.with_info_before_validator_function(func, schema=schema, field_name=handler.field_name)
        else:
            func = cast(core_schema.NoInfoValidatorFunction, self.func)
            return core_schema.no_info_before_validator_function(func, schema=schema)

@dataclasses.dataclass(frozen=True, **_internal_dataclass.slots_true)
class PlainValidator:
    """Usage docs: https://docs.pydantic.dev/2.6/concepts/validators/#annotated-validators

    A metadata class that indicates that a validation should be applied **instead** of the inner validation logic.

    Attributes:
        func: The validator function.

    Example:
        ```py
        from typing_extensions import Annotated

        from pydantic import BaseModel, PlainValidator

        MyInt = Annotated[int, PlainValidator(lambda v: int(v) + 1)]

        class Model(BaseModel):
            a: MyInt

        print(Model(a='1').a)
        #> 2
        ```
    """
    func: core_schema.NoInfoValidatorFunction | core_schema.WithInfoValidatorFunction

    def __get_pydantic_core_schema__(self, source_type: Any, handler: _GetCoreSchemaHandler) -> core_schema.CoreSchema:
        if False:
            while True:
                i = 10
        info_arg = _inspect_validator(self.func, 'plain')
        if info_arg:
            func = cast(core_schema.WithInfoValidatorFunction, self.func)
            return core_schema.with_info_plain_validator_function(func, field_name=handler.field_name)
        else:
            func = cast(core_schema.NoInfoValidatorFunction, self.func)
            return core_schema.no_info_plain_validator_function(func)

@dataclasses.dataclass(frozen=True, **_internal_dataclass.slots_true)
class WrapValidator:
    """Usage docs: https://docs.pydantic.dev/2.6/concepts/validators/#annotated-validators

    A metadata class that indicates that a validation should be applied **around** the inner validation logic.

    Attributes:
        func: The validator function.

    ```py
    from datetime import datetime

    from typing_extensions import Annotated

    from pydantic import BaseModel, ValidationError, WrapValidator

    def validate_timestamp(v, handler):
        if v == 'now':
            # we don't want to bother with further validation, just return the new value
            return datetime.now()
        try:
            return handler(v)
        except ValidationError:
            # validation failed, in this case we want to return a default value
            return datetime(2000, 1, 1)

    MyTimestamp = Annotated[datetime, WrapValidator(validate_timestamp)]

    class Model(BaseModel):
        a: MyTimestamp

    print(Model(a='now').a)
    #> 2032-01-02 03:04:05.000006
    print(Model(a='invalid').a)
    #> 2000-01-01 00:00:00
    ```
    """
    func: core_schema.NoInfoWrapValidatorFunction | core_schema.WithInfoWrapValidatorFunction

    def __get_pydantic_core_schema__(self, source_type: Any, handler: _GetCoreSchemaHandler) -> core_schema.CoreSchema:
        if False:
            while True:
                i = 10
        schema = handler(source_type)
        info_arg = _inspect_validator(self.func, 'wrap')
        if info_arg:
            func = cast(core_schema.WithInfoWrapValidatorFunction, self.func)
            return core_schema.with_info_wrap_validator_function(func, schema=schema, field_name=handler.field_name)
        else:
            func = cast(core_schema.NoInfoWrapValidatorFunction, self.func)
            return core_schema.no_info_wrap_validator_function(func, schema=schema)
if TYPE_CHECKING:

    class _OnlyValueValidatorClsMethod(Protocol):

        def __call__(self, __cls: Any, __value: Any) -> Any:
            if False:
                return 10
            ...

    class _V2ValidatorClsMethod(Protocol):

        def __call__(self, __cls: Any, __input_value: Any, __info: _core_schema.ValidationInfo) -> Any:
            if False:
                print('Hello World!')
            ...

    class _V2WrapValidatorClsMethod(Protocol):

        def __call__(self, __cls: Any, __input_value: Any, __validator: _core_schema.ValidatorFunctionWrapHandler, __info: _core_schema.ValidationInfo) -> Any:
            if False:
                while True:
                    i = 10
            ...
    _V2Validator = Union[_V2ValidatorClsMethod, _core_schema.WithInfoValidatorFunction, _OnlyValueValidatorClsMethod, _core_schema.NoInfoValidatorFunction]
    _V2WrapValidator = Union[_V2WrapValidatorClsMethod, _core_schema.WithInfoWrapValidatorFunction]
    _PartialClsOrStaticMethod: TypeAlias = Union[classmethod[Any, Any, Any], staticmethod[Any, Any], partialmethod[Any]]
    _V2BeforeAfterOrPlainValidatorType = TypeVar('_V2BeforeAfterOrPlainValidatorType', _V2Validator, _PartialClsOrStaticMethod)
    _V2WrapValidatorType = TypeVar('_V2WrapValidatorType', _V2WrapValidator, _PartialClsOrStaticMethod)

@overload
def field_validator(__field: str, *fields: str, mode: Literal['before', 'after', 'plain']=..., check_fields: bool | None=...) -> Callable[[_V2BeforeAfterOrPlainValidatorType], _V2BeforeAfterOrPlainValidatorType]:
    if False:
        i = 10
        return i + 15
    ...

@overload
def field_validator(__field: str, *fields: str, mode: Literal['wrap'], check_fields: bool | None=...) -> Callable[[_V2WrapValidatorType], _V2WrapValidatorType]:
    if False:
        print('Hello World!')
    ...
FieldValidatorModes: TypeAlias = Literal['before', 'after', 'wrap', 'plain']

def field_validator(__field: str, *fields: str, mode: FieldValidatorModes='after', check_fields: bool | None=None) -> Callable[[Any], Any]:
    if False:
        print('Hello World!')
    'Usage docs: https://docs.pydantic.dev/2.6/concepts/validators/#field-validators\n\n    Decorate methods on the class indicating that they should be used to validate fields.\n\n    Example usage:\n    ```py\n    from typing import Any\n\n    from pydantic import (\n        BaseModel,\n        ValidationError,\n        field_validator,\n    )\n\n    class Model(BaseModel):\n        a: str\n\n        @field_validator(\'a\')\n        @classmethod\n        def ensure_foobar(cls, v: Any):\n            if \'foobar\' not in v:\n                raise ValueError(\'"foobar" not found in a\')\n            return v\n\n    print(repr(Model(a=\'this is foobar good\')))\n    #> Model(a=\'this is foobar good\')\n\n    try:\n        Model(a=\'snap\')\n    except ValidationError as exc_info:\n        print(exc_info)\n        \'\'\'\n        1 validation error for Model\n        a\n          Value error, "foobar" not found in a [type=value_error, input_value=\'snap\', input_type=str]\n        \'\'\'\n    ```\n\n    For more in depth examples, see [Field Validators](../concepts/validators.md#field-validators).\n\n    Args:\n        __field: The first field the `field_validator` should be called on; this is separate\n            from `fields` to ensure an error is raised if you don\'t pass at least one.\n        *fields: Additional field(s) the `field_validator` should be called on.\n        mode: Specifies whether to validate the fields before or after validation.\n        check_fields: Whether to check that the fields actually exist on the model.\n\n    Returns:\n        A decorator that can be used to decorate a function to be used as a field_validator.\n\n    Raises:\n        PydanticUserError:\n            - If `@field_validator` is used bare (with no fields).\n            - If the args passed to `@field_validator` as fields are not strings.\n            - If `@field_validator` applied to instance methods.\n    '
    if isinstance(__field, FunctionType):
        raise PydanticUserError("`@field_validator` should be used with fields and keyword arguments, not bare. E.g. usage should be `@validator('<field_name>', ...)`", code='validator-no-fields')
    fields = (__field, *fields)
    if not all((isinstance(field, str) for field in fields)):
        raise PydanticUserError("`@field_validator` fields should be passed as separate string args. E.g. usage should be `@validator('<field_name_1>', '<field_name_2>', ...)`", code='validator-invalid-fields')

    def dec(f: Callable[..., Any] | staticmethod[Any, Any] | classmethod[Any, Any, Any]) -> _decorators.PydanticDescriptorProxy[Any]:
        if False:
            while True:
                i = 10
        if _decorators.is_instance_method_from_sig(f):
            raise PydanticUserError('`@field_validator` cannot be applied to instance methods', code='validator-instance-method')
        f = _decorators.ensure_classmethod_based_on_signature(f)
        dec_info = _decorators.FieldValidatorDecoratorInfo(fields=fields, mode=mode, check_fields=check_fields)
        return _decorators.PydanticDescriptorProxy(f, dec_info)
    return dec
_ModelType = TypeVar('_ModelType')
_ModelTypeCo = TypeVar('_ModelTypeCo', covariant=True)

class ModelWrapValidatorHandler(_core_schema.ValidatorFunctionWrapHandler, Protocol[_ModelTypeCo]):
    """@model_validator decorated function handler argument type. This is used when `mode='wrap'`."""

    def __call__(self, input_value: Any, outer_location: str | int | None=None) -> _ModelTypeCo:
        if False:
            i = 10
            return i + 15
        ...

class ModelWrapValidatorWithoutInfo(Protocol[_ModelType]):
    """A @model_validator decorated function signature.
    This is used when `mode='wrap'` and the function does not have info argument.
    """

    def __call__(self, cls: type[_ModelType], __value: Any, __handler: ModelWrapValidatorHandler[_ModelType]) -> _ModelType:
        if False:
            print('Hello World!')
        ...

class ModelWrapValidator(Protocol[_ModelType]):
    """A @model_validator decorated function signature. This is used when `mode='wrap'`."""

    def __call__(self, cls: type[_ModelType], __value: Any, __handler: ModelWrapValidatorHandler[_ModelType], __info: _core_schema.ValidationInfo) -> _ModelType:
        if False:
            return 10
        ...

class ModelBeforeValidatorWithoutInfo(Protocol):
    """A @model_validator decorated function signature.
    This is used when `mode='before'` and the function does not have info argument.
    """

    def __call__(self, cls: Any, __value: Any) -> Any:
        if False:
            while True:
                i = 10
        ...

class ModelBeforeValidator(Protocol):
    """A `@model_validator` decorated function signature. This is used when `mode='before'`."""

    def __call__(self, cls: Any, __value: Any, __info: _core_schema.ValidationInfo) -> Any:
        if False:
            print('Hello World!')
        ...
ModelAfterValidatorWithoutInfo = Callable[[_ModelType], _ModelType]
"A `@model_validator` decorated function signature. This is used when `mode='after'` and the function does not\nhave info argument.\n"
ModelAfterValidator = Callable[[_ModelType, _core_schema.ValidationInfo], _ModelType]
"A `@model_validator` decorated function signature. This is used when `mode='after'`."
_AnyModelWrapValidator = Union[ModelWrapValidator[_ModelType], ModelWrapValidatorWithoutInfo[_ModelType]]
_AnyModeBeforeValidator = Union[ModelBeforeValidator, ModelBeforeValidatorWithoutInfo]
_AnyModelAfterValidator = Union[ModelAfterValidator[_ModelType], ModelAfterValidatorWithoutInfo[_ModelType]]

@overload
def model_validator(*, mode: Literal['wrap']) -> Callable[[_AnyModelWrapValidator[_ModelType]], _decorators.PydanticDescriptorProxy[_decorators.ModelValidatorDecoratorInfo]]:
    if False:
        return 10
    ...

@overload
def model_validator(*, mode: Literal['before']) -> Callable[[_AnyModeBeforeValidator], _decorators.PydanticDescriptorProxy[_decorators.ModelValidatorDecoratorInfo]]:
    if False:
        return 10
    ...

@overload
def model_validator(*, mode: Literal['after']) -> Callable[[_AnyModelAfterValidator[_ModelType]], _decorators.PydanticDescriptorProxy[_decorators.ModelValidatorDecoratorInfo]]:
    if False:
        print('Hello World!')
    ...

def model_validator(*, mode: Literal['wrap', 'before', 'after']) -> Any:
    if False:
        i = 10
        return i + 15
    "Usage docs: https://docs.pydantic.dev/2.6/concepts/validators/#model-validators\n\n    Decorate model methods for validation purposes.\n\n    Example usage:\n    ```py\n    from typing import Optional\n\n    from pydantic import BaseModel, ValidationError, model_validator\n\n    class Square(BaseModel):\n        width: float\n        height: float\n\n        @model_validator(mode='after')\n        def verify_square(self) -> 'Rectangle':\n            if self.width != self.height:\n                raise ValueError('width and height do not match')\n            return self\n\n    s = Square(width=1, height=1)\n    print(repr(s))\n    #> Square(width=1.0, height=1.0)\n\n    try:\n        Square(width=1, height=2)\n    except ValidationError as e:\n        print(e)\n        '''\n        1 validation error for Square\n        __root__\n          width and height do not match (type=value_error)\n        '''\n    ```\n\n    For more in depth examples, see [Model Validators](../concepts/validators.md#model-validators).\n\n    Args:\n        mode: A required string literal that specifies the validation mode.\n            It can be one of the following: 'wrap', 'before', or 'after'.\n\n    Returns:\n        A decorator that can be used to decorate a function to be used as a model validator.\n    "

    def dec(f: Any) -> _decorators.PydanticDescriptorProxy[Any]:
        if False:
            print('Hello World!')
        f = _decorators.ensure_classmethod_based_on_signature(f)
        dec_info = _decorators.ModelValidatorDecoratorInfo(mode=mode)
        return _decorators.PydanticDescriptorProxy(f, dec_info)
    return dec
AnyType = TypeVar('AnyType')
if TYPE_CHECKING:
    InstanceOf = Annotated[AnyType, ...]
else:

    @dataclasses.dataclass(**_internal_dataclass.slots_true)
    class InstanceOf:
        '''Generic type for annotating a type that is an instance of a given class.

        Example:
            ```py
            from pydantic import BaseModel, InstanceOf

            class Foo:
                ...

            class Bar(BaseModel):
                foo: InstanceOf[Foo]

            Bar(foo=Foo())
            try:
                Bar(foo=42)
            except ValidationError as e:
                print(e)
                """
                [
                │   {
                │   │   'type': 'is_instance_of',
                │   │   'loc': ('foo',),
                │   │   'msg': 'Input should be an instance of Foo',
                │   │   'input': 42,
                │   │   'ctx': {'class': 'Foo'},
                │   │   'url': 'https://errors.pydantic.dev/0.38.0/v/is_instance_of'
                │   }
                ]
                """
            ```
        '''

        @classmethod
        def __class_getitem__(cls, item: AnyType) -> AnyType:
            if False:
                while True:
                    i = 10
            return Annotated[item, cls()]

        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
            if False:
                i = 10
                return i + 15
            from pydantic import PydanticSchemaGenerationError
            instance_of_schema = core_schema.is_instance_schema(_generics.get_origin(source) or source)
            try:
                original_schema = handler(source)
            except PydanticSchemaGenerationError:
                return instance_of_schema
            else:
                instance_of_schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(function=lambda v, h: h(v), schema=original_schema)
                return core_schema.json_or_python_schema(python_schema=instance_of_schema, json_schema=original_schema)
        __hash__ = object.__hash__
if TYPE_CHECKING:
    SkipValidation = Annotated[AnyType, ...]
else:

    @dataclasses.dataclass(**_internal_dataclass.slots_true)
    class SkipValidation:
        """If this is applied as an annotation (e.g., via `x: Annotated[int, SkipValidation]`), validation will be
            skipped. You can also use `SkipValidation[int]` as a shorthand for `Annotated[int, SkipValidation]`.

        This can be useful if you want to use a type annotation for documentation/IDE/type-checking purposes,
        and know that it is safe to skip validation for one or more of the fields.

        Because this converts the validation schema to `any_schema`, subsequent annotation-applied transformations
        may not have the expected effects. Therefore, when used, this annotation should generally be the final
        annotation applied to a type.
        """

        def __class_getitem__(cls, item: Any) -> Any:
            if False:
                for i in range(10):
                    print('nop')
            return Annotated[item, SkipValidation()]

        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
            if False:
                print('Hello World!')
            original_schema = handler(source)
            metadata = _core_metadata.build_metadata_dict(js_annotation_functions=[lambda _c, h: h(original_schema)])
            return core_schema.any_schema(metadata=metadata, serialization=core_schema.wrap_serializer_function_ser_schema(function=lambda v, h: h(v), schema=original_schema))
        __hash__ = object.__hash__