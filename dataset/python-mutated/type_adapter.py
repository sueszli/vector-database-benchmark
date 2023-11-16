"""Type adapter specification."""
from __future__ import annotations as _annotations
import sys
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any, Dict, Generic, Iterable, Set, TypeVar, Union, cast, overload
from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator, Some
from typing_extensions import Literal, is_typeddict
from pydantic.errors import PydanticUserError
from pydantic.main import BaseModel
from ._internal import _config, _generate_schema, _typing_extra
from .config import ConfigDict
from .json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaKeyT, JsonSchemaMode, JsonSchemaValue
from .plugin._schema_validator import create_schema_validator
T = TypeVar('T')
if TYPE_CHECKING:
    IncEx = Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any]]

def _get_schema(type_: Any, config_wrapper: _config.ConfigWrapper, parent_depth: int) -> CoreSchema:
    if False:
        while True:
            i = 10
    '`BaseModel` uses its own `__module__` to find out where it was defined\n    and then look for symbols to resolve forward references in those globals.\n    On the other hand this function can be called with arbitrary objects,\n    including type aliases where `__module__` (always `typing.py`) is not useful.\n    So instead we look at the globals in our parent stack frame.\n\n    This works for the case where this function is called in a module that\n    has the target of forward references in its scope, but\n    does not work for more complex cases.\n\n    For example, take the following:\n\n    a.py\n    ```python\n    from typing import Dict, List\n\n    IntList = List[int]\n    OuterDict = Dict[str, \'IntList\']\n    ```\n\n    b.py\n    ```python test="skip"\n    from a import OuterDict\n\n    from pydantic import TypeAdapter\n\n    IntList = int  # replaces the symbol the forward reference is looking for\n    v = TypeAdapter(OuterDict)\n    v({\'x\': 1})  # should fail but doesn\'t\n    ```\n\n    If OuterDict were a `BaseModel`, this would work because it would resolve\n    the forward reference within the `a.py` namespace.\n    But `TypeAdapter(OuterDict)`\n    can\'t know what module OuterDict came from.\n\n    In other words, the assumption that _all_ forward references exist in the\n    module we are being called from is not technically always true.\n    Although most of the time it is and it works fine for recursive models and such,\n    `BaseModel`\'s behavior isn\'t perfect either and _can_ break in similar ways,\n    so there is no right or wrong between the two.\n\n    But at the very least this behavior is _subtly_ different from `BaseModel`\'s.\n    '
    local_ns = _typing_extra.parent_frame_namespace(parent_depth=parent_depth)
    global_ns = sys._getframe(max(parent_depth - 1, 1)).f_globals.copy()
    global_ns.update(local_ns or {})
    gen = _generate_schema.GenerateSchema(config_wrapper, types_namespace=global_ns, typevars_map={})
    schema = gen.generate_schema(type_)
    schema = gen.clean_schema(schema)
    return schema

def _getattr_no_parents(obj: Any, attribute: str) -> Any:
    if False:
        for i in range(10):
            print('nop')
    'Returns the attribute value without attempting to look up attributes from parent types.'
    if hasattr(obj, '__dict__'):
        try:
            return obj.__dict__[attribute]
        except KeyError:
            pass
    slots = getattr(obj, '__slots__', None)
    if slots is not None and attribute in slots:
        return getattr(obj, attribute)
    else:
        raise AttributeError(attribute)

class TypeAdapter(Generic[T]):
    """Type adapters provide a flexible way to perform validation and serialization based on a Python type.

    A `TypeAdapter` instance exposes some of the functionality from `BaseModel` instance methods
    for types that do not have such methods (such as dataclasses, primitive types, and more).

    Note that `TypeAdapter` is not an actual type, so you cannot use it in type annotations.

    Attributes:
        core_schema: The core schema for the type.
        validator (SchemaValidator): The schema validator for the type.
        serializer: The schema serializer for the type.
    """
    if TYPE_CHECKING:

        @overload
        def __new__(cls, __type: type[T], *, config: ConfigDict | None=...) -> TypeAdapter[T]:
            if False:
                while True:
                    i = 10
            ...

        @overload
        def __new__(cls, __type: T, *, config: ConfigDict | None=...) -> TypeAdapter[T]:
            if False:
                print('Hello World!')
            ...

        def __new__(cls, __type: Any, *, config: ConfigDict | None=None) -> TypeAdapter[T]:
            if False:
                for i in range(10):
                    print('nop')
            'A class representing the type adapter.'
            raise NotImplementedError

        @overload
        def __init__(self, type: type[T], *, config: ConfigDict | None=None, _parent_depth: int=2, module: str | None=None) -> None:
            if False:
                i = 10
                return i + 15
            ...

        @overload
        def __init__(self, type: T, *, config: ConfigDict | None=None, _parent_depth: int=2, module: str | None=None) -> None:
            if False:
                print('Hello World!')
            ...

    def __init__(self, type: Any, *, config: ConfigDict | None=None, _parent_depth: int=2, module: str | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Initializes the TypeAdapter object.\n\n        Args:\n            type: The type associated with the `TypeAdapter`.\n            config: Configuration for the `TypeAdapter`, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].\n            _parent_depth: depth at which to search the parent namespace to construct the local namespace.\n            module: The module that passes to plugin if provided.\n\n        !!! note\n            You cannot use the `config` argument when instantiating a `TypeAdapter` if the type you're using has its own\n            config that cannot be overridden (ex: `BaseModel`, `TypedDict`, and `dataclass`). A\n            [`type-adapter-config-unused`](../errors/usage_errors.md#type-adapter-config-unused) error will be raised in this case.\n\n        !!! note\n            The `_parent_depth` argument is named with an underscore to suggest its private nature and discourage use.\n            It may be deprecated in a minor version, so we only recommend using it if you're\n            comfortable with potential change in behavior / support.\n\n        Returns:\n            A type adapter configured for the specified `type`.\n        "
        config_wrapper = _config.ConfigWrapper(config)
        try:
            type_has_config = issubclass(type, BaseModel) or is_dataclass(type) or is_typeddict(type)
        except TypeError:
            type_has_config = False
        if type_has_config and config is not None:
            raise PydanticUserError('Cannot use `config` when the type is a BaseModel, dataclass or TypedDict. These types can have their own config and setting the config via the `config` parameter to TypeAdapter will not override it, thus the `config` you passed to TypeAdapter becomes meaningless, which is probably not what you want.', code='type-adapter-config-unused')
        core_schema: CoreSchema
        try:
            core_schema = _getattr_no_parents(type, '__pydantic_core_schema__')
        except AttributeError:
            core_schema = _get_schema(type, config_wrapper, parent_depth=_parent_depth + 1)
        core_config = config_wrapper.core_config(None)
        validator: SchemaValidator
        try:
            validator = _getattr_no_parents(type, '__pydantic_validator__')
        except AttributeError:
            if module is None:
                f = sys._getframe(1)
                module = cast(str, f.f_globals['__name__'])
            validator = create_schema_validator(core_schema, type, module, str(type), 'TypeAdapter', core_config, config_wrapper.plugin_settings)
        serializer: SchemaSerializer
        try:
            serializer = _getattr_no_parents(type, '__pydantic_serializer__')
        except AttributeError:
            serializer = SchemaSerializer(core_schema, core_config)
        self.core_schema = core_schema
        self.validator = validator
        self.serializer = serializer

    def validate_python(self, __object: Any, *, strict: bool | None=None, from_attributes: bool | None=None, context: dict[str, Any] | None=None) -> T:
        if False:
            for i in range(10):
                print('nop')
        'Validate a Python object against the model.\n\n        Args:\n            __object: The Python object to validate against the model.\n            strict: Whether to strictly check types.\n            from_attributes: Whether to extract data from object attributes.\n            context: Additional context to pass to the validator.\n\n        !!! note\n            When using `TypeAdapter` with a Pydantic `dataclass`, the use of the `from_attributes`\n            argument is not supported.\n\n        Returns:\n            The validated object.\n        '
        return self.validator.validate_python(__object, strict=strict, from_attributes=from_attributes, context=context)

    def validate_json(self, __data: str | bytes, *, strict: bool | None=None, context: dict[str, Any] | None=None) -> T:
        if False:
            i = 10
            return i + 15
        'Usage docs: https://docs.pydantic.dev/2.6/concepts/json/#json-parsing\n\n        Validate a JSON string or bytes against the model.\n\n        Args:\n            __data: The JSON data to validate against the model.\n            strict: Whether to strictly check types.\n            context: Additional context to use during validation.\n\n        Returns:\n            The validated object.\n        '
        return self.validator.validate_json(__data, strict=strict, context=context)

    def validate_strings(self, __obj: Any, *, strict: bool | None=None, context: dict[str, Any] | None=None) -> T:
        if False:
            for i in range(10):
                print('nop')
        'Validate object contains string data against the model.\n\n        Args:\n            __obj: The object contains string data to validate.\n            strict: Whether to strictly check types.\n            context: Additional context to use during validation.\n\n        Returns:\n            The validated object.\n        '
        return self.validator.validate_strings(__obj, strict=strict, context=context)

    def get_default_value(self, *, strict: bool | None=None, context: dict[str, Any] | None=None) -> Some[T] | None:
        if False:
            for i in range(10):
                print('nop')
        'Get the default value for the wrapped type.\n\n        Args:\n            strict: Whether to strictly check types.\n            context: Additional context to pass to the validator.\n\n        Returns:\n            The default value wrapped in a `Some` if there is one or None if not.\n        '
        return self.validator.get_default_value(strict=strict, context=context)

    def dump_python(self, __instance: T, *, mode: Literal['json', 'python']='python', include: IncEx | None=None, exclude: IncEx | None=None, by_alias: bool=False, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False, round_trip: bool=False, warnings: bool=True) -> Any:
        if False:
            while True:
                i = 10
        'Dump an instance of the adapted type to a Python object.\n\n        Args:\n            __instance: The Python object to serialize.\n            mode: The output format.\n            include: Fields to include in the output.\n            exclude: Fields to exclude from the output.\n            by_alias: Whether to use alias names for field names.\n            exclude_unset: Whether to exclude unset fields.\n            exclude_defaults: Whether to exclude fields with default values.\n            exclude_none: Whether to exclude fields with None values.\n            round_trip: Whether to output the serialized data in a way that is compatible with deserialization.\n            warnings: Whether to display serialization warnings.\n\n        Returns:\n            The serialized object.\n        '
        return self.serializer.to_python(__instance, mode=mode, by_alias=by_alias, include=include, exclude=exclude, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, round_trip=round_trip, warnings=warnings)

    def dump_json(self, __instance: T, *, indent: int | None=None, include: IncEx | None=None, exclude: IncEx | None=None, by_alias: bool=False, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False, round_trip: bool=False, warnings: bool=True) -> bytes:
        if False:
            print('Hello World!')
        'Usage docs: https://docs.pydantic.dev/2.6/concepts/json/#json-serialization\n\n        Serialize an instance of the adapted type to JSON.\n\n        Args:\n            __instance: The instance to be serialized.\n            indent: Number of spaces for JSON indentation.\n            include: Fields to include.\n            exclude: Fields to exclude.\n            by_alias: Whether to use alias names for field names.\n            exclude_unset: Whether to exclude unset fields.\n            exclude_defaults: Whether to exclude fields with default values.\n            exclude_none: Whether to exclude fields with a value of `None`.\n            round_trip: Whether to serialize and deserialize the instance to ensure round-tripping.\n            warnings: Whether to emit serialization warnings.\n\n        Returns:\n            The JSON representation of the given instance as bytes.\n        '
        return self.serializer.to_json(__instance, indent=indent, include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, round_trip=round_trip, warnings=warnings)

    def json_schema(self, *, by_alias: bool=True, ref_template: str=DEFAULT_REF_TEMPLATE, schema_generator: type[GenerateJsonSchema]=GenerateJsonSchema, mode: JsonSchemaMode='validation') -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Generate a JSON schema for the adapted type.\n\n        Args:\n            by_alias: Whether to use alias names for field names.\n            ref_template: The format string used for generating $ref strings.\n            schema_generator: The generator class used for creating the schema.\n            mode: The mode to use for schema generation.\n\n        Returns:\n            The JSON schema for the model as a dictionary.\n        '
        schema_generator_instance = schema_generator(by_alias=by_alias, ref_template=ref_template)
        return schema_generator_instance.generate(self.core_schema, mode=mode)

    @staticmethod
    def json_schemas(__inputs: Iterable[tuple[JsonSchemaKeyT, JsonSchemaMode, TypeAdapter[Any]]], *, by_alias: bool=True, title: str | None=None, description: str | None=None, ref_template: str=DEFAULT_REF_TEMPLATE, schema_generator: type[GenerateJsonSchema]=GenerateJsonSchema) -> tuple[dict[tuple[JsonSchemaKeyT, JsonSchemaMode], JsonSchemaValue], JsonSchemaValue]:
        if False:
            return 10
        'Generate a JSON schema including definitions from multiple type adapters.\n\n        Args:\n            __inputs: Inputs to schema generation. The first two items will form the keys of the (first)\n                output mapping; the type adapters will provide the core schemas that get converted into\n                definitions in the output JSON schema.\n            by_alias: Whether to use alias names.\n            title: The title for the schema.\n            description: The description for the schema.\n            ref_template: The format string used for generating $ref strings.\n            schema_generator: The generator class used for creating the schema.\n\n        Returns:\n            A tuple where:\n\n                - The first element is a dictionary whose keys are tuples of JSON schema key type and JSON mode, and\n                    whose values are the JSON schema corresponding to that pair of inputs. (These schemas may have\n                    JsonRef references to definitions that are defined in the second returned element.)\n                - The second element is a JSON schema containing all definitions referenced in the first returned\n                    element, along with the optional title and description keys.\n\n        '
        schema_generator_instance = schema_generator(by_alias=by_alias, ref_template=ref_template)
        inputs = [(key, mode, adapter.core_schema) for (key, mode, adapter) in __inputs]
        (json_schemas_map, definitions) = schema_generator_instance.generate_definitions(inputs)
        json_schema: dict[str, Any] = {}
        if definitions:
            json_schema['$defs'] = definitions
        if title:
            json_schema['title'] = title
        if description:
            json_schema['description'] = description
        return (json_schemas_map, json_schema)