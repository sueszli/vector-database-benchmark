import inspect
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar, Union
from typing_extensions import Annotated, get_args, get_origin
from dagster import Enum as DagsterEnum
from dagster._config.config_type import Array, ConfigType, Noneable
from dagster._config.post_process import resolve_defaults
from dagster._config.source import BoolSource, IntSource, StringSource
from dagster._config.validate import validate_config
from dagster._core.definitions.definition_config_schema import DefinitionConfigSchema
from dagster._core.errors import DagsterInvalidConfigDefinitionError, DagsterInvalidConfigError, DagsterInvalidDefinitionError, DagsterInvalidPythonicConfigDefinitionError
from .attach_other_object_to_context import IAttachDifferentObjectToOpContext as IAttachDifferentObjectToOpContext
try:
    from functools import cached_property
except ImportError:

    class cached_property:
        pass
import dagster._check as check
from dagster import Field, Selector
from dagster._config.field_utils import FIELD_NO_DEFAULT_PROVIDED, Map, convert_potential_field
from .pydantic_compat_layer import ModelFieldCompat, PydanticUndefined, model_fields
from .type_check_utils import is_optional, safe_is_subclass

def _apply_defaults_to_schema_field(field: Field, additional_default_values: Any) -> Field:
    if False:
        print('Hello World!')
    evr = validate_config(field.config_type, additional_default_values)
    if not evr.success:
        raise DagsterInvalidConfigError('Incorrect values passed to .configured', evr.errors, additional_default_values)
    if field.default_provided:
        defaults_processed_evr = resolve_defaults(field.config_type, additional_default_values)
        check.invariant(defaults_processed_evr.success, 'Since validation passed, this should always work.')
        default_to_pass = defaults_processed_evr.value
        return copy_with_default(field, default_to_pass)
    else:
        return copy_with_default(field, additional_default_values)

def copy_with_default(old_field: Field, new_config_value: Any) -> Field:
    if False:
        for i in range(10):
            print('nop')
    return Field(config=old_field.config_type, default_value=new_config_value, is_required=False, description=old_field.description)

def _curry_config_schema(schema_field: Field, data: Any) -> DefinitionConfigSchema:
    if False:
        return 10
    'Return a new config schema configured with the passed in data.'
    return DefinitionConfigSchema(_apply_defaults_to_schema_field(schema_field, data))
TResValue = TypeVar('TResValue')

def _convert_pydantic_field(pydantic_field: ModelFieldCompat, model_cls: Optional[Type]=None) -> Field:
    if False:
        for i in range(10):
            print('nop')
    'Transforms a Pydantic field into a corresponding Dagster config field.\n\n\n    Args:\n        pydantic_field (ModelFieldCompat): The Pydantic field to convert.\n        model_cls (Optional[Type]): The Pydantic model class that the field belongs to. This is\n            used for error messages.\n    '
    from .config import Config, infer_schema_from_config_class
    if pydantic_field.discriminator:
        return _convert_pydantic_discriminated_union_field(pydantic_field)
    field_type = pydantic_field.annotation
    if safe_is_subclass(field_type, Config):
        inferred_field = infer_schema_from_config_class(field_type, description=pydantic_field.description)
        return inferred_field
    else:
        if not pydantic_field.is_required() and (not is_optional(field_type)):
            field_type = Optional[field_type]
        config_type = _config_type_for_type_on_pydantic_field(field_type)
        return Field(config=config_type, description=pydantic_field.description, is_required=pydantic_field.is_required() and (not is_optional(field_type)), default_value=pydantic_field.default if pydantic_field.default is not PydanticUndefined else FIELD_NO_DEFAULT_PROVIDED)

def strip_wrapping_annotated_types(potentially_annotated_type: Any) -> Any:
    if False:
        return 10
    'For a type that is wrapped in Annotated, return the unwrapped type. Recursive,\n    so it will unwrap nested Annotated types.\n\n    e.g. Annotated[Annotated[List[str], "foo"], "bar] -> List[str]\n    '
    while get_origin(potentially_annotated_type) == Annotated:
        potentially_annotated_type = get_args(potentially_annotated_type)[0]
    return potentially_annotated_type

def _config_type_for_type_on_pydantic_field(potential_dagster_type: Any) -> ConfigType:
    if False:
        return 10
    "Generates a Dagster ConfigType from a Pydantic field's Python type.\n\n    Args:\n        potential_dagster_type (Any): The Python type of the Pydantic field.\n    "
    potential_dagster_type = strip_wrapping_annotated_types(potential_dagster_type)
    try:
        from pydantic import ConstrainedFloat, ConstrainedInt, ConstrainedStr
        if safe_is_subclass(potential_dagster_type, ConstrainedStr):
            return StringSource
        elif safe_is_subclass(potential_dagster_type, ConstrainedFloat):
            potential_dagster_type = float
        elif safe_is_subclass(potential_dagster_type, ConstrainedInt):
            return IntSource
    except ImportError:
        pass
    if safe_is_subclass(get_origin(potential_dagster_type), List):
        list_inner_type = get_args(potential_dagster_type)[0]
        return Array(_config_type_for_type_on_pydantic_field(list_inner_type))
    elif is_optional(potential_dagster_type):
        optional_inner_type = next((arg for arg in get_args(potential_dagster_type) if arg is not type(None)))
        return Noneable(_config_type_for_type_on_pydantic_field(optional_inner_type))
    elif safe_is_subclass(get_origin(potential_dagster_type), Dict) or safe_is_subclass(get_origin(potential_dagster_type), Mapping):
        (key_type, value_type) = get_args(potential_dagster_type)
        return Map(key_type, _config_type_for_type_on_pydantic_field(value_type))
    from .config import Config, infer_schema_from_config_class
    if safe_is_subclass(potential_dagster_type, Config):
        inferred_field = infer_schema_from_config_class(potential_dagster_type)
        return inferred_field.config_type
    if safe_is_subclass(potential_dagster_type, Enum):
        return DagsterEnum.from_python_enum_direct_values(potential_dagster_type)
    if potential_dagster_type is str:
        return StringSource
    elif potential_dagster_type is int:
        return IntSource
    elif potential_dagster_type is bool:
        return BoolSource
    else:
        return convert_potential_field(potential_dagster_type).config_type

def _convert_pydantic_discriminated_union_field(pydantic_field: ModelFieldCompat) -> Field:
    if False:
        for i in range(10):
            print('nop')
    'Builds a Selector config field from a Pydantic field which is a discriminated union.\n\n    For example:\n\n    class Cat(Config):\n        pet_type: Literal["cat"]\n        meows: int\n\n    class Dog(Config):\n        pet_type: Literal["dog"]\n        barks: float\n\n    class OpConfigWithUnion(Config):\n        pet: Union[Cat, Dog] = Field(..., discriminator="pet_type")\n\n    Becomes:\n\n    Shape({\n      "pet": Selector({\n          "cat": Shape({"meows": Int}),\n          "dog": Shape({"barks": Float}),\n      })\n    })\n    '
    from .config import Config, infer_schema_from_config_class
    field_type = pydantic_field.annotation
    discriminator = pydantic_field.discriminator if pydantic_field.discriminator else None
    if not get_origin(field_type) == Union:
        raise DagsterInvalidDefinitionError('Discriminated union must be a Union type.')
    sub_fields = get_args(field_type)
    if not all((issubclass(sub_field, Config) for sub_field in sub_fields)):
        raise NotImplementedError('Discriminated unions with non-Config types are not supported.')
    sub_fields_mapping = {}
    if discriminator:
        for sub_field in sub_fields:
            sub_field_annotation = model_fields(sub_field)[discriminator].annotation
            for sub_field_key in get_args(sub_field_annotation):
                sub_fields_mapping[sub_field_key] = sub_field
    dagster_config_field_mapping = {discriminator_value: infer_schema_from_config_class(field, fields_to_omit={discriminator} if discriminator else None) for (discriminator_value, field) in sub_fields_mapping.items()}
    return Field(config=Selector(fields=dagster_config_field_mapping))

def infer_schema_from_config_annotation(model_cls: Any, config_arg_default: Any) -> Field:
    if False:
        return 10
    'Parses a structured config class or primitive type and returns a corresponding Dagster config Field.'
    from .config import Config, infer_schema_from_config_class
    if safe_is_subclass(model_cls, Config):
        check.invariant(config_arg_default is inspect.Parameter.empty, 'Cannot provide a default value when using a Config class')
        return infer_schema_from_config_class(model_cls)
    try:
        inner_config_type = _config_type_for_type_on_pydantic_field(model_cls)
    except (DagsterInvalidDefinitionError, DagsterInvalidConfigDefinitionError):
        raise DagsterInvalidPythonicConfigDefinitionError(invalid_type=model_cls, config_class=None, field_name=None)
    return Field(config=inner_config_type, default_value=FIELD_NO_DEFAULT_PROVIDED if config_arg_default is inspect.Parameter.empty else config_arg_default)