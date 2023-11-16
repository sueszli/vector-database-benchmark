"""Private logic for creating pydantic dataclasses."""
from __future__ import annotations as _annotations
import dataclasses
import inspect
import typing
import warnings
from functools import partial, wraps
from inspect import Parameter, Signature
from typing import Any, Callable, ClassVar
from pydantic_core import ArgsKwargs, PydanticUndefined, SchemaSerializer, SchemaValidator, core_schema
from typing_extensions import TypeGuard
from ..errors import PydanticUndefinedAnnotation
from ..fields import FieldInfo
from ..plugin._schema_validator import create_schema_validator
from ..warnings import PydanticDeprecatedSince20
from . import _config, _decorators, _typing_extra
from ._config import ConfigWrapper
from ._fields import collect_dataclass_fields
from ._generate_schema import GenerateSchema, generate_pydantic_signature
from ._generics import get_standard_typevars_map
from ._mock_val_ser import set_dataclass_mocks
from ._schema_generation_shared import CallbackGetCoreSchemaHandler
from ._utils import is_valid_identifier
if typing.TYPE_CHECKING:
    from ..config import ConfigDict

    class StandardDataclass(typing.Protocol):
        __dataclass_fields__: ClassVar[dict[str, Any]]
        __dataclass_params__: ClassVar[Any]
        __post_init__: ClassVar[Callable[..., None]]

        def __init__(self, *args: object, **kwargs: object) -> None:
            if False:
                return 10
            pass

    class PydanticDataclass(StandardDataclass, typing.Protocol):
        """A protocol containing attributes only available once a class has been decorated as a Pydantic dataclass.

        Attributes:
            __pydantic_config__: Pydantic-specific configuration settings for the dataclass.
            __pydantic_complete__: Whether dataclass building is completed, or if there are still undefined fields.
            __pydantic_core_schema__: The pydantic-core schema used to build the SchemaValidator and SchemaSerializer.
            __pydantic_decorators__: Metadata containing the decorators defined on the dataclass.
            __pydantic_fields__: Metadata about the fields defined on the dataclass.
            __pydantic_serializer__: The pydantic-core SchemaSerializer used to dump instances of the dataclass.
            __pydantic_validator__: The pydantic-core SchemaValidator used to validate instances of the dataclass.
        """
        __pydantic_config__: ClassVar[ConfigDict]
        __pydantic_complete__: ClassVar[bool]
        __pydantic_core_schema__: ClassVar[core_schema.CoreSchema]
        __pydantic_decorators__: ClassVar[_decorators.DecoratorInfos]
        __pydantic_fields__: ClassVar[dict[str, FieldInfo]]
        __pydantic_serializer__: ClassVar[SchemaSerializer]
        __pydantic_validator__: ClassVar[SchemaValidator]
else:
    DeprecationWarning = PydanticDeprecatedSince20

def set_dataclass_fields(cls: type[StandardDataclass], types_namespace: dict[str, Any] | None=None) -> None:
    if False:
        while True:
            i = 10
    'Collect and set `cls.__pydantic_fields__`.\n\n    Args:\n        cls: The class.\n        types_namespace: The types namespace, defaults to `None`.\n    '
    typevars_map = get_standard_typevars_map(cls)
    fields = collect_dataclass_fields(cls, types_namespace, typevars_map=typevars_map)
    cls.__pydantic_fields__ = fields

def complete_dataclass(cls: type[Any], config_wrapper: _config.ConfigWrapper, *, raise_errors: bool=True, types_namespace: dict[str, Any] | None) -> bool:
    if False:
        return 10
    'Finish building a pydantic dataclass.\n\n    This logic is called on a class which has already been wrapped in `dataclasses.dataclass()`.\n\n    This is somewhat analogous to `pydantic._internal._model_construction.complete_model_class`.\n\n    Args:\n        cls: The class.\n        config_wrapper: The config wrapper instance.\n        raise_errors: Whether to raise errors, defaults to `True`.\n        types_namespace: The types namespace.\n\n    Returns:\n        `True` if building a pydantic dataclass is successfully completed, `False` otherwise.\n\n    Raises:\n        PydanticUndefinedAnnotation: If `raise_error` is `True` and there is an undefined annotations.\n    '
    if hasattr(cls, '__post_init_post_parse__'):
        warnings.warn('Support for `__post_init_post_parse__` has been dropped, the method will not be called', DeprecationWarning)
    if types_namespace is None:
        types_namespace = _typing_extra.get_cls_types_namespace(cls)
    set_dataclass_fields(cls, types_namespace)
    typevars_map = get_standard_typevars_map(cls)
    gen_schema = GenerateSchema(config_wrapper, types_namespace, typevars_map)
    sig = generate_dataclass_signature(cls, cls.__pydantic_fields__, config_wrapper)

    def __init__(__dataclass_self__: PydanticDataclass, *args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        __tracebackhide__ = True
        s = __dataclass_self__
        s.__pydantic_validator__.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
    __init__.__qualname__ = f'{cls.__qualname__}.__init__'
    cls.__init__ = __init__
    cls.__pydantic_config__ = config_wrapper.config_dict
    cls.__signature__ = sig
    get_core_schema = getattr(cls, '__get_pydantic_core_schema__', None)
    try:
        if get_core_schema:
            schema = get_core_schema(cls, CallbackGetCoreSchemaHandler(partial(gen_schema.generate_schema, from_dunder_get_core_schema=False), gen_schema, ref_mode='unpack'))
        else:
            schema = gen_schema.generate_schema(cls, from_dunder_get_core_schema=False)
    except PydanticUndefinedAnnotation as e:
        if raise_errors:
            raise
        set_dataclass_mocks(cls, cls.__name__, f'`{e.name}`')
        return False
    core_config = config_wrapper.core_config(cls)
    try:
        schema = gen_schema.clean_schema(schema)
    except gen_schema.CollectedInvalid:
        set_dataclass_mocks(cls, cls.__name__, 'all referenced types')
        return False
    cls = typing.cast('type[PydanticDataclass]', cls)
    cls.__pydantic_core_schema__ = schema
    cls.__pydantic_validator__ = validator = create_schema_validator(schema, cls, cls.__module__, cls.__qualname__, 'dataclass', core_config, config_wrapper.plugin_settings)
    cls.__pydantic_serializer__ = SchemaSerializer(schema, core_config)
    if config_wrapper.validate_assignment:

        @wraps(cls.__setattr__)
        def validated_setattr(instance: Any, __field: str, __value: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            validator.validate_assignment(instance, __field, __value)
        cls.__setattr__ = validated_setattr.__get__(None, cls)
    return True

def process_param_defaults(param: Parameter) -> Parameter:
    if False:
        return 10
    'Custom processing where the parameter default is of type FieldInfo\n\n    Args:\n        param (Parameter): The parameter\n\n    Returns:\n        Parameter: The custom processed parameter\n    '
    param_default = param.default
    if isinstance(param_default, FieldInfo):
        annotation = param.annotation
        if annotation == 'Any':
            annotation = Any
        name = param.name
        alias = param_default.alias
        validation_alias = param_default.validation_alias
        if validation_alias is None and isinstance(alias, str) and is_valid_identifier(alias):
            name = alias
        elif isinstance(validation_alias, str) and is_valid_identifier(validation_alias):
            name = validation_alias
        default = param_default.default
        if default is PydanticUndefined:
            if param_default.default_factory is PydanticUndefined:
                default = inspect.Signature.empty
            else:
                default = dataclasses._HAS_DEFAULT_FACTORY
        return param.replace(annotation=annotation, name=name, default=default)
    return param

def generate_dataclass_signature(cls: type[StandardDataclass], fields: dict[str, FieldInfo], config_wrapper: ConfigWrapper) -> Signature:
    if False:
        i = 10
        return i + 15
    'Generate signature for a pydantic dataclass.\n\n    Args:\n        cls: The dataclass.\n        fields: The model fields.\n        config_wrapper: The config wrapper instance.\n\n    Returns:\n        The dataclass signature.\n    '
    return generate_pydantic_signature(init=cls.__init__, fields=fields, config_wrapper=config_wrapper, post_process_parameter=process_param_defaults)

def is_builtin_dataclass(_cls: type[Any]) -> TypeGuard[type[StandardDataclass]]:
    if False:
        print('Hello World!')
    "Returns True if a class is a stdlib dataclass and *not* a pydantic dataclass.\n\n    We check that\n    - `_cls` is a dataclass\n    - `_cls` does not inherit from a processed pydantic dataclass (and thus have a `__pydantic_validator__`)\n    - `_cls` does not have any annotations that are not dataclass fields\n    e.g.\n    ```py\n    import dataclasses\n\n    import pydantic.dataclasses\n\n    @dataclasses.dataclass\n    class A:\n        x: int\n\n    @pydantic.dataclasses.dataclass\n    class B(A):\n        y: int\n    ```\n    In this case, when we first check `B`, we make an extra check and look at the annotations ('y'),\n    which won't be a superset of all the dataclass fields (only the stdlib fields i.e. 'x')\n\n    Args:\n        cls: The class.\n\n    Returns:\n        `True` if the class is a stdlib dataclass, `False` otherwise.\n    "
    return dataclasses.is_dataclass(_cls) and (not hasattr(_cls, '__pydantic_validator__')) and set(_cls.__dataclass_fields__).issuperset(set(getattr(_cls, '__annotations__', {})))