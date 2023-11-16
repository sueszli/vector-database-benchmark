"""Private logic for creating models."""
from __future__ import annotations as _annotations
import operator
import typing
import warnings
import weakref
from abc import ABCMeta
from functools import partial
from types import FunctionType
from typing import Any, Callable, Generic, Mapping
import typing_extensions
from pydantic_core import PydanticUndefined, SchemaSerializer
from typing_extensions import dataclass_transform, deprecated
from ..errors import PydanticUndefinedAnnotation, PydanticUserError
from ..plugin._schema_validator import create_schema_validator
from ..warnings import GenericBeforeBaseModelWarning, PydanticDeprecatedSince20
from ._config import ConfigWrapper
from ._decorators import DecoratorInfos, PydanticDescriptorProxy, get_attribute_from_bases
from ._fields import collect_model_fields, is_valid_field_name, is_valid_privateattr_name
from ._generate_schema import GenerateSchema, generate_pydantic_signature
from ._generics import PydanticGenericMetadata, get_model_typevars_map
from ._mock_val_ser import MockValSer, set_model_mocks
from ._schema_generation_shared import CallbackGetCoreSchemaHandler
from ._typing_extra import get_cls_types_namespace, is_annotated, is_classvar, parent_frame_namespace
from ._utils import ClassAttribute
from ._validate_call import ValidateCallWrapper
if typing.TYPE_CHECKING:
    from inspect import Signature
    from ..fields import Field as PydanticModelField
    from ..fields import FieldInfo, ModelPrivateAttr
    from ..main import BaseModel
else:
    DeprecationWarning = PydanticDeprecatedSince20
    PydanticModelField = object()
object_setattr = object.__setattr__

class _ModelNamespaceDict(dict):
    """A dictionary subclass that intercepts attribute setting on model classes and
    warns about overriding of decorators.
    """

    def __setitem__(self, k: str, v: object) -> None:
        if False:
            print('Hello World!')
        existing: Any = self.get(k, None)
        if existing and v is not existing and isinstance(existing, PydanticDescriptorProxy):
            warnings.warn(f'`{k}` overrides an existing Pydantic `{existing.decorator_info.decorator_repr}` decorator')
        return super().__setitem__(k, v)

@dataclass_transform(kw_only_default=True, field_specifiers=(PydanticModelField,))
class ModelMetaclass(ABCMeta):

    def __new__(mcs, cls_name: str, bases: tuple[type[Any], ...], namespace: dict[str, Any], __pydantic_generic_metadata__: PydanticGenericMetadata | None=None, __pydantic_reset_parent_namespace__: bool=True, _create_model_module: str | None=None, **kwargs: Any) -> type:
        if False:
            while True:
                i = 10
        'Metaclass for creating Pydantic models.\n\n        Args:\n            cls_name: The name of the class to be created.\n            bases: The base classes of the class to be created.\n            namespace: The attribute dictionary of the class to be created.\n            __pydantic_generic_metadata__: Metadata for generic models.\n            __pydantic_reset_parent_namespace__: Reset parent namespace.\n            _create_model_module: The module of the class to be created, if created by `create_model`.\n            **kwargs: Catch-all for any other keyword arguments.\n\n        Returns:\n            The new class created by the metaclass.\n        '
        if bases:
            (base_field_names, class_vars, base_private_attributes) = mcs._collect_bases_data(bases)
            config_wrapper = ConfigWrapper.for_model(bases, namespace, kwargs)
            namespace['model_config'] = config_wrapper.config_dict
            private_attributes = inspect_namespace(namespace, config_wrapper.ignored_types, class_vars, base_field_names)
            if private_attributes:
                original_model_post_init = get_model_post_init(namespace, bases)
                if original_model_post_init is not None:

                    def wrapped_model_post_init(self: BaseModel, __context: Any) -> None:
                        if False:
                            i = 10
                            return i + 15
                        'We need to both initialize private attributes and call the user-defined model_post_init\n                        method.\n                        '
                        init_private_attributes(self, __context)
                        original_model_post_init(self, __context)
                    namespace['model_post_init'] = wrapped_model_post_init
                else:
                    namespace['model_post_init'] = init_private_attributes
            namespace['__class_vars__'] = class_vars
            namespace['__private_attributes__'] = {**base_private_attributes, **private_attributes}
            cls: type[BaseModel] = super().__new__(mcs, cls_name, bases, namespace, **kwargs)
            from ..main import BaseModel
            mro = cls.__mro__
            if Generic in mro and mro.index(Generic) < mro.index(BaseModel):
                warnings.warn(GenericBeforeBaseModelWarning('Classes should inherit from `BaseModel` before generic classes (e.g. `typing.Generic[T]`) for pydantic generics to work properly.'), stacklevel=2)
            cls.__pydantic_custom_init__ = not getattr(cls.__init__, '__pydantic_base_init__', False)
            cls.__pydantic_post_init__ = None if cls.model_post_init is BaseModel.model_post_init else 'model_post_init'
            cls.__pydantic_decorators__ = DecoratorInfos.build(cls)
            if __pydantic_generic_metadata__:
                cls.__pydantic_generic_metadata__ = __pydantic_generic_metadata__
            else:
                parent_parameters = getattr(cls, '__pydantic_generic_metadata__', {}).get('parameters', ())
                parameters = getattr(cls, '__parameters__', None) or parent_parameters
                if parameters and parent_parameters and (not all((x in parameters for x in parent_parameters))):
                    combined_parameters = parent_parameters + tuple((x for x in parameters if x not in parent_parameters))
                    parameters_str = ', '.join([str(x) for x in combined_parameters])
                    generic_type_label = f'typing.Generic[{parameters_str}]'
                    error_message = f'All parameters must be present on typing.Generic; you should inherit from {generic_type_label}.'
                    if Generic not in bases:
                        bases_str = ', '.join([x.__name__ for x in bases] + [generic_type_label])
                        error_message += f' Note: `typing.Generic` must go last: `class {cls.__name__}({bases_str}): ...`)'
                    raise TypeError(error_message)
                cls.__pydantic_generic_metadata__ = {'origin': None, 'args': (), 'parameters': parameters}
            cls.__pydantic_complete__ = False
            for (name, obj) in private_attributes.items():
                obj.__set_name__(cls, name)
            if __pydantic_reset_parent_namespace__:
                cls.__pydantic_parent_namespace__ = build_lenient_weakvaluedict(parent_frame_namespace())
            parent_namespace = getattr(cls, '__pydantic_parent_namespace__', None)
            if isinstance(parent_namespace, dict):
                parent_namespace = unpack_lenient_weakvaluedict(parent_namespace)
            types_namespace = get_cls_types_namespace(cls, parent_namespace)
            set_model_fields(cls, bases, config_wrapper, types_namespace)
            if config_wrapper.frozen and '__hash__' not in namespace:
                set_default_hash_func(cls, bases)
            complete_model_class(cls, cls_name, config_wrapper, raise_errors=False, types_namespace=types_namespace, create_model_module=_create_model_module)
            super(cls, cls).__pydantic_init_subclass__(**kwargs)
            return cls
        else:
            return super().__new__(mcs, cls_name, bases, namespace, **kwargs)
    if not typing.TYPE_CHECKING:

        def __getattr__(self, item: str) -> Any:
            if False:
                i = 10
                return i + 15
            'This is necessary to keep attribute access working for class attribute access.'
            private_attributes = self.__dict__.get('__private_attributes__')
            if private_attributes and item in private_attributes:
                return private_attributes[item]
            if item == '__pydantic_core_schema__':
                maybe_mock_validator = getattr(self, '__pydantic_validator__', None)
                if isinstance(maybe_mock_validator, MockValSer):
                    rebuilt_validator = maybe_mock_validator.rebuild()
                    if rebuilt_validator is not None:
                        return getattr(self, '__pydantic_core_schema__')
            raise AttributeError(item)

    @classmethod
    def __prepare__(cls, *args: Any, **kwargs: Any) -> Mapping[str, object]:
        if False:
            i = 10
            return i + 15
        return _ModelNamespaceDict()

    def __instancecheck__(self, instance: Any) -> bool:
        if False:
            while True:
                i = 10
        "Avoid calling ABC _abc_subclasscheck unless we're pretty sure.\n\n        See #3829 and python/cpython#92810\n        "
        return hasattr(instance, '__pydantic_validator__') and super().__instancecheck__(instance)

    @staticmethod
    def _collect_bases_data(bases: tuple[type[Any], ...]) -> tuple[set[str], set[str], dict[str, ModelPrivateAttr]]:
        if False:
            print('Hello World!')
        from ..main import BaseModel
        field_names: set[str] = set()
        class_vars: set[str] = set()
        private_attributes: dict[str, ModelPrivateAttr] = {}
        for base in bases:
            if issubclass(base, BaseModel) and base is not BaseModel:
                field_names.update(getattr(base, 'model_fields', {}).keys())
                class_vars.update(base.__class_vars__)
                private_attributes.update(base.__private_attributes__)
        return (field_names, class_vars, private_attributes)

    @property
    @deprecated('The `__fields__` attribute is deprecated, use `model_fields` instead.', category=PydanticDeprecatedSince20)
    def __fields__(self) -> dict[str, FieldInfo]:
        if False:
            print('Hello World!')
        warnings.warn('The `__fields__` attribute is deprecated, use `model_fields` instead.', DeprecationWarning)
        return self.model_fields

def init_private_attributes(self: BaseModel, __context: Any) -> None:
    if False:
        i = 10
        return i + 15
    "This function is meant to behave like a BaseModel method to initialise private attributes.\n\n    It takes context as an argument since that's what pydantic-core passes when calling it.\n\n    Args:\n        self: The BaseModel instance.\n        __context: The context.\n    "
    if getattr(self, '__pydantic_private__', None) is None:
        pydantic_private = {}
        for (name, private_attr) in self.__private_attributes__.items():
            default = private_attr.get_default()
            if default is not PydanticUndefined:
                pydantic_private[name] = default
        object_setattr(self, '__pydantic_private__', pydantic_private)

def get_model_post_init(namespace: dict[str, Any], bases: tuple[type[Any], ...]) -> Callable[..., Any] | None:
    if False:
        print('Hello World!')
    'Get the `model_post_init` method from the namespace or the class bases, or `None` if not defined.'
    if 'model_post_init' in namespace:
        return namespace['model_post_init']
    from ..main import BaseModel
    model_post_init = get_attribute_from_bases(bases, 'model_post_init')
    if model_post_init is not BaseModel.model_post_init:
        return model_post_init

def inspect_namespace(namespace: dict[str, Any], ignored_types: tuple[type[Any], ...], base_class_vars: set[str], base_class_fields: set[str]) -> dict[str, ModelPrivateAttr]:
    if False:
        print('Hello World!')
    'Iterate over the namespace and:\n    * gather private attributes\n    * check for items which look like fields but are not (e.g. have no annotation) and warn.\n\n    Args:\n        namespace: The attribute dictionary of the class to be created.\n        ignored_types: A tuple of ignore types.\n        base_class_vars: A set of base class class variables.\n        base_class_fields: A set of base class fields.\n\n    Returns:\n        A dict contains private attributes info.\n\n    Raises:\n        TypeError: If there is a `__root__` field in model.\n        NameError: If private attribute name is invalid.\n        PydanticUserError:\n            - If a field does not have a type annotation.\n            - If a field on base class was overridden by a non-annotated attribute.\n    '
    from ..fields import FieldInfo, ModelPrivateAttr, PrivateAttr
    all_ignored_types = ignored_types + default_ignored_types()
    private_attributes: dict[str, ModelPrivateAttr] = {}
    raw_annotations = namespace.get('__annotations__', {})
    if '__root__' in raw_annotations or '__root__' in namespace:
        raise TypeError("To define root models, use `pydantic.RootModel` rather than a field called '__root__'")
    ignored_names: set[str] = set()
    for (var_name, value) in list(namespace.items()):
        if var_name == 'model_config':
            continue
        elif isinstance(value, type) and value.__module__ == namespace['__module__'] and value.__qualname__.startswith(namespace['__qualname__']):
            continue
        elif isinstance(value, all_ignored_types) or value.__class__.__module__ == 'functools':
            ignored_names.add(var_name)
            continue
        elif isinstance(value, ModelPrivateAttr):
            if var_name.startswith('__'):
                raise NameError(f'Private attributes must not use dunder names; use a single underscore prefix instead of {var_name!r}.')
            elif is_valid_field_name(var_name):
                raise NameError(f"Private attributes must not use valid field names; use sunder names, e.g. {'_' + var_name!r} instead of {var_name!r}.")
            private_attributes[var_name] = value
            del namespace[var_name]
        elif isinstance(value, FieldInfo) and (not is_valid_field_name(var_name)):
            suggested_name = var_name.lstrip('_') or 'my_field'
            raise NameError(f'Fields must not use names with leading underscores; e.g., use {suggested_name!r} instead of {var_name!r}.')
        elif var_name.startswith('__'):
            continue
        elif is_valid_privateattr_name(var_name):
            if var_name not in raw_annotations or not is_classvar(raw_annotations[var_name]):
                private_attributes[var_name] = PrivateAttr(default=value)
                del namespace[var_name]
        elif var_name in base_class_vars:
            continue
        elif var_name not in raw_annotations:
            if var_name in base_class_fields:
                raise PydanticUserError(f'Field {var_name!r} defined on a base class was overridden by a non-annotated attribute. All field definitions, including overrides, require a type annotation.', code='model-field-overridden')
            elif isinstance(value, FieldInfo):
                raise PydanticUserError(f'Field {var_name!r} requires a type annotation', code='model-field-missing-annotation')
            else:
                raise PydanticUserError(f"A non-annotated attribute was detected: `{var_name} = {value!r}`. All model fields require a type annotation; if `{var_name}` is not meant to be a field, you may be able to resolve this error by annotating it as a `ClassVar` or updating `model_config['ignored_types']`.", code='model-field-missing-annotation')
    for (ann_name, ann_type) in raw_annotations.items():
        if is_valid_privateattr_name(ann_name) and ann_name not in private_attributes and (ann_name not in ignored_names) and (not is_classvar(ann_type)) and (ann_type not in all_ignored_types) and (getattr(ann_type, '__module__', None) != 'functools'):
            if is_annotated(ann_type):
                (_, *metadata) = typing_extensions.get_args(ann_type)
                private_attr = next((v for v in metadata if isinstance(v, ModelPrivateAttr)), None)
                if private_attr is not None:
                    private_attributes[ann_name] = private_attr
                    continue
            private_attributes[ann_name] = PrivateAttr()
    return private_attributes

def set_default_hash_func(cls: type[BaseModel], bases: tuple[type[Any], ...]) -> None:
    if False:
        return 10
    base_hash_func = get_attribute_from_bases(bases, '__hash__')
    new_hash_func = make_hash_func(cls)
    if base_hash_func in {None, object.__hash__} or getattr(base_hash_func, '__code__', None) == new_hash_func.__code__:
        cls.__hash__ = new_hash_func

def make_hash_func(cls: type[BaseModel]) -> Any:
    if False:
        i = 10
        return i + 15
    getter = operator.itemgetter(*cls.model_fields.keys()) if cls.model_fields else lambda _: 0

    def hash_func(self: Any) -> int:
        if False:
            i = 10
            return i + 15
        try:
            return hash(getter(self.__dict__))
        except KeyError:
            return hash(getter(FallbackDict(self.__dict__)))
    return hash_func

class FallbackDict:

    def __init__(self, inner):
        if False:
            print('Hello World!')
        self.inner = inner

    def __getitem__(self, key):
        if False:
            return 10
        return self.inner.get(key)

def set_model_fields(cls: type[BaseModel], bases: tuple[type[Any], ...], config_wrapper: ConfigWrapper, types_namespace: dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    'Collect and set `cls.model_fields` and `cls.__class_vars__`.\n\n    Args:\n        cls: BaseModel or dataclass.\n        bases: Parents of the class, generally `cls.__bases__`.\n        config_wrapper: The config wrapper instance.\n        types_namespace: Optional extra namespace to look for types in.\n    '
    typevars_map = get_model_typevars_map(cls)
    (fields, class_vars) = collect_model_fields(cls, bases, config_wrapper, types_namespace, typevars_map=typevars_map)
    cls.model_fields = fields
    cls.__class_vars__.update(class_vars)
    for k in class_vars:
        value = cls.__private_attributes__.pop(k, None)
        if value is not None and value.default is not PydanticUndefined:
            setattr(cls, k, value.default)

def complete_model_class(cls: type[BaseModel], cls_name: str, config_wrapper: ConfigWrapper, *, raise_errors: bool=True, types_namespace: dict[str, Any] | None, create_model_module: str | None=None) -> bool:
    if False:
        i = 10
        return i + 15
    'Finish building a model class.\n\n    This logic must be called after class has been created since validation functions must be bound\n    and `get_type_hints` requires a class object.\n\n    Args:\n        cls: BaseModel or dataclass.\n        cls_name: The model or dataclass name.\n        config_wrapper: The config wrapper instance.\n        raise_errors: Whether to raise errors.\n        types_namespace: Optional extra namespace to look for types in.\n        create_model_module: The module of the class to be created, if created by `create_model`.\n\n    Returns:\n        `True` if the model is successfully completed, else `False`.\n\n    Raises:\n        PydanticUndefinedAnnotation: If `PydanticUndefinedAnnotation` occurs in`__get_pydantic_core_schema__`\n            and `raise_errors=True`.\n    '
    typevars_map = get_model_typevars_map(cls)
    gen_schema = GenerateSchema(config_wrapper, types_namespace, typevars_map)
    handler = CallbackGetCoreSchemaHandler(partial(gen_schema.generate_schema, from_dunder_get_core_schema=False), gen_schema, ref_mode='unpack')
    if config_wrapper.defer_build:
        set_model_mocks(cls, cls_name)
        return False
    try:
        schema = cls.__get_pydantic_core_schema__(cls, handler)
    except PydanticUndefinedAnnotation as e:
        if raise_errors:
            raise
        set_model_mocks(cls, cls_name, f'`{e.name}`')
        return False
    core_config = config_wrapper.core_config(cls)
    try:
        schema = gen_schema.clean_schema(schema)
    except gen_schema.CollectedInvalid:
        set_model_mocks(cls, cls_name)
        return False
    cls.__pydantic_core_schema__ = schema
    cls.__pydantic_validator__ = create_schema_validator(schema, cls, create_model_module or cls.__module__, cls.__qualname__, 'create_model' if create_model_module else 'BaseModel', core_config, config_wrapper.plugin_settings)
    cls.__pydantic_serializer__ = SchemaSerializer(schema, core_config)
    cls.__pydantic_complete__ = True
    cls.__signature__ = ClassAttribute('__signature__', generate_model_signature(cls.__init__, cls.model_fields, config_wrapper))
    return True

def generate_model_signature(init: Callable[..., None], fields: dict[str, FieldInfo], config_wrapper: ConfigWrapper) -> Signature:
    if False:
        return 10
    'Generate signature for model based on its fields.\n\n    Args:\n        init: The class init.\n        fields: The model fields.\n        config_wrapper: The config wrapper instance.\n\n    Returns:\n        The model signature.\n    '
    return generate_pydantic_signature(init, fields, config_wrapper)

class _PydanticWeakRef:
    """Wrapper for `weakref.ref` that enables `pickle` serialization.

    Cloudpickle fails to serialize `weakref.ref` objects due to an arcane error related
    to abstract base classes (`abc.ABC`). This class works around the issue by wrapping
    `weakref.ref` instead of subclassing it.

    See https://github.com/pydantic/pydantic/issues/6763 for context.

    Semantics:
        - If not pickled, behaves the same as a `weakref.ref`.
        - If pickled along with the referenced object, the same `weakref.ref` behavior
          will be maintained between them after unpickling.
        - If pickled without the referenced object, after unpickling the underlying
          reference will be cleared (`__call__` will always return `None`).
    """

    def __init__(self, obj: Any):
        if False:
            return 10
        if obj is None:
            self._wr = None
        else:
            self._wr = weakref.ref(obj)

    def __call__(self) -> Any:
        if False:
            print('Hello World!')
        if self._wr is None:
            return None
        else:
            return self._wr()

    def __reduce__(self) -> tuple[Callable, tuple[weakref.ReferenceType | None]]:
        if False:
            i = 10
            return i + 15
        return (_PydanticWeakRef, (self(),))

def build_lenient_weakvaluedict(d: dict[str, Any] | None) -> dict[str, Any] | None:
    if False:
        return 10
    "Takes an input dictionary, and produces a new value that (invertibly) replaces the values with weakrefs.\n\n    We can't just use a WeakValueDictionary because many types (including int, str, etc.) can't be stored as values\n    in a WeakValueDictionary.\n\n    The `unpack_lenient_weakvaluedict` function can be used to reverse this operation.\n    "
    if d is None:
        return None
    result = {}
    for (k, v) in d.items():
        try:
            proxy = _PydanticWeakRef(v)
        except TypeError:
            proxy = v
        result[k] = proxy
    return result

def unpack_lenient_weakvaluedict(d: dict[str, Any] | None) -> dict[str, Any] | None:
    if False:
        while True:
            i = 10
    'Inverts the transform performed by `build_lenient_weakvaluedict`.'
    if d is None:
        return None
    result = {}
    for (k, v) in d.items():
        if isinstance(v, _PydanticWeakRef):
            v = v()
            if v is not None:
                result[k] = v
        else:
            result[k] = v
    return result

def default_ignored_types() -> tuple[type[Any], ...]:
    if False:
        for i in range(10):
            print('nop')
    from ..fields import ComputedFieldInfo
    return (FunctionType, property, classmethod, staticmethod, PydanticDescriptorProxy, ComputedFieldInfo, ValidateCallWrapper)