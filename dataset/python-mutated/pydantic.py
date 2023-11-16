from functools import partial
from typing import Any, Callable, Generic, Type, TypeVar, cast, overload
from prefect._internal.pydantic import HAS_PYDANTIC_V2
if HAS_PYDANTIC_V2:
    import pydantic.v1 as pydantic
else:
    import pydantic
from jsonpatch import JsonPatch as JsonPatchBase
from typing_extensions import Self
from prefect.utilities.dispatch import get_dispatch_key, lookup_type, register_base_type
from prefect.utilities.importtools import from_qualified_name, to_qualified_name
D = TypeVar('D', bound=Any)
M = TypeVar('M', bound=pydantic.BaseModel)

def _reduce_model(model: pydantic.BaseModel):
    if False:
        return 10
    '\n    Helper for serializing a cythonized model with cloudpickle.\n\n    Keyword arguments can provide additional settings to the `json` call. Since\n    `__reduce__` takes no arguments, these are set on the `__reduce_kwargs__` attr.\n    '
    return (_unreduce_model, (to_qualified_name(type(model)), model.json(**getattr(model, '__reduce_kwargs__', {}))))

def _unreduce_model(model_name, json):
    if False:
        while True:
            i = 10
    'Helper for restoring model after serialization'
    model = from_qualified_name(model_name)
    return model.parse_raw(json)

@overload
def add_cloudpickle_reduction(__model_cls: Type[M]) -> Type[M]:
    if False:
        while True:
            i = 10
    ...

@overload
def add_cloudpickle_reduction(**kwargs: Any) -> Callable[[Type[M]], Type[M]]:
    if False:
        print('Hello World!')
    ...

def add_cloudpickle_reduction(__model_cls: Type[M]=None, **kwargs: Any):
    if False:
        while True:
            i = 10
    '\n    Adds a `__reducer__` to the given class that ensures it is cloudpickle compatible.\n\n    Workaround for issues with cloudpickle when using cythonized pydantic which\n    throws exceptions when attempting to pickle the class which has "compiled"\n    validator methods dynamically attached to it.\n\n    We cannot define this utility in the model class itself because the class is the\n    type that contains unserializable methods.\n\n    Any model using some features of Pydantic (e.g. `Path` validation) with a Cython\n    compiled Pydantic installation may encounter pickling issues.\n\n    See related issue at https://github.com/cloudpipe/cloudpickle/issues/408\n    '
    if __model_cls:
        __model_cls.__reduce__ = _reduce_model
        __model_cls.__reduce_kwargs__ = kwargs
        return __model_cls
    else:
        return cast(Callable[[Type[M]], Type[M]], partial(add_cloudpickle_reduction, **kwargs))

def get_class_fields_only(model: Type[pydantic.BaseModel]) -> set:
    if False:
        i = 10
        return i + 15
    '\n    Gets all the field names defined on the model class but not any parent classes.\n    Any fields that are on the parent but redefined on the subclass are included.\n    '
    subclass_class_fields = set(model.__annotations__.keys())
    parent_class_fields = set()
    for base in model.__class__.__bases__:
        if issubclass(base, pydantic.BaseModel):
            parent_class_fields.update(base.__annotations__.keys())
    return subclass_class_fields - parent_class_fields | subclass_class_fields & parent_class_fields

def add_type_dispatch(model_cls: Type[M]) -> Type[M]:
    if False:
        return 10
    "\n    Extend a Pydantic model to add a 'type' field that is used a discriminator field\n    to dynamically determine the subtype that when deserializing models.\n\n    This allows automatic resolution to subtypes of the decorated model.\n\n    If a type field already exists, it should be a string literal field that has a\n    constant value for each subclass. The default value of this field will be used as\n    the dispatch key.\n\n    If a type field does not exist, one will be added. In this case, the value of the\n    field will be set to the value of the `__dispatch_key__`. The base class should\n    define a `__dispatch_key__` class method that is used to determine the unique key\n    for each subclass. Alternatively, each subclass can define the `__dispatch_key__`\n    as a string literal.\n\n    The base class must not define a 'type' field. If it is not desirable to add a field\n    to the model and the dispatch key can be tracked separately, the lower level\n    utilities in `prefect.utilities.dispatch` should be used directly.\n    "
    defines_dispatch_key = hasattr(model_cls, '__dispatch_key__') or '__dispatch_key__' in getattr(model_cls, '__annotations__', {})
    defines_type_field = 'type' in model_cls.__fields__
    if not defines_dispatch_key and (not defines_type_field):
        raise ValueError(f'Model class {model_cls.__name__!r} does not define a `__dispatch_key__` or a type field. One of these is required for dispatch.')
    elif defines_dispatch_key and (not defines_type_field):
        model_cls.__fields__['type'] = pydantic.fields.ModelField(name='type', type_=str, required=True, class_validators=None, model_config=model_cls.__config__)
    elif not defines_dispatch_key and defines_type_field:
        field_type_annotation = model_cls.__fields__['type'].type_
        if field_type_annotation != str:
            raise TypeError(f"Model class {model_cls.__name__!r} defines a 'type' field with type {field_type_annotation.__name__!r} but it must be 'str'.")

        @classmethod
        def dispatch_key_from_type_field(cls):
            if False:
                for i in range(10):
                    print('nop')
            return cls.__fields__['type'].default
        model_cls.__dispatch_key__ = dispatch_key_from_type_field
    else:
        raise ValueError(f'Model class {model_cls.__name__!r} defines a `__dispatch_key__` and a type field. Only one of these may be defined for dispatch.')
    cls_init = model_cls.__init__
    cls_new = model_cls.__new__

    def __init__(__pydantic_self__, **data: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        type_string = get_dispatch_key(__pydantic_self__) if type(__pydantic_self__) != model_cls else '__base__'
        data.setdefault('type', type_string)
        cls_init(__pydantic_self__, **data)

    def __new__(cls: Type[Self], **kwargs) -> Self:
        if False:
            while True:
                i = 10
        if 'type' in kwargs:
            try:
                subcls = lookup_type(cls, dispatch_key=kwargs['type'])
            except KeyError as exc:
                raise pydantic.ValidationError(errors=[exc], model=cls)
            return cls_new(subcls)
        else:
            return cls_new(cls)
    model_cls.__init__ = __init__
    model_cls.__new__ = __new__
    register_base_type(model_cls)
    return model_cls

class PartialModel(Generic[M]):
    """
    A utility for creating a Pydantic model in several steps.

    Fields may be set at initialization, via attribute assignment, or at finalization
    when the concrete model is returned.

    Pydantic validation does not occur until finalization.

    Each field can only be set once and a `ValueError` will be raised on assignment if
    a field already has a value.

    Example:
        >>> class MyModel(pydantic.BaseModel):
        >>>     x: int
        >>>     y: str
        >>>     z: float
        >>>
        >>> partial_model = PartialModel(MyModel, x=1)
        >>> partial_model.y = "two"
        >>> model = partial_model.finalize(z=3.0)
    """

    def __init__(self, __model_cls: Type[M], **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self.fields = kwargs
        self.model_cls = __model_cls
        for name in kwargs.keys():
            self.raise_if_not_in_model(name)

    def finalize(self, **kwargs: Any) -> M:
        if False:
            for i in range(10):
                print('nop')
        for name in kwargs.keys():
            self.raise_if_already_set(name)
            self.raise_if_not_in_model(name)
        return self.model_cls(**self.fields, **kwargs)

    def raise_if_already_set(self, name):
        if False:
            i = 10
            return i + 15
        if name in self.fields:
            raise ValueError(f'Field {name!r} has already been set.')

    def raise_if_not_in_model(self, name):
        if False:
            i = 10
            return i + 15
        if name not in self.model_cls.__fields__:
            raise ValueError(f'Field {name!r} is not present in the model.')

    def __setattr__(self, __name: str, __value: Any) -> None:
        if False:
            print('Hello World!')
        if __name in {'fields', 'model_cls'}:
            return super().__setattr__(__name, __value)
        self.raise_if_already_set(__name)
        self.raise_if_not_in_model(__name)
        self.fields[__name] = __value

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        dsp_fields = ', '.join((f'{key}={repr(value)}' for (key, value) in self.fields.items()))
        return f'PartialModel(cls={self.model_cls.__name__}, {dsp_fields})'

class JsonPatch(JsonPatchBase):

    @classmethod
    def __modify_schema__(cls, field_schema):
        if False:
            return 10
        field_schema.update({'type': 'array', 'format': 'rfc6902', 'items': {'type': 'object', 'additionalProperties': {'type': 'string'}}})