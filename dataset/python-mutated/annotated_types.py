import sys
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, NamedTuple, Type
from .fields import Required
from .main import BaseModel, create_model
from .typing import is_typeddict, is_typeddict_special
if TYPE_CHECKING:
    from typing_extensions import TypedDict
if sys.version_info < (3, 11):

    def is_legacy_typeddict(typeddict_cls: Type['TypedDict']) -> bool:
        if False:
            return 10
        return is_typeddict(typeddict_cls) and type(typeddict_cls).__module__ == 'typing'
else:

    def is_legacy_typeddict(_: Any) -> Any:
        if False:
            i = 10
            return i + 15
        return False

def create_model_from_typeddict(typeddict_cls: Type['TypedDict'], **kwargs: Any) -> Type['BaseModel']:
    if False:
        print('Hello World!')
    '\n    Create a `BaseModel` based on the fields of a `TypedDict`.\n    Since `typing.TypedDict` in Python 3.8 does not store runtime information about optional keys,\n    we raise an error if this happens (see https://bugs.python.org/issue38834).\n    '
    field_definitions: Dict[str, Any]
    if not hasattr(typeddict_cls, '__required_keys__'):
        raise TypeError('You should use `typing_extensions.TypedDict` instead of `typing.TypedDict` with Python < 3.9.2. Without it, there is no way to differentiate required and optional fields when subclassed.')
    if is_legacy_typeddict(typeddict_cls) and any((is_typeddict_special(t) for t in typeddict_cls.__annotations__.values())):
        raise TypeError('You should use `typing_extensions.TypedDict` instead of `typing.TypedDict` with Python < 3.11. Without it, there is no way to reflect Required/NotRequired keys.')
    required_keys: FrozenSet[str] = typeddict_cls.__required_keys__
    field_definitions = {field_name: (field_type, Required if field_name in required_keys else None) for (field_name, field_type) in typeddict_cls.__annotations__.items()}
    return create_model(typeddict_cls.__name__, **kwargs, **field_definitions)

def create_model_from_namedtuple(namedtuple_cls: Type['NamedTuple'], **kwargs: Any) -> Type['BaseModel']:
    if False:
        while True:
            i = 10
    '\n    Create a `BaseModel` based on the fields of a named tuple.\n    A named tuple can be created with `typing.NamedTuple` and declared annotations\n    but also with `collections.namedtuple`, in this case we consider all fields\n    to have type `Any`.\n    '
    namedtuple_annotations: Dict[str, Type[Any]] = getattr(namedtuple_cls, '__annotations__', None) or {k: Any for k in namedtuple_cls._fields}
    field_definitions: Dict[str, Any] = {field_name: (field_type, Required) for (field_name, field_type) in namedtuple_annotations.items()}
    return create_model(namedtuple_cls.__name__, **kwargs, **field_definitions)