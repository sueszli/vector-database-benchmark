from __future__ import annotations
import sys
import types
import typing
from collections import ChainMap
from contextlib import contextmanager
from contextvars import ContextVar
from types import prepare_class
from typing import TYPE_CHECKING, Any, Iterator, List, Mapping, MutableMapping, Tuple, TypeVar
from weakref import WeakValueDictionary
import typing_extensions
from ._core_utils import get_type_ref
from ._forward_ref import PydanticRecursiveRef
from ._typing_extra import TypeVarType, typing_base
from ._utils import all_identical, is_model_class
if sys.version_info >= (3, 10):
    from typing import _UnionGenericAlias
if TYPE_CHECKING:
    from ..main import BaseModel
GenericTypesCacheKey = Tuple[Any, Any, Tuple[Any, ...]]
KT = TypeVar('KT')
VT = TypeVar('VT')
_LIMITED_DICT_SIZE = 100
if TYPE_CHECKING:

    class LimitedDict(dict, MutableMapping[KT, VT]):

        def __init__(self, size_limit: int=_LIMITED_DICT_SIZE):
            if False:
                return 10
            ...
else:

    class LimitedDict(dict):
        """Limit the size/length of a dict used for caching to avoid unlimited increase in memory usage.

        Since the dict is ordered, and we always remove elements from the beginning, this is effectively a FIFO cache.
        """

        def __init__(self, size_limit: int=_LIMITED_DICT_SIZE):
            if False:
                for i in range(10):
                    print('nop')
            self.size_limit = size_limit
            super().__init__()

        def __setitem__(self, __key: Any, __value: Any) -> None:
            if False:
                for i in range(10):
                    print('nop')
            super().__setitem__(__key, __value)
            if len(self) > self.size_limit:
                excess = len(self) - self.size_limit + self.size_limit // 10
                to_remove = list(self.keys())[:excess]
                for key in to_remove:
                    del self[key]
if sys.version_info >= (3, 9):
    GenericTypesCache = WeakValueDictionary[GenericTypesCacheKey, 'type[BaseModel]']
else:
    GenericTypesCache = WeakValueDictionary
if TYPE_CHECKING:

    class DeepChainMap(ChainMap[KT, VT]):
        ...
else:

    class DeepChainMap(ChainMap):
        """Variant of ChainMap that allows direct updates to inner scopes.

        Taken from https://docs.python.org/3/library/collections.html#collections.ChainMap,
        with some light modifications for this use case.
        """

        def clear(self) -> None:
            if False:
                return 10
            for mapping in self.maps:
                mapping.clear()

        def __setitem__(self, key: KT, value: VT) -> None:
            if False:
                print('Hello World!')
            for mapping in self.maps:
                mapping[key] = value

        def __delitem__(self, key: KT) -> None:
            if False:
                for i in range(10):
                    print('nop')
            hit = False
            for mapping in self.maps:
                if key in mapping:
                    del mapping[key]
                    hit = True
            if not hit:
                raise KeyError(key)
_GENERIC_TYPES_CACHE = GenericTypesCache()

class PydanticGenericMetadata(typing_extensions.TypedDict):
    origin: type[BaseModel] | None
    args: tuple[Any, ...]
    parameters: tuple[type[Any], ...]

def create_generic_submodel(model_name: str, origin: type[BaseModel], args: tuple[Any, ...], params: tuple[Any, ...]) -> type[BaseModel]:
    if False:
        for i in range(10):
            print('nop')
    'Dynamically create a submodel of a provided (generic) BaseModel.\n\n    This is used when producing concrete parametrizations of generic models. This function\n    only *creates* the new subclass; the schema/validators/serialization must be updated to\n    reflect a concrete parametrization elsewhere.\n\n    Args:\n        model_name: The name of the newly created model.\n        origin: The base class for the new model to inherit from.\n        args: A tuple of generic metadata arguments.\n        params: A tuple of generic metadata parameters.\n\n    Returns:\n        The created submodel.\n    '
    namespace: dict[str, Any] = {'__module__': origin.__module__}
    bases = (origin,)
    (meta, ns, kwds) = prepare_class(model_name, bases)
    namespace.update(ns)
    created_model = meta(model_name, bases, namespace, __pydantic_generic_metadata__={'origin': origin, 'args': args, 'parameters': params}, __pydantic_reset_parent_namespace__=False, **kwds)
    (model_module, called_globally) = _get_caller_frame_info(depth=3)
    if called_globally:
        object_by_reference = None
        reference_name = model_name
        reference_module_globals = sys.modules[created_model.__module__].__dict__
        while object_by_reference is not created_model:
            object_by_reference = reference_module_globals.setdefault(reference_name, created_model)
            reference_name += '_'
    return created_model

def _get_caller_frame_info(depth: int=2) -> tuple[str | None, bool]:
    if False:
        print('Hello World!')
    'Used inside a function to check whether it was called globally.\n\n    Args:\n        depth: The depth to get the frame.\n\n    Returns:\n        A tuple contains `module_nam` and `called_globally`.\n\n    Raises:\n        RuntimeError: If the function is not called inside a function.\n    '
    try:
        previous_caller_frame = sys._getframe(depth)
    except ValueError as e:
        raise RuntimeError('This function must be used inside another function') from e
    except AttributeError:
        return (None, False)
    frame_globals = previous_caller_frame.f_globals
    return (frame_globals.get('__name__'), previous_caller_frame.f_locals is frame_globals)
DictValues: type[Any] = {}.values().__class__

def iter_contained_typevars(v: Any) -> Iterator[TypeVarType]:
    if False:
        for i in range(10):
            print('nop')
    "Recursively iterate through all subtypes and type args of `v` and yield any typevars that are found.\n\n    This is inspired as an alternative to directly accessing the `__parameters__` attribute of a GenericAlias,\n    since __parameters__ of (nested) generic BaseModel subclasses won't show up in that list.\n    "
    if isinstance(v, TypeVar):
        yield v
    elif is_model_class(v):
        yield from v.__pydantic_generic_metadata__['parameters']
    elif isinstance(v, (DictValues, list)):
        for var in v:
            yield from iter_contained_typevars(var)
    else:
        args = get_args(v)
        for arg in args:
            yield from iter_contained_typevars(arg)

def get_args(v: Any) -> Any:
    if False:
        print('Hello World!')
    pydantic_generic_metadata: PydanticGenericMetadata | None = getattr(v, '__pydantic_generic_metadata__', None)
    if pydantic_generic_metadata:
        return pydantic_generic_metadata.get('args')
    return typing_extensions.get_args(v)

def get_origin(v: Any) -> Any:
    if False:
        print('Hello World!')
    pydantic_generic_metadata: PydanticGenericMetadata | None = getattr(v, '__pydantic_generic_metadata__', None)
    if pydantic_generic_metadata:
        return pydantic_generic_metadata.get('origin')
    return typing_extensions.get_origin(v)

def get_standard_typevars_map(cls: type[Any]) -> dict[TypeVarType, Any] | None:
    if False:
        return 10
    "Package a generic type's typevars and parametrization (if present) into a dictionary compatible with the\n    `replace_types` function. Specifically, this works with standard typing generics and typing._GenericAlias.\n    "
    origin = get_origin(cls)
    if origin is None:
        return None
    if not hasattr(origin, '__parameters__'):
        return None
    args: tuple[Any, ...] = cls.__args__
    parameters: tuple[TypeVarType, ...] = origin.__parameters__
    return dict(zip(parameters, args))

def get_model_typevars_map(cls: type[BaseModel]) -> dict[TypeVarType, Any] | None:
    if False:
        return 10
    "Package a generic BaseModel's typevars and concrete parametrization (if present) into a dictionary compatible\n    with the `replace_types` function.\n\n    Since BaseModel.__class_getitem__ does not produce a typing._GenericAlias, and the BaseModel generic info is\n    stored in the __pydantic_generic_metadata__ attribute, we need special handling here.\n    "
    generic_metadata = cls.__pydantic_generic_metadata__
    origin = generic_metadata['origin']
    args = generic_metadata['args']
    return dict(zip(iter_contained_typevars(origin), args))

def replace_types(type_: Any, type_map: Mapping[Any, Any] | None) -> Any:
    if False:
        for i in range(10):
            print('nop')
    'Return type with all occurrences of `type_map` keys recursively replaced with their values.\n\n    Args:\n        type_: The class or generic alias.\n        type_map: Mapping from `TypeVar` instance to concrete types.\n\n    Returns:\n        A new type representing the basic structure of `type_` with all\n        `typevar_map` keys recursively replaced.\n\n    Example:\n        ```py\n        from typing import List, Tuple, Union\n\n        from pydantic._internal._generics import replace_types\n\n        replace_types(Tuple[str, Union[List[str], float]], {str: int})\n        #> Tuple[int, Union[List[int], float]]\n        ```\n    '
    if not type_map:
        return type_
    type_args = get_args(type_)
    origin_type = get_origin(type_)
    if origin_type is typing_extensions.Annotated:
        (annotated_type, *annotations) = type_args
        annotated = replace_types(annotated_type, type_map)
        for annotation in annotations:
            annotated = typing_extensions.Annotated[annotated, annotation]
        return annotated
    if type_args:
        resolved_type_args = tuple((replace_types(arg, type_map) for arg in type_args))
        if all_identical(type_args, resolved_type_args):
            return type_
        if origin_type is not None and isinstance(type_, typing_base) and (not isinstance(origin_type, typing_base)) and (getattr(type_, '_name', None) is not None):
            origin_type = getattr(typing, type_._name)
        assert origin_type is not None
        if sys.version_info >= (3, 10) and origin_type is types.UnionType:
            return _UnionGenericAlias(origin_type, resolved_type_args)
        return origin_type[resolved_type_args[0] if len(resolved_type_args) == 1 else resolved_type_args]
    if not origin_type and is_model_class(type_):
        parameters = type_.__pydantic_generic_metadata__['parameters']
        if not parameters:
            return type_
        resolved_type_args = tuple((replace_types(t, type_map) for t in parameters))
        if all_identical(parameters, resolved_type_args):
            return type_
        return type_[resolved_type_args]
    if isinstance(type_, (List, list)):
        resolved_list = list((replace_types(element, type_map) for element in type_))
        if all_identical(type_, resolved_list):
            return type_
        return resolved_list
    return type_map.get(type_, type_)

def has_instance_in_type(type_: Any, isinstance_target: Any) -> bool:
    if False:
        while True:
            i = 10
    'Checks if the type, or any of its arbitrary nested args, satisfy\n    `isinstance(<type>, isinstance_target)`.\n    '
    if isinstance(type_, isinstance_target):
        return True
    type_args = get_args(type_)
    origin_type = get_origin(type_)
    if origin_type is typing_extensions.Annotated:
        (annotated_type, *annotations) = type_args
        return has_instance_in_type(annotated_type, isinstance_target)
    if any((has_instance_in_type(a, isinstance_target) for a in type_args)):
        return True
    if isinstance(type_, (List, list)) and (not isinstance(type_, typing_extensions.ParamSpec)):
        if any((has_instance_in_type(element, isinstance_target) for element in type_)):
            return True
    return False

def check_parameters_count(cls: type[BaseModel], parameters: tuple[Any, ...]) -> None:
    if False:
        i = 10
        return i + 15
    'Check the generic model parameters count is equal.\n\n    Args:\n        cls: The generic model.\n        parameters: A tuple of passed parameters to the generic model.\n\n    Raises:\n        TypeError: If the passed parameters count is not equal to generic model parameters count.\n    '
    actual = len(parameters)
    expected = len(cls.__pydantic_generic_metadata__['parameters'])
    if actual != expected:
        description = 'many' if actual > expected else 'few'
        raise TypeError(f'Too {description} parameters for {cls}; actual {actual}, expected {expected}')
_generic_recursion_cache: ContextVar[set[str] | None] = ContextVar('_generic_recursion_cache', default=None)

@contextmanager
def generic_recursion_self_type(origin: type[BaseModel], args: tuple[Any, ...]) -> Iterator[PydanticRecursiveRef | None]:
    if False:
        print('Hello World!')
    'This contextmanager should be placed around the recursive calls used to build a generic type,\n    and accept as arguments the generic origin type and the type arguments being passed to it.\n\n    If the same origin and arguments are observed twice, it implies that a self-reference placeholder\n    can be used while building the core schema, and will produce a schema_ref that will be valid in the\n    final parent schema.\n    '
    previously_seen_type_refs = _generic_recursion_cache.get()
    if previously_seen_type_refs is None:
        previously_seen_type_refs = set()
        token = _generic_recursion_cache.set(previously_seen_type_refs)
    else:
        token = None
    try:
        type_ref = get_type_ref(origin, args_override=args)
        if type_ref in previously_seen_type_refs:
            self_type = PydanticRecursiveRef(type_ref=type_ref)
            yield self_type
        else:
            previously_seen_type_refs.add(type_ref)
            yield None
    finally:
        if token:
            _generic_recursion_cache.reset(token)

def recursively_defined_type_refs() -> set[str]:
    if False:
        i = 10
        return i + 15
    visited = _generic_recursion_cache.get()
    if not visited:
        return set()
    return visited.copy()

def get_cached_generic_type_early(parent: type[BaseModel], typevar_values: Any) -> type[BaseModel] | None:
    if False:
        print('Hello World!')
    'The use of a two-stage cache lookup approach was necessary to have the highest performance possible for\n    repeated calls to `__class_getitem__` on generic types (which may happen in tighter loops during runtime),\n    while still ensuring that certain alternative parametrizations ultimately resolve to the same type.\n\n    As a concrete example, this approach was necessary to make Model[List[T]][int] equal to Model[List[int]].\n    The approach could be modified to not use two different cache keys at different points, but the\n    _early_cache_key is optimized to be as quick to compute as possible (for repeated-access speed), and the\n    _late_cache_key is optimized to be as "correct" as possible, so that two types that will ultimately be the\n    same after resolving the type arguments will always produce cache hits.\n\n    If we wanted to move to only using a single cache key per type, we would either need to always use the\n    slower/more computationally intensive logic associated with _late_cache_key, or would need to accept\n    that Model[List[T]][int] is a different type than Model[List[T]][int]. Because we rely on subclass relationships\n    during validation, I think it is worthwhile to ensure that types that are functionally equivalent are actually\n    equal.\n    '
    return _GENERIC_TYPES_CACHE.get(_early_cache_key(parent, typevar_values))

def get_cached_generic_type_late(parent: type[BaseModel], typevar_values: Any, origin: type[BaseModel], args: tuple[Any, ...]) -> type[BaseModel] | None:
    if False:
        return 10
    'See the docstring of `get_cached_generic_type_early` for more information about the two-stage cache lookup.'
    cached = _GENERIC_TYPES_CACHE.get(_late_cache_key(origin, args, typevar_values))
    if cached is not None:
        set_cached_generic_type(parent, typevar_values, cached, origin, args)
    return cached

def set_cached_generic_type(parent: type[BaseModel], typevar_values: tuple[Any, ...], type_: type[BaseModel], origin: type[BaseModel] | None=None, args: tuple[Any, ...] | None=None) -> None:
    if False:
        i = 10
        return i + 15
    'See the docstring of `get_cached_generic_type_early` for more information about why items are cached with\n    two different keys.\n    '
    _GENERIC_TYPES_CACHE[_early_cache_key(parent, typevar_values)] = type_
    if len(typevar_values) == 1:
        _GENERIC_TYPES_CACHE[_early_cache_key(parent, typevar_values[0])] = type_
    if origin and args:
        _GENERIC_TYPES_CACHE[_late_cache_key(origin, args, typevar_values)] = type_

def _union_orderings_key(typevar_values: Any) -> Any:
    if False:
        i = 10
        return i + 15
    'This is intended to help differentiate between Union types with the same arguments in different order.\n\n    Thanks to caching internal to the `typing` module, it is not possible to distinguish between\n    List[Union[int, float]] and List[Union[float, int]] (and similarly for other "parent" origins besides List)\n    because `typing` considers Union[int, float] to be equal to Union[float, int].\n\n    However, you _can_ distinguish between (top-level) Union[int, float] vs. Union[float, int].\n    Because we parse items as the first Union type that is successful, we get slightly more consistent behavior\n    if we make an effort to distinguish the ordering of items in a union. It would be best if we could _always_\n    get the exact-correct order of items in the union, but that would require a change to the `typing` module itself.\n    (See https://github.com/python/cpython/issues/86483 for reference.)\n    '
    if isinstance(typevar_values, tuple):
        args_data = []
        for value in typevar_values:
            args_data.append(_union_orderings_key(value))
        return tuple(args_data)
    elif typing_extensions.get_origin(typevar_values) is typing.Union:
        return get_args(typevar_values)
    else:
        return ()

def _early_cache_key(cls: type[BaseModel], typevar_values: Any) -> GenericTypesCacheKey:
    if False:
        i = 10
        return i + 15
    "This is intended for minimal computational overhead during lookups of cached types.\n\n    Note that this is overly simplistic, and it's possible that two different cls/typevar_values\n    inputs would ultimately result in the same type being created in BaseModel.__class_getitem__.\n    To handle this, we have a fallback _late_cache_key that is checked later if the _early_cache_key\n    lookup fails, and should result in a cache hit _precisely_ when the inputs to __class_getitem__\n    would result in the same type.\n    "
    return (cls, typevar_values, _union_orderings_key(typevar_values))

def _late_cache_key(origin: type[BaseModel], args: tuple[Any, ...], typevar_values: Any) -> GenericTypesCacheKey:
    if False:
        i = 10
        return i + 15
    'This is intended for use later in the process of creating a new type, when we have more information\n    about the exact args that will be passed. If it turns out that a different set of inputs to\n    __class_getitem__ resulted in the same inputs to the generic type creation process, we can still\n    return the cached type, and update the cache with the _early_cache_key as well.\n    '
    return (_union_orderings_key(typevar_values), origin, args)