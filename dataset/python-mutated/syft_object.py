from collections import defaultdict
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from collections.abc import Set
from hashlib import sha256
import inspect
from inspect import Signature
import re
import types
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import KeysView
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
import warnings
import pandas as pd
import pydantic
from pydantic import EmailStr
from pydantic.fields import Undefined
from result import OkErr
from typeguard import check_type
from ..node.credentials import SyftVerifyKey
from ..serde.recursive_primitives import recursive_serde_register_type
from ..serde.serialize import _serialize as serialize
from ..util.autoreload import autoreload_enabled
from ..util.markdown import as_markdown_python_code
from ..util.notebook_ui.notebook_addons import create_table_template
from ..util.util import aggressive_set_attr
from ..util.util import full_name_with_qualname
from ..util.util import get_qualname_for
from .dicttuple import DictTuple
from .syft_metaclass import Empty
from .syft_metaclass import PartialModelMetaclass
from .uid import UID
IntStr = Union[int, str]
AbstractSetIntStr = Set[IntStr]
MappingIntStrAny = Mapping[IntStr, Any]
SYFT_OBJECT_VERSION_1 = 1
SYFT_OBJECT_VERSION_2 = 2
SYFT_OBJECT_VERSION_3 = 3
supported_object_versions = [SYFT_OBJECT_VERSION_1, SYFT_OBJECT_VERSION_2, SYFT_OBJECT_VERSION_3]
HIGHEST_SYFT_OBJECT_VERSION = max(supported_object_versions)
LOWEST_SYFT_OBJECT_VERSION = min(supported_object_versions)
DYNAMIC_SYFT_ATTRIBUTES = ['syft_node_location', 'syft_client_verify_key']

class SyftHashableObject:
    __hash_exclude_attrs__ = []

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return int.from_bytes(self.__sha256__(), byteorder='big')

    def __sha256__(self) -> bytes:
        if False:
            return 10
        self.__hash_exclude_attrs__.extend(DYNAMIC_SYFT_ATTRIBUTES)
        _bytes = serialize(self, to_bytes=True, for_hashing=True)
        return sha256(_bytes).digest()

    def hash(self) -> str:
        if False:
            print('Hello World!')
        return self.__sha256__().hex()

class SyftBaseObject(pydantic.BaseModel, SyftHashableObject):

    class Config:
        arbitrary_types_allowed = True
    __canonical_name__: str
    __version__: int
    syft_node_location: Optional[UID]
    syft_client_verify_key: Optional[SyftVerifyKey]

    def _set_obj_location_(self, node_uid, credentials):
        if False:
            return 10
        self.syft_node_location = node_uid
        self.syft_client_verify_key = credentials

class Context(SyftBaseObject):
    __canonical_name__ = 'Context'
    __version__ = SYFT_OBJECT_VERSION_1
    pass

class SyftObjectRegistry:
    __object_version_registry__: Dict[str, Type['SyftObject']] = {}
    __object_transform_registry__: Dict[str, Callable] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init_subclass__(**kwargs)
        if hasattr(cls, '__canonical_name__') and hasattr(cls, '__version__'):
            mapping_string = f'{cls.__canonical_name__}_{cls.__version__}'
            if mapping_string in cls.__object_version_registry__ and (not autoreload_enabled()):
                current_cls = cls.__object_version_registry__[mapping_string]
                if cls == current_cls:
                    return None
                if 'syft.user' in cls.__module__:
                    return None
                else:
                    raise Exception(f'Duplicate mapping for {mapping_string} and {cls}')
            else:
                cls.__object_version_registry__[mapping_string] = cls

    @classmethod
    def versioned_class(cls, name: str, version: int) -> Optional[Type['SyftObject']]:
        if False:
            i = 10
            return i + 15
        mapping_string = f'{name}_{version}'
        if mapping_string not in cls.__object_version_registry__:
            return None
        return cls.__object_version_registry__[mapping_string]

    @classmethod
    def add_transform(cls, klass_from: str, version_from: int, klass_to: str, version_to: int, method: Callable) -> None:
        if False:
            while True:
                i = 10
        mapping_string = f'{klass_from}_{version_from}_x_{klass_to}_{version_to}'
        cls.__object_transform_registry__[mapping_string] = method

    @classmethod
    def get_transform(cls, type_from: Type['SyftObject'], type_to: Type['SyftObject']) -> Callable:
        if False:
            while True:
                i = 10
        for type_from_mro in type_from.mro():
            if issubclass(type_from_mro, SyftObject):
                klass_from = type_from_mro.__canonical_name__
                version_from = type_from_mro.__version__
            else:
                klass_from = type_from_mro.__name__
                version_from = None
            for type_to_mro in type_to.mro():
                if issubclass(type_to_mro, SyftBaseObject):
                    klass_to = type_to_mro.__canonical_name__
                    version_to = type_to_mro.__version__
                else:
                    klass_to = type_to_mro.__name__
                    version_to = None
                mapping_string = f'{klass_from}_{version_from}_x_{klass_to}_{version_to}'
                if mapping_string in cls.__object_transform_registry__:
                    return cls.__object_transform_registry__[mapping_string]
        raise Exception(f'No mapping found for: {type_from} to {type_to} inthe registry: {cls.__object_transform_registry__.keys()}')

class SyftMigrationRegistry:
    __migration_version_registry__: Dict[str, Dict[int, str]] = {}
    __migration_transform_registry__: Dict[str, Dict[str, Callable]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Populate the `__migration_version_registry__` dictionary with format\n        __migration_version_registry__ = {\n            "canonical_name": {version_number: "klass_name"}\n        }\n        For example\n        __migration_version_registry__ = {\n            \'APIEndpoint\': {1: \'syft.client.api.APIEndpoint\'},\n            \'Action\':      {1: \'syft.service.action.action_object.Action\'},\n        }\n        '
        super().__init_subclass__(**kwargs)
        klass = type(cls) if not isinstance(cls, type) else cls
        cls.register_version(klass=klass)

    @classmethod
    def register_version(cls, klass: type):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(klass, '__canonical_name__') and hasattr(klass, '__version__'):
            mapping_string = klass.__canonical_name__
            klass_version = klass.__version__
            fqn = f'{klass.__module__}.{klass.__name__}'
            if mapping_string in cls.__migration_version_registry__ and (not autoreload_enabled()):
                versions = cls.__migration_version_registry__[mapping_string]
                versions[klass_version] = fqn
            else:
                cls.__migration_version_registry__[mapping_string] = {klass_version: fqn}

    @classmethod
    def get_versions(cls, canonical_name: str) -> List[int]:
        if False:
            print('Hello World!')
        available_versions: Dict = cls.__migration_version_registry__.get(canonical_name, {})
        return list(available_versions.keys())

    @classmethod
    def register_transform(cls, klass_type_str: str, version_from: int, version_to: int, method: Callable) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Populate the __migration_transform_registry__ dictionary with format\n        __migration_version_registry__ = {\n            "canonical_name": {"version_from x version_to": <function transform_function>}\n        }\n        For example\n        {\'NodeMetadata\': {\'1x2\': <function transform_function>,\n                          \'2x1\': <function transform_function>}}\n        '
        if klass_type_str not in cls.__migration_version_registry__:
            raise Exception(f'{klass_type_str} is not yet registered.')
        available_versions = cls.__migration_version_registry__[klass_type_str]
        versions_exists = version_from in available_versions and version_to in available_versions
        if versions_exists:
            mapping_string = f'{version_from}x{version_to}'
            if klass_type_str not in cls.__migration_transform_registry__:
                cls.__migration_transform_registry__[klass_type_str] = {}
            cls.__migration_transform_registry__[klass_type_str][mapping_string] = method
        else:
            raise Exception(f"Available versions for {klass_type_str} are: {available_versions}.You're trying to add a transform from version: {version_from} to version: {version_to}")

    @classmethod
    def get_migration(cls, type_from: Type[SyftBaseObject], type_to: Type[SyftBaseObject]) -> Callable:
        if False:
            while True:
                i = 10
        for type_from_mro in type_from.mro():
            if issubclass(type_from_mro, SyftBaseObject) and type_from_mro != SyftBaseObject:
                klass_from = type_from_mro.__canonical_name__
                version_from = type_from_mro.__version__
                for type_to_mro in type_to.mro():
                    if issubclass(type_to_mro, SyftBaseObject) and type_to_mro != SyftBaseObject:
                        klass_to = type_to_mro.__canonical_name__
                        version_to = type_to_mro.__version__
                    if klass_from == klass_to:
                        mapping_string = f'{version_from}x{version_to}'
                        if mapping_string in cls.__migration_transform_registry__[klass_from]:
                            return cls.__migration_transform_registry__[klass_from][mapping_string]

    @classmethod
    def get_migration_for_version(cls, type_from: Type[SyftBaseObject], version_to: int) -> Callable:
        if False:
            i = 10
            return i + 15
        canonical_name = type_from.__canonical_name__
        for type_from_mro in type_from.mro():
            if issubclass(type_from_mro, SyftBaseObject) and type_from_mro != SyftBaseObject:
                klass_from = type_from_mro.__canonical_name__
                if klass_from != canonical_name:
                    continue
                version_from = type_from_mro.__version__
                mapping_string = f'{version_from}x{version_to}'
                if mapping_string in cls.__migration_transform_registry__[type_from.__canonical_name__]:
                    return cls.__migration_transform_registry__[klass_from][mapping_string]
        raise Exception(f'No migration found for class type: {type_from} to version: {version_to} in the migration registry.')
print_type_cache = defaultdict(list)

class SyftObject(SyftBaseObject, SyftObjectRegistry, SyftMigrationRegistry):
    __canonical_name__ = 'SyftObject'
    __version__ = SYFT_OBJECT_VERSION_1

    class Config:
        arbitrary_types_allowed = True
    id: UID

    @pydantic.root_validator(pre=True)
    def make_id(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        id_field = cls.__fields__['id']
        if 'id' not in values and id_field.required:
            values['id'] = id_field.type_()
        return values
    __attr_searchable__: ClassVar[List[str]] = []
    __attr_unique__: ClassVar[List[str]] = []
    __serde_overrides__: Dict[str, Sequence[Callable]] = {}
    __owner__: str
    __repr_attrs__: ClassVar[List[str]] = []
    __attr_custom_repr__: ClassVar[List[str]] = None

    def __syft_get_funcs__(self) -> List[Tuple[str, Signature]]:
        if False:
            while True:
                i = 10
        funcs = print_type_cache[type(self)]
        if len(funcs) > 0:
            return funcs
        for attr in dir(type(self)):
            obj = getattr(type(self), attr, None)
            if 'SyftObject' in getattr(obj, '__qualname__', '') and callable(obj) and (not isinstance(obj, type)) and (not attr.startswith('__')):
                sig = inspect.signature(obj)
                funcs.append((attr, sig))
        print_type_cache[type(self)] = funcs
        return funcs

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        try:
            fqn = full_name_with_qualname(type(self))
            return fqn
        except Exception:
            return str(type(self))

    def __str__(self) -> str:
        if False:
            return 10
        return self.__repr__()

    def _repr_debug_(self) -> str:
        if False:
            return 10
        class_name = get_qualname_for(type(self))
        _repr_str = f'class {class_name}:\n'
        fields = getattr(self, '__fields__', {})
        for attr in fields.keys():
            if attr in DYNAMIC_SYFT_ATTRIBUTES:
                continue
            value = getattr(self, attr, '<Missing>')
            value_type = full_name_with_qualname(type(attr))
            value_type = value_type.replace('builtins.', '')
            if hasattr(value, 'syft_action_data_str_'):
                value = value.syft_action_data_str_
            value = f'"{value}"' if isinstance(value, str) else value
            _repr_str += f'  {attr}: {value_type} = {value}\n'
        return _repr_str

    def _repr_markdown_(self, wrap_as_python=True, indent=0) -> str:
        if False:
            while True:
                i = 10
        s_indent = ' ' * indent * 2
        class_name = get_qualname_for(type(self))
        if self.__attr_custom_repr__ is not None:
            fields = self.__attr_custom_repr__
        elif self.__repr_attrs__ is not None:
            fields = self.__repr_attrs__
        else:
            fields = list(getattr(self, '__fields__', {}).keys())
        if 'id' not in fields:
            fields = ['id'] + fields
        dynam_attrs = set(DYNAMIC_SYFT_ATTRIBUTES)
        fields = [x for x in fields if x not in dynam_attrs]
        _repr_str = f'{s_indent}class {class_name}:\n'
        for attr in fields:
            value = self
            if '.' in attr:
                for _attr in attr.split('.'):
                    value = getattr(value, _attr, '<Missing>')
            else:
                value = getattr(value, attr, '<Missing>')
            value_type = full_name_with_qualname(type(attr))
            value_type = value_type.replace('builtins.', '')
            if hasattr(value, '__repr_syft_nested__'):
                value = value.__repr_syft_nested__()
            if isinstance(value, list):
                value = [elem.__repr_syft_nested__() if hasattr(elem, '__repr_syft_nested__') else elem for elem in value]
            value = f'"{value}"' if isinstance(value, str) else value
            _repr_str += f'{s_indent}  {attr}: {value_type} = {value}\n'
        if wrap_as_python:
            return as_markdown_python_code(_repr_str)
        else:
            return _repr_str

    def keys(self) -> KeysView[str]:
        if False:
            print('Hello World!')
        return self.__dict__.keys()

    def __getitem__(self, key: str) -> Any:
        if False:
            print('Hello World!')
        return self.__dict__.__getitem__(key)

    def _upgrade_version(self, latest: bool=True) -> 'SyftObject':
        if False:
            return 10
        constructor = SyftObjectRegistry.versioned_class(name=self.__canonical_name__, version=self.__version__ + 1)
        if not constructor:
            return self
        else:
            upgraded = constructor._from_previous_version(self)
            if latest:
                upgraded = upgraded._upgrade_version(latest=latest)
            return upgraded

    def to(self, projection: type, context: Optional[Context]=None) -> Any:
        if False:
            for i in range(10):
                print('nop')
        transform = SyftObjectRegistry.get_transform(type(self), projection)
        return transform(self, context)

    def to_dict(self, exclude_none: bool=False, exclude_empty: bool=False) -> Dict[str, Any]:
        if False:
            return 10
        warnings.warn('`SyftObject.to_dict` is deprecated and will be removed in a future version', PendingDeprecationWarning, stacklevel=2)
        if not exclude_none and (not exclude_empty):
            return self.dict()
        else:
            new_dict = {}
            for (k, v) in dict(self).items():
                if k in DYNAMIC_SYFT_ATTRIBUTES:
                    continue
                if exclude_empty and v is not Empty:
                    new_dict[k] = v
                if exclude_none and v is not None:
                    new_dict[k] = v
            return new_dict

    def dict(self, *, include: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']]=None, exclude: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']]=None, by_alias: bool=False, skip_defaults: Optional[bool]=None, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False):
        if False:
            return 10
        if exclude is None:
            exclude = set()
        for attr in DYNAMIC_SYFT_ATTRIBUTES:
            exclude.add(attr)
        return super().dict(include=include, exclude=exclude, by_alias=by_alias, skip_defaults=skip_defaults, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none)

    def __post_init__(self) -> None:
        if False:
            print('Hello World!')
        pass

    def _syft_set_validate_private_attrs_(self, **kwargs):
        if False:
            i = 10
            return i + 15
        for (attr, decl) in self.__private_attributes__.items():
            value = kwargs.get(attr, decl.get_default())
            var_annotation = self.__annotations__.get(attr)
            if value is not Undefined:
                if decl.default_factory:
                    value = decl.default_factory(value)
                elif var_annotation is not None:
                    check_type(attr, value, var_annotation)
                setattr(self, attr, value)
            else:
                is_optional_attr = type(None) in getattr(var_annotation, '__args__', [])
                if not is_optional_attr:
                    raise ValueError(f'{attr}\n field required (type=value_error.missing)')

    def __init__(self, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self._syft_set_validate_private_attrs_(**kwargs)
        self.__post_init__()

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        return int.from_bytes(self.__sha256__(), byteorder='big')

    @classmethod
    def _syft_keys_types_dict(cls, attr_name: str) -> Dict[str, type]:
        if False:
            return 10
        kt_dict = {}
        for key in getattr(cls, attr_name, []):
            if key in cls.__fields__:
                type_ = cls.__fields__[key].type_
            else:
                try:
                    method = getattr(cls, key)
                    if isinstance(method, types.FunctionType):
                        type_ = method.__annotations__['return']
                except Exception as e:
                    print(f'Failed to get attribute from key {key} type for {cls} storage. {e}')
                    raise e
            if type(type_) is type and issubclass(type_, EmailStr):
                type_ = str
            kt_dict[key] = type_
        return kt_dict

    @classmethod
    def _syft_unique_keys_dict(cls) -> Dict[str, type]:
        if False:
            for i in range(10):
                print('nop')
        return cls._syft_keys_types_dict('__attr_unique__')

    @classmethod
    def _syft_searchable_keys_dict(cls) -> Dict[str, type]:
        if False:
            while True:
                i = 10
        return cls._syft_keys_types_dict('__attr_searchable__')

    def migrate_to(self, version: int, context: Optional[Context]=None) -> Any:
        if False:
            i = 10
            return i + 15
        if self.__version__ != version:
            migration_transform = SyftMigrationRegistry.get_migration_for_version(type_from=type(self), version_to=version)
            return migration_transform(self, context)
        return self

def short_qual_name(name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return name.split('.')[-1]

def short_uid(uid: UID) -> str:
    if False:
        return 10
    if uid is None:
        return uid
    else:
        return str(uid)[:6] + '...'

def get_repr_values_table(_self, is_homogenous, extra_fields=None):
    if False:
        print('Hello World!')
    if extra_fields is None:
        extra_fields = []
    cols = defaultdict(list)
    for item in iter(_self.items() if isinstance(_self, Mapping) else _self):
        if isinstance(_self, Mapping):
            (key, item) = item
            cols['key'].append(key)
        id_ = getattr(item, 'id', None)
        if id_ is not None:
            cols['id'].append({'value': str(id_), 'type': 'clipboard'})
        if type(item) == type:
            t = full_name_with_qualname(item)
        else:
            try:
                t = item.__class__.__name__
            except Exception:
                t = item.__repr__()
        if not is_homogenous:
            cols['type'].append(t)
        if hasattr(item, '_coll_repr_'):
            ret_val = item._coll_repr_()
            if 'id' in ret_val:
                del ret_val['id']
            for key in ret_val.keys():
                cols[key].append(ret_val[key])
        else:
            for field in extra_fields:
                value = item
                try:
                    attrs = field.split('.')
                    for (i, attr) in enumerate(attrs):
                        res = re.search('\\[[+-]?\\d+\\]', attr)
                        has_index = False
                        if res:
                            has_index = True
                            index_str = res.group()
                            index = int(index_str.replace('[', '').replace(']', ''))
                            attr = attr.replace(index_str, '')
                        value = getattr(value, attr, None)
                        if isinstance(value, list) and has_index:
                            value = value[index]
                        if hasattr(value, '__repr_syft_nested__') and i == len(attrs) - 1:
                            value = value.__repr_syft_nested__()
                        if isinstance(value, list) and i == len(attrs) - 1 and (len(value) > 0) and hasattr(value[0], '__repr_syft_nested__'):
                            value = [x.__repr_syft_nested__() if hasattr(x, '__repr_syft_nested__') else x for x in value]
                    if value is None:
                        value = 'n/a'
                except Exception as e:
                    print(e)
                    value = None
                cols[field].append(str(value))
    df = pd.DataFrame(cols)
    if 'created_at' in df.columns:
        df.sort_values(by='created_at', ascending=False, inplace=True)
    return df.to_dict('records')

def list_dict_repr_html(self) -> str:
    if False:
        i = 10
        return i + 15
    try:
        max_check = 1
        items_checked = 0
        has_syft = False
        extra_fields = []
        if isinstance(self, Mapping):
            values = list(self.values())
        elif isinstance(self, Set):
            values = list(self)
        else:
            values = self
        if len(values) == 0:
            return self.__repr__()
        for item in iter(self.values() if isinstance(self, Mapping) else self):
            items_checked += 1
            if items_checked > max_check:
                break
            if hasattr(type(item), 'mro') and type(item) != type:
                mro = type(item).mro()
            elif hasattr(item, 'mro') and type(item) != type:
                mro = item.mro()
            else:
                mro = str(self)
            if 'syft' in str(mro).lower():
                has_syft = True
                extra_fields = getattr(item, '__repr_attrs__', [])
                break
        if has_syft:
            table_icon = None
            if hasattr(values[0], 'icon'):
                table_icon = values[0].icon
            is_homogenous = len({type(x) for x in values}) == 1
            first_value = values[0]
            if is_homogenous:
                cls_name = first_value.__class__.__name__
            else:
                cls_name = ''
            vals = get_repr_values_table(self, is_homogenous, extra_fields=extra_fields)
            return create_table_template(vals, f'{cls_name} {self.__class__.__name__.capitalize()}', table_icon=table_icon)
    except Exception as e:
        print(f'error representing {type(self)} of objects. {e}')
        pass
    import html
    return html.escape(self.__repr__())
aggressive_set_attr(type([]), '_repr_html_', list_dict_repr_html)
aggressive_set_attr(type({}), '_repr_html_', list_dict_repr_html)
aggressive_set_attr(type(set()), '_repr_html_', list_dict_repr_html)
aggressive_set_attr(tuple, '_repr_html_', list_dict_repr_html)

class StorableObjectType:

    def to(self, projection: type, context: Optional[Context]=None) -> Any:
        if False:
            while True:
                i = 10
        transform = SyftObjectRegistry.get_transform(type(self), projection)
        return transform(self, context)

class PartialSyftObject(SyftObject, metaclass=PartialModelMetaclass):
    """Syft Object to which partial arguments can be provided."""
    __canonical_name__ = 'PartialSyftObject'
    __version__ = SYFT_OBJECT_VERSION_1

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        (args_, kwargs_) = ((), {})
        for arg in args:
            if arg is not Empty:
                args_.append(arg)
        for (key, val) in kwargs.items():
            if val is not Empty:
                kwargs_[key] = val
        super().__init__(*args_, **kwargs_)
        fields_with_default = set()
        for (_field_name, _field) in self.__fields__.items():
            if _field.default or _field.allow_none:
                fields_with_default.add(_field_name)
        fields_set_via_validator = []
        for _field_name in self.__validators__.keys():
            _field = self.__fields__[_field_name]
            if self.__dict__[_field_name] is None:
                if _field.allow_none or _field.default is None:
                    fields_set_via_validator.append(_field)
        unset_fields = set(self.__fields__) - set(self.__fields_set__) - set(fields_set_via_validator)
        empty_fields = unset_fields - fields_with_default
        for field_name in empty_fields:
            self.__dict__[field_name] = Empty
recursive_serde_register_type(PartialSyftObject)

def attach_attribute_to_syft_object(result: Any, attr_dict: Dict[str, Any]) -> Any:
    if False:
        return 10
    constructor = None
    extra_args = []
    single_entity = False
    if isinstance(result, OkErr):
        constructor = type(result)
        result = result.value
    if isinstance(result, MutableMapping):
        iterable_keys = result.keys()
    elif isinstance(result, MutableSequence):
        iterable_keys = range(len(result))
    elif isinstance(result, tuple):
        iterable_keys = range(len(result))
        constructor = type(result)
        if isinstance(result, DictTuple):
            extra_args.append(result.keys())
        result = list(result)
    else:
        iterable_keys = range(1)
        result = [result]
        single_entity = True
    for key in iterable_keys:
        _object = result[key]
        if isinstance(_object, SyftBaseObject):
            for (attr_name, attr_value) in attr_dict.items():
                setattr(_object, attr_name, attr_value)
            for (field_name, attr) in _object.__dict__.items():
                updated_attr = attach_attribute_to_syft_object(attr, attr_dict)
                setattr(_object, field_name, updated_attr)
        result[key] = _object
    wrapped_result = result[0] if single_entity else result
    if constructor is not None:
        wrapped_result = constructor(wrapped_result, *extra_args)
    return wrapped_result