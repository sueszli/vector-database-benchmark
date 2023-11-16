from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Set, cast
import dagster._check as check
from dagster._serdes import whitelist_for_serdes
from .config_type import ConfigScalarKind, ConfigType, ConfigTypeKind
from .field import Field

def get_recursive_type_keys(config_type_snap: 'ConfigTypeSnap', config_schema_snapshot: 'ConfigSchemaSnapshot') -> Set[str]:
    if False:
        print('Hello World!')
    check.inst_param(config_type_snap, 'config_type_snap', ConfigTypeSnap)
    check.inst_param(config_schema_snapshot, 'config_schema_snapshot', ConfigSchemaSnapshot)
    result_keys = set()
    for type_key in config_type_snap.get_child_type_keys():
        result_keys.add(type_key)
        for recurse_key in get_recursive_type_keys(config_schema_snapshot.get_config_snap(type_key), config_schema_snapshot):
            result_keys.add(recurse_key)
    return result_keys

@whitelist_for_serdes
class ConfigSchemaSnapshot(NamedTuple('_ConfigSchemaSnapshot', [('all_config_snaps_by_key', Mapping[str, 'ConfigTypeSnap'])])):

    def __new__(cls, all_config_snaps_by_key: Mapping[str, 'ConfigTypeSnap']):
        if False:
            i = 10
            return i + 15
        return super(ConfigSchemaSnapshot, cls).__new__(cls, all_config_snaps_by_key=check.mapping_param(all_config_snaps_by_key, 'all_config_snaps_by_key', key_type=str, value_type=ConfigTypeSnap))

    @property
    def all_config_keys(self) -> Sequence[str]:
        if False:
            while True:
                i = 10
        return list(self.all_config_snaps_by_key.keys())

    def get_config_snap(self, key: str) -> 'ConfigTypeSnap':
        if False:
            while True:
                i = 10
        check.str_param(key, 'key')
        return self.all_config_snaps_by_key[key]

    def has_config_snap(self, key: str) -> bool:
        if False:
            i = 10
            return i + 15
        check.str_param(key, 'key')
        return key in self.all_config_snaps_by_key

@whitelist_for_serdes(skip_when_empty_fields={'field_aliases'})
class ConfigTypeSnap(NamedTuple('_ConfigTypeSnap', [('kind', ConfigTypeKind), ('key', str), ('given_name', Optional[str]), ('description', Optional[str]), ('type_param_keys', Optional[Sequence[str]]), ('enum_values', Optional[Sequence['ConfigEnumValueSnap']]), ('fields', Optional[Sequence['ConfigFieldSnap']]), ('scalar_kind', Optional[ConfigScalarKind]), ('field_aliases', Optional[Mapping[str, str]])])):

    def __new__(cls, kind, key, given_name, description, type_param_keys, enum_values, fields, scalar_kind=None, field_aliases=None):
        if False:
            i = 10
            return i + 15
        return super(ConfigTypeSnap, cls).__new__(cls, kind=check.inst_param(kind, 'kind', ConfigTypeKind), key=check.str_param(key, 'key'), given_name=check.opt_str_param(given_name, 'given_name'), type_param_keys=None if type_param_keys is None else check.list_param(type_param_keys, 'type_param_keys', of_type=str), enum_values=None if enum_values is None else check.list_param(enum_values, 'enum_values', of_type=ConfigEnumValueSnap), fields=None if fields is None else sorted(check.list_param(fields, 'field', of_type=ConfigFieldSnap), key=lambda ct: ct.name), description=check.opt_str_param(description, 'description'), scalar_kind=check.opt_inst_param(scalar_kind, 'scalar_kind', ConfigScalarKind), field_aliases=check.opt_dict_param(field_aliases, 'field_aliases'))

    @property
    def key_type_key(self) -> str:
        if False:
            return 10
        'For a type which has keys such as Map, returns the type of the key.'
        check.invariant(self.kind == ConfigTypeKind.MAP)
        type_param_keys = check.is_list(self.type_param_keys, of_type=str)
        check.invariant(len(type_param_keys) == 2)
        return type_param_keys[0]

    @property
    def inner_type_key(self) -> str:
        if False:
            i = 10
            return i + 15
        'For container types such as Array or Noneable, the contained type. For a Map, the value type.'
        check.invariant(self.kind == ConfigTypeKind.NONEABLE or self.kind == ConfigTypeKind.ARRAY or self.kind == ConfigTypeKind.MAP)
        type_param_keys = check.is_list(self.type_param_keys, of_type=str)
        if self.kind == ConfigTypeKind.MAP:
            check.invariant(len(type_param_keys) == 2)
            return type_param_keys[1]
        else:
            check.invariant(len(type_param_keys) == 1)
            return type_param_keys[0]

    @property
    def scalar_type_key(self) -> str:
        if False:
            while True:
                i = 10
        check.invariant(self.kind == ConfigTypeKind.SCALAR_UNION)
        type_param_keys = check.is_list(self.type_param_keys, of_type=str)
        return type_param_keys[0]

    @property
    def non_scalar_type_key(self) -> str:
        if False:
            print('Hello World!')
        check.invariant(self.kind == ConfigTypeKind.SCALAR_UNION)
        type_param_keys = check.is_list(self.type_param_keys, of_type=str)
        return type_param_keys[1]

    def _get_field(self, name: str) -> Optional['ConfigFieldSnap']:
        if False:
            while True:
                i = 10
        check.str_param(name, 'name')
        check.invariant(ConfigTypeKind.has_fields(self.kind))
        fields = check.is_list(self.fields, of_type=ConfigFieldSnap)
        for f in fields:
            if f.name == name:
                return f
        return None

    def get_field(self, name: str) -> 'ConfigFieldSnap':
        if False:
            i = 10
            return i + 15
        field = self._get_field(name)
        if not field:
            check.failed(f'Field {name} not found')
        return field

    def has_field(self, name: str) -> bool:
        if False:
            while True:
                i = 10
        return bool(self._get_field(name))

    @property
    def field_names(self) -> Sequence[str]:
        if False:
            while True:
                i = 10
        fields = check.is_list(self.fields, of_type=ConfigFieldSnap)
        return [fs.name for fs in fields]

    def get_child_type_keys(self) -> Sequence[str]:
        if False:
            return 10
        if ConfigTypeKind.is_closed_generic(self.kind):
            return cast(List[str], self.type_param_keys)
        elif ConfigTypeKind.has_fields(self.kind):
            return [field.type_key for field in cast(List[ConfigFieldSnap], check.not_none(self.fields))]
        else:
            return []

    def has_enum_value(self, value: object) -> bool:
        if False:
            i = 10
            return i + 15
        check.invariant(self.kind == ConfigTypeKind.ENUM)
        for enum_value in cast(List[ConfigEnumValueSnap], self.enum_values):
            if enum_value.value == value:
                return True
        return False

@whitelist_for_serdes
class ConfigEnumValueSnap(NamedTuple('_ConfigEnumValueSnap', [('value', str), ('description', Optional[str])])):

    def __new__(cls, value: str, description: Optional[str]):
        if False:
            print('Hello World!')
        return super(ConfigEnumValueSnap, cls).__new__(cls, value=check.str_param(value, 'value'), description=check.opt_str_param(description, 'description'))

@whitelist_for_serdes
class ConfigFieldSnap(NamedTuple('_ConfigFieldSnap', [('name', Optional[str]), ('type_key', str), ('is_required', bool), ('default_provided', bool), ('default_value_as_json_str', Optional[str]), ('description', Optional[str])])):

    def __new__(cls, name, type_key, is_required, default_provided, default_value_as_json_str, description):
        if False:
            i = 10
            return i + 15
        return super(ConfigFieldSnap, cls).__new__(cls, name=check.opt_str_param(name, 'name'), type_key=check.str_param(type_key, 'type_key'), is_required=check.bool_param(is_required, 'is_required'), default_provided=check.bool_param(default_provided, 'default_provided'), default_value_as_json_str=check.opt_str_param(default_value_as_json_str, 'default_value_as_json_str'), description=check.opt_str_param(description, 'description'))

def snap_from_field(name: str, field: Field):
    if False:
        print('Hello World!')
    return ConfigFieldSnap(name=name, type_key=field.config_type.key, is_required=field.is_required, default_provided=field.default_provided, default_value_as_json_str=field.default_value_as_json_str if field.default_provided else None, description=field.description)

def snap_from_config_type(config_type: ConfigType) -> ConfigTypeSnap:
    if False:
        i = 10
        return i + 15
    return ConfigTypeSnap(key=config_type.key, given_name=config_type.given_name, kind=config_type.kind, description=config_type.description, type_param_keys=[ct.key for ct in config_type.type_params] if config_type.type_params else [config_type.scalar_type.key, config_type.non_scalar_type.key] if config_type.kind == ConfigTypeKind.SCALAR_UNION else None, enum_values=[ConfigEnumValueSnap(ev.config_value, ev.description) for ev in config_type.enum_values] if config_type.kind == ConfigTypeKind.ENUM else None, fields=[snap_from_field(name, field) for (name, field) in config_type.fields.items()] if ConfigTypeKind.has_fields(config_type.kind) else None, scalar_kind=config_type.scalar_kind if config_type.kind == ConfigTypeKind.SCALAR else None, field_aliases=config_type.field_aliases if config_type.kind == ConfigTypeKind.STRICT_SHAPE else None)

def minimal_config_for_type_snap(config_schema_snap: ConfigSchemaSnapshot, config_type_snap: ConfigTypeSnap) -> Any:
    if False:
        i = 10
        return i + 15
    check.inst_param(config_schema_snap, 'config_schema_snap', ConfigSchemaSnapshot)
    check.inst_param(config_type_snap, 'config_type_snap', ConfigTypeSnap)
    if ConfigTypeKind.has_fields(config_type_snap.kind):
        default_dict = {}
        if ConfigTypeKind.is_selector(config_type_snap.kind):
            return '<selector>'
        for field in config_type_snap.fields:
            if not field.is_required:
                continue
            default_dict[field.name] = minimal_config_for_type_snap(config_schema_snap, config_schema_snap.get_config_snap(field.type_key))
        return default_dict
    elif config_type_snap.kind == ConfigTypeKind.ANY:
        return 'AnyType'
    elif config_type_snap.kind == ConfigTypeKind.SCALAR:
        defaults = {'String': '...', 'Int': 0, 'Float': 0.0, 'Bool': True}
        return defaults.get(config_type_snap.given_name, '<unknown>')
    elif config_type_snap.kind == ConfigTypeKind.ARRAY:
        return []
    elif config_type_snap.kind == ConfigTypeKind.MAP:
        return {}
    elif config_type_snap.kind == ConfigTypeKind.ENUM:
        return config_type_snap.enum_values[0].value if config_type_snap.enum_values else '<unknown>'
    elif config_type_snap.kind == ConfigTypeKind.SCALAR_UNION:
        return minimal_config_for_type_snap(config_schema_snap, config_schema_snap.get_config_snap(config_type_snap.type_param_keys[0]))
    else:
        return '<unknown>'