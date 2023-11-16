import json
from typing import Any, Mapping, Optional
from dagster._config.snap import ConfigSchemaSnapshot, ConfigTypeSnap
from dagster._utils.yaml_utils import dump_run_config_yaml

def _safe_json_loads(json_str: Optional[str]) -> object:
    if False:
        return 10
    try:
        return json.loads(json_str) if json_str else None
    except json.JSONDecodeError:
        return None
PRIORITY_CONFIG_KEYS = ('ops', 'resources')

def _filter_empty_dicts(to_filter: Any) -> Any:
    if False:
        i = 10
        return i + 15
    if not isinstance(to_filter, Mapping):
        return to_filter
    else:
        filtered_dict = {k: _filter_empty_dicts(v) for (k, v) in to_filter.items()}
        return {k: v for (k, v) in filtered_dict.items() if v is not None and v != {}}

def _cleanup_run_config_dict(run_config_dict: Any) -> Any:
    if False:
        i = 10
        return i + 15
    'Performs cleanup of the run config dict to remove empty dicts and strip the default executor\n    config if it has not been overridden, to make the output more readable.\n    '
    return _filter_empty_dicts(run_config_dict)

def default_values_yaml_from_type_snap(snapshot: ConfigSchemaSnapshot, type_snap: ConfigTypeSnap) -> str:
    if False:
        i = 10
        return i + 15
    'Returns a YAML representation of the default values for the given type snap.'
    run_config_dict = _cleanup_run_config_dict(default_values_from_type_snap(type_snap, snapshot))
    run_config_dict_sorted: Mapping[str, Any] = dict(((k, run_config_dict.get(k)) for k in [*PRIORITY_CONFIG_KEYS, *run_config_dict.keys()] if k in run_config_dict))
    return dump_run_config_yaml(run_config_dict_sorted, sort_keys=False)

def default_values_from_type_snap(type_snap: ConfigTypeSnap, snapshot: ConfigSchemaSnapshot) -> Any:
    if False:
        i = 10
        return i + 15
    'Given a type snap and a snapshot, returns a dictionary of default values for the type\n    snap, recursively assembling a default if the type snap does not have a default value\n    explicitly set.\n    '
    if not type_snap.fields:
        return {}
    defaults_by_field = {}
    for field_name in type_snap.field_names:
        field = type_snap.get_field(field_name)
        default_value_as_json = field.default_value_as_json_str
        field_snap = snapshot.get_config_snap(field.type_key) if snapshot.has_config_snap(field.type_key) else None
        if default_value_as_json:
            defaults_by_field[field_name] = _safe_json_loads(default_value_as_json)
        elif field_snap and field_snap.fields:
            defaults_by_field[field_name] = default_values_from_type_snap(field_snap, snapshot)
    return defaults_by_field