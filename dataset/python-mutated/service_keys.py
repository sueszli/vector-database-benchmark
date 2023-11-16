from typing import Any, Dict
from typing import Dict

def get_similarity_config_keys_values(config: Dict[str, str | Dict[str, str]]) -> Dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Recursively flattens a nested dictionary into a single-level dictionary with keys\n    in the format "parent_key__child_key" for all nested keys. Returns the resulting\n    flattened dictionary.\n\n    Args:\n        config (Dict[str, str | Dict[str, str]]): The nested dictionary to flatten.\n\n    Returns:\n        Dict[str, str]: The resulting flattened dictionary.\n    '
    result: Dict[str, str] = {}
    for (key, value) in config.items():
        if isinstance(value, dict):
            subkeys: Dict[str, str] = get_similarity_config_keys_values(value)
            result.update({f'{key}__{subkey_key}': subkey_value for (subkey_key, subkey_value) in subkeys.items()})
        else:
            result[key] = value
    return result
from typing import Dict, Any

def update_yaml_config(keys: Dict[str, str]) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    "\n    Recursively updates a dictionary with keys that contain double underscores ('__') in their names.\n    The double underscores are used to indicate nested keys in a YAML file.\n\n    Args:\n        keys (Dict[str, str]): A dictionary containing keys and values to update.\n\n    Returns:\n        Dict[str, Any]: The updated dictionary. Note: the type should be Dict[str, str | Dict[str, str]], but the type checker doesn't like that. Need to figure out how to correctly type this 🤔\n    "
    config: Dict[str, Any] = {}
    for (key, value) in keys.items():
        if '__' in key:
            (parent_key, child_key) = key.split('__', 1)
            if parent_key not in config:
                config[parent_key] = {}
            config[parent_key].update(update_yaml_config({child_key: value}))
        else:
            config[key] = value
    return config