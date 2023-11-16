from typing import Any, List, Mapping
import dpath.util

def get_secret_paths(spec: Mapping[str, Any]) -> List[List[str]]:
    if False:
        i = 10
        return i + 15
    paths = []

    def traverse_schema(schema_item: Any, path: List[str]):
        if False:
            while True:
                i = 10
        "\n        schema_item can be any property or value in the originally input jsonschema, depending on how far down the recursion stack we go\n        path is the path to that schema item in the original input\n        for example if we have the input {'password': {'type': 'string', 'airbyte_secret': True}} then the arguments will evolve\n        as follows:\n        schema_item=<whole_object>, path=[]\n        schema_item={'type': 'string', 'airbyte_secret': True}, path=['password']\n        schema_item='string', path=['password', 'type']\n        schema_item=True, path=['password', 'airbyte_secret']\n        "
        if isinstance(schema_item, dict):
            for (k, v) in schema_item.items():
                traverse_schema(v, [*path, k])
        elif isinstance(schema_item, list):
            for i in schema_item:
                traverse_schema(i, path)
        elif path[-1] == 'airbyte_secret' and schema_item is True:
            filtered_path = [p for p in path[:-1] if p not in ['properties', 'oneOf']]
            paths.append(filtered_path)
    traverse_schema(spec, [])
    return paths

def get_secrets(connection_specification: Mapping[str, Any], config: Mapping[str, Any]) -> List[Any]:
    if False:
        print('Hello World!')
    '\n    Get a list of secret values from the source config based on the source specification\n    :type connection_specification: the connection_specification field of an AirbyteSpecification i.e the JSONSchema definition\n    '
    secret_paths = get_secret_paths(connection_specification.get('properties', {}))
    result = []
    for path in secret_paths:
        try:
            result.append(dpath.util.get(config, path))
        except KeyError:
            pass
    return result
__SECRETS_FROM_CONFIG: List[str] = []

def update_secrets(secrets: List[str]):
    if False:
        while True:
            i = 10
    'Update the list of secrets to be replaced'
    global __SECRETS_FROM_CONFIG
    __SECRETS_FROM_CONFIG = secrets

def filter_secrets(string: str) -> str:
    if False:
        i = 10
        return i + 15
    'Filter secrets from a string by replacing them with ****'
    for secret in __SECRETS_FROM_CONFIG:
        if secret:
            string = string.replace(str(secret), '****')
    return string