"""Validates handler args against its schema by calling schema utils.
Also contains a list of handler class names which does not contain the schema.
"""
from __future__ import annotations
from core import schema_utils
from typing import Any, Dict, List, Tuple, Union

def get_schema_type(arg_schema: Dict[str, Any]) -> str:
    if False:
        i = 10
        return i + 15
    'Returns the schema type for an argument.\n\n    Args:\n        arg_schema: dict(str, *). Schema for an argument.\n\n    Returns:\n        str. Returns schema type by extracting it from schema.\n    '
    schema_type: str = arg_schema['schema']['type']
    return schema_type

def get_corresponding_key_for_object(arg_schema: Dict[str, Any]) -> str:
    if False:
        return 10
    'Returns the new key for an argument from its schema.\n\n    Args:\n        arg_schema: dict(str, *). Schema for an argument.\n\n    Returns:\n        str. The new argument name.\n    '
    new_key_for_argument: str = arg_schema['schema']['new_key_for_argument']
    return new_key_for_argument

def validate_arguments_against_schema(handler_args: Dict[str, Any], handler_args_schemas: Dict[str, Any], allowed_extra_args: bool, allow_string_to_bool_conversion: bool=False) -> Tuple[Dict[str, Any], List[str]]:
    if False:
        for i in range(10):
            print('nop')
    'Calls schema utils for normalization of object against its schema\n    and collects all the errors.\n\n    Args:\n        handler_args: Dict(str, *). Object for normalization.\n        handler_args_schemas: dict. Schema for args.\n        allowed_extra_args: bool. Whether extra args are allowed in handler.\n        allow_string_to_bool_conversion: bool. Whether to allow string to\n            boolean conversion.\n\n    Returns:\n        *. A two tuple, where the first element represents the normalized value\n        in dict format and the second element represents the lists of errors\n        after validation.\n    '
    errors = []
    normalized_values = {}
    for (arg_key, arg_schema) in handler_args_schemas.items():
        if arg_key not in handler_args or handler_args[arg_key] is None:
            if 'default_value' in arg_schema:
                if arg_schema['default_value'] is None:
                    continue
                if arg_schema['default_value'] is not None:
                    handler_args[arg_key] = arg_schema['default_value']
            else:
                errors.append('Missing key in handler args: %s.' % arg_key)
                continue
        if allow_string_to_bool_conversion and get_schema_type(arg_schema) == schema_utils.SCHEMA_TYPE_BOOL and isinstance(handler_args[arg_key], str):
            handler_args[arg_key] = convert_string_to_bool(handler_args[arg_key])
        try:
            normalized_value = schema_utils.normalize_against_schema(handler_args[arg_key], arg_schema['schema'])
            if 'new_key_for_argument' in arg_schema['schema']:
                arg_key = get_corresponding_key_for_object(arg_schema)
            normalized_values[arg_key] = normalized_value
        except Exception as e:
            errors.append("Schema validation for '%s' failed: %s" % (arg_key, e))
    extra_args = set(handler_args.keys()) - set(handler_args_schemas.keys())
    if not allowed_extra_args and extra_args:
        errors.append('Found extra args: %s.' % list(extra_args))
    return (normalized_values, errors)

def convert_string_to_bool(param: str) -> Union[bool, str]:
    if False:
        while True:
            i = 10
    "Converts a request param of type string into expected bool type.\n\n    Args:\n        param: str. The params which needs normalization.\n\n    Returns:\n        Union[bool, str]. Returns a boolean value if the param is either a\n        'true' or 'false' string literal, and returns string value otherwise.\n    "
    case_insensitive_param = param.lower()
    if case_insensitive_param == 'true':
        return True
    elif case_insensitive_param == 'false':
        return False
    else:
        return param