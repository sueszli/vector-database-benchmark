"""
This script is responsible for connectors spec.json file validation.

Input:
List of spec files

Output:
exit code 0 - check is success
exit code 1 - check failed for at least one spec file

How spec file validation works:
1. read spec file and serialize it as python dict object
2. get properties field from spec object
3. check if all fields from FIELDS_TO_CHECK exist in each property
4. if field has oneOf attribute - fetch all subobjects and for each of them goto step (2)
"""
import json
import logging
import sys
from typing import Any, List, Mapping, Optional, Tuple
FIELDS_TO_CHECK = {'title', 'description'}
logging.basicConfig(format='%(message)s')

def read_spec_file(spec_path: str) -> bool:
    if False:
        return 10
    '\n    Parses spec file and applies validation rules.\n    Returns True if spec is valid else False\n    '
    errors: List[Tuple[str, Optional[str]]] = []
    with open(spec_path) as json_file:
        try:
            root_schema = json.load(json_file)['connectionSpecification']['properties']
        except (KeyError, TypeError):
            errors.append(("Couldn't find properties in connector spec.json", None))
        except json.JSONDecodeError:
            errors.append(("Couldn't parse json file", None))
        else:
            errors.extend(validate_schema(spec_path, root_schema))
    for (err_msg, err_field) in errors:
        print_error(spec_path, err_msg, err_field)
    return False if errors else True

def print_error(spec_path: str, error_message: str, failed_field: Optional[str]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Logs error in following format: <BOLD>SPEC PATH</BOLD> ERROR MSG <RED>FIELD NAME</RED>\n    '
    error = f'\x1b[1m{spec_path}\x1b[0m: {error_message}'
    if failed_field:
        error += f' \x1b[31;1m{failed_field}\x1b[0m'
    logging.error(error)

def validate_schema(spec_path: str, schema: Mapping[str, Any], parent_fields: Optional[List[str]]=None) -> List[Tuple[str, str]]:
    if False:
        i = 10
        return i + 15
    '\n    Validates given spec dictionary object. Returns list of errors\n    '
    errors: List[Tuple[str, str]] = []
    parent_fields = parent_fields if parent_fields else []
    for (field_name, field_schema) in schema.items():
        field_errors = validate_field(field_name, field_schema, parent_fields)
        errors.extend(field_errors)
        if field_errors:
            continue
        for (index, oneof_schema) in enumerate(fetch_oneof_schemas(field_schema)):
            errors.extend(validate_schema(spec_path, oneof_schema['properties'], parent_fields + [field_name, str(index)]))
    return errors

def fetch_oneof_schemas(schema: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Finds subschemas in oneOf field\n    '
    return [spec for spec in schema.get('oneOf', []) if spec.get('properties')]

def validate_field(field_name: str, schema: Mapping[str, Any], parent_fields: Optional[List[str]]=None) -> List[Tuple[str, str]]:
    if False:
        while True:
            i = 10
    '\n    Validates single field objects and return errors if they exist\n    '
    if 'const' in schema.keys():
        return []
    errors: List[Tuple[str, str]] = []
    full_field_name = get_full_field_name(field_name, parent_fields)
    if not FIELDS_TO_CHECK.issubset(schema.keys()):
        errors.append(('Check failed for field', full_field_name))
    if schema.get('oneOf') and (schema['type'] != 'object' or not isinstance(schema['oneOf'], list)):
        errors.append(('Incorrect oneOf schema in field', full_field_name))
    return errors

def get_full_field_name(field_name: str, parent_fields: Optional[List[str]]=None) -> str:
    if False:
        print('Hello World!')
    '\n    Returns full path to a field.\n    e.g. root.middle.child, root.oneof.1.attr\n    '
    return '.'.join(parent_fields + [field_name]) if parent_fields else field_name
if __name__ == '__main__':
    spec_files = sys.argv[1:]
    if not all([read_spec_file(file_path) for file_path in spec_files]):
        exit(1)