"""
This is refactored from cmake.py to avoid circular imports issue with env.py,
which calls get_cmake_cache_variables_from_file
"""
import re
from typing import Dict, IO, Optional, Union
CMakeValue = Optional[Union[bool, str]]

def convert_cmake_value_to_python_value(cmake_value: str, cmake_type: str) -> CMakeValue:
    if False:
        print('Hello World!')
    'Convert a CMake value in a string form to a Python value.\n\n    Args:\n      cmake_value (string): The CMake value in a string form (e.g., "ON", "OFF", "1").\n      cmake_type (string): The CMake type of :attr:`cmake_value`.\n\n    Returns:\n      A Python value corresponding to :attr:`cmake_value` with type :attr:`cmake_type`.\n    '
    cmake_type = cmake_type.upper()
    up_val = cmake_value.upper()
    if cmake_type == 'BOOL':
        return not (up_val in ('FALSE', 'OFF', 'N', 'NO', '0', '', 'NOTFOUND') or up_val.endswith('-NOTFOUND'))
    elif cmake_type == 'FILEPATH':
        if up_val.endswith('-NOTFOUND'):
            return None
        else:
            return cmake_value
    else:
        return cmake_value

def get_cmake_cache_variables_from_file(cmake_cache_file: IO[str]) -> Dict[str, CMakeValue]:
    if False:
        while True:
            i = 10
    'Gets values in CMakeCache.txt into a dictionary.\n\n    Args:\n      cmake_cache_file: A CMakeCache.txt file object.\n    Returns:\n      dict: A ``dict`` containing the value of cached CMake variables.\n    '
    results = {}
    for (i, line) in enumerate(cmake_cache_file, 1):
        line = line.strip()
        if not line or line.startswith(('#', '//')):
            continue
        matched = re.match('("?)(.+?)\\1(?::\\s*([a-zA-Z_-][a-zA-Z0-9_-]*)?)?\\s*=\\s*(.*)', line)
        if matched is None:
            raise ValueError(f'Unexpected line {i} in {repr(cmake_cache_file)}: {line}')
        (_, variable, type_, value) = matched.groups()
        if type_ is None:
            type_ = ''
        if type_.upper() in ('INTERNAL', 'STATIC'):
            continue
        results[variable] = convert_cmake_value_to_python_value(value, type_)
    return results