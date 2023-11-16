from __future__ import annotations
import os
import re
from ast import literal_eval
from typing import Any
from hatchling.build import *

def read_dependencies() -> list[str]:
    if False:
        i = 10
        return i + 15
    pattern = '^dependencies = (\\[.*?\\])$'
    with open(os.path.join(os.getcwd(), 'pyproject.toml'), encoding='utf-8') as f:
        contents = '\n'.join((line.rstrip() for line in f))
    match = re.search(pattern, contents, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        message = 'No dependencies found'
        raise ValueError(message)
    return literal_eval(match.group(1))

def get_requires_for_build_sdist(config_settings: dict[str, Any] | None=None) -> list[str]:
    if False:
        print('Hello World!')
    '\n    https://peps.python.org/pep-0517/#get-requires-for-build-sdist\n    '
    return read_dependencies()

def get_requires_for_build_wheel(config_settings: dict[str, Any] | None=None) -> list[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    https://peps.python.org/pep-0517/#get-requires-for-build-wheel\n    '
    return read_dependencies()

def get_requires_for_build_editable(config_settings: dict[str, Any] | None=None) -> list[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    https://peps.python.org/pep-0660/#get-requires-for-build-editable\n    '
    return read_dependencies()