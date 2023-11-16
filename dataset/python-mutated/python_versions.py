from __future__ import annotations
import sys
from airflow_breeze.global_constants import ALLOWED_PYTHON_MAJOR_MINOR_VERSIONS
from airflow_breeze.utils.console import get_console

def get_python_version_list(python_versions: str) -> list[str]:
    if False:
        print('Hello World!')
    '\n    Retrieve and validate space-separated list of Python versions and return them in the form of list.\n    :param python_versions: space separated list of Python versions\n    :return: List of python versions\n    '
    python_version_list = python_versions.split(' ')
    errors = False
    for python in python_version_list:
        if python not in ALLOWED_PYTHON_MAJOR_MINOR_VERSIONS:
            get_console().print(f'[error]The Python version {python} passed in {python_versions} is wrong.[/]')
            errors = True
    if errors:
        get_console().print(f'\nSome of the Python versions passed are not in the list: {ALLOWED_PYTHON_MAJOR_MINOR_VERSIONS}. Quitting.\n')
        sys.exit(1)
    return python_version_list