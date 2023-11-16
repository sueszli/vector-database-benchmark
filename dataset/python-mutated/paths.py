"""Paths that are useful throughout the project."""
from pathlib import Path

def get_project_root() -> Path:
    if False:
        for i in range(10):
            print('nop')
    'Returns the path to the project root folder.\n\n    Returns:\n        The path to the project root folder.\n    '
    return Path(__file__).parent.parent.parent.parent

def get_config(file_name: str) -> Path:
    if False:
        print('Hello World!')
    'Returns the path a config file.\n\n    Returns:\n        The path to a config file.\n    '
    return Path(__file__).parent.parent / file_name

def get_data_path() -> Path:
    if False:
        for i in range(10):
            print('nop')
    'Returns the path to the dataset cache ([root] / data)\n\n    Returns:\n        The path to the dataset cache\n    '
    return get_project_root() / 'data'

def get_html_template_path() -> Path:
    if False:
        for i in range(10):
            print('nop')
    'Returns the path to the HTML templates\n\n    Returns:\n        The path to the HTML templates\n    '
    return Path(__file__).parent.parent / 'report' / 'presentation' / 'flavours' / 'html' / 'templates'