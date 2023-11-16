from __future__ import annotations
import json
import os
from glob import glob
from pathlib import Path
from typing import Any
CONSOLE_WIDTH = 180
ROOT_DIR = Path(__file__).parents[5].resolve()
PROVIDER_DATA_SCHEMA_PATH = ROOT_DIR / 'airflow' / 'provider.yaml.schema.json'

def _load_schema() -> dict[str, Any]:
    if False:
        return 10
    with open(PROVIDER_DATA_SCHEMA_PATH) as schema_file:
        content = json.load(schema_file)
    return content

def _filepath_to_module(filepath: str):
    if False:
        print('Hello World!')
    return str(Path(filepath).relative_to(ROOT_DIR)).replace('/', '.')

def _filepath_to_system_tests(filepath: str):
    if False:
        print('Hello World!')
    return str(ROOT_DIR / 'tests' / 'system' / 'providers' / Path(filepath).relative_to(ROOT_DIR / 'airflow' / 'providers'))

def get_provider_yaml_paths():
    if False:
        print('Hello World!')
    'Returns list of provider.yaml files'
    return sorted(glob(f'{ROOT_DIR}/airflow/providers/**/provider.yaml', recursive=True))

def pretty_format_path(path: str, start: str) -> str:
    if False:
        while True:
            i = 10
    'Formats path nicely.'
    relpath = os.path.relpath(path, start)
    if relpath == path:
        return path
    return f'{start}/{relpath}'

def prepare_code_snippet(file_path: str, line_no: int, context_lines_count: int=5) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Prepare code snippet with line numbers and  a specific line marked.\n\n    :param file_path: File name\n    :param line_no: Line number\n    :param context_lines_count: The number of lines that will be cut before and after.\n    :return: str\n    '
    with open(file_path) as text_file:
        code = text_file.read()
        code_lines = code.splitlines()
        code_lines = [f'>{lno:3} | {line}' if line_no == lno else f'{lno:4} | {line}' for (lno, line) in enumerate(code_lines, 1)]
        start_line_no = max(0, line_no - context_lines_count - 1)
        end_line_no = line_no + context_lines_count
        code_lines = code_lines[start_line_no:end_line_no]
        code = '\n'.join(code_lines)
    return code