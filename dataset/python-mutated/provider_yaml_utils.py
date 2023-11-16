from __future__ import annotations
import json
import os
from glob import glob
from pathlib import Path
from typing import Any
import jsonschema
import yaml
ROOT_DIR = Path(__file__).parents[2].resolve()
PROVIDER_DATA_SCHEMA_PATH = ROOT_DIR / 'airflow' / 'provider.yaml.schema.json'

def _load_schema() -> dict[str, Any]:
    if False:
        while True:
            i = 10
    with open(PROVIDER_DATA_SCHEMA_PATH) as schema_file:
        content = json.load(schema_file)
    return content

def _filepath_to_module(filepath: str):
    if False:
        i = 10
        return i + 15
    return str(Path(filepath).relative_to(ROOT_DIR)).replace('/', '.')

def _filepath_to_system_tests(filepath: str):
    if False:
        i = 10
        return i + 15
    return str(ROOT_DIR / 'tests' / 'system' / 'providers' / Path(filepath).relative_to(ROOT_DIR / 'airflow' / 'providers'))

def get_provider_yaml_paths():
    if False:
        for i in range(10):
            print('nop')
    'Returns list of provider.yaml files'
    return sorted(glob(f'{ROOT_DIR}/airflow/providers/**/provider.yaml', recursive=True))

def load_package_data(include_suspended: bool=False) -> list[dict[str, Any]]:
    if False:
        return 10
    '\n    Load all data from providers files\n\n    :return: A list containing the contents of all provider.yaml files.\n    '
    schema = _load_schema()
    result = []
    for provider_yaml_path in get_provider_yaml_paths():
        with open(provider_yaml_path) as yaml_file:
            provider = yaml.safe_load(yaml_file)
        try:
            jsonschema.validate(provider, schema=schema)
        except jsonschema.ValidationError:
            raise Exception(f'Unable to parse: {provider_yaml_path}.')
        if provider['suspended'] and (not include_suspended):
            continue
        provider_yaml_dir = os.path.dirname(provider_yaml_path)
        provider['python-module'] = _filepath_to_module(provider_yaml_dir)
        provider['package-dir'] = provider_yaml_dir
        provider['system-tests-dir'] = _filepath_to_system_tests(provider_yaml_dir)
        result.append(provider)
    return result