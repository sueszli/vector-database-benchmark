import os
import shutil
from difflib import unified_diff
import pytest
from dash.development._py_components_generation import generate_class_string, generate_class_file
from . import _dir, has_trailing_space
import_string = '# AUTO GENERATED FILE - DO NOT EDIT\n\n' + 'from dash.development.base_component import' + ' Component, _explicitize_args\n\n\n'

@pytest.fixture
def make_component_dir(load_test_metadata_json):
    if False:
        while True:
            i = 10
    os.makedirs('TableComponents')
    yield load_test_metadata_json
    shutil.rmtree('TableComponents')

@pytest.fixture
def expected_class_string():
    if False:
        while True:
            i = 10
    expected_string_path = os.path.join(_dir, 'metadata_test.py')
    with open(expected_string_path, 'r') as f:
        return f.read()

@pytest.fixture
def component_class_string(make_component_dir):
    if False:
        i = 10
        return i + 15
    return import_string + generate_class_string(typename='Table', props=make_component_dir['props'], description=make_component_dir['description'], namespace='TableComponents')

@pytest.fixture
def written_class_string(make_component_dir):
    if False:
        for i in range(10):
            print('nop')
    generate_class_file(typename='Table', props=make_component_dir['props'], description=make_component_dir['description'], namespace='TableComponents')
    written_file_path = os.path.join('TableComponents', 'Table.py')
    with open(written_file_path, 'r') as f:
        return f.read()

def test_class_string(expected_class_string, component_class_string):
    if False:
        return 10
    assert not list(unified_diff(expected_class_string.splitlines(), component_class_string.splitlines()))
    assert not has_trailing_space(component_class_string)

def test_class_file(expected_class_string, written_class_string):
    if False:
        for i in range(10):
            print('nop')
    assert not list(unified_diff(expected_class_string.splitlines(), written_class_string.splitlines()))
    assert not has_trailing_space(written_class_string)