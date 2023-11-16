import pytest
from source_gridly.helpers import Helpers

@pytest.fixture
def gridly_column_type():
    if False:
        for i in range(10):
            print('nop')
    return 'singleLine'

@pytest.fixture
def expected_data_type():
    if False:
        return 10
    return 'string'

@pytest.fixture
def view_response():
    if False:
        print('Hello World!')
    return {'id': 'view1', 'name': 'Default view', 'columns': [{'id': '_recordId'}, {'id': 'column1', 'type': 'singleLine'}, {'id': 'column2', 'type': 'number'}]}

@pytest.fixture
def record_response():
    if False:
        return 10
    return [{'id': 'record1', 'cells': [{'columnId': 'column1', 'value': 'Value 1'}, {'columnId': 'column2', 'value': 1}]}, {'id': 'record2', 'cells': [{'columnId': 'column1', 'value': 'Value 2'}, {'columnId': 'column2', 'value': 2}]}]

@pytest.fixture
def expected_json_schema():
    if False:
        for i in range(10):
            print('nop')
    return {'$schema': 'http://json-schema.org/draft-07/schema#', 'properties': {'_recordId': {'type': ['null', 'string']}, 'column1': {'type': ['null', 'string']}, 'column2': {'type': ['null', 'number']}}, 'type': 'object'}

@pytest.fixture
def expected_transformed_record():
    if False:
        while True:
            i = 10
    return {'_recordId': 'record1', 'column1': 'Value 1', 'column2': 1}

def test_to_airbyte_data_type(gridly_column_type, expected_data_type):
    if False:
        while True:
            i = 10
    assert expected_data_type == Helpers.to_airbyte_data_type(gridly_column_type)

def test_get_json_schema(view_response, expected_json_schema):
    if False:
        print('Hello World!')
    json_schema = Helpers.get_json_schema(view_response)
    assert json_schema == expected_json_schema

def test_get_airbyte_stream(view_response, expected_json_schema):
    if False:
        return 10
    stream = Helpers.get_airbyte_stream(view_response)
    assert stream
    assert stream.name == view_response.get('name')
    assert stream.json_schema == expected_json_schema

def test_transform_record(view_response, record_response, expected_transformed_record):
    if False:
        while True:
            i = 10
    json_schema = Helpers.get_json_schema(view_response)
    record1 = record_response[0]
    transformed_record = Helpers.transform_record(record1, json_schema)
    assert expected_transformed_record == transformed_record