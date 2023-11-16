import json
import unittest.mock as mock
import pytest
import spec_linter

def test_get_full_field_name():
    if False:
        i = 10
        return i + 15
    assert spec_linter.get_full_field_name('field') == 'field'
    assert spec_linter.get_full_field_name('field', ['root']) == 'root.field'
    assert spec_linter.get_full_field_name('field', ['root', 'fake_field', '0']) == 'root.fake_field.0.field'

def test_fetch_oneof_schemas():
    if False:
        print('Hello World!')
    root_schema = {'oneOf': [{'properties': {1: 1}}, {'values': [1, 2, 3]}]}
    schemas = spec_linter.fetch_oneof_schemas(root_schema)
    assert len(schemas) == 1
    assert schemas[0] == {'properties': {1: 1}}
    root_schema = {'oneOf': [{'properties': {1: 1}}, {'properties': {2: 2}}]}
    schemas = spec_linter.fetch_oneof_schemas(root_schema)
    assert len(schemas) == 2
    assert schemas[0] == {'properties': {1: 1}}
    assert schemas[1] == {'properties': {2: 2}}

@pytest.mark.parametrize('schema,error_text', [({'type': 'string', 'title': 'Field'}, 'Check failed for field'), ({'type': 'string', 'description': 'Format: YYYY-MM-DDTHH:mm:ss[Z].'}, 'Check failed for field'), ({'type': 'string', 'title': 'Field', 'description': 'Format: YYYY-MM-DDTHH:mm:ss[Z].', 'oneOf': 'invalid'}, 'Incorrect oneOf schema in field'), ({'type': 'string', 'title': 'Field', 'description': 'Format: YYYY-MM-DDTHH:mm:ss[Z].', 'examples': ['2020-01-01T00:00:00Z'], 'oneOf': [1, 2, 3]}, 'Incorrect oneOf schema in field')])
def test_validate_field(schema, error_text):
    if False:
        while True:
            i = 10
    errors = spec_linter.validate_field('field', schema, [])
    assert len(errors) == 1
    assert error_text in errors[0]

def test_validate_field_invalid_schema_and_oneof():
    if False:
        print('Hello World!')
    schema = {'type': 'string', 'description': 'Format: YYYY-MM-DDTHH:mm:ss[Z].', 'examples': ['2020-01-01T00:00:00Z'], 'oneOf': [1, 2, 3]}
    errors = spec_linter.validate_field('field', schema, ['root'])
    assert len(errors) == 2
    assert 'Check failed for field' in errors[0]
    assert 'Incorrect oneOf schema in field' in errors[1]

def test_read_spec_file():
    if False:
        while True:
            i = 10
    with mock.patch('builtins.open', mock.mock_open(read_data='test')):
        assert not spec_linter.read_spec_file('path_1')
    with mock.patch('builtins.open', mock.mock_open(read_data='{"connectionSpecification": "test"}')):
        assert not spec_linter.read_spec_file('path_1')
    valid_schema = {'connectionSpecification': {'properties': {}}}
    with mock.patch('builtins.open', mock.mock_open(read_data=json.dumps(valid_schema))):
        assert spec_linter.read_spec_file('path_1')
    invalid_schema = {'connectionSpecification': {'properties': {'field': {'title': 'Field', 'type': 'string'}}}}
    with mock.patch('builtins.open', mock.mock_open(read_data=json.dumps(invalid_schema))):
        assert not spec_linter.read_spec_file('path_1')

def test_validate_schema_failed():
    if False:
        i = 10
        return i + 15
    schema = {'access_token': {'type': 'string', 'airbyte_secret': True, 'description': 'API Key.'}, 'store_name': {'type': 'string', 'title': 'Store name.'}, 'start_date': {'title': 'Start Date', 'type': 'string', 'description': "The date from which you'd like to replicate the data", 'examples': ['2021-01-01T00:00:00Z']}}
    errors = spec_linter.validate_schema('path', schema, ['root'])
    assert len(errors) == 2
    assert 'Check failed for field' in errors[0] and 'root.access_token' in errors[0]
    assert 'Check failed for field' in errors[1] and 'root.store_name' in errors[1]

def test_validate_schema_success():
    if False:
        return 10
    schema = {'access_token': {'type': 'string', 'airbyte_secret': True, 'description': 'API Key.', 'title': 'Key'}, 'store_name': {'type': 'string', 'description': 'My description', 'title': 'My name'}, 'limit': {'title': 'Records Limit', 'type': 'integer', 'description': 'Just a limit'}}
    errors = spec_linter.validate_schema('path', schema, ['root'])
    assert len(errors) == 0

def test_validate_schema_with_nested_oneof():
    if False:
        for i in range(10):
            print('nop')
    schema = {'store_name': {'type': 'string', 'description': 'Store name.'}, 'start_date': {'title': 'Start Date', 'type': 'string', 'description': "The date from which you'd like to replicate the data"}, 'nested_field': {'type': 'object', 'title': 'Nested field title', 'description': 'Nested field description', 'oneOf': [{'type': 'object', 'properties': {'settings': {'type': 'object', 'title': 'Settings', 'description': 'blah-blah-blah', 'oneOf': [{'type': 'object', 'properties': {'access_token': {'type': 'object'}}}, {'type': 'string', 'multipleOf': 3}]}}}, {'type': 'string', 'title': 'Start Date'}]}}
    errors = spec_linter.validate_schema('path', schema, [])
    assert len(errors) == 2
    assert 'Check failed for field' == errors[0][0]
    assert 'Check failed for field' == errors[1][0]
    assert 'store_name' == errors[0][1]
    assert 'nested_field.0.settings.0.access_token' == errors[1][1]