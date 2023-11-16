import pytest
from airbyte_cdk.utils.mapping_helpers import combine_mappings

def test_basic_merge():
    if False:
        for i in range(10):
            print('nop')
    mappings = [{'a': 1}, {'b': 2}, {'c': 3}, {}]
    result = combine_mappings(mappings)
    assert result == {'a': 1, 'b': 2, 'c': 3}

def test_combine_with_string():
    if False:
        for i in range(10):
            print('nop')
    mappings = [{'a': 1}, 'option']
    with pytest.raises(ValueError, match='Cannot combine multiple options if one is a string'):
        combine_mappings(mappings)

def test_overlapping_keys():
    if False:
        while True:
            i = 10
    mappings = [{'a': 1, 'b': 2}, {'b': 3}]
    with pytest.raises(ValueError, match='Duplicate keys found'):
        combine_mappings(mappings)

def test_multiple_strings():
    if False:
        for i in range(10):
            print('nop')
    mappings = ['option1', 'option2']
    with pytest.raises(ValueError, match='Cannot combine multiple string options'):
        combine_mappings(mappings)

def test_handle_none_values():
    if False:
        while True:
            i = 10
    mappings = [{'a': 1}, None, {'b': 2}]
    result = combine_mappings(mappings)
    assert result == {'a': 1, 'b': 2}

def test_empty_mappings():
    if False:
        for i in range(10):
            print('nop')
    mappings = []
    result = combine_mappings(mappings)
    assert result == {}

def test_single_mapping():
    if False:
        while True:
            i = 10
    mappings = [{'a': 1}]
    result = combine_mappings(mappings)
    assert result == {'a': 1}

def test_combine_with_string_and_empty_mappings():
    if False:
        for i in range(10):
            print('nop')
    mappings = ['option', {}]
    result = combine_mappings(mappings)
    assert result == 'option'