import pytest
from custom_auth.oauth.exceptions import GithubError
from custom_auth.oauth.helpers.github_helpers import convert_response_data_to_dictionary, get_first_and_last_name

def test_convert_response_data_to_dictionary_success():
    if False:
        print('Hello World!')
    response_string = 'key_1=value_1&key_2=value_2&key_3=value_3'
    response_dict = convert_response_data_to_dictionary(response_string)
    assert response_dict == {'key_1': 'value_1', 'key_2': 'value_2', 'key_3': 'value_3'}

def test_convert_response_data_to_dictionary_fail():
    if False:
        while True:
            i = 10
    response_string = 'key_1value_1&key_2=value_2=value_2'
    with pytest.raises(GithubError):
        convert_response_data_to_dictionary(response_string)

def test_get_first_and_last_name_success():
    if False:
        while True:
            i = 10
    full_name = 'tommy tester'
    (first_name, last_name) = get_first_and_last_name(full_name)
    assert first_name == 'tommy'
    assert last_name == 'tester'

def test_get_first_and_last_name_too_many_names():
    if False:
        while True:
            i = 10
    full_name = 'tommy tester the third king among testers'
    (first_name, last_name) = get_first_and_last_name(full_name)
    assert first_name == full_name
    assert last_name == ''

def test_get_first_and_last_name_too_few_names():
    if False:
        print('Hello World!')
    full_name = 'wall-e'
    (first_name, last_name) = get_first_and_last_name(full_name)
    assert first_name == full_name
    assert last_name == ''