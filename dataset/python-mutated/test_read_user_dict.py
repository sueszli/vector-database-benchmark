"""Test `process_json`, `read_user_dict` functions in `cookiecutter.prompt`."""
import click
import pytest
from cookiecutter.prompt import process_json, read_user_dict, JsonPrompt
from rich.prompt import InvalidResponse

def test_process_json_invalid_json():
    if False:
        return 10
    'Test `process_json` for correct error on malformed input.'
    with pytest.raises(InvalidResponse) as exc_info:
        process_json('nope]')
    assert str(exc_info.value) == 'Unable to decode to JSON.'

def test_process_json_non_dict():
    if False:
        while True:
            i = 10
    'Test `process_json` for correct error on non-JSON input.'
    with pytest.raises(InvalidResponse) as exc_info:
        process_json('[1, 2]')
    assert str(exc_info.value) == 'Requires JSON dict.'

def test_process_json_valid_json():
    if False:
        print('Hello World!')
    'Test `process_json` for correct output on JSON input.\n\n    Test for simple dict with list.\n    '
    user_value = '{"name": "foobar", "bla": ["a", 1, "b", false]}'
    assert process_json(user_value) == {'name': 'foobar', 'bla': ['a', 1, 'b', False]}

def test_process_json_deep_dict():
    if False:
        return 10
    'Test `process_json` for correct output on JSON input.\n\n    Test for dict in dict case.\n    '
    user_value = '{\n        "key": "value",\n        "integer_key": 37,\n        "dict_key": {\n            "deep_key": "deep_value",\n            "deep_integer": 42,\n            "deep_list": [\n                "deep value 1",\n                "deep value 2",\n                "deep value 3"\n            ]\n        },\n        "list_key": [\n            "value 1",\n            "value 2",\n            "value 3"\n        ]\n    }'
    assert process_json(user_value) == {'key': 'value', 'integer_key': 37, 'dict_key': {'deep_key': 'deep_value', 'deep_integer': 42, 'deep_list': ['deep value 1', 'deep value 2', 'deep value 3']}, 'list_key': ['value 1', 'value 2', 'value 3']}

def test_should_raise_type_error(mocker):
    if False:
        while True:
            i = 10
    'Test `default_value` arg verification in `read_user_dict` function.'
    prompt = mocker.patch('cookiecutter.prompt.JsonPrompt.ask')
    with pytest.raises(TypeError):
        read_user_dict('name', 'russell')
    assert not prompt.called

def test_should_call_prompt_with_process_json(mocker):
    if False:
        print('Hello World!')
    'Test to make sure that `process_json` is actually being used.\n\n    Verifies generation of a processor for the user input.\n    '
    mock_prompt = mocker.patch('cookiecutter.prompt.JsonPrompt.ask', autospec=True)
    read_user_dict('name', {'project_slug': 'pytest-plugin'})
    print(mock_prompt.call_args)
    (args, kwargs) = mock_prompt.call_args
    assert args == ('name [cyan bold](default)[/]',)
    assert kwargs['default'] == {'project_slug': 'pytest-plugin'}

def test_should_not_load_json_from_sentinel(mocker):
    if False:
        i = 10
        return i + 15
    'Make sure that `json.loads` is not called when using default value.'
    mock_json_loads = mocker.patch('cookiecutter.prompt.json.loads', autospec=True, return_value={})
    runner = click.testing.CliRunner()
    with runner.isolation(input='\n'):
        read_user_dict('name', {'project_slug': 'pytest-plugin'})
    mock_json_loads.assert_not_called()

@pytest.mark.parametrize('input', ['\n', '\ndefault\n'])
def test_read_user_dict_default_value(mocker, input):
    if False:
        print('Hello World!')
    'Make sure that `read_user_dict` returns the default value.\n\n    Verify return of a dict variable rather than the display value.\n    '
    runner = click.testing.CliRunner()
    with runner.isolation(input=input):
        val = read_user_dict('name', {'project_slug': 'pytest-plugin'})
    assert val == {'project_slug': 'pytest-plugin'}

def test_json_prompt_process_response():
    if False:
        for i in range(10):
            print('nop')
    'Test `JsonPrompt` process_response to convert str to json.'
    jp = JsonPrompt()
    assert jp.process_response('{"project_slug": "something"}') == {'project_slug': 'something'}