"""Tests around prompting for and handling of choice variables."""
import pytest
from cookiecutter.prompt import read_user_choice
OPTIONS = ['hello', 'world', 'foo', 'bar']
OPTIONS_INDEX = ['1', '2', '3', '4']
EXPECTED_PROMPT = 'Select varname\n    [bold magenta]1[/] - [bold]hello[/]\n    [bold magenta]2[/] - [bold]world[/]\n    [bold magenta]3[/] - [bold]foo[/]\n    [bold magenta]4[/] - [bold]bar[/]\n    Choose from'

@pytest.mark.parametrize('user_choice, expected_value', enumerate(OPTIONS, 1))
def test_click_invocation(mocker, user_choice, expected_value):
    if False:
        i = 10
        return i + 15
    'Test click function called correctly by cookiecutter.\n\n    Test for choice type invocation.\n    '
    prompt = mocker.patch('rich.prompt.Prompt.ask')
    prompt.return_value = f'{user_choice}'
    assert read_user_choice('varname', OPTIONS) == expected_value
    prompt.assert_called_once_with(EXPECTED_PROMPT, choices=OPTIONS_INDEX, default='1')

def test_raise_if_options_is_not_a_non_empty_list():
    if False:
        i = 10
        return i + 15
    'Test function called by cookiecutter raise expected errors.\n\n    Test for choice type invocation.\n    '
    with pytest.raises(TypeError):
        read_user_choice('foo', 'NOT A LIST')
    with pytest.raises(ValueError):
        read_user_choice('foo', [])