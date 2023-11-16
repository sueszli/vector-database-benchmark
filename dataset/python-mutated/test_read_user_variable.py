"""test_read_user_variable."""
import pytest
from cookiecutter.prompt import read_user_variable
VARIABLE = 'project_name'
DEFAULT = 'Kivy Project'

@pytest.fixture
def mock_prompt(mocker):
    if False:
        for i in range(10):
            print('nop')
    "Return a mocked version of the 'Prompt.ask' function."
    return mocker.patch('rich.prompt.Prompt.ask')

def test_click_invocation(mock_prompt):
    if False:
        for i in range(10):
            print('nop')
    'Test click function called correctly by cookiecutter.\n\n    Test for string type invocation.\n    '
    mock_prompt.return_value = DEFAULT
    assert read_user_variable(VARIABLE, DEFAULT) == DEFAULT
    mock_prompt.assert_called_once_with(VARIABLE, default=DEFAULT)

def test_input_loop_with_null_default_value(mock_prompt):
    if False:
        return 10
    'Test `Prompt.ask` is run repeatedly until a valid answer is provided.\n\n    Test for `default_value` parameter equal to None.\n    '
    mock_prompt.side_effect = [None, DEFAULT]
    assert read_user_variable(VARIABLE, None) == DEFAULT
    assert mock_prompt.call_count == 2