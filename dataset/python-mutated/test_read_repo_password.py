"""Tests around handling repositories which require authentication."""
from cookiecutter.prompt import read_repo_password

def test_click_invocation(mocker):
    if False:
        i = 10
        return i + 15
    'Test click function called correctly by cookiecutter.\n\n    Test for password (hidden input) type invocation.\n    '
    prompt = mocker.patch('rich.prompt.Prompt.ask')
    prompt.return_value = 'sekrit'
    assert read_repo_password('Password') == 'sekrit'
    prompt.assert_called_once_with('Password', password=True)