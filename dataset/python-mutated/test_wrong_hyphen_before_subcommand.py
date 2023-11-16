import pytest
from thefuck.rules.wrong_hyphen_before_subcommand import match, get_new_command
from thefuck.types import Command

@pytest.fixture(autouse=True)
def get_all_executables(mocker):
    if False:
        i = 10
        return i + 15
    mocker.patch('thefuck.rules.wrong_hyphen_before_subcommand.get_all_executables', return_value=['git', 'apt', 'apt-get', 'ls', 'pwd'])

@pytest.mark.parametrize('script', ['git-log', 'apt-install python'])
def test_match(script):
    if False:
        for i in range(10):
            print('nop')
    assert match(Command(script, ''))

@pytest.mark.parametrize('script', ['ls -la', 'git2-make', 'apt-get install python'])
def test_not_match(script):
    if False:
        i = 10
        return i + 15
    assert not match(Command(script, ''))

@pytest.mark.parametrize('script, new_command', [('git-log', 'git log'), ('apt-install python', 'apt install python')])
def test_get_new_command(script, new_command):
    if False:
        while True:
            i = 10
    assert get_new_command(Command(script, '')) == new_command