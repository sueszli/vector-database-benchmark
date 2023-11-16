import pytest
from thefuck.rules.git_diff_no_index import match, get_new_command
from thefuck.types import Command

@pytest.mark.parametrize('command', [Command('git diff foo bar', '')])
def test_match(command):
    if False:
        print('Hello World!')
    assert match(command)

@pytest.mark.parametrize('command', [Command('git diff --no-index foo bar', ''), Command('git diff foo', ''), Command('git diff foo bar baz', '')])
def test_not_match(command):
    if False:
        print('Hello World!')
    assert not match(command)

@pytest.mark.parametrize('command, new_command', [(Command('git diff foo bar', ''), 'git diff --no-index foo bar')])
def test_get_new_command(command, new_command):
    if False:
        i = 10
        return i + 15
    assert get_new_command(command) == new_command