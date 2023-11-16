import pytest
from thefuck.rules.git_rm_recursive import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def output(target):
    if False:
        for i in range(10):
            print('nop')
    return "fatal: not removing '{}' recursively without -r".format(target)

@pytest.mark.parametrize('script, target', [('git rm foo', 'foo'), ('git rm foo bar', 'foo bar')])
def test_match(output, script, target):
    if False:
        for i in range(10):
            print('nop')
    assert match(Command(script, output))

@pytest.mark.parametrize('script', ['git rm foo', 'git rm foo bar'])
def test_not_match(script):
    if False:
        i = 10
        return i + 15
    assert not match(Command(script, ''))

@pytest.mark.parametrize('script, target, new_command', [('git rm foo', 'foo', 'git rm -r foo'), ('git rm foo bar', 'foo bar', 'git rm -r foo bar')])
def test_get_new_command(output, script, target, new_command):
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(Command(script, output)) == new_command