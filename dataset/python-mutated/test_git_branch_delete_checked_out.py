import pytest
from thefuck.rules.git_branch_delete_checked_out import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def output():
    if False:
        print('Hello World!')
    return "error: Cannot delete branch 'foo' checked out at '/bar/foo'"

@pytest.mark.parametrize('script', ['git branch -d foo', 'git branch -D foo'])
def test_match(script, output):
    if False:
        for i in range(10):
            print('nop')
    assert match(Command(script, output))

@pytest.mark.parametrize('script', ['git branch -d foo', 'git branch -D foo'])
def test_not_match(script):
    if False:
        return 10
    assert not match(Command(script, 'Deleted branch foo (was a1b2c3d).'))

@pytest.mark.parametrize('script, new_command', [('git branch -d foo', 'git checkout master && git branch -D foo'), ('git branch -D foo', 'git checkout master && git branch -D foo')])
def test_get_new_command(script, new_command, output):
    if False:
        print('Hello World!')
    assert get_new_command(Command(script, output)) == new_command