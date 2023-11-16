import pytest
from thefuck.rules.git_pull_uncommitted_changes import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def output():
    if False:
        for i in range(10):
            print('nop')
    return 'error: Cannot pull with rebase: You have unstaged changes.'

def test_match(output):
    if False:
        print('Hello World!')
    assert match(Command('git pull', output))
    assert not match(Command('git pull', ''))
    assert not match(Command('ls', output))

def test_get_new_command(output):
    if False:
        while True:
            i = 10
    assert get_new_command(Command('git pull', output)) == 'git stash && git pull && git stash pop'