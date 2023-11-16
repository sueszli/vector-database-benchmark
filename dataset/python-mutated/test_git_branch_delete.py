import pytest
from thefuck.rules.git_branch_delete import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def output():
    if False:
        print('Hello World!')
    return "error: The branch 'branch' is not fully merged.\nIf you are sure you want to delete it, run 'git branch -D branch'.\n\n"

def test_match(output):
    if False:
        while True:
            i = 10
    assert match(Command('git branch -d branch', output))
    assert not match(Command('git branch -d branch', ''))
    assert not match(Command('ls', output))

def test_get_new_command(output):
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(Command('git branch -d branch', output)) == 'git branch -D branch'