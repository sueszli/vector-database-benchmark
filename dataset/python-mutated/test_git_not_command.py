import pytest
from thefuck.rules.git_not_command import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def git_not_command():
    if False:
        while True:
            i = 10
    return "git: 'brnch' is not a git command. See 'git --help'.\n\nThe most similar command is\nbranch\n"

@pytest.fixture
def git_not_command_one_of_this():
    if False:
        print('Hello World!')
    return "git: 'st' is not a git command. See 'git --help'.\n\nThe most similar commands are\nstatus\nreset\nstage\nstash\nstats\n"

@pytest.fixture
def git_not_command_closest():
    if False:
        print('Hello World!')
    return "git: 'tags' is not a git command. See 'git --help'.\n\nThe most similar commands are\n\tstage\n\ttag\n"

@pytest.fixture
def git_command():
    if False:
        for i in range(10):
            print('nop')
    return '* master'

def test_match(git_not_command, git_command, git_not_command_one_of_this):
    if False:
        return 10
    assert match(Command('git brnch', git_not_command))
    assert match(Command('git st', git_not_command_one_of_this))
    assert not match(Command('ls brnch', git_not_command))
    assert not match(Command('git branch', git_command))

def test_get_new_command(git_not_command, git_not_command_one_of_this, git_not_command_closest):
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(Command('git brnch', git_not_command)) == ['git branch']
    assert get_new_command(Command('git st', git_not_command_one_of_this)) == ['git stats', 'git stash', 'git stage']
    assert get_new_command(Command('git tags', git_not_command_closest)) == ['git tag', 'git stage']