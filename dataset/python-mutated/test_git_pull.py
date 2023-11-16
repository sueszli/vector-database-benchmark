import pytest
from thefuck.rules.git_pull import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def output():
    if False:
        while True:
            i = 10
    return 'There is no tracking information for the current branch.\nPlease specify which branch you want to merge with.\nSee git-pull(1) for details\n\n    git pull <remote> <branch>\n\nIf you wish to set tracking information for this branch you can do so with:\n\n    git branch --set-upstream-to=<remote>/<branch> master\n\n'

def test_match(output):
    if False:
        i = 10
        return i + 15
    assert match(Command('git pull', output))
    assert not match(Command('git pull', ''))
    assert not match(Command('ls', output))

def test_get_new_command(output):
    if False:
        i = 10
        return i + 15
    assert get_new_command(Command('git pull', output)) == 'git branch --set-upstream-to=origin/master master && git pull'