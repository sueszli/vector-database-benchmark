import pytest
from thefuck.rules.git_pull_clone import match, get_new_command
from thefuck.types import Command
git_err = '\nfatal: Not a git repository (or any parent up to mount point /home)\nStopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).\n'

@pytest.mark.parametrize('command', [Command('git pull git@github.com:mcarton/thefuck.git', git_err)])
def test_match(command):
    if False:
        return 10
    assert match(command)

@pytest.mark.parametrize('command, output', [(Command('git pull git@github.com:mcarton/thefuck.git', git_err), 'git clone git@github.com:mcarton/thefuck.git')])
def test_get_new_command(command, output):
    if False:
        i = 10
        return i + 15
    assert get_new_command(command) == output