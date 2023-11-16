import pytest
from thefuck.rules.git_stash import match, get_new_command
from thefuck.types import Command
cherry_pick_error = 'error: Your local changes would be overwritten by cherry-pick.\nhint: Commit your changes or stash them to proceed.\nfatal: cherry-pick failed'
rebase_error = 'Cannot rebase: Your index contains uncommitted changes.\nPlease commit or stash them.'

@pytest.mark.parametrize('command', [Command('git cherry-pick a1b2c3d', cherry_pick_error), Command('git rebase -i HEAD~7', rebase_error)])
def test_match(command):
    if False:
        i = 10
        return i + 15
    assert match(command)

@pytest.mark.parametrize('command', [Command('git cherry-pick a1b2c3d', ''), Command('git rebase -i HEAD~7', '')])
def test_not_match(command):
    if False:
        print('Hello World!')
    assert not match(command)

@pytest.mark.parametrize('command, new_command', [(Command('git cherry-pick a1b2c3d', cherry_pick_error), 'git stash && git cherry-pick a1b2c3d'), (Command('git rebase -i HEAD~7', rebase_error), 'git stash && git rebase -i HEAD~7')])
def test_get_new_command(command, new_command):
    if False:
        while True:
            i = 10
    assert get_new_command(command) == new_command