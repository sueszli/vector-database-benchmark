import pytest
from thefuck.rules.git_push_different_branch_names import get_new_command, match
from thefuck.types import Command
output = "fatal: The upstream branch of your current branch does not match\nthe name of your current branch.  To push to the upstream branch\non the remote, use\n\n    git push origin HEAD:%s\n\nTo push to the branch of the same name on the remote, use\n\n    git push origin %s\n\nTo choose either option permanently, see push.default in 'git help config'.\n"

def error_msg(localbranch, remotebranch):
    if False:
        for i in range(10):
            print('nop')
    return output % (remotebranch, localbranch)

def test_match():
    if False:
        print('Hello World!')
    assert match(Command('git push', error_msg('foo', 'bar')))

@pytest.mark.parametrize('command', [Command('vim', ''), Command('git status', error_msg('foo', 'bar')), Command('git push', '')])
def test_not_match(command):
    if False:
        return 10
    assert not match(command)

def test_get_new_command():
    if False:
        print('Hello World!')
    new_command = get_new_command(Command('git push', error_msg('foo', 'bar')))
    assert new_command == 'git push origin HEAD:bar'