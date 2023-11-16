import pytest
from thefuck.rules.git_rebase_no_changes import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def output():
    if False:
        return 10
    return 'Applying: Test commit\nNo changes - did you forget to use \'git add\'?\nIf there is nothing left to stage, chances are that something else\nalready introduced the same changes; you might want to skip this patch.\n\nWhen you have resolved this problem, run "git rebase --continue".\nIf you prefer to skip this patch, run "git rebase --skip" instead.\nTo check out the original branch and stop rebasing, run "git rebase --abort".\n\n'

def test_match(output):
    if False:
        while True:
            i = 10
    assert match(Command('git rebase --continue', output))
    assert not match(Command('git rebase --continue', ''))
    assert not match(Command('git rebase --skip', ''))

def test_get_new_command(output):
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(Command('git rebase --continue', output)) == 'git rebase --skip'