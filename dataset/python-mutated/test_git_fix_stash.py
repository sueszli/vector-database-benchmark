import pytest
from thefuck.rules.git_fix_stash import match, get_new_command
from thefuck.types import Command
git_stash_err = '\nusage: git stash list [<options>]\n   or: git stash show [<stash>]\n   or: git stash drop [-q|--quiet] [<stash>]\n   or: git stash ( pop | apply ) [--index] [-q|--quiet] [<stash>]\n   or: git stash branch <branchname> [<stash>]\n   or: git stash [save [--patch] [-k|--[no-]keep-index] [-q|--quiet]\n\t\t       [-u|--include-untracked] [-a|--all] [<message>]]\n   or: git stash clear\n'

@pytest.mark.parametrize('wrong', ['git stash opp', 'git stash Some message', 'git stash saev Some message'])
def test_match(wrong):
    if False:
        print('Hello World!')
    assert match(Command(wrong, git_stash_err))

def test_not_match():
    if False:
        for i in range(10):
            print('nop')
    assert not match(Command('git', git_stash_err))

@pytest.mark.parametrize('wrong,fixed', [('git stash opp', 'git stash pop'), ('git stash Some message', 'git stash save Some message'), ('git stash saev Some message', 'git stash save Some message')])
def test_get_new_command(wrong, fixed):
    if False:
        i = 10
        return i + 15
    assert get_new_command(Command(wrong, git_stash_err)) == fixed