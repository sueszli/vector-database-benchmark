import pytest
from thefuck.rules.git_commit_reset import match, get_new_command
from thefuck.types import Command

@pytest.mark.parametrize('script, output', [('git commit -m "test"', 'test output'), ('git commit', '')])
def test_match(output, script):
    if False:
        while True:
            i = 10
    assert match(Command(script, output))

@pytest.mark.parametrize('script', ['git branch foo', 'git checkout feature/test_commit', 'git push'])
def test_not_match(script):
    if False:
        while True:
            i = 10
    assert not match(Command(script, ''))

@pytest.mark.parametrize('script', ['git commit -m "test commit"', 'git commit'])
def test_get_new_command(script):
    if False:
        return 10
    assert get_new_command(Command(script, '')) == 'git reset HEAD~'