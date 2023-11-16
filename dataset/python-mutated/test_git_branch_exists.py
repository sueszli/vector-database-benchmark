import pytest
from thefuck.rules.git_branch_exists import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def output(src_branch_name):
    if False:
        return 10
    return "fatal: A branch named '{}' already exists.".format(src_branch_name)

@pytest.fixture
def new_command(branch_name):
    if False:
        for i in range(10):
            print('nop')
    return [cmd.format(branch_name) for cmd in ['git branch -d {0} && git branch {0}', 'git branch -d {0} && git checkout -b {0}', 'git branch -D {0} && git branch {0}', 'git branch -D {0} && git checkout -b {0}', 'git checkout {0}']]

@pytest.mark.parametrize('script, src_branch_name, branch_name', [('git branch foo', 'foo', 'foo'), ('git checkout bar', 'bar', 'bar'), ('git checkout -b "let\'s-push-this"', '"let\'s-push-this"', '"let\'s-push-this"')])
def test_match(output, script, branch_name):
    if False:
        print('Hello World!')
    assert match(Command(script, output))

@pytest.mark.parametrize('script', ['git branch foo', 'git checkout bar', 'git checkout -b "let\'s-push-this"'])
def test_not_match(script):
    if False:
        i = 10
        return i + 15
    assert not match(Command(script, ''))

@pytest.mark.parametrize('script, src_branch_name, branch_name', [('git branch foo', 'foo', 'foo'), ('git checkout bar', 'bar', 'bar'), ('git checkout -b "let\'s-push-this"', "let's-push-this", "let\\'s-push-this")])
def test_get_new_command(output, new_command, script, src_branch_name, branch_name):
    if False:
        i = 10
        return i + 15
    assert get_new_command(Command(script, output)) == new_command