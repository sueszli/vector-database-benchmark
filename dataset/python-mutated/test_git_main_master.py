import pytest
from thefuck.rules.git_main_master import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def output(branch_name):
    if False:
        for i in range(10):
            print('nop')
    if not branch_name:
        return ''
    output_str = u"error: pathspec '{}' did not match any file(s) known to git"
    return output_str.format(branch_name)

@pytest.mark.parametrize('script, branch_name', [('git checkout main', 'main'), ('git checkout master', 'master'), ('git show main', 'main')])
def test_match(script, branch_name, output):
    if False:
        print('Hello World!')
    assert match(Command(script, output))

@pytest.mark.parametrize('script, branch_name', [('git checkout master', ''), ('git checkout main', ''), ('git checkout wibble', 'wibble')])
def test_not_match(script, branch_name, output):
    if False:
        return 10
    assert not match(Command(script, output))

@pytest.mark.parametrize('script, branch_name, new_command', [('git checkout main', 'main', 'git checkout master'), ('git checkout master', 'master', 'git checkout main'), ('git checkout wibble', 'wibble', 'git checkout wibble')])
def test_get_new_command(script, branch_name, new_command, output):
    if False:
        while True:
            i = 10
    assert get_new_command(Command(script, output)) == new_command