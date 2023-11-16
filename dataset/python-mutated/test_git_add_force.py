import pytest
from thefuck.rules.git_add_force import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def output():
    if False:
        print('Hello World!')
    return 'The following paths are ignored by one of your .gitignore files:\ndist/app.js\ndist/background.js\ndist/options.js\nUse -f if you really want to add them.\n'

def test_match(output):
    if False:
        while True:
            i = 10
    assert match(Command('git add dist/*.js', output))
    assert not match(Command('git add dist/*.js', ''))

def test_get_new_command(output):
    if False:
        return 10
    assert get_new_command(Command('git add dist/*.js', output)) == 'git add --force dist/*.js'