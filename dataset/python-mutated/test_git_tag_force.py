import pytest
from thefuck.rules.git_tag_force import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def output():
    if False:
        print('Hello World!')
    return "fatal: tag 'alert' already exists"

def test_match(output):
    if False:
        while True:
            i = 10
    assert match(Command('git tag alert', output))
    assert not match(Command('git tag alert', ''))

def test_get_new_command(output):
    if False:
        while True:
            i = 10
    assert get_new_command(Command('git tag alert', output)) == 'git tag --force alert'