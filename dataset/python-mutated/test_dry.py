import pytest
from thefuck.rules.dry import match, get_new_command
from thefuck.types import Command

@pytest.mark.parametrize('command', [Command('cd cd foo', ''), Command('git git push origin/master', '')])
def test_match(command):
    if False:
        i = 10
        return i + 15
    assert match(command)

@pytest.mark.parametrize('command, new_command', [(Command('cd cd foo', ''), 'cd foo'), (Command('git git push origin/master', ''), 'git push origin/master')])
def test_get_new_command(command, new_command):
    if False:
        print('Hello World!')
    assert get_new_command(command) == new_command