import pytest
from thefuck.rules.remove_trailing_cedilla import match, get_new_command, CEDILLA
from thefuck.types import Command

@pytest.mark.parametrize('command', [Command('wrong' + CEDILLA, ''), Command('wrong with args' + CEDILLA, '')])
def test_match(command):
    if False:
        for i in range(10):
            print('nop')
    assert match(command)

@pytest.mark.parametrize('command, new_command', [(Command('wrong' + CEDILLA, ''), 'wrong'), (Command('wrong with args' + CEDILLA, ''), 'wrong with args')])
def test_get_new_command(command, new_command):
    if False:
        i = 10
        return i + 15
    assert get_new_command(command) == new_command