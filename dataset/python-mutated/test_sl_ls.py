from thefuck.rules.sl_ls import match, get_new_command
from thefuck.types import Command

def test_match():
    if False:
        for i in range(10):
            print('nop')
    assert match(Command('sl', ''))
    assert not match(Command('ls', ''))

def test_get_new_command():
    if False:
        i = 10
        return i + 15
    assert get_new_command(Command('sl', '')) == 'ls'