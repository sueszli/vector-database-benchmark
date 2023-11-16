import pytest
from thefuck.rules.rm_root import match, get_new_command
from thefuck.types import Command

def test_match():
    if False:
        for i in range(10):
            print('nop')
    assert match(Command('rm -rf /', 'add --no-preserve-root'))

@pytest.mark.parametrize('command', [Command('ls', 'add --no-preserve-root'), Command('rm --no-preserve-root /', 'add --no-preserve-root'), Command('rm -rf /', '')])
def test_not_match(command):
    if False:
        i = 10
        return i + 15
    assert not match(command)

def test_get_new_command():
    if False:
        while True:
            i = 10
    assert get_new_command(Command('rm -rf /', '')) == 'rm -rf / --no-preserve-root'