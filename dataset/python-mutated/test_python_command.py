from thefuck.rules.python_command import match, get_new_command
from thefuck.types import Command

def test_match():
    if False:
        for i in range(10):
            print('nop')
    assert match(Command('temp.py', 'Permission denied'))
    assert not match(Command('', ''))

def test_get_new_command():
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(Command('./test_sudo.py', '')) == 'python ./test_sudo.py'