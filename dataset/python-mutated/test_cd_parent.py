from thefuck.rules.cd_parent import match, get_new_command
from thefuck.types import Command

def test_match():
    if False:
        while True:
            i = 10
    assert match(Command('cd..', 'cd..: command not found'))
    assert not match(Command('', ''))

def test_get_new_command():
    if False:
        while True:
            i = 10
    assert get_new_command(Command('cd..', '')) == 'cd ..'