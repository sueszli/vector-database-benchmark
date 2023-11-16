from thefuck.rules.cd_cs import match, get_new_command
from thefuck.types import Command

def test_match():
    if False:
        for i in range(10):
            print('nop')
    assert match(Command('cs', 'cs: command not found'))
    assert match(Command('cs /etc/', 'cs: command not found'))

def test_get_new_command():
    if False:
        i = 10
        return i + 15
    assert get_new_command(Command('cs /etc/', 'cs: command not found')) == 'cd /etc/'