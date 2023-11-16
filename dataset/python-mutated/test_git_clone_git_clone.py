from thefuck.rules.git_clone_git_clone import match, get_new_command
from thefuck.types import Command
output_clean = '\nfatal: Too many arguments.\n\nusage: git clone [<options>] [--] <repo> [<dir>]\n'

def test_match():
    if False:
        for i in range(10):
            print('nop')
    assert match(Command('git clone git clone foo', output_clean))

def test_not_match():
    if False:
        for i in range(10):
            print('nop')
    assert not match(Command('', ''))
    assert not match(Command('git branch', ''))
    assert not match(Command('git clone foo', ''))
    assert not match(Command('git clone foo bar baz', output_clean))

def test_get_new_command():
    if False:
        i = 10
        return i + 15
    assert get_new_command(Command('git clone git clone foo', output_clean)) == 'git clone foo'