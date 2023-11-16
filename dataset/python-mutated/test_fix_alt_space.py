from thefuck.rules.fix_alt_space import match, get_new_command
from thefuck.types import Command

def test_match():
    if False:
        while True:
            i = 10
    "The character before 'grep' is Alt+Space, which happens frequently\n    on the Mac when typing the pipe character (Alt+7), and holding the Alt\n    key pressed for longer than necessary.\n\n    "
    assert match(Command(u'ps -ef |\xa0grep foo', u'-bash: \xa0grep: command not found'))
    assert not match(Command('ps -ef | grep foo', ''))
    assert not match(Command('', ''))

def test_get_new_command():
    if False:
        i = 10
        return i + 15
    ' Replace the Alt+Space character by a simple space '
    assert get_new_command(Command(u'ps -ef |\xa0grep foo', '')) == 'ps -ef | grep foo'