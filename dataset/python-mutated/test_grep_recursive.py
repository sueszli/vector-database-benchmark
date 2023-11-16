from thefuck.rules.grep_recursive import match, get_new_command
from thefuck.types import Command

def test_match():
    if False:
        return 10
    assert match(Command('grep blah .', 'grep: .: Is a directory'))
    assert match(Command(u'grep café .', 'grep: .: Is a directory'))
    assert not match(Command('', ''))

def test_get_new_command():
    if False:
        i = 10
        return i + 15
    assert get_new_command(Command('grep blah .', '')) == 'grep -r blah .'
    assert get_new_command(Command(u'grep café .', '')) == u'grep -r café .'