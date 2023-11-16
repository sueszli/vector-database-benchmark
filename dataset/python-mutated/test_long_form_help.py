import pytest
from thefuck.rules.long_form_help import match, get_new_command
from thefuck.types import Command

@pytest.mark.parametrize('output', ["Try 'grep --help' for more information."])
def test_match(output):
    if False:
        i = 10
        return i + 15
    assert match(Command('grep -h', output))

def test_not_match():
    if False:
        for i in range(10):
            print('nop')
    assert not match(Command('', ''))

@pytest.mark.parametrize('before, after', [('grep -h', 'grep --help'), ('tar -h', 'tar --help'), ('docker run -h', 'docker run --help'), ('cut -h', 'cut --help')])
def test_get_new_command(before, after):
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(Command(before, '')) == after