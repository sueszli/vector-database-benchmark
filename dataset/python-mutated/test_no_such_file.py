import pytest
from thefuck.rules.no_such_file import match, get_new_command
from thefuck.types import Command

@pytest.mark.parametrize('command', [Command('mv foo bar/foo', "mv: cannot move 'foo' to 'bar/foo': No such file or directory"), Command('mv foo bar/', "mv: cannot move 'foo' to 'bar/': No such file or directory")])
def test_match(command):
    if False:
        while True:
            i = 10
    assert match(command)

@pytest.mark.parametrize('command', [Command('mv foo bar/', ''), Command('mv foo bar/foo', 'mv: permission denied')])
def test_not_match(command):
    if False:
        return 10
    assert not match(command)

@pytest.mark.parametrize('command, new_command', [(Command('mv foo bar/foo', "mv: cannot move 'foo' to 'bar/foo': No such file or directory"), 'mkdir -p bar && mv foo bar/foo'), (Command('mv foo bar/', "mv: cannot move 'foo' to 'bar/': No such file or directory"), 'mkdir -p bar && mv foo bar/')])
def test_get_new_command(command, new_command):
    if False:
        i = 10
        return i + 15
    assert get_new_command(command) == new_command