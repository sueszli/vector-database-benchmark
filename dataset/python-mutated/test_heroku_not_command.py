import pytest
from thefuck.types import Command
from thefuck.rules.heroku_not_command import match, get_new_command
suggest_output = '\n ▸    log is not a heroku command.\n ▸    Perhaps you meant logs?\n ▸    Run heroku _ to run heroku logs.\n ▸    Run heroku help for a list of available commands.'

@pytest.mark.parametrize('cmd', ['log'])
def test_match(cmd):
    if False:
        print('Hello World!')
    assert match(Command('heroku {}'.format(cmd), suggest_output))

@pytest.mark.parametrize('script, output', [('cat log', suggest_output)])
def test_not_match(script, output):
    if False:
        return 10
    assert not match(Command(script, output))

@pytest.mark.parametrize('cmd, result', [('log', 'heroku logs')])
def test_get_new_command(cmd, result):
    if False:
        i = 10
        return i + 15
    command = Command('heroku {}'.format(cmd), suggest_output)
    assert get_new_command(command) == result