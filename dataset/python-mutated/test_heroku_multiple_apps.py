import pytest
from thefuck.types import Command
from thefuck.rules.heroku_multiple_apps import match, get_new_command
suggest_output = "\n ▸    Multiple apps in git remotes\n ▸    Usage: --remote heroku-dev\n ▸    or: --app myapp-dev\n ▸    Your local git repository has more than 1 app referenced in git remotes.\n ▸    Because of this, we can't determine which app you want to run this command against.\n ▸    Specify the app you want with --app or --remote.\n ▸    Heroku remotes in repo:\n ▸    myapp (heroku)\n ▸    myapp-dev (heroku-dev)\n ▸\n ▸    https://devcenter.heroku.com/articles/multiple-environments\n"
not_match_output = '\n=== HEROKU_POSTGRESQL_TEAL_URL, DATABASE_URL\nPlan:                  Hobby-basic\nStatus:                Available\nConnections:           20/20\nPG Version:            9.6.4\nCreated:               2017-01-01 00:00 UTC\nData Size:             99.9 MB\nTables:                99\nRows:                  12345/10000000 (In compliance)\nFork/Follow:           Unsupported\nRollback:              Unsupported\nContinuous Protection: Off\nAdd-on:                postgresql-round-12345\n'

@pytest.mark.parametrize('cmd', ['pg'])
def test_match(cmd):
    if False:
        for i in range(10):
            print('nop')
    assert match(Command('heroku {}'.format(cmd), suggest_output))

@pytest.mark.parametrize('script, output', [('heroku pg', not_match_output)])
def test_not_match(script, output):
    if False:
        while True:
            i = 10
    assert not match(Command(script, output))

@pytest.mark.parametrize('cmd, result', [('pg', ['heroku pg --app myapp', 'heroku pg --app myapp-dev'])])
def test_get_new_command(cmd, result):
    if False:
        while True:
            i = 10
    command = Command('heroku {}'.format(cmd), suggest_output)
    assert get_new_command(command) == result