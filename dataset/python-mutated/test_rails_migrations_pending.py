import pytest
from thefuck.rules.rails_migrations_pending import match, get_new_command
from thefuck.types import Command
output_env_development = '\nMigrations are pending. To resolve this issue, run:\n\n        rails db:migrate RAILS_ENV=development\n'
output_env_test = '\nMigrations are pending. To resolve this issue, run:\n\n        bin/rails db:migrate RAILS_ENV=test\n'

@pytest.mark.parametrize('command', [Command('', output_env_development), Command('', output_env_test)])
def test_match(command):
    if False:
        for i in range(10):
            print('nop')
    assert match(command)

@pytest.mark.parametrize('command', [Command('Environment data not found in the schema. To resolve this issue, run: \n\n', '')])
def test_not_match(command):
    if False:
        for i in range(10):
            print('nop')
    assert not match(command)

@pytest.mark.parametrize('command, new_command', [(Command('bin/rspec', output_env_development), 'rails db:migrate RAILS_ENV=development && bin/rspec'), (Command('bin/rspec', output_env_test), 'bin/rails db:migrate RAILS_ENV=test && bin/rspec')])
def test_get_new_command(command, new_command):
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(command) == new_command