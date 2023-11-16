import pytest
from thefuck.rules.yarn_alias import match, get_new_command
from thefuck.types import Command
output_remove = 'error Did you mean `yarn remove`?'
output_etl = 'error Command "etil" not found. Did you mean "etl"?'
output_list = 'error Did you mean `yarn list`?'

@pytest.mark.parametrize('command', [Command('yarn rm', output_remove), Command('yarn etil', output_etl), Command('yarn ls', output_list)])
def test_match(command):
    if False:
        return 10
    assert match(command)

@pytest.mark.parametrize('command, new_command', [(Command('yarn rm', output_remove), 'yarn remove'), (Command('yarn etil', output_etl), 'yarn etl'), (Command('yarn ls', output_list), 'yarn list')])
def test_get_new_command(command, new_command):
    if False:
        print('Hello World!')
    assert get_new_command(command) == new_command