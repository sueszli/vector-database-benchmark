import pytest
from thefuck.rules.az_cli import match, get_new_command
from thefuck.types import Command
no_suggestions = 'az provider: error: the following arguments are required: _subcommand\nusage: az provider [-h] {list,show,register,unregister,operation} ...\n'
misspelled_command = "az: 'providers' is not in the 'az' command group. See 'az --help'.\n\nThe most similar choice to 'providers' is:\n    provider\n"
misspelled_subcommand = "az provider: 'lis' is not in the 'az provider' command group. See 'az provider --help'.\n\nThe most similar choice to 'lis' is:\n    list\n"

@pytest.mark.parametrize('command', [Command('az providers', misspelled_command), Command('az provider lis', misspelled_subcommand)])
def test_match(command):
    if False:
        i = 10
        return i + 15
    assert match(command)

def test_not_match():
    if False:
        print('Hello World!')
    assert not match(Command('az provider', no_suggestions))

@pytest.mark.parametrize('command, result', [(Command('az providers list', misspelled_command), ['az provider list']), (Command('az provider lis', misspelled_subcommand), ['az provider list'])])
def test_get_new_command(command, result):
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(command) == result