import pytest
from thefuck.rules.cargo_no_command import match, get_new_command
from thefuck.types import Command
no_such_subcommand_old = 'No such subcommand\n\n        Did you mean `build`?\n'
no_such_subcommand = 'error: no such subcommand\n\n\tDid you mean `build`?\n'

@pytest.mark.parametrize('command', [Command('cargo buid', no_such_subcommand_old), Command('cargo buils', no_such_subcommand)])
def test_match(command):
    if False:
        print('Hello World!')
    assert match(command)

@pytest.mark.parametrize('command, new_command', [(Command('cargo buid', no_such_subcommand_old), 'cargo build'), (Command('cargo buils', no_such_subcommand), 'cargo build')])
def test_get_new_command(command, new_command):
    if False:
        while True:
            i = 10
    assert get_new_command(command) == new_command