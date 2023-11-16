import pytest
from thefuck.types import Command
from thefuck.rules.yarn_command_replaced import match, get_new_command
output = 'error `install` has been replaced with `add` to add new dependencies. Run "yarn add {}" instead.'.format

@pytest.mark.parametrize('command', [Command('yarn install redux', output('redux')), Command('yarn install moment', output('moment')), Command('yarn install lodash', output('lodash'))])
def test_match(command):
    if False:
        while True:
            i = 10
    assert match(command)

@pytest.mark.parametrize('command', [Command('yarn install', '')])
def test_not_match(command):
    if False:
        while True:
            i = 10
    assert not match(command)

@pytest.mark.parametrize('command, new_command', [(Command('yarn install redux', output('redux')), 'yarn add redux'), (Command('yarn install moment', output('moment')), 'yarn add moment'), (Command('yarn install lodash', output('lodash')), 'yarn add lodash')])
def test_get_new_command(command, new_command):
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(command) == new_command