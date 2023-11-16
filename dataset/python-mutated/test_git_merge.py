import pytest
from thefuck.rules.git_merge import match, get_new_command
from thefuck.types import Command
output = 'merge: local - not something we can merge\n\nDid you mean this?\n\tremote/local'

def test_match():
    if False:
        i = 10
        return i + 15
    assert match(Command('git merge test', output))
    assert not match(Command('git merge master', ''))
    assert not match(Command('ls', output))

@pytest.mark.parametrize('command, new_command', [(Command('git merge local', output), 'git merge remote/local'), (Command('git merge -m "test" local', output), 'git merge -m "test" remote/local'), (Command('git merge -m "test local" local', output), 'git merge -m "test local" remote/local')])
def test_get_new_command(command, new_command):
    if False:
        print('Hello World!')
    assert get_new_command(command) == new_command