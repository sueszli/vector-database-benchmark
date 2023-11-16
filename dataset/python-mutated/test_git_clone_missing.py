import pytest
from thefuck.rules.git_clone_missing import match, get_new_command
from thefuck.types import Command
valid_urls = ['https://github.com/nvbn/thefuck.git', 'https://github.com/nvbn/thefuck', 'http://github.com/nvbn/thefuck.git', 'git@github.com:nvbn/thefuck.git', 'git@github.com:nvbn/thefuck', 'ssh://git@github.com:nvbn/thefuck.git']
invalid_urls = ['', 'notacommand', 'ssh git@github.com:nvbn/thefrick.git', 'git clone foo', 'git clone https://github.com/nvbn/thefuck.git', 'github.com/nvbn/thefuck.git', 'github.com:nvbn/thefuck.git', 'git clone git clone ssh://git@github.com:nvbn/thefrick.git', 'https:/github.com/nvbn/thefuck.git']
outputs = ['No such file or directory', 'not found', 'is not recognised as']

@pytest.mark.parametrize('cmd', valid_urls)
@pytest.mark.parametrize('output', outputs)
def test_match(cmd, output):
    if False:
        while True:
            i = 10
    c = Command(cmd, output)
    assert match(c)

@pytest.mark.parametrize('cmd', invalid_urls)
@pytest.mark.parametrize('output', outputs + ['some other output'])
def test_not_match(cmd, output):
    if False:
        i = 10
        return i + 15
    c = Command(cmd, output)
    assert not match(c)

@pytest.mark.parametrize('script', valid_urls)
@pytest.mark.parametrize('output', outputs)
def test_get_new_command(script, output):
    if False:
        return 10
    command = Command(script, output)
    new_command = 'git clone ' + script
    assert get_new_command(command) == new_command