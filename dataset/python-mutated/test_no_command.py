import pytest
from thefuck.rules.no_command import match, get_new_command
from thefuck.types import Command

@pytest.fixture(autouse=True)
def get_all_executables(mocker):
    if False:
        return 10
    mocker.patch('thefuck.rules.no_command.get_all_executables', return_value=['vim', 'fsck', 'git', 'go', 'python'])

@pytest.fixture(autouse=True)
def history_without_current(mocker):
    if False:
        i = 10
        return i + 15
    return mocker.patch('thefuck.rules.no_command.get_valid_history_without_current', return_value=['git commit'])

@pytest.mark.usefixtures('no_memoize')
@pytest.mark.parametrize('script, output', [('vom file.py', 'vom: not found'), ('fucck', 'fucck: not found'), ('puthon', "'puthon' is not recognized as an internal or external command"), ('got commit', 'got: command not found'), ('gti commit -m "new commit"', 'gti: command not found')])
def test_match(mocker, script, output):
    if False:
        i = 10
        return i + 15
    mocker.patch('thefuck.rules.no_command.which', return_value=None)
    assert match(Command(script, output))

@pytest.mark.usefixtures('no_memoize')
@pytest.mark.parametrize('script, output, which', [('qweqwe', 'qweqwe: not found', None), ('vom file.py', 'some text', None), ('vim file.py', 'vim: not found', 'vim')])
def test_not_match(mocker, script, output, which):
    if False:
        while True:
            i = 10
    mocker.patch('thefuck.rules.no_command.which', return_value=which)
    assert not match(Command(script, output))

@pytest.mark.usefixtures('no_memoize')
@pytest.mark.parametrize('script, result', [('vom file.py', ['vim file.py']), ('fucck', ['fsck']), ('got commit', ['git commit', 'go commit']), ('gti commit -m "new commit"', ['git commit -m "new commit"'])])
def test_get_new_command(script, result):
    if False:
        while True:
            i = 10
    assert get_new_command(Command(script, '')) == result