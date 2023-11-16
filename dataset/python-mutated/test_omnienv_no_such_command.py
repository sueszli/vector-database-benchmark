import pytest
from thefuck.rules.omnienv_no_such_command import get_new_command, match
from thefuck.types import Command

@pytest.fixture
def output(pyenv_cmd):
    if False:
        for i in range(10):
            print('nop')
    return "pyenv: no such command `{}'".format(pyenv_cmd)

@pytest.fixture(autouse=True)
def Popen(mocker):
    if False:
        while True:
            i = 10
    mock = mocker.patch('thefuck.rules.omnienv_no_such_command.Popen')
    mock.return_value.stdout.readlines.return_value = b'--version\nactivate\ncommands\ncompletions\ndeactivate\nexec_\nglobal\nhelp\nhooks\ninit\ninstall\nlocal\nprefix_\nrealpath.dylib\nrehash\nroot\nshell\nshims\nuninstall\nversion_\nversion-file\nversion-file-read\nversion-file-write\nversion-name_\nversion-origin\nversions\nvirtualenv\nvirtualenv-delete_\nvirtualenv-init\nvirtualenv-prefix\nvirtualenvs_\nvirtualenvwrapper\nvirtualenvwrapper_lazy\nwhence\nwhich_\n'.split()
    return mock

@pytest.mark.parametrize('script, pyenv_cmd', [('pyenv globe', 'globe'), ('pyenv intall 3.8.0', 'intall'), ('pyenv list', 'list')])
def test_match(script, pyenv_cmd, output):
    if False:
        for i in range(10):
            print('nop')
    assert match(Command(script, output=output))

def test_match_goenv_output_quote():
    if False:
        return 10
    "test goenv's specific output with quotes (')"
    assert match(Command('goenv list', output="goenv: no such command 'list'"))

@pytest.mark.parametrize('script, output', [('pyenv global', 'system'), ('pyenv versions', '  3.7.0\n  3.7.1\n* 3.7.2\n'), ('pyenv install --list', '  3.7.0\n  3.7.1\n  3.7.2\n')])
def test_not_match(script, output):
    if False:
        return 10
    assert not match(Command(script, output=output))

@pytest.mark.parametrize('script, pyenv_cmd, result', [('pyenv globe', 'globe', 'pyenv global'), ('pyenv intall 3.8.0', 'intall', 'pyenv install 3.8.0'), ('pyenv list', 'list', 'pyenv install --list'), ('pyenv remove 3.8.0', 'remove', 'pyenv uninstall 3.8.0')])
def test_get_new_command(script, pyenv_cmd, output, result):
    if False:
        i = 10
        return i + 15
    assert result in get_new_command(Command(script, output))