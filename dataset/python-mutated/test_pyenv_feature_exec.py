import os
import sys
import pytest
from pathlib import Path
from tempenv import TemporaryEnvironment
from test_pyenv_helpers import Native

@pytest.fixture()
def settings():
    if False:
        return 10
    return lambda : {'versions': [Native('3.7.7'), Native('3.8.9'), Native('3.10.0')], 'global_ver': Native('3.7.7'), 'local_ver': [Native('3.7.7'), Native('3.8.9')]}

@pytest.fixture()
def env(pyenv_path):
    if False:
        print('Hello World!')
    env = {'PATH': f"{os.path.dirname(sys.executable)};{str(Path(pyenv_path, 'bin'))};{str(Path(pyenv_path, 'shims'))};{os.environ['PATH']}"}
    environment = TemporaryEnvironment(env)
    with environment:
        yield env

@pytest.fixture(autouse=True)
def remove_python_exe(pyenv, pyenv_path, settings):
    if False:
        while True:
            i = 10
    '\n    We do not have any python version installed.\n    But we prepend the path with sys.executable dir.\n    And we remote fake python.exe (empty file generated) to ensure sys.executable is found and used.\n    This method allows us to execute python.exe.\n    But it cannot be used to play with many python versions.\n    '
    pyenv.rehash()
    for v in settings()['versions']:
        os.unlink(str(pyenv_path / 'versions' / v / 'python.exe'))

@pytest.mark.parametrize('command', [lambda path: [str(path / 'bin' / 'pyenv.bat'), 'exec', 'python'], lambda path: [str(path / 'shims' / 'python.bat')]], ids=['pyenv exec', 'python shim'])
@pytest.mark.parametrize('arg', ['Hello', 'Hello World', "Hello 'World'", 'Hello "World"', 'Hello %World%', 'Hello !World!', 'Hello #World#', "Hello World'", 'Hello World"', "Hello ''World'", 'Hello ""World"'], ids=['One Word', 'Two Words', 'Single Quote', 'Double Quote', 'Percentage', 'Exclamation Mark', 'Pound', 'One Single Quote', 'One Double Quote', 'Imbalance Single Quote', 'Imbalance Double Quote'])
def test_exec_arg(command, arg, env, pyenv_path, run):
    if False:
        while True:
            i = 10
    env['World'] = 'Earth'
    (stdout, stderr) = run(*command(pyenv_path), '-c', 'import sys; print(sys.argv[1])', arg, env=env)
    assert (stdout, stderr) == (arg.replace('%World%', 'Earth'), '')

@pytest.mark.parametrize('args', [['--help', 'exec'], ['help', 'exec'], ['exec', '--help']], ids=['--help exec', 'help exec', 'exec --help'])
def test_exec_help(args, env, pyenv):
    if False:
        while True:
            i = 10
    (stdout, stderr) = pyenv(*args, env=env)
    assert ('\r\n'.join(stdout.splitlines()[:1]), stderr) == (pyenv_exec_help(), '')

def test_path_not_updated(pyenv_path, local_path, env, run):
    if False:
        return 10
    python = str(pyenv_path / 'shims' / 'python.bat')
    tmp_bat = str(Path(local_path, 'tmp.bat'))
    with open(tmp_bat, 'w') as f:
        print(f'@echo %PATH%', file=f)
        print(f'@call "{python}" -V>nul', file=f)
        print(f'@echo %PATH%', file=f)
    (stdout, stderr) = run('call', tmp_bat, env=env)
    path = os.environ['PATH']
    assert (stdout, stderr) == (f'{path}\r\n{path}', '')

def test_many_paths(pyenv_path, env, pyenv):
    if False:
        print('Hello World!')
    (stdout, stderr) = pyenv.exec('python', '-c', "import os; print(os.environ['PATH'])", env=env)
    assert stderr == ''
    assert stdout.startswith(f"{pyenv_path}\\versions\\{Native('3.7.7')};{pyenv_path}\\versions\\{Native('3.7.7')}\\Scripts;{pyenv_path}\\versions\\{Native('3.8.9')};{pyenv_path}\\versions\\{Native('3.8.9')}\\Scripts;")
    assert pyenv.exec('version.bat') == ('3.7.7', '')

def test_bat_shim(pyenv):
    if False:
        i = 10
        return i + 15
    assert pyenv.exec('hello') == ('Hello world!', '')

def test_removes_shims_from_path(pyenv):
    if False:
        print('Hello World!')
    assert pyenv.exec('python310') == ('', "'python310' is not recognized as an internal or external command,\r\noperable program or batch file.")

def pyenv_exec_help():
    if False:
        return 10
    return 'Usage: pyenv exec <command> [arg1 arg2...]'