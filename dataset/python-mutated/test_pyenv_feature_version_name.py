import pytest
from test_pyenv_helpers import Native

def test_version_name_help(pyenv):
    if False:
        while True:
            i = 10
    for args in [['--help', 'version-name'], ['help', 'version-name'], ['version-name', '--help']]:
        (stdout, stderr) = pyenv(*args)
        stdout = '\r\n'.join(stdout.splitlines()[:2]).strip()
        assert (stdout, stderr) == ('Usage: pyenv version-name', '')

def test_vname_help(pyenv):
    if False:
        i = 10
        return i + 15
    for args in [['--help', 'vname'], ['help', 'vname'], ['vname', '--help']]:
        (stdout, stderr) = pyenv(*args)
        stdout = '\r\n'.join(stdout.splitlines()[:2]).strip()
        assert (stdout, stderr) == ('Usage: pyenv vname', '')

@pytest.mark.parametrize('command', ['version-name', 'vname'])
def test_no_version(command, pyenv):
    if False:
        print('Hello World!')
    assert pyenv(command) == ('No global/local python version has been set yet. Please set the global/local version by typing:\r\npyenv global <python-version>\r\npyenv global 3.7.4\r\npyenv local <python-version>\r\npyenv local 3.7.4', '')

@pytest.mark.parametrize('settings', [lambda : {'global_ver': Native('3.7.4')}])
@pytest.mark.parametrize('command', ['version-name', 'vname'])
def test_global_version(command, pyenv):
    if False:
        return 10
    assert pyenv(command) == (Native('3.7.4'), '')

@pytest.mark.parametrize('settings', [lambda : {'global_ver': Native('3.7.4'), 'local_ver': Native('3.9.1')}])
@pytest.mark.parametrize('command', ['version-name', 'vname'])
def test_one_local_version(command, pyenv):
    if False:
        return 10
    assert pyenv(command) == (Native('3.9.1'), '')

@pytest.mark.parametrize('settings', [lambda : {'global_ver': Native('3.7.5'), 'local_ver': Native('3.8.6')}])
@pytest.mark.parametrize('command', ['version-name', 'vname'])
def test_shell_version(command, pyenv):
    if False:
        while True:
            i = 10
    env = {'PYENV_VERSION': Native('3.9.2')}
    assert pyenv(command, env=env) == (Native('3.9.2'), '')

@pytest.mark.parametrize('settings', [lambda : {'global_ver': Native('3.7.4'), 'local_ver': [Native('3.8.8'), Native('3.9.1')]}])
@pytest.mark.parametrize('command', ['version-name', 'vname'])
def test_many_local_versions(command, pyenv):
    if False:
        print('Hello World!')
    assert pyenv(command) == ('\r\n'.join([Native('3.8.8'), Native('3.9.1')]), '')