import pytest
from test_pyenv_helpers import Native

def assert_paths_equal(actual, expected):
    if False:
        while True:
            i = 10
    assert actual.lower() == expected.lower()

def pyenv_which_usage():
    if False:
        for i in range(10):
            print('nop')
    return f"Usage: pyenv which <command>\r\n\r\nShows the full path of the executable\r\nselected. To obtain the full path, use `pyenv which pip'."

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.7.7')]}])
def test_which_no_arg(pyenv):
    if False:
        return 10
    assert pyenv.which() == (pyenv_which_usage(), '')
    assert pyenv.which('--help') == (pyenv_which_usage(), '')
    assert pyenv('--help', 'which') == (pyenv_which_usage(), '')
    assert pyenv('help', 'which') == (pyenv_which_usage(), '')

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.8.5')], 'global_ver': Native('3.8.5')}])
def test_which_exists_is_global(pyenv_path, pyenv):
    if False:
        i = 10
        return i + 15
    for name in ['python', 'python3', 'python38', 'pip3', 'pip3.8']:
        sub_dir = '' if 'python' in name else 'Scripts\\'
        (stdout, stderr) = pyenv.which(name)
        assert_paths_equal(stdout, f"{pyenv_path}\\versions\\{Native('3.8.5')}\\{sub_dir}{name}.exe")
        assert stderr == ''

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.8.5')], 'local_ver': Native('3.8.5')}])
def test_which_exists_is_local(pyenv_path, pyenv):
    if False:
        i = 10
        return i + 15
    for name in ['python', 'python3', 'python38', 'pip3', 'pip3.8']:
        sub_dir = '' if 'python' in name else 'Scripts\\'
        (stdout, stderr) = pyenv.which(name)
        assert_paths_equal(stdout, f"{pyenv_path}\\versions\\{Native('3.8.5')}\\{sub_dir}{name}.exe")
        assert stderr == ''

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.8.5')]}])
def test_which_exists_is_shell(pyenv_path, pyenv):
    if False:
        for i in range(10):
            print('nop')
    env = {'PYENV_VERSION': Native('3.8.5')}
    for name in ['python', 'python3', 'python38', 'pip3', 'pip3.8']:
        sub_dir = '' if 'python' in name else 'Scripts\\'
        (stdout, stderr) = pyenv.which(name, env=env)
        assert_paths_equal(stdout, f"{pyenv_path}\\versions\\{Native('3.8.5')}\\{sub_dir}{name}.exe")
        assert stderr == ''

@pytest.mark.parametrize('settings', [lambda : {'global_ver': Native('3.8.5')}])
def test_which_exists_is_global_not_installed(pyenv):
    if False:
        for i in range(10):
            print('nop')
    for name in ['python', 'python3', 'python38', 'pip3', 'pip3.8']:
        assert pyenv.which(name) == (f"pyenv: version '{Native('3.8.5')}' is not installed (set by {Native('3.8.5')})", '')

@pytest.mark.parametrize('settings', [lambda : {'local_ver': Native('3.8.5')}])
def test_which_exists_is_local_not_installed(pyenv):
    if False:
        while True:
            i = 10
    for name in ['python', 'python3', 'python38', 'pip3', 'pip3.8']:
        assert pyenv.which(name) == (f"pyenv: version '{Native('3.8.5')}' is not installed (set by {Native('3.8.5')})", '')

def test_which_exists_is_shell_not_installed(pyenv):
    if False:
        return 10
    env = {'PYENV_VERSION': Native('3.8.5')}
    for name in ['python', 'python3', 'python38', 'pip3', 'pip3.8']:
        assert pyenv.which(name, env=env) == (f"pyenv: version '{Native('3.8.5')}' is not installed (set by {Native('3.8.5')})", '')

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.8.2'), Native('3.8.6'), Native('3.9.1')], 'global_ver': Native('3.9.1')}])
def test_which_exists_is_global_other_version(pyenv):
    if False:
        print('Hello World!')
    for name in ['python38', 'pip3.8']:
        assert pyenv.which(name) == (f"pyenv: {name}: command not found\r\n\r\nThe '{name}' command exists in these Python versions:\r\n  {Native('3.8.2')}\r\n  {Native('3.8.6')}\r\n  ", '')

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.8.2'), Native('3.8.6'), Native('3.9.1')], 'local_ver': Native('3.9.1')}])
def test_which_exists_is_local_other_version(pyenv):
    if False:
        print('Hello World!')
    for name in ['python38', 'pip3.8']:
        assert pyenv.which(name) == (f"pyenv: {name}: command not found\r\n\r\nThe '{name}' command exists in these Python versions:\r\n  {Native('3.8.2')}\r\n  {Native('3.8.6')}\r\n  ", '')

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.8.2'), Native('3.8.6'), Native('3.9.1')]}])
def test_which_exists_is_shell_other_version(pyenv):
    if False:
        print('Hello World!')
    env = {'PYENV_VERSION': Native('3.9.1')}
    for name in ['python38', 'python3.8', 'pip3.8']:
        assert pyenv.which(name, env=env) == (f"pyenv: {name}: command not found\r\n\r\nThe '{name}' command exists in these Python versions:\r\n  {Native('3.8.2')}\r\n  {Native('3.8.6')}\r\n  ", '')

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.8.6')], 'global_ver': Native('3.8.6')}])
def test_which_command_not_found(pyenv):
    if False:
        i = 10
        return i + 15
    for name in ['unknown3.8']:
        assert pyenv.which(name) == (f'pyenv: {name}: command not found', '')

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.8.6')]}])
def test_which_no_version_defined(pyenv):
    if False:
        while True:
            i = 10
    for name in ['python']:
        assert pyenv.which(name) == ('No global/local python version has been set yet. Please set the global/local version by typing:\r\npyenv global <python-version>\r\npyenv global 3.7.4\r\npyenv local <python-version>\r\npyenv local 3.7.4', '')

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.7.7'), Native('3.8.2'), Native('3.9.1')], 'local_ver': [Native('3.7.7'), Native('3.8.2')]}])
def test_which_many_local_versions(pyenv_path, pyenv):
    if False:
        for i in range(10):
            print('nop')
    cases = [('python37', f"{Native('3.7.7')}\\python37.exe"), ('python38', f"{Native('3.8.2')}\\python38.exe"), ('pip3.7', f"{Native('3.7.7')}\\Scripts\\pip3.7.exe"), ('pip3.8', f"{Native('3.8.2')}\\Scripts\\pip3.8.exe")]
    for (name, path) in cases:
        (stdout, stderr) = pyenv.which(name)
        assert_paths_equal(stdout, f'{pyenv_path}\\versions\\{path}')
        assert stderr == ''