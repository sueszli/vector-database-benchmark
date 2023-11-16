import os
import pytest

@pytest.fixture(scope='module')
def install():
    if False:
        return 10
    pytest.helpers.clean_env()
    pytest.helpers.old_install()
    pytest.helpers.run_command([pytest.INST_BIN, '/S', '/default-config', '/master=cli_master'])
    yield
    pytest.helpers.clean_env()

def test_ssm_present_old_location(install):
    if False:
        for i in range(10):
            print('nop')
    assert os.path.exists(f'{pytest.OLD_DIR}\\bin\\ssm.exe')

def test_binaries_present_old_location(install):
    if False:
        i = 10
        return i + 15
    assert os.path.exists(f'{pytest.OLD_DIR}\\bin\\python.exe')

def test_config_present_old_location(install):
    if False:
        print('Hello World!')
    assert os.path.exists(f'{pytest.OLD_DIR}\\conf\\minion')

def test_config_correct(install):
    if False:
        return 10
    expected = ['# Default config from test suite line 1/6\n', 'master: cli_master\n', '# Default config from test suite line 2/6\n', '#id:\n', '# Default config from test suite line 3/6\n', '# Default config from test suite line 4/6\n', '# Default config from test suite line 5/6\n', '# Default config from test suite line 6/6\n']
    with open(f'{pytest.OLD_DIR}\\conf\\minion') as f:
        result = f.readlines()
    assert result == expected