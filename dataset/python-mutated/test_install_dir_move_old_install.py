import os
import pytest

@pytest.fixture(scope='module')
def inst_dir():
    if False:
        for i in range(10):
            print('nop')
    return 'C:\\custom_location'

@pytest.fixture(scope='module')
def install(inst_dir):
    if False:
        for i in range(10):
            print('nop')
    pytest.helpers.clean_env()
    pytest.helpers.old_install()
    pytest.helpers.run_command([pytest.INST_BIN, '/S', f'/install-dir={inst_dir}', '/move-config'])
    yield
    pytest.helpers.clean_env()

def test_binaries_present_old_location(install):
    if False:
        while True:
            i = 10
    assert os.path.exists(f'{pytest.OLD_DIR}\\bin\\ssm.exe')
    assert os.path.exists(f'{pytest.OLD_DIR}\\bin\\python.exe')

def test_config_present(install):
    if False:
        while True:
            i = 10
    assert os.path.exists(f'{pytest.DATA_DIR}\\conf\\minion')

def test_config_correct(install):
    if False:
        return 10
    expected = pytest.OLD_CONTENT
    with open(f'{pytest.DATA_DIR}\\conf\\minion') as f:
        result = f.readlines()
    assert result == expected