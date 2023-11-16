import os
import pytest

@pytest.fixture(scope='module')
def install():
    if False:
        print('Hello World!')
    pytest.helpers.clean_env()
    pytest.helpers.old_install()
    pytest.helpers.run_command([pytest.INST_BIN, '/S', '/move-config'])
    yield
    pytest.helpers.clean_env()

def test_ssm_present_old_location(install):
    if False:
        print('Hello World!')
    assert os.path.exists(f'{pytest.OLD_DIR}\\bin\\ssm.exe')

def test_binaries_present_old_location(install):
    if False:
        print('Hello World!')
    assert os.path.exists(f'{pytest.OLD_DIR}\\bin\\python.exe')

def test_config_present_old_location(install):
    if False:
        for i in range(10):
            print('nop')
    assert os.path.exists(f'{pytest.DATA_DIR}\\conf\\minion')

def test_config_correct(install):
    if False:
        for i in range(10):
            print('nop')
    expected = pytest.OLD_CONTENT
    with open(f'{pytest.DATA_DIR}\\conf\\minion') as f:
        result = f.readlines()
    assert result == expected