import os
import pytest

@pytest.fixture(scope='module')
def install():
    if False:
        return 10
    pytest.helpers.clean_env()
    pytest.helpers.old_install()
    pytest.helpers.run_command([pytest.INST_BIN, '/S'])
    yield
    pytest.helpers.clean_env()

def test_ssm_present_old_location(install):
    if False:
        print('Hello World!')
    assert os.path.exists(f'{pytest.OLD_DIR}\\bin\\ssm.exe')

def test_binaries_present_old_location(install):
    if False:
        while True:
            i = 10
    assert os.path.exists(f'{pytest.OLD_DIR}\\bin\\python.exe')

def test_config_present_old_location(install):
    if False:
        i = 10
        return i + 15
    assert os.path.exists(f'{pytest.OLD_DIR}\\conf\\minion')

def test_config_correct(install):
    if False:
        i = 10
        return i + 15
    expected = pytest.OLD_CONTENT
    with open(f'{pytest.OLD_DIR}\\conf\\minion') as f:
        result = f.readlines()
    assert result == expected