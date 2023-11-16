import os
import pytest

@pytest.fixture(scope='module')
def install():
    if False:
        while True:
            i = 10
    pytest.helpers.clean_env()
    pytest.helpers.custom_config()
    pytest.helpers.run_command([pytest.INST_BIN, '/custom-config=custom_conf'])
    yield
    pytest.helpers.clean_env()

def test_binaries_present(install):
    if False:
        i = 10
        return i + 15
    assert os.path.exists(f'{pytest.INST_DIR}\\ssm.exe')

def test_config_present(install):
    if False:
        return 10
    assert os.path.exists(f'{pytest.DATA_DIR}\\conf\\minion')

def test_config_correct(install):
    if False:
        for i in range(10):
            print('nop')
    with open(f'{pytest.REPO_DIR}\\custom_conf') as f:
        expected = f.readlines()
    with open(f'{pytest.DATA_DIR}\\conf\\minion') as f:
        result = f.readlines()
    assert result == expected