import os
import pytest

@pytest.fixture(scope='module')
def install():
    if False:
        while True:
            i = 10
    pytest.helpers.clean_env()
    pytest.helpers.run_command([pytest.INST_BIN])
    yield
    pytest.helpers.clean_env()

def test_binaries_present(install):
    if False:
        print('Hello World!')
    assert os.path.exists(f'{pytest.INST_DIR}\\ssm.exe')

def test_config_present(install):
    if False:
        return 10
    assert os.path.exists(f'{pytest.DATA_DIR}\\conf\\minion')

def test_config_correct(install):
    if False:
        print('Hello World!')
    with open(f'{pytest.REPO_DIR}\\tests\\_files\\minion') as f:
        expected = f.readlines()
    with open(f'{pytest.DATA_DIR}\\conf\\minion') as f:
        result = f.readlines()
    assert result == expected