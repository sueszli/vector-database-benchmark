import os
import pytest

@pytest.fixture(scope='module')
def install():
    if False:
        print('Hello World!')
    pytest.helpers.clean_env()
    pytest.helpers.existing_config()
    pytest.helpers.run_command([pytest.INST_BIN])
    yield
    pytest.helpers.clean_env()

def test_binaries_present(install):
    if False:
        while True:
            i = 10
    assert os.path.exists(f'{pytest.INST_DIR}\\ssm.exe')

def test_config_present(install):
    if False:
        while True:
            i = 10
    assert os.path.exists(f'{pytest.DATA_DIR}\\conf\\minion')

def test_config_correct(install):
    if False:
        i = 10
        return i + 15
    expected = pytest.EXISTING_CONTENT
    with open(f'{pytest.DATA_DIR}\\conf\\minion') as f:
        result = f.readlines()
    assert result == expected