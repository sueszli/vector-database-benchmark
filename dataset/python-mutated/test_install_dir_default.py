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
    pytest.helpers.clean_env(inst_dir)
    pytest.helpers.run_command([pytest.INST_BIN, '/S', f'/install-dir={inst_dir}'])
    yield
    pytest.helpers.clean_env(inst_dir)

def test_binaries_present(install, inst_dir):
    if False:
        print('Hello World!')
    assert os.path.exists(f'{inst_dir}\\ssm.exe')

def test_config_present(install):
    if False:
        return 10
    assert os.path.exists(f'{pytest.DATA_DIR}\\conf\\minion')

def test_config_correct(install):
    if False:
        return 10
    with open(f'{pytest.REPO_DIR}\\_files\\minion') as f:
        expected = f.readlines()
    with open(f'{pytest.DATA_DIR}\\conf\\minion') as f:
        result = f.readlines()
    assert result == expected