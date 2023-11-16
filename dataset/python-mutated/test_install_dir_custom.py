import os
import pytest

@pytest.fixture(scope='module')
def inst_dir():
    if False:
        return 10
    return 'C:\\custom_location'

@pytest.fixture(scope='module')
def install(inst_dir):
    if False:
        while True:
            i = 10
    pytest.helpers.clean_env(inst_dir)
    pytest.helpers.custom_config()
    pytest.helpers.run_command([pytest.INST_BIN, '/S', f'/install-dir={inst_dir}', '/custom-config=custom_conf'])
    yield
    pytest.helpers.clean_env(inst_dir)

def test_binaries_present(install, inst_dir):
    if False:
        print('Hello World!')
    assert os.path.exists(f'{inst_dir}\\ssm.exe')

def test_config_present(install):
    if False:
        i = 10
        return i + 15
    assert os.path.exists(f'{pytest.DATA_DIR}\\conf\\minion')

def test_config_correct(install):
    if False:
        i = 10
        return i + 15
    with open(f'{pytest.REPO_DIR}\\custom_conf') as f:
        expected = f.readlines()
    with open(f'{pytest.DATA_DIR}\\conf\\minion') as f:
        result = f.readlines()
    assert result == expected