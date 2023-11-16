import os
import pytest

@pytest.fixture(scope='module')
def inst_dir():
    if False:
        print('Hello World!')
    return 'C:\\custom_location'

@pytest.fixture(scope='module')
def install(inst_dir):
    if False:
        i = 10
        return i + 15
    pytest.helpers.clean_env(inst_dir)
    pytest.helpers.existing_config()
    pytest.helpers.run_command([pytest.INST_BIN, '/S', f'/install-dir={inst_dir}'])
    yield
    pytest.helpers.clean_env(inst_dir)

def test_binaries_present(install, inst_dir):
    if False:
        while True:
            i = 10
    assert os.path.exists(f'{inst_dir}\\ssm.exe')

def test_config_present(install):
    if False:
        print('Hello World!')
    assert os.path.exists(f'{pytest.DATA_DIR}\\conf\\minion')

def test_config_correct(install):
    if False:
        print('Hello World!')
    expected = pytest.EXISTING_CONTENT
    with open(f'{pytest.DATA_DIR}\\conf\\minion') as f:
        result = f.readlines()
    assert result == expected