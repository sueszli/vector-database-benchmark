import os
import pytest

@pytest.fixture(scope='module')
def inst_dir():
    if False:
        i = 10
        return i + 15
    return 'C:\\custom_location'

@pytest.fixture(scope='module')
def install(inst_dir):
    if False:
        while True:
            i = 10
    pytest.helpers.clean_env(inst_dir)
    pytest.helpers.run_command([pytest.INST_BIN, '/S', f'/install-dir={inst_dir}', '/minion-name=cli_minion'])
    yield {'inst_dir': inst_dir}
    pytest.helpers.clean_env(inst_dir)

def test_binaries_present(install, inst_dir):
    if False:
        for i in range(10):
            print('nop')
    assert os.path.exists(f'{inst_dir}\\ssm.exe')

def test_config_present(install):
    if False:
        print('Hello World!')
    assert os.path.exists(f'{pytest.DATA_DIR}\\conf\\minion')

def test_config_correct(install):
    if False:
        return 10
    expected = ['# Default config from test suite line 1/6\n', '#master: salt\n', '# Default config from test suite line 2/6\n', 'id: cli_minion\n', '# Default config from test suite line 3/6\n', '# Default config from test suite line 4/6\n', '# Default config from test suite line 5/6\n', '# Default config from test suite line 6/6\n']
    with open(f'{pytest.DATA_DIR}\\conf\\minion') as f:
        result = f.readlines()
    assert result == expected