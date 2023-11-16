import os
import pytest

@pytest.fixture(scope='module')
def install():
    if False:
        for i in range(10):
            print('nop')
    pytest.helpers.clean_env()
    pytest.helpers.existing_config()
    pytest.helpers.custom_config()
    pytest.helpers.run_command([pytest.INST_BIN, '/custom-config=custom_conf', '/master=cli_master'])
    yield
    pytest.helpers.clean_env()

def test_binaries_present(install):
    if False:
        return 10
    assert os.path.exists(f'{pytest.INST_DIR}\\ssm.exe')

def test_config_present(install):
    if False:
        return 10
    assert os.path.exists(f'{pytest.DATA_DIR}\\conf\\minion')

def test_config_correct(install):
    if False:
        print('Hello World!')
    expected = ['# Custom config from test suite line 1/6\n', 'master: cli_master\n', '# Custom config from test suite line 2/6\n', 'id: custom_minion\n', '# Custom config from test suite line 3/6\n', '# Custom config from test suite line 4/6\n', '# Custom config from test suite line 5/6\n', '# Custom config from test suite line 6/6\n']
    with open(f'{pytest.DATA_DIR}\\conf\\minion') as f:
        result = f.readlines()
    assert result == expected