import pathlib
import pytest
from tests.support.runtests import RUNTIME_VARS
pytestmark = [pytest.mark.slow_test]

@pytest.fixture(scope='module')
def reset_pillar(salt_call_cli):
    if False:
        i = 10
        return i + 15
    try:
        yield
    finally:
        ret = salt_call_cli.run('saltutil.refresh_pillar', wait=True)
        assert ret.returncode == 0
        assert ret.data is True

@pytest.fixture(scope='module')
def pillar_test_true(base_env_pillar_tree_root_dir, salt_minion, salt_call_cli, reset_pillar):
    if False:
        return 10
    top_file = "\n    base:\n      '{}':\n        - basic\n    ".format(salt_minion.id)
    basic_pillar_file = '\n    test: true\n    '
    with pytest.helpers.temp_file('top.sls', top_file, base_env_pillar_tree_root_dir), pytest.helpers.temp_file('basic.sls', basic_pillar_file, base_env_pillar_tree_root_dir):
        ret = salt_call_cli.run('saltutil.refresh_pillar', wait=True)
        assert ret.returncode == 0
        assert ret.data is True
        yield

@pytest.fixture(scope='module')
def pillar_test_empty(base_env_pillar_tree_root_dir, salt_minion, salt_call_cli, reset_pillar):
    if False:
        for i in range(10):
            print('nop')
    top_file = "\n    base:\n      '{}':\n        - basic\n    ".format(salt_minion.id)
    basic_pillar_file = ''
    with pytest.helpers.temp_file('top.sls', top_file, base_env_pillar_tree_root_dir), pytest.helpers.temp_file('basic.sls', basic_pillar_file, base_env_pillar_tree_root_dir):
        ret = salt_call_cli.run('saltutil.refresh_pillar', wait=True)
        assert ret.returncode == 0
        assert ret.data is True
        yield

@pytest.fixture(scope='module')
def pillar_test_false(base_env_pillar_tree_root_dir, salt_minion, salt_call_cli, reset_pillar):
    if False:
        while True:
            i = 10
    top_file = "\n    base:\n      '{}':\n        - basic\n    ".format(salt_minion.id)
    basic_pillar_file = '\n    test: false\n    '
    with pytest.helpers.temp_file('top.sls', top_file, base_env_pillar_tree_root_dir), pytest.helpers.temp_file('basic.sls', basic_pillar_file, base_env_pillar_tree_root_dir):
        ret = salt_call_cli.run('saltutil.refresh_pillar', wait=True)
        assert ret.returncode == 0
        assert ret.data is True
        yield

@pytest.fixture
def testfile_path(tmp_path, base_env_state_tree_root_dir):
    if False:
        for i in range(10):
            print('nop')
    testfile = tmp_path / 'testfile'
    sls_contents = '\n    {}:\n      file:\n        - managed\n        - source: salt://testfile\n        - makedirs: true\n    '.format(testfile)
    with pytest.helpers.temp_file('sls-id-test.sls', sls_contents, base_env_state_tree_root_dir):
        yield testfile

@pytest.mark.usefixtures('pillar_test_true')
def test_state_sls_id_test(salt_call_cli, testfile_path):
    if False:
        i = 10
        return i + 15
    '\n    test state.sls_id when test is set\n    to true in pillar data\n    '
    expected_comment = 'The file {} is set to be changed\nNote: No changes made, actual changes may\nbe different due to other states.'.format(testfile_path)
    ret = salt_call_cli.run('state.sls', 'sls-id-test')
    assert ret.returncode == 0
    for val in ret.data.values():
        assert val['comment'] == expected_comment
        assert val['changes'] == {'newfile': str(testfile_path)}

@pytest.mark.usefixtures('pillar_test_true')
def test_state_sls_id_test_state_test_post_run(salt_call_cli, testfile_path):
    if False:
        return 10
    '\n    test state.sls_id when test is set to\n    true post the state already being run previously\n    '
    source = pathlib.Path(RUNTIME_VARS.BASE_FILES, 'testfile')
    testfile_path.write_text(source.read_text())
    testfile_path.chmod(420)
    ret = salt_call_cli.run('state.sls', 'sls-id-test')
    assert ret.returncode == 0
    for val in ret.data.values():
        assert val['comment'] == 'The file {} is in the correct state'.format(testfile_path)
        assert val['changes'] == {}

@pytest.mark.usefixtures('pillar_test_empty')
def test_state_sls_id_test_true(salt_call_cli, testfile_path):
    if False:
        return 10
    '\n    test state.sls_id when test=True is passed as arg\n    '
    expected_comment = 'The file {} is set to be changed\nNote: No changes made, actual changes may\nbe different due to other states.'.format(testfile_path)
    ret = salt_call_cli.run('state.sls', 'sls-id-test', test=True)
    assert ret.returncode == 0
    for val in ret.data.values():
        assert val['comment'] == expected_comment
        assert val['changes'] == {'newfile': str(testfile_path)}

@pytest.mark.usefixtures('pillar_test_empty')
def test_state_sls_id_test_true_post_run(salt_call_cli, testfile_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    test state.sls_id when test is set to true as an\n    arg post the state already being run previously\n    '
    ret = salt_call_cli.run('state.sls', 'sls-id-test')
    assert ret.returncode == 0
    assert testfile_path.exists()
    for val in ret.data.values():
        assert val['comment'] == 'File {} updated'.format(testfile_path)
        assert val['changes']['diff'] == 'New file'
    ret = salt_call_cli.run('state.sls', 'sls-id-test', test=True)
    assert ret.returncode == 0
    for val in ret.data.values():
        assert val['comment'] == 'The file {} is in the correct state'.format(testfile_path)
        assert val['changes'] == {}

@pytest.mark.usefixtures('pillar_test_true')
def test_state_sls_id_test_false_pillar_true(salt_call_cli, testfile_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    test state.sls_id when test is set to false as an\n    arg and minion_state_test is set to True. Should\n    return test=False.\n    '
    ret = salt_call_cli.run('state.sls', 'sls-id-test', test=False)
    assert ret.returncode == 0
    for val in ret.data.values():
        assert val['comment'] == 'File {} updated'.format(testfile_path)
        assert val['changes']['diff'] == 'New file'

@pytest.mark.usefixtures('pillar_test_false')
def test_state_test_pillar_false(salt_call_cli, testfile_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    test state.test forces test kwarg to True even when pillar is set to False\n    '
    expected_comment = 'The file {} is set to be changed\nNote: No changes made, actual changes may\nbe different due to other states.'.format(testfile_path)
    ret = salt_call_cli.run('state.test', 'sls-id-test')
    assert ret.returncode == 0
    for val in ret.data.values():
        assert val['comment'] == expected_comment
        assert val['changes'] == {'newfile': str(testfile_path)}

@pytest.mark.usefixtures('pillar_test_false')
def test_state_test_test_false_pillar_false(salt_call_cli, testfile_path):
    if False:
        i = 10
        return i + 15
    '\n    test state.test forces test kwarg to True even when pillar and kwarg are set\n    to False\n    '
    expected_comment = 'The file {} is set to be changed\nNote: No changes made, actual changes may\nbe different due to other states.'.format(testfile_path)
    ret = salt_call_cli.run('state.test', 'sls-id-test', test=False)
    assert ret.returncode == 0
    for val in ret.data.values():
        assert val['comment'] == expected_comment
        assert val['changes'] == {'newfile': str(testfile_path)}