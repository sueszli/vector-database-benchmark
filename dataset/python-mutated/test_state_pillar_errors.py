import textwrap
import pytest
from saltfactories.utils.functional import StateResult
pytestmark = [pytest.mark.slow_test]

@pytest.fixture(scope='module')
def reset_pillar(salt_call_cli):
    if False:
        while True:
            i = 10
    try:
        yield
    finally:
        ret = salt_call_cli.run('saltutil.refresh_pillar', wait=True)
        assert ret.returncode == 0
        assert ret.data is True

@pytest.fixture
def testfile_path(tmp_path, base_env_state_tree_root_dir):
    if False:
        print('Hello World!')
    testfile = tmp_path / 'testfile'
    sls_contents = textwrap.dedent('\n        {}:\n          file:\n            - managed\n            - source: salt://testfile\n            - makedirs: true\n            - mode: 644\n        '.format(testfile))
    with pytest.helpers.temp_file('sls-id-test.sls', sls_contents, base_env_state_tree_root_dir):
        yield testfile

@pytest.mark.usefixtures('testfile_path', 'reset_pillar')
def test_state_apply_aborts_on_pillar_error(salt_cli, salt_minion, base_env_pillar_tree_root_dir):
    if False:
        while True:
            i = 10
    '\n    Test state.apply with error in pillar.\n    '
    pillar_top_file = textwrap.dedent("\n        base:\n          '{}':\n            - basic\n        ".format(salt_minion.id))
    basic_pillar_file = textwrap.dedent('\n        syntax_error\n        ')
    with pytest.helpers.temp_file('top.sls', pillar_top_file, base_env_pillar_tree_root_dir), pytest.helpers.temp_file('basic.sls', basic_pillar_file, base_env_pillar_tree_root_dir):
        expected_comment = ['Pillar failed to render with the following messages:', "SLS 'basic' does not render to a dictionary"]
        shell_result = salt_cli.run('state.apply', 'sls-id-test', minion_tgt=salt_minion.id)
        assert shell_result.returncode == 1
        assert shell_result.data == expected_comment

@pytest.mark.usefixtures('testfile_path', 'reset_pillar')
def test_state_apply_continues_after_pillar_error_is_fixed(salt_cli, salt_minion, base_env_pillar_tree_root_dir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test state.apply with error in pillar.\n    '
    pillar_top_file = textwrap.dedent("\n        base:\n          '{}':\n            - basic\n        ").format(salt_minion.id)
    basic_pillar_file_error = textwrap.dedent('\n        syntax_error\n        ')
    basic_pillar_file = textwrap.dedent('\n        syntax_error: Fixed!\n        ')
    with pytest.helpers.temp_file('top.sls', pillar_top_file, base_env_pillar_tree_root_dir), pytest.helpers.temp_file('basic.sls', basic_pillar_file_error, base_env_pillar_tree_root_dir):
        shell_result = salt_cli.run('saltutil.refresh_pillar', minion_tgt=salt_minion.id)
        assert shell_result.returncode == 0
    with pytest.helpers.temp_file('top.sls', pillar_top_file, base_env_pillar_tree_root_dir), pytest.helpers.temp_file('basic.sls', basic_pillar_file, base_env_pillar_tree_root_dir):
        shell_result = salt_cli.run('state.apply', 'sls-id-test', minion_tgt=salt_minion.id)
        assert shell_result.returncode == 0
        state_result = StateResult(shell_result.data)
        assert state_result.result is True
        assert state_result.changes == {'diff': 'New file', 'mode': '0644'}