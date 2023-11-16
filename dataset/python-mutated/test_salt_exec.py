from sys import platform
import pytest
pytestmark = [pytest.mark.skip_on_windows]

@pytest.fixture
def cat_file(tmp_path):
    if False:
        i = 10
        return i + 15
    fp = tmp_path / 'cat-file'
    fp.write_text(str(fp))
    return fp

def test_salt_cmd_run(salt_cli, salt_minion, cat_file):
    if False:
        i = 10
        return i + 15
    "\n    Test salt cmd.run 'ipconfig' or 'cat <file>'\n    "
    ret = None
    if platform.startswith('win'):
        ret = salt_cli.run('cmd.run', 'ipconfig', minion_tgt=salt_minion.id)
    else:
        ret = salt_cli.run('cmd.run', f'cat {str(cat_file)}', minion_tgt=salt_minion.id)
    assert ret
    assert ret.stdout

def test_salt_list_users(salt_cli, salt_minion):
    if False:
        i = 10
        return i + 15
    '\n    Test salt user.list_users\n    '
    ret = salt_cli.run('user.list_users', minion_tgt=salt_minion.id)
    if platform.startswith('win'):
        assert 'Administrator' in ret.stdout
    else:
        assert 'root' in ret.stdout