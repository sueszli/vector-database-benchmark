"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.rsyn
"""
import pytest
import salt.modules.rsync as rsync
from salt.exceptions import CommandExecutionError, SaltInvocationError
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {rsync: {}}

def test_rsync():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for rsync files from src to dst\n    '
    with patch.dict(rsync.__salt__, {'config.option': MagicMock(return_value=False)}):
        pytest.raises(SaltInvocationError, rsync.rsync, '', '')
    with patch.dict(rsync.__salt__, {'config.option': MagicMock(return_value='A'), 'cmd.run_all': MagicMock(side_effect=[OSError(1, 'f'), 'A'])}):
        with patch.object(rsync, '_check', return_value=['A']):
            pytest.raises(CommandExecutionError, rsync.rsync, 'a', 'b')
            assert rsync.rsync('src', 'dst') == 'A'

def test_version():
    if False:
        return 10
    '\n    Test for return rsync version\n    '
    mock = MagicMock(side_effect=[OSError(1, 'f'), 'A B C\n'])
    with patch.dict(rsync.__salt__, {'cmd.run_stdout': mock}):
        pytest.raises(CommandExecutionError, rsync.version)
        assert rsync.version() == 'C'

def test_rsync_excludes_list():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for rsync files from src to dst with a list of excludes\n    '
    mock = {'config.option': MagicMock(return_value=False), 'cmd.run_all': MagicMock()}
    with patch.dict(rsync.__salt__, mock):
        rsync.rsync('src', 'dst', exclude=['test/one', 'test/two'])
    mock['cmd.run_all'].assert_called_once_with(['rsync', '-avz', '--exclude', 'test/one', '--exclude', 'test/two', 'src', 'dst'], python_shell=False)

def test_rsync_excludes_str():
    if False:
        return 10
    '\n    Test for rsync files from src to dst with one exclude\n    '
    mock = {'config.option': MagicMock(return_value=False), 'cmd.run_all': MagicMock()}
    with patch.dict(rsync.__salt__, mock):
        rsync.rsync('src', 'dst', exclude='test/one')
    mock['cmd.run_all'].assert_called_once_with(['rsync', '-avz', '--exclude', 'test/one', 'src', 'dst'], python_shell=False)