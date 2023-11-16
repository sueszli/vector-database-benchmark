"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>

    Test cases for salt.modules.hadoop
"""
import pytest
import salt.modules.hadoop as hadoop
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {hadoop: {}}

def test_version():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Return version from hadoop version\n    '
    mock = MagicMock(return_value='A \nB \n')
    with patch.dict(hadoop.__salt__, {'cmd.run': mock}):
        assert hadoop.version() == 'B'

def test_dfs():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Execute a command on DFS\n    '
    with patch.object(hadoop, '_hadoop_cmd', return_value='A'):
        assert hadoop.dfs('command') == 'A'
    assert hadoop.dfs() == 'Error: command must be provided'

def test_dfs_present():
    if False:
        i = 10
        return i + 15
    '\n    Test for Check if a file or directory is present on the distributed FS.\n    '
    with patch.object(hadoop, '_hadoop_cmd', side_effect=['No such file or directory', 'A']):
        assert not hadoop.dfs_present('path')
        assert hadoop.dfs_present('path')

def test_dfs_absent():
    if False:
        return 10
    '\n    Test for Check if a file or directory is absent on the distributed FS.\n    '
    with patch.object(hadoop, '_hadoop_cmd', side_effect=['No such file or directory', 'A']):
        assert hadoop.dfs_absent('path')
        assert not hadoop.dfs_absent('path')

def test_namenode_format():
    if False:
        return 10
    '\n    Test for Format a name node\n    '
    with patch.object(hadoop, '_hadoop_cmd', return_value='A'):
        assert hadoop.namenode_format('force') == 'A'