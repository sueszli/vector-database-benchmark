"""
    :codeauthor: Nicole Thomas <nicole@saltstack.com>
"""
import pytest
import salt.modules.mac_group as mac_group
from salt.exceptions import CommandExecutionError, SaltInvocationError
from tests.support.mock import MagicMock, patch
grp = pytest.importorskip('grp', reason="Missing required library 'grp'")

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {mac_group: {}}

def test_add_group_exists():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests if the group already exists or not\n    '
    mock_group = {'passwd': '*', 'gid': 0, 'name': 'test', 'members': ['root']}
    with patch('salt.modules.mac_group.info', MagicMock(return_value=mock_group)):
        pytest.raises(CommandExecutionError, mac_group.add, 'test')

def test_add_whitespace():
    if False:
        while True:
            i = 10
    '\n    Tests if there is whitespace in the group name\n    '
    with patch('salt.modules.mac_group.info', MagicMock(return_value={})):
        pytest.raises(SaltInvocationError, mac_group.add, 'white space')

def test_add_underscore():
    if False:
        return 10
    '\n    Tests if the group name starts with an underscore or not\n    '
    with patch('salt.modules.mac_group.info', MagicMock(return_value={})):
        pytest.raises(SaltInvocationError, mac_group.add, '_Test')

def test_add_gid_int():
    if False:
        return 10
    '\n    Tests if the gid is an int or not\n    '
    with patch('salt.modules.mac_group.info', MagicMock(return_value={})):
        pytest.raises(SaltInvocationError, mac_group.add, 'foo', 'foo')

def test_add_gid_exists():
    if False:
        return 10
    '\n    Tests if the gid is already in use or not\n    '
    with patch('salt.modules.mac_group.info', MagicMock(return_value={})), patch('salt.modules.mac_group._list_gids', MagicMock(return_value=['3456'])):
        pytest.raises(CommandExecutionError, mac_group.add, 'foo', 3456)

def test_add():
    if False:
        while True:
            i = 10
    '\n    Tests if specified group was added\n    '
    mock_ret = MagicMock(return_value=0)
    with patch.dict(mac_group.__salt__, {'cmd.retcode': mock_ret}), patch('salt.modules.mac_group.info', MagicMock(return_value={})), patch('salt.modules.mac_group._list_gids', MagicMock(return_value=[])):
        assert mac_group.add('test', 500)

def test_delete_whitespace():
    if False:
        print('Hello World!')
    '\n    Tests if there is whitespace in the group name\n    '
    pytest.raises(SaltInvocationError, mac_group.delete, 'white space')

def test_delete_underscore():
    if False:
        while True:
            i = 10
    '\n    Tests if the group name starts with an underscore or not\n    '
    pytest.raises(SaltInvocationError, mac_group.delete, '_Test')

def test_delete_group_exists():
    if False:
        print('Hello World!')
    '\n    Tests if the group to be deleted exists or not\n    '
    with patch('salt.modules.mac_group.info', MagicMock(return_value={})):
        assert mac_group.delete('test')

def test_delete():
    if False:
        while True:
            i = 10
    '\n    Tests if the specified group was deleted\n    '
    mock_ret = MagicMock(return_value=0)
    mock_group = {'passwd': '*', 'gid': 0, 'name': 'test', 'members': ['root']}
    with patch.dict(mac_group.__salt__, {'cmd.retcode': mock_ret}), patch('salt.modules.mac_group.info', MagicMock(return_value=mock_group)):
        assert mac_group.delete('test')

def test_info_whitespace():
    if False:
        return 10
    '\n    Tests if there is whitespace in the group name\n    '
    pytest.raises(SaltInvocationError, mac_group.info, 'white space')

def test_info():
    if False:
        return 10
    '\n    Tests the return of group information\n    '
    mock_getgrall = [grp.struct_group(('foo', '*', 20, ['test']))]
    with patch('grp.getgrall', MagicMock(return_value=mock_getgrall)):
        ret = {'passwd': '*', 'gid': 20, 'name': 'foo', 'members': ['test']}
        assert mac_group.info('foo') == ret

def test_format_info():
    if False:
        i = 10
        return i + 15
    '\n    Tests the formatting of returned group information\n    '
    data = grp.struct_group(('wheel', '*', 0, ['root']))
    ret = {'passwd': '*', 'gid': 0, 'name': 'wheel', 'members': ['root']}
    assert mac_group._format_info(data) == ret

def test_getent():
    if False:
        i = 10
        return i + 15
    '\n    Tests the return of information on all groups\n    '
    mock_getgrall = [grp.struct_group(('foo', '*', 20, ['test']))]
    with patch('grp.getgrall', MagicMock(return_value=mock_getgrall)):
        ret = [{'passwd': '*', 'gid': 20, 'name': 'foo', 'members': ['test']}]
        assert mac_group.getent() == ret

def test_chgid_gid_int():
    if False:
        i = 10
        return i + 15
    '\n    Tests if gid is an integer or not\n    '
    pytest.raises(SaltInvocationError, mac_group.chgid, 'foo', 'foo')

def test_chgid_group_exists():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests if the group id exists or not\n    '
    mock_pre_gid = MagicMock(return_value='')
    with patch.dict(mac_group.__salt__, {'file.group_to_gid': mock_pre_gid}), patch('salt.modules.mac_group.info', MagicMock(return_value={})):
        pytest.raises(CommandExecutionError, mac_group.chgid, 'foo', 4376)

def test_chgid_gid_same():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests if the group id is the same as argument\n    '
    mock_group = {'passwd': '*', 'gid': 0, 'name': 'test', 'members': ['root']}
    mock_pre_gid = MagicMock(return_value=0)
    with patch.dict(mac_group.__salt__, {'file.group_to_gid': mock_pre_gid}), patch('salt.modules.mac_group.info', MagicMock(return_value=mock_group)):
        assert mac_group.chgid('test', 0)

def test_chgid():
    if False:
        print('Hello World!')
    '\n    Tests the gid for a named group was changed\n    '
    mock_group = {'passwd': '*', 'gid': 0, 'name': 'test', 'members': ['root']}
    mock_pre_gid = MagicMock(return_value=0)
    mock_ret = MagicMock(return_value=0)
    with patch.dict(mac_group.__salt__, {'file.group_to_gid': mock_pre_gid}), patch.dict(mac_group.__salt__, {'cmd.retcode': mock_ret}), patch('salt.modules.mac_group.info', MagicMock(return_value=mock_group)):
        assert mac_group.chgid('test', 500)