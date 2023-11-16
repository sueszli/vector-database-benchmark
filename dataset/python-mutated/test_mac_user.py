"""
    :codeauthor: Nicole Thomas <nicole@saltstack.com>
"""
import pytest
import salt.modules.mac_user as mac_user
from salt.exceptions import CommandExecutionError, SaltInvocationError
from tests.support.mock import MagicMock, patch
pwd = pytest.importorskip('pwd')

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {mac_user: {}}

@pytest.fixture
def mock_pwall():
    if False:
        i = 10
        return i + 15
    return [pwd.struct_passwd(('_amavisd', '*', 83, 83, 'AMaViS Daemon', '/var/virusmails', '/usr/bin/false')), pwd.struct_passwd(('_appleevents', '*', 55, 55, 'AppleEvents Daemon', '/var/empty', '/usr/bin/false')), pwd.struct_passwd(('_appowner', '*', 87, 87, 'Application Owner', '/var/empty', '/usr/bin/false'))]

@pytest.fixture
def mock_info_ret():
    if False:
        i = 10
        return i + 15
    return {'shell': '/bin/bash', 'name': 'test', 'gid': 4376, 'groups': ['TEST_GROUP'], 'home': '/Users/foo', 'fullname': 'TEST USER', 'uid': 4376}

@pytest.mark.skip(reason='Waiting on some clarifications from bug report #10594')
def _test_flush_dscl_cache():
    if False:
        i = 10
        return i + 15
    pass

def test_dscl():
    if False:
        return 10
    '\n    Tests the creation of a dscl node\n    '
    mac_mock = MagicMock(return_value={'pid': 4948, 'retcode': 0, 'stderr': '', 'stdout': ''})
    with patch.dict(mac_user.__salt__, {'cmd.run_all': mac_mock}):
        with patch.dict(mac_user.__grains__, {'kernel': 'Darwin', 'osrelease': '10.9.1', 'osrelease_info': (10, 9, 1)}):
            assert mac_user._dscl(['username', 'UniqueID', 501]) == {'pid': 4948, 'retcode': 0, 'stderr': '', 'stdout': ''}

def test_first_avail_uid(mock_pwall):
    if False:
        i = 10
        return i + 15
    '\n    Tests the availability of the next uid\n    '
    with patch('pwd.getpwall', MagicMock(return_value=mock_pwall)):
        assert mac_user._first_avail_uid() == 501

def test_add_user_exists(mock_info_ret):
    if False:
        i = 10
        return i + 15
    '\n    Tests if the user exists or not\n    '
    with patch('salt.modules.mac_user.info', MagicMock(return_value=mock_info_ret)):
        pytest.raises(CommandExecutionError, mac_user.add, 'test')

def test_add_whitespace():
    if False:
        return 10
    '\n    Tests if there is whitespace in the user name\n    '
    with patch('salt.modules.mac_user.info', MagicMock(return_value={})):
        pytest.raises(SaltInvocationError, mac_user.add, 'foo bar')

def test_add_uid_int():
    if False:
        i = 10
        return i + 15
    '\n    Tests if the uid is an int\n    '
    with patch('salt.modules.mac_user.info', MagicMock(return_value={})):
        pytest.raises(SaltInvocationError, mac_user.add, 'foo', 'foo')

def test_add_gid_int():
    if False:
        print('Hello World!')
    '\n    Tests if the gid is an int\n    '
    with patch('salt.modules.mac_user.info', MagicMock(return_value={})):
        pytest.raises(SaltInvocationError, mac_user.add, 'foo', 20, 'foo')

def test_delete_whitespace():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests if there is whitespace in the user name\n    '
    pytest.raises(SaltInvocationError, mac_user.delete, 'foo bar')

def test_delete_user_exists():
    if False:
        print('Hello World!')
    '\n    Tests if the user exists or not\n    '
    with patch('salt.modules.mac_user.info', MagicMock(return_value={})):
        assert mac_user.delete('foo')

def test_getent(mock_pwall):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests the list of information for all users\n    '
    with patch('pwd.getpwall', MagicMock(return_value=mock_pwall)), patch('salt.modules.mac_user.list_groups', MagicMock(return_value=['TEST_GROUP'])):
        ret = [{'shell': '/usr/bin/false', 'name': '_amavisd', 'gid': 83, 'groups': ['TEST_GROUP'], 'home': '/var/virusmails', 'fullname': 'AMaViS Daemon', 'uid': 83}, {'shell': '/usr/bin/false', 'name': '_appleevents', 'gid': 55, 'groups': ['TEST_GROUP'], 'home': '/var/empty', 'fullname': 'AppleEvents Daemon', 'uid': 55}, {'shell': '/usr/bin/false', 'name': '_appowner', 'gid': 87, 'groups': ['TEST_GROUP'], 'home': '/var/empty', 'fullname': 'Application Owner', 'uid': 87}]
        assert mac_user.getent() == ret

def test_chuid_int():
    if False:
        return 10
    '\n    Tests if the uid is an int\n    '
    pytest.raises(SaltInvocationError, mac_user.chuid, 'foo', 'foo')

def test_chuid_user_exists():
    if False:
        return 10
    '\n    Tests if the user exists or not\n    '
    with patch('salt.modules.mac_user.info', MagicMock(return_value={})):
        pytest.raises(CommandExecutionError, mac_user.chuid, 'foo', 4376)

def test_chuid_same_uid(mock_info_ret):
    if False:
        while True:
            i = 10
    "\n    Tests if the user's uid is the same as as the argument\n    "
    with patch('salt.modules.mac_user.info', MagicMock(return_value=mock_info_ret)):
        assert mac_user.chuid('foo', 4376)

def test_chgid_int():
    if False:
        print('Hello World!')
    '\n    Tests if the gid is an int\n    '
    pytest.raises(SaltInvocationError, mac_user.chgid, 'foo', 'foo')

def test_chgid_user_exists():
    if False:
        return 10
    '\n    Tests if the user exists or not\n    '
    with patch('salt.modules.mac_user.info', MagicMock(return_value={})):
        pytest.raises(CommandExecutionError, mac_user.chgid, 'foo', 4376)

def test_chgid_same_gid(mock_info_ret):
    if False:
        for i in range(10):
            print('nop')
    "\n    Tests if the user's gid is the same as as the argument\n    "
    with patch('salt.modules.mac_user.info', MagicMock(return_value=mock_info_ret)):
        assert mac_user.chgid('foo', 4376)

def test_chshell_user_exists():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests if the user exists or not\n    '
    with patch('salt.modules.mac_user.info', MagicMock(return_value={})):
        pytest.raises(CommandExecutionError, mac_user.chshell, 'foo', '/bin/bash')

def test_chshell_same_shell(mock_info_ret):
    if False:
        i = 10
        return i + 15
    "\n    Tests if the user's shell is the same as the argument\n    "
    with patch('salt.modules.mac_user.info', MagicMock(return_value=mock_info_ret)):
        assert mac_user.chshell('foo', '/bin/bash')

def test_chhome_user_exists():
    if False:
        while True:
            i = 10
    '\n    Test if the user exists or not\n    '
    with patch('salt.modules.mac_user.info', MagicMock(return_value={})):
        pytest.raises(CommandExecutionError, mac_user.chhome, 'foo', '/Users/foo')

def test_chhome_same_home(mock_info_ret):
    if False:
        i = 10
        return i + 15
    "\n    Tests if the user's home is the same as the argument\n    "
    with patch('salt.modules.mac_user.info', MagicMock(return_value=mock_info_ret)):
        assert mac_user.chhome('foo', '/Users/foo')

def test_chfullname_user_exists():
    if False:
        i = 10
        return i + 15
    '\n    Tests if the user exists or not\n    '
    with patch('salt.modules.mac_user.info', MagicMock(return_value={})):
        pytest.raises(CommandExecutionError, mac_user.chfullname, 'test', 'TEST USER')

def test_chfullname_same_name(mock_info_ret):
    if False:
        for i in range(10):
            print('nop')
    "\n    Tests if the user's full name is the same as the argument\n    "
    with patch('salt.modules.mac_user.info', MagicMock(return_value=mock_info_ret)):
        assert mac_user.chfullname('test', 'TEST USER')

def test_chgroups_user_exists():
    if False:
        print('Hello World!')
    '\n    Tests if the user exists or not\n    '
    with patch('salt.modules.mac_user.info', MagicMock(return_value={})):
        pytest.raises(CommandExecutionError, mac_user.chgroups, 'foo', 'wheel,root')

def test_chgroups_bad_groups(mock_info_ret):
    if False:
        while True:
            i = 10
    '\n    Test if there is white space in groups argument\n    '
    with patch('salt.modules.mac_user.info', MagicMock(return_value=mock_info_ret)):
        pytest.raises(SaltInvocationError, mac_user.chgroups, 'test', 'bad group')

def test_chgroups_same_desired(mock_info_ret):
    if False:
        return 10
    "\n    Tests if the user's list of groups is the same as the arguments\n    "
    mock_primary = MagicMock(return_value='wheel')
    with patch.dict(mac_user.__salt__, {'file.gid_to_group': mock_primary}), patch('salt.modules.mac_user.info', MagicMock(return_value=mock_info_ret)), patch('salt.modules.mac_user.list_groups', MagicMock(return_value=('wheel', 'root'))):
        assert mac_user.chgroups('test', 'wheel,root')

def test_info():
    if False:
        while True:
            i = 10
    '\n    Tests the return of user information\n    '
    mock_pwnam = pwd.struct_passwd(('root', '*', 0, 0, 'TEST USER', '/var/test', '/bin/bash'))
    ret = {'shell': '/bin/bash', 'name': 'root', 'gid': 0, 'groups': ['_TEST_GROUP'], 'home': '/var/test', 'fullname': 'TEST USER', 'uid': 0}
    with patch('pwd.getpwall', MagicMock(return_value=[mock_pwnam])), patch('salt.modules.mac_user.list_groups', MagicMock(return_value=['_TEST_GROUP'])):
        assert mac_user.info('root') == ret

def test_format_info():
    if False:
        while True:
            i = 10
    '\n    Tests the formatting of returned user information\n    '
    data = pwd.struct_passwd(('_TEST_GROUP', '*', 83, 83, 'AMaViS Daemon', '/var/virusmails', '/usr/bin/false'))
    ret = {'shell': '/usr/bin/false', 'name': '_TEST_GROUP', 'gid': 83, 'groups': ['_TEST_GROUP'], 'home': '/var/virusmails', 'fullname': 'AMaViS Daemon', 'uid': 83}
    with patch('salt.modules.mac_user.list_groups', MagicMock(return_value=['_TEST_GROUP'])):
        assert mac_user._format_info(data) == ret

def test_list_users():
    if False:
        i = 10
        return i + 15
    '\n    Tests the list of all users\n    '
    expected = ['spongebob', 'patrick', 'squidward']
    mock_run = MagicMock(return_value={'pid': 4948, 'retcode': 0, 'stderr': '', 'stdout': '\n'.join(expected)})
    with patch.dict(mac_user.__grains__, {'osrelease_info': (10, 9, 1)}), patch.dict(mac_user.__salt__, {'cmd.run_all': mock_run}):
        assert mac_user.list_users() == expected

def test_kcpassword():
    if False:
        for i in range(10):
            print('nop')
    hashes = {'0': '4d 89 f9 91 1f 7a 46 5e f7 a8 11 ff', 'password': '0d e8 21 50 a5 d3 af 8e a3 de d9 14', 'shorterpwd': '0e e1 3d 51 a6 d9 af 9a d4 dd 1f 27', 'Squarepants': '2e f8 27 42 a0 d9 ad 8b cd cd 6c 7d', 'longerpasswd': '11 e6 3c 44 b7 ce ad 8b d0 ca 68 19 89 b1 65 ae 7e 89 12 b8 51 f8 f0 ff', 'ridiculouslyextendedpass': '0f e0 36 4a b1 c9 b1 85 d6 ca 73 04 ec 2a 57 b7 d2 b9 8f c7 c9 7e 0e fa 52 7b 71 e6 f8 b7 a6 ae 47 94 d7 86'}
    for (password, hash) in hashes.items():
        kcpass = mac_user._kcpassword(password)
        hash = bytes.fromhex(hash)
        length = len(password) + 1
        assert kcpass[:length] == hash[:length]
        assert len(kcpass) == len(hash)