"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.ilo
"""
import tempfile
import pytest
import salt.modules.file
import salt.modules.ilo as ilo
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {ilo: {'__opts__': {'cachedir': tempfile.gettempdir()}, '__salt__': {'file.remove': salt.modules.file.remove}}}

def test_execute_cmd():
    if False:
        while True:
            i = 10
    '\n    Test if __execute_command opens the temporary file\n    properly when calling global_settings.\n    '
    mock_cmd_run = MagicMock(return_value={'retcode': 0, 'stdout': ''})
    with patch.dict(ilo.__salt__, {'cmd.run_all': mock_cmd_run}):
        ret = ilo.global_settings()
        assert ret

def test_global_settings():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it shows global_settings\n    '
    with patch('salt.modules.ilo.__execute_cmd', MagicMock(return_value={'Global Settings': {}})):
        assert ilo.global_settings() == {'Global Settings': {}}

def test_set_http_port():
    if False:
        print('Hello World!')
    '\n    Test if it configure the port HTTP should listen on\n    '
    with patch.object(ilo, 'global_settings', return_value={'Global Settings': {'HTTP_PORT': {'VALUE': 80}}}):
        assert ilo.set_http_port()
    with patch.object(ilo, 'global_settings', return_value={'Global Settings': {'HTTP_PORT': {'VALUE': 40}}}):
        with patch.object(ilo, '__execute_cmd', return_value={'Set HTTP Port': {}}):
            assert ilo.set_http_port() == {'Set HTTP Port': {}}

def test_set_https_port():
    if False:
        while True:
            i = 10
    '\n    Test if it configure the port HTTPS should listen on\n    '
    with patch.object(ilo, 'global_settings', return_value={'Global Settings': {'HTTP_PORT': {'VALUE': 443}}}):
        assert ilo.set_https_port()
    with patch.object(ilo, 'global_settings', return_value={'Global Settings': {'HTTP_PORT': {'VALUE': 80}}}):
        with patch.object(ilo, '__execute_cmd', return_value={'Set HTTPS Port': {}}):
            assert ilo.set_https_port() == {'Set HTTPS Port': {}}

def test_enable_ssh():
    if False:
        i = 10
        return i + 15
    '\n    Test if it enable the SSH daemon\n    '
    with patch.object(ilo, 'global_settings', return_value={'Global Settings': {'SSH_STATUS': {'VALUE': 'Y'}}}):
        assert ilo.enable_ssh()
    with patch.object(ilo, 'global_settings', return_value={'Global Settings': {'SSH_STATUS': {'VALUE': 'N'}}}):
        with patch.object(ilo, '__execute_cmd', return_value={'Enable SSH': {}}):
            assert ilo.enable_ssh() == {'Enable SSH': {}}

def test_disable_ssh():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it disable the SSH daemon\n    '
    with patch.object(ilo, 'global_settings', return_value={'Global Settings': {'SSH_STATUS': {'VALUE': 'N'}}}):
        assert ilo.disable_ssh()
    with patch.object(ilo, 'global_settings', return_value={'Global Settings': {'SSH_STATUS': {'VALUE': 'Y'}}}):
        with patch.object(ilo, '__execute_cmd', return_value={'Disable SSH': {}}):
            assert ilo.disable_ssh() == {'Disable SSH': {}}

def test_set_ssh_port():
    if False:
        i = 10
        return i + 15
    '\n    Test if it enable SSH on a user defined port\n    '
    with patch.object(ilo, 'global_settings', return_value={'Global Settings': {'SSH_PORT': {'VALUE': 22}}}):
        assert ilo.set_ssh_port()
    with patch.object(ilo, 'global_settings', return_value={'Global Settings': {'SSH_PORT': {'VALUE': 20}}}):
        with patch.object(ilo, '__execute_cmd', return_value={'Configure SSH Port': {}}):
            assert ilo.set_ssh_port() == {'Configure SSH Port': {}}

def test_set_ssh_key():
    if False:
        i = 10
        return i + 15
    '\n    Test if it configure SSH public keys for specific users\n    '
    with patch('salt.modules.ilo.__execute_cmd', MagicMock(return_value={'Import SSH Publickey': {}})):
        assert ilo.set_ssh_key('ssh-rsa AAAAB3Nza Salt') == {'Import SSH Publickey': {}}

def test_delete_ssh_key():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it delete a users SSH key from the ILO\n    '
    with patch('salt.modules.ilo.__execute_cmd', MagicMock(return_value={'Delete user SSH key': {}})):
        assert ilo.delete_ssh_key('Salt') == {'Delete user SSH key': {}}

def test_list_users():
    if False:
        while True:
            i = 10
    '\n    Test if it list all users\n    '
    with patch('salt.modules.ilo.__execute_cmd', MagicMock(return_value={'All users': {}})):
        assert ilo.list_users() == {'All users': {}}

def test_list_users_info():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it List all users in detail\n    '
    with patch('salt.modules.ilo.__execute_cmd', MagicMock(return_value={'All users info': {}})):
        assert ilo.list_users_info() == {'All users info': {}}

def test_create_user():
    if False:
        while True:
            i = 10
    '\n    Test if it create user\n    '
    with patch('salt.modules.ilo.__execute_cmd', MagicMock(return_value={'Create user': {}})):
        assert ilo.create_user('Salt', 'secretagent', 'VIRTUAL_MEDIA_PRIV') == {'Create user': {}}

def test_delete_user():
    if False:
        i = 10
        return i + 15
    '\n    Test if it delete a user\n    '
    with patch('salt.modules.ilo.__execute_cmd', MagicMock(return_value={'Delete user': {}})):
        assert ilo.delete_user('Salt') == {'Delete user': {}}

def test_get_user():
    if False:
        print('Hello World!')
    '\n    Test if it returns local user information, excluding the password\n    '
    with patch('salt.modules.ilo.__execute_cmd', MagicMock(return_value={'User Info': {}})):
        assert ilo.get_user('Salt') == {'User Info': {}}

def test_change_username():
    if False:
        i = 10
        return i + 15
    '\n    Test if it change a username\n    '
    with patch('salt.modules.ilo.__execute_cmd', MagicMock(return_value={'Change username': {}})):
        assert ilo.change_username('Salt', 'SALT') == {'Change username': {}}

def test_change_password():
    if False:
        i = 10
        return i + 15
    '\n    Test if it reset a users password\n    '
    with patch('salt.modules.ilo.__execute_cmd', MagicMock(return_value={'Change password': {}})):
        assert ilo.change_password('Salt', 'saltpasswd') == {'Change password': {}}

def test_network():
    if False:
        while True:
            i = 10
    '\n    Test if it grab the current network settings\n    '
    with patch('salt.modules.ilo.__execute_cmd', MagicMock(return_value={'Network Settings': {}})):
        assert ilo.network() == {'Network Settings': {}}

def test_configure_network():
    if False:
        i = 10
        return i + 15
    '\n    Test if it configure Network Interface\n    '
    with patch('salt.modules.ilo.__execute_cmd', MagicMock(return_value={'Configure_Network': {}})):
        ret = {'Network Settings': {'IP_ADDRESS': {'VALUE': '10.0.0.10'}, 'SUBNET_MASK': {'VALUE': '255.255.255.0'}, 'GATEWAY_IP_ADDRESS': {'VALUE': '10.0.0.1'}}}
        with patch.object(ilo, 'network', return_value=ret):
            assert ilo.configure_network('10.0.0.10', '255.255.255.0', '10.0.0.1')
        with patch.object(ilo, 'network', return_value=ret):
            with patch.object(ilo, '__execute_cmd', return_value={'Network Settings': {}}):
                assert ilo.configure_network('10.0.0.100', '255.255.255.10', '10.0.0.10') == {'Network Settings': {}}

def test_enable_dhcp():
    if False:
        while True:
            i = 10
    '\n    Test if it enable DHCP\n    '
    with patch.object(ilo, 'network', return_value={'Network Settings': {'DHCP_ENABLE': {'VALUE': 'Y'}}}):
        assert ilo.enable_dhcp()
    with patch.object(ilo, 'network', return_value={'Network Settings': {'DHCP_ENABLE': {'VALUE': 'N'}}}):
        with patch.object(ilo, '__execute_cmd', return_value={'Enable DHCP': {}}):
            assert ilo.enable_dhcp() == {'Enable DHCP': {}}

def test_disable_dhcp():
    if False:
        print('Hello World!')
    '\n    Test if it disable DHCP\n    '
    with patch.object(ilo, 'network', return_value={'Network Settings': {'DHCP_ENABLE': {'VALUE': 'N'}}}):
        assert ilo.disable_dhcp()
    with patch.object(ilo, 'network', return_value={'Network Settings': {'DHCP_ENABLE': {'VALUE': 'Y'}}}):
        with patch.object(ilo, '__execute_cmd', return_value={'Disable DHCP': {}}):
            assert ilo.disable_dhcp() == {'Disable DHCP': {}}

def test_configure_snmp():
    if False:
        i = 10
        return i + 15
    '\n    Test if it configure SNMP\n    '
    with patch('salt.modules.ilo.__execute_cmd', MagicMock(return_value={'Configure SNMP': {}})):
        assert ilo.configure_snmp('Salt') == {'Configure SNMP': {}}