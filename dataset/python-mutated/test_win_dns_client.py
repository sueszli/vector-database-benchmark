"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import types
import pytest
import salt.modules.win_dns_client as win_dns_client
import salt.utils.stringutils
from tests.support.mock import MagicMock, Mock, patch
wmi = pytest.importorskip('wmi', reason='WMI only available on Windows')

class Mockwmi:
    """
    Mock wmi class
    """

    def __init__(self):
        if False:
            print('Hello World!')
        pass
    NetConnectionID = 'Local Area Connection'
    Index = 0
    DNSServerSearchOrder = ['10.1.1.10']
    Description = 'Local Area Connection'
    DHCPEnabled = True

class Mockwinapi:
    """
    Mock winapi class
    """

    def __init__(self):
        if False:
            return 10
        pass

    class winapi:
        """
        Mock winapi class
        """

        def __init__(self):
            if False:
                i = 10
                return i + 15
            pass

        @staticmethod
        def Com():
            if False:
                while True:
                    i = 10
            '\n            Mock Com method\n            '
            return True

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    mock_pythoncom = types.ModuleType(salt.utils.stringutils.to_str('pythoncom'))
    with patch.dict('sys.modules', {'pythoncom': mock_pythoncom}):
        yield {win_dns_client: {'wmi': wmi}}

def test_get_dns_servers():
    if False:
        print('Hello World!')
    '\n    Test if it return a list of the configured DNS servers\n    of the specified interface.\n    '
    WMI = Mock()
    with patch('salt.utils.winapi.Com', MagicMock()), patch.object(WMI, 'Win32_NetworkAdapter', return_value=[Mockwmi()]), patch.object(WMI, 'Win32_NetworkAdapterConfiguration', return_value=[Mockwmi()]), patch.object(wmi, 'WMI', Mock(return_value=WMI)):
        assert win_dns_client.get_dns_servers('Local Area Connection') == ['10.1.1.10']
        assert not win_dns_client.get_dns_servers('Ethernet')

def test_rm_dns():
    if False:
        i = 10
        return i + 15
    '\n    Test if it remove the DNS server from the network interface.\n    '
    with patch.dict(win_dns_client.__salt__, {'cmd.retcode': MagicMock(return_value=0)}):
        assert win_dns_client.rm_dns('10.1.1.10')

def test_add_dns():
    if False:
        while True:
            i = 10
    '\n    Test if it add the DNS server to the network interface.\n    '
    WMI = Mock()
    with patch('salt.utils.winapi.Com', MagicMock()), patch.object(WMI, 'Win32_NetworkAdapter', return_value=[Mockwmi()]), patch.object(WMI, 'Win32_NetworkAdapterConfiguration', return_value=[Mockwmi()]), patch.object(wmi, 'WMI', Mock(return_value=WMI)):
        assert not win_dns_client.add_dns('10.1.1.10', 'Ethernet')
        assert win_dns_client.add_dns('10.1.1.10', 'Local Area Connection')
    with patch.object(win_dns_client, 'get_dns_servers', MagicMock(return_value=['10.1.1.10'])), patch.dict(win_dns_client.__salt__, {'cmd.retcode': MagicMock(return_value=0)}), patch.object(wmi, 'WMI', Mock(return_value=WMI)):
        assert win_dns_client.add_dns('10.1.1.0', 'Local Area Connection')

def test_dns_dhcp():
    if False:
        return 10
    '\n    Test if it configure the interface to get its\n    DNS servers from the DHCP server\n    '
    with patch.dict(win_dns_client.__salt__, {'cmd.retcode': MagicMock(return_value=0)}):
        assert win_dns_client.dns_dhcp()

def test_get_dns_config():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it get the type of DNS configuration (dhcp / static)\n    '
    WMI = Mock()
    with patch('salt.utils.winapi.Com', MagicMock()), patch.object(WMI, 'Win32_NetworkAdapter', return_value=[Mockwmi()]), patch.object(WMI, 'Win32_NetworkAdapterConfiguration', return_value=[Mockwmi()]), patch.object(wmi, 'WMI', Mock(return_value=WMI)):
        assert win_dns_client.get_dns_config()

def test___virtual__non_windows():
    if False:
        for i in range(10):
            print('nop')
    mock_false = Mock(return_value=False)
    with patch('salt.utils.platform.is_windows', mock_false):
        result = win_dns_client.__virtual__()
        expected = (False, 'Module win_dns_client: module only works on Windows systems')
        assert result == expected

def test___virtual__missing_libs():
    if False:
        while True:
            i = 10
    with patch.object(win_dns_client, 'HAS_LIBS', False):
        result = win_dns_client.__virtual__()
        expected = (False, 'Module win_dns_client: missing required libraries')
        assert result == expected

def test___virtual__():
    if False:
        i = 10
        return i + 15
    result = win_dns_client.__virtual__()
    expected = 'win_dns_client'
    assert result == expected