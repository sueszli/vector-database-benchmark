"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import socket
import pytest
import salt.modules.win_network as win_network
import salt.utils.network
from tests.support.mock import MagicMock, Mock, patch
try:
    import wmi
    HAS_WMI = True
except ImportError:
    HAS_WMI = False

class Mockwmi:
    """
    Mock wmi class
    """
    NetConnectionID = 'Ethernet'

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

class Mockwinapi:
    """
    Mock winapi class
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    class winapi:
        """
        Mock winapi class
        """

        def __init__(self):
            if False:
                return 10
            pass

        class Com:
            """
            Mock Com method
            """

            def __enter__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self

            def __exit__(self, *exc_info):
                if False:
                    print('Hello World!')
                return False

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {win_network: {}}

def test_ping():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it performs a ping to a host.\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(win_network.__salt__, {'cmd.run': mock}):
        assert win_network.ping('127.0.0.1')

def test_netstat():
    if False:
        i = 10
        return i + 15
    '\n    Test if it return information on open ports and states\n    '
    ret = '  Proto  Local Address    Foreign Address    State    PID\n  TCP    127.0.0.1:1434    0.0.0.0:0    LISTENING    1728\n  UDP    127.0.0.1:1900    *:*        4240'
    mock = MagicMock(return_value=ret)
    with patch.dict(win_network.__salt__, {'cmd.run': mock}):
        assert win_network.netstat() == [{'local-address': '127.0.0.1:1434', 'program': '1728', 'proto': 'TCP', 'remote-address': '0.0.0.0:0', 'state': 'LISTENING'}, {'local-address': '127.0.0.1:1900', 'program': '4240', 'proto': 'UDP', 'remote-address': '*:*', 'state': None}]

def test_traceroute():
    if False:
        print('Hello World!')
    '\n    Test if it performs a traceroute to a 3rd party host\n    '
    ret = '  1     1 ms    <1 ms    <1 ms  172.27.104.1\n  2     1 ms    <1 ms     1 ms  121.242.35.1.s[121.242.35.1]\n  3     3 ms     2 ms     2 ms  121.242.4.53.s[121.242.4.53]\n'
    mock = MagicMock(return_value=ret)
    with patch.dict(win_network.__salt__, {'cmd.run': mock}):
        assert win_network.traceroute('google.com') == [{'count': '1', 'hostname': None, 'ip': '172.27.104.1', 'ms1': '1', 'ms2': '<1', 'ms3': '<1'}, {'count': '2', 'hostname': None, 'ip': '121.242.35.1.s[121.242.35.1]', 'ms1': '1', 'ms2': '<1', 'ms3': '1'}, {'count': '3', 'hostname': None, 'ip': '121.242.4.53.s[121.242.4.53]', 'ms1': '3', 'ms2': '2', 'ms3': '2'}]

def test_nslookup():
    if False:
        print('Hello World!')
    '\n    Test if it query DNS for information about a domain or ip address\n    '
    ret = 'Server:  ct-dc-3-2.cybage.com\nAddress:  172.27.172.12\nNon-authoritative answer:\nName:    google.com\nAddresses:  2404:6800:4007:806::200e\n216.58.196.110\n'
    mock = MagicMock(return_value=ret)
    with patch.dict(win_network.__salt__, {'cmd.run': mock}):
        assert win_network.nslookup('google.com') == [{'Server': 'ct-dc-3-2.cybage.com'}, {'Address': '172.27.172.12'}, {'Name': 'google.com'}, {'Addresses': ['2404:6800:4007:806::200e', '216.58.196.110']}]

def test_dig():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it performs a DNS lookup with dig\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(win_network.__salt__, {'cmd.run': mock}):
        assert win_network.dig('google.com')

@pytest.mark.skipif(HAS_WMI is False, reason='WMI is available only on Windows')
def test_interfaces_names():
    if False:
        print('Hello World!')
    '\n    Test if it return a list of all the interfaces names\n    '
    WMI = Mock()
    WMI.Win32_NetworkAdapter = MagicMock(return_value=Mockwmi)
    with patch('salt.utils.winapi.Com', MagicMock()), patch.object(WMI, 'Win32_NetworkAdapter', return_value=[Mockwmi()]), patch('salt.utils', Mockwinapi), patch.object(wmi, 'WMI', Mock(return_value=WMI)):
        assert win_network.interfaces_names() == ['Ethernet']

def test_interfaces():
    if False:
        while True:
            i = 10
    '\n    Test if it return information about all the interfaces on the minion\n    '
    with patch.object(salt.utils.network, 'win_interfaces', MagicMock(return_value=True)):
        assert win_network.interfaces()

def test_hw_addr():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it return the hardware address (a.k.a. MAC address)\n    for a given interface\n    '
    with patch.object(salt.utils.network, 'hw_addr', MagicMock(return_value='Ethernet')):
        assert win_network.hw_addr('Ethernet') == 'Ethernet'

def test_subnets():
    if False:
        print('Hello World!')
    '\n    Test if it returns a list of subnets to which the host belongs\n    '
    with patch.object(salt.utils.network, 'subnets', MagicMock(return_value='10.1.1.0/24')):
        assert win_network.subnets() == '10.1.1.0/24'

def test_in_subnet():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it returns True if host is within specified subnet,\n    otherwise False\n    '
    with patch.object(salt.utils.network, 'in_subnet', MagicMock(return_value=True)):
        assert win_network.in_subnet('10.1.1.0/16')

def test_get_route():
    if False:
        i = 10
        return i + 15
    '\n    Test if it return information on open ports and states\n    '
    ret = '\n\nIPAddress         : 10.0.0.15\nInterfaceIndex    : 3\nInterfaceAlias    : Wi-Fi\nAddressFamily     : IPv4\nType              : Unicast\nPrefixLength      : 24\nPrefixOrigin      : Dhcp\nSuffixOrigin      : Dhcp\nAddressState      : Preferred\nValidLifetime     : 6.17:52:39\nPreferredLifetime : 6.17:52:39\nSkipAsSource      : False\nPolicyStore       : ActiveStore\n\n\nCaption            :\nDescription        :\nElementName        :\nInstanceID         : :8:8:8:9:55=55;:8;8;:8;55;\nAdminDistance      :\nDestinationAddress :\nIsStatic           :\nRouteMetric        : 0\nTypeOfRoute        : 3\nAddressFamily      : IPv4\nCompartmentId      : 1\nDestinationPrefix  : 0.0.0.0/0\nInterfaceAlias     : Wi-Fi\nInterfaceIndex     : 3\nNextHop            : 10.0.0.1\nPreferredLifetime  : 6.23:14:43\nProtocol           : NetMgmt\nPublish            : No\nStore              : ActiveStore\nValidLifetime      : 6.23:14:43\nPSComputerName     :\nifIndex            : 3'
    mock = MagicMock(return_value=ret)
    with patch.dict(win_network.__salt__, {'cmd.run': mock}):
        assert win_network.get_route('192.0.0.8') == {'destination': '192.0.0.8', 'gateway': '10.0.0.1', 'interface': 'Wi-Fi', 'source': '10.0.0.15'}

def test_connect_53371():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that UnboundLocalError is not thrown on socket.gaierror\n    as reported in #53371\n    '
    with patch('socket.getaddrinfo', autospec=True, side_effect=socket.gaierror('[Errno 11004] getaddrinfo failed')):
        rtn = win_network.connect('test-server', 80)
        assert rtn
        assert not rtn['result']
        assert rtn['comment'] == 'Unable to connect to test-server (unknown) on tcp port 80'