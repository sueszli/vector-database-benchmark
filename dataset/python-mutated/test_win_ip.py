"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.win_ip
"""
import pytest
import salt.modules.win_ip as win_ip
from salt.exceptions import CommandExecutionError, SaltInvocationError
from tests.support.mock import MagicMock, call, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {win_ip: {}}

@pytest.fixture
def ethernet_config():
    if False:
        for i in range(10):
            print('nop')
    return 'Configuration for interface "Ethernet"\nDHCP enabled: Yes\nIP Address: 1.2.3.74\nSubnet Prefix: 1.2.3.0/24 (mask 255.255.255.0)\nDefault Gateway: 1.2.3.1\nGateway Metric: 0\nInterfaceMetric: 20\nDNS servers configured through DHCP: 1.2.3.4\nRegister with which suffix: Primary only\nWINS servers configured through DHCP: None\n'

@pytest.fixture
def ethernet_enable():
    if False:
        i = 10
        return i + 15
    return 'Ethernet\nType: Dedicated\nAdministrative state: Enabled\nConnect state: Connected'

def test_raw_interface_configs(ethernet_config):
    if False:
        print('Hello World!')
    '\n    Test if it return raw configs for all interfaces.\n    '
    mock_cmd = MagicMock(return_value=ethernet_config)
    with patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd}):
        assert win_ip.raw_interface_configs() == ethernet_config

def test_get_all_interfaces(ethernet_config):
    if False:
        return 10
    '\n    Test if it return configs for all interfaces.\n    '
    ret = {'Ethernet': {'DHCP enabled': 'Yes', 'DNS servers configured through DHCP': ['1.2.3.4'], 'Default Gateway': '1.2.3.1', 'Gateway Metric': '0', 'InterfaceMetric': '20', 'Register with which suffix': 'Primary only', 'WINS servers configured through DHCP': ['None'], 'ip_addrs': [{'IP Address': '1.2.3.74', 'Netmask': '255.255.255.0', 'Subnet': '1.2.3.0/24'}]}}
    mock_cmd = MagicMock(return_value=ethernet_config)
    with patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd}):
        assert win_ip.get_all_interfaces() == ret

def test_get_interface(ethernet_config):
    if False:
        i = 10
        return i + 15
    '\n    Test if it return the configuration of a network interface.\n    '
    ret = {'DHCP enabled': 'Yes', 'DNS servers configured through DHCP': ['1.2.3.4'], 'Default Gateway': '1.2.3.1', 'Gateway Metric': '0', 'InterfaceMetric': '20', 'Register with which suffix': 'Primary only', 'WINS servers configured through DHCP': ['None'], 'ip_addrs': [{'IP Address': '1.2.3.74', 'Netmask': '255.255.255.0', 'Subnet': '1.2.3.0/24'}]}
    mock_cmd = MagicMock(return_value=ethernet_config)
    with patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd}):
        assert win_ip.get_interface('Ethernet') == ret

def test_is_enabled(ethernet_enable):
    if False:
        while True:
            i = 10
    '\n    Test if it returns `True` if interface is enabled, otherwise `False`.\n    '
    mock_cmd = MagicMock(side_effect=[ethernet_enable, ''])
    with patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd}):
        assert win_ip.is_enabled('Ethernet')
        pytest.raises(CommandExecutionError, win_ip.is_enabled, 'Ethernet')

def test_is_disabled(ethernet_enable):
    if False:
        return 10
    '\n    Test if it returns `True` if interface is disabled, otherwise `False`.\n    '
    mock_cmd = MagicMock(return_value=ethernet_enable)
    with patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd}):
        assert not win_ip.is_disabled('Ethernet')

def test_enable():
    if False:
        i = 10
        return i + 15
    '\n    Test if it enable an interface.\n    '
    with patch.object(win_ip, 'is_enabled', return_value=True):
        assert win_ip.enable('Ethernet')
    mock_cmd = MagicMock()
    with patch.object(win_ip, 'is_enabled', side_effect=[False, True]), patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd}):
        assert win_ip.enable('Ethernet')
    mock_cmd.assert_called_once_with(['netsh', 'interface', 'set', 'interface', 'name=Ethernet', 'admin=ENABLED'], python_shell=False)

def test_disable():
    if False:
        print('Hello World!')
    '\n    Test if it disable an interface.\n    '
    with patch.object(win_ip, 'is_disabled', return_value=True):
        assert win_ip.disable('Ethernet')
    mock_cmd = MagicMock()
    with patch.object(win_ip, 'is_disabled', side_effect=[False, True]), patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd}):
        assert win_ip.disable('Ethernet')
    mock_cmd.assert_called_once_with(['netsh', 'interface', 'set', 'interface', 'name=Ethernet', 'admin=DISABLED'], python_shell=False)

def test_get_subnet_length():
    if False:
        i = 10
        return i + 15
    '\n    Test if it disable an interface.\n    '
    assert win_ip.get_subnet_length('255.255.255.0') == 24
    pytest.raises(SaltInvocationError, win_ip.get_subnet_length, '255.255.0')

@pytest.mark.slow_test
def test_set_static_ip(ethernet_config):
    if False:
        while True:
            i = 10
    '\n    Test if it set static IP configuration on a Windows NIC.\n    '
    pytest.raises(SaltInvocationError, win_ip.set_static_ip, 'Local Area Connection', '10.1.2/24')
    mock_cmd = MagicMock(return_value=ethernet_config)
    mock_all = MagicMock(return_value={'retcode': 1, 'stderr': 'Error'})
    with patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd, 'cmd.run_all': mock_all}):
        pytest.raises(CommandExecutionError, win_ip.set_static_ip, 'Ethernet', '1.2.3.74/24', append=True)
        pytest.raises(CommandExecutionError, win_ip.set_static_ip, 'Ethernet', '1.2.3.74/24')
    mock_all = MagicMock(return_value={'retcode': 0})
    with patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd, 'cmd.run_all': mock_all}):
        assert win_ip.set_static_ip('Local Area Connection', '1.2.3.74/24') == {}
        assert win_ip.set_static_ip('Ethernet', '1.2.3.74/24') == {'Address Info': {'IP Address': '1.2.3.74', 'Netmask': '255.255.255.0', 'Subnet': '1.2.3.0/24'}}

def test_set_dhcp_ip(ethernet_config):
    if False:
        i = 10
        return i + 15
    '\n    Test if it set Windows NIC to get IP from DHCP.\n    '
    mock_cmd = MagicMock(return_value=ethernet_config)
    with patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd}):
        assert win_ip.set_dhcp_ip('Ethernet') == {'DHCP enabled': 'Yes', 'Interface': 'Ethernet'}

def test_set_static_dns():
    if False:
        print('Hello World!')
    '\n    Test if it set static DNS configuration on a Windows NIC.\n    '
    mock_cmd = MagicMock()
    with patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd}):
        assert win_ip.set_static_dns('Ethernet', '192.168.1.252', '192.168.1.253') == {'DNS Server': ('192.168.1.252', '192.168.1.253'), 'Interface': 'Ethernet'}
        mock_cmd.assert_has_calls([call(['netsh', 'interface', 'ip', 'set', 'dns', 'name=Ethernet', 'source=static', 'address=192.168.1.252', 'register=primary'], python_shell=False), call(['netsh', 'interface', 'ip', 'add', 'dns', 'name=Ethernet', 'address=192.168.1.253', 'index=2'], python_shell=False)])

def test_set_static_dns_clear():
    if False:
        while True:
            i = 10
    '\n    Test if it set static DNS configuration on a Windows NIC.\n    '
    mock_cmd = MagicMock()
    with patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd}):
        assert win_ip.set_static_dns('Ethernet', []) == {'DNS Server': [], 'Interface': 'Ethernet'}
        mock_cmd.assert_called_once_with(['netsh', 'interface', 'ip', 'set', 'dns', 'name=Ethernet', 'source=static', 'address=none'], python_shell=False)

def test_set_static_dns_no_action():
    if False:
        print('Hello World!')
    '\n    Test if it set static DNS configuration on a Windows NIC.\n    '
    assert win_ip.set_static_dns('Ethernet') == {'DNS Server': 'No Changes', 'Interface': 'Ethernet'}
    assert win_ip.set_static_dns('Ethernet', None) == {'DNS Server': 'No Changes', 'Interface': 'Ethernet'}
    assert win_ip.set_static_dns('Ethernet', 'None') == {'DNS Server': 'No Changes', 'Interface': 'Ethernet'}

def test_set_dhcp_dns(ethernet_config):
    if False:
        while True:
            i = 10
    '\n    Test if it set DNS source to DHCP on Windows.\n    '
    mock_cmd = MagicMock(return_value=ethernet_config)
    with patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd}):
        assert win_ip.set_dhcp_dns('Ethernet') == {'DNS Server': 'DHCP', 'Interface': 'Ethernet'}

def test_set_dhcp_all(ethernet_config):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it set both IP Address and DNS to DHCP.\n    '
    mock_cmd = MagicMock(return_value=ethernet_config)
    with patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd}):
        assert win_ip.set_dhcp_all('Ethernet') == {'Interface': 'Ethernet', 'DNS Server': 'DHCP', 'DHCP enabled': 'Yes'}

def test_get_default_gateway(ethernet_config):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it set DNS source to DHCP on Windows.\n    '
    mock_cmd = MagicMock(return_value=ethernet_config)
    with patch.dict(win_ip.__salt__, {'cmd.run': mock_cmd}):
        assert win_ip.get_default_gateway() == '1.2.3.1'