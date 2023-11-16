"""
integration tests for nilirt_ip
"""
import configparser
import re
import shutil
import time
import pytest
import salt.utils.files
import salt.utils.platform
try:
    import pyiface
    from pyiface.ifreqioctls import IFF_LOOPBACK, IFF_RUNNING
except ImportError:
    pyiface = None
try:
    from requests.structures import CaseInsensitiveDict
except ImportError:
    CaseInsensitiveDict = None
INTERFACE_FOR_TEST = 'eth1'
pytestmark = [pytest.mark.skip_if_not_root, pytest.mark.destructive_test, pytest.mark.skipif('grains["os_family"] != "NILinuxRT"', reason='Tests applicable only to NILinuxRT'), pytest.mark.skip_initial_gh_actions_failure(reason='This was skipped on older golden images and is failing on newer.'), pytest.mark.skipif(pyiface is None, reason='The python pyiface package is not installed'), pytest.mark.skipif(CaseInsensitiveDict is None, reason='The python package requests is not installed')]

def _connected(interface):
    if False:
        i = 10
        return i + 15
    '\n    Check if an interface is up or down\n    :param interface: pyiface.Interface object\n    :return: True, if interface is up, otherwise False.\n    '
    return interface.flags & IFF_RUNNING != 0

@pytest.fixture(scope='module')
def interfaces():
    if False:
        print('Hello World!')
    return [interface for interface in pyiface.getIfaces() if interface.flags & IFF_LOOPBACK == 0]

@pytest.fixture(scope='module')
def ethercat_installed(grains):
    if False:
        i = 10
        return i + 15
    '\n    Check if ethercat is installed.\n\n    :return: True if ethercat is installed, otherwise False.\n    '
    if grains['lsb_distrib_id'] != 'nilrt':
        return False
    with salt.utils.files.fopen('/etc/natinst/share/ni-rt.ini', 'r') as config_file:
        config_parser = configparser.RawConfigParser(dict_type=CaseInsensitiveDict)
        config_parser.readfp(config_file)
        protocols = config_parser.get('lvrt', 'AdditionalNetworkProtocols', fallback='').lower()
        return 'ethercat' in protocols

@pytest.fixture
def ip(modules, grains, interfaces):
    if False:
        print('Hello World!')
    try:
        if grains.get('lsb_distrib_id') == 'nilrt':
            shutil.move('/etc/natinst/share/ni-rt.ini', '/tmp/ni-rt.ini')
        else:
            shutil.move('/var/lib/connman', '/tmp/connman')
        yield modules.ip
    finally:
        if grains.get('lsb_distrib_id') == 'nilrt':
            shutil.move('/tmp/ni-rt.ini', '/etc/natinst/share/ni-rt.ini')
            modules.cmd.run('/etc/init.d/networking restart')
        else:
            shutil.move('/tmp/connman', '/var/lib/connman')
            modules.service.restart('connman')
        time.sleep(10)
        for interface in interfaces:
            modules.ip.up(interface.name)

def test_down(ip, grains, interfaces):
    if False:
        i = 10
        return i + 15
    '\n    Test ip.down function\n    '
    for interface in interfaces:
        result = ip.down(interface.name)
        assert result
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if grains['lsb_distrib_id'] == 'nilrt':
            assert interface['adapter_mode'] == 'disabled'
        assert not _connected(pyiface.Interface(name=interface['connectionid']))

def test_up(ip, grains, interfaces):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test ip.up function\n    '
    for interface in interfaces:
        ip.down(interface.name)
        assert not _connected(interface)
    for interface in interfaces:
        result = ip.up(interface.name)
        assert result
    if grains['lsb_distrib_id'] == 'nilrt':
        info = ip.get_interfaces_details(timeout=300)
        for interface in info['interfaces']:
            assert interface['adapter_mode'] == 'tcpip'

def test_set_dhcp_linklocal_all(ip, grains, interfaces):
    if False:
        while True:
            i = 10
    '\n    Test ip.set_dhcp_linklocal_all function\n    '
    for interface in interfaces:
        result = ip.set_dhcp_linklocal_all(interface.name)
        assert result
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        assert interface['ipv4']['requestmode'] == 'dhcp_linklocal'
        if grains['lsb_distrib_id'] == 'nilrt':
            assert interface['adapter_mode'] == 'tcpip'

def test_set_dhcp_only_all(ip, grains, interfaces):
    if False:
        return 10
    '\n    Test ip.set_dhcp_only_all function\n    '
    if grains['lsb_distrib_id'] != 'nilrt':
        raise pytest.skip.Exception('Test not applicable to newer nilrt', _use_item_location=True)
    for interface in interfaces:
        result = ip.set_dhcp_only_all(interface.name)
        assert result
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        assert interface['ipv4']['requestmode'] == 'dhcp_only'
        assert interface['adapter_mode'] == 'tcpip'

def test_set_linklocal_only_all(ip, grains, interfaces):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test ip.set_linklocal_only_all function\n    '
    if grains['lsb_distrib_id'] != 'nilrt':
        raise pytest.skip.Exception('Test not applicable to newer nilrt', _use_item_location=True)
    for interface in interfaces:
        result = ip.set_linklocal_only_all(interface.name)
        assert result
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        assert interface['ipv4']['requestmode'] == 'linklocal_only'
        assert interface['adapter_mode'] == 'tcpip'

def test_static_all(ip, grains, interfaces):
    if False:
        i = 10
        return i + 15
    '\n    Test ip.set_static_all function\n    '
    for interface in interfaces:
        result = ip.set_static_all(interface.name, '192.168.10.4', '255.255.255.0', '192.168.10.1', '8.8.4.4 8.8.8.8')
        assert result
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if grains['lsb_distrib_id'] != 'nilrt':
            assert '8.8.4.4' in interface['ipv4']['dns']
            assert '8.8.8.8' in interface['ipv4']['dns']
        else:
            assert interface['ipv4']['dns'] == ['8.8.4.4']
            assert interface['adapter_mode'] == 'tcpip'
        assert interface['ipv4']['requestmode'] == 'static'
        assert interface['ipv4']['address'] == '192.168.10.4'
        assert interface['ipv4']['netmask'] == '255.255.255.0'
        assert interface['ipv4']['gateway'] == '192.168.10.1'

def test_supported_adapter_modes(ip, grains, ethercat_installed):
    if False:
        i = 10
        return i + 15
    '\n    Test supported adapter modes for each interface\n    '
    if grains['lsb_distrib_id'] != 'nilrt':
        raise pytest.skip.Exception('Test is just for older nilrt distros', _use_item_location=True)
    interface_pattern = re.compile('^eth[0-9]+$')
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if interface['connectionid'] == 'eth0':
            assert interface['supported_adapter_modes'] == ['tcpip']
        else:
            assert 'tcpip' in interface['supported_adapter_modes']
            if not interface_pattern.match(interface['connectionid']):
                assert 'ethercat' not in interface['supported_adapter_modes']
            elif ethercat_installed:
                assert 'ethercat' in interface['supported_adapter_modes']

def test_ethercat(ip, ethercat_installed):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test ip.set_ethercat function\n    '
    if not ethercat_installed:
        raise pytest.skip.Exception('Test is just for systems with Ethercat', _use_item_location=True)
    assert ip.set_ethercat(INTERFACE_FOR_TEST, 19)
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if interface['connectionid'] == INTERFACE_FOR_TEST:
            assert interface['adapter_mode'] == 'ethercat'
            assert int(interface['ethercat']['masterid']) == 19
            break
    assert ip.set_dhcp_linklocal_all(INTERFACE_FOR_TEST)
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if interface['connectionid'] == INTERFACE_FOR_TEST:
            assert interface['adapter_mode'] == 'tcpip'
            assert interface['ipv4']['requestmode'] == 'dhcp_linklocal'
            break

@pytest.mark.destructive_test
def test_dhcp_disable(ip, grains):
    if False:
        return 10
    '\n    Test cases:\n        - dhcp -> disable\n        - disable -> dhcp\n    '
    if grains['lsb_distrib_id'] == 'nilrt':
        raise pytest.skip.Exception('Test is just for newer nilrt distros', _use_item_location=True)
    assert ip.set_dhcp_linklocal_all(INTERFACE_FOR_TEST)
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if interface['connectionid'] == INTERFACE_FOR_TEST:
            assert interface['ipv4']['requestmode'] == 'dhcp_linklocal'
            break
    assert ip.disable(INTERFACE_FOR_TEST)
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if interface['connectionid'] == INTERFACE_FOR_TEST:
            assert interface['ipv4']['requestmode'] == 'disabled'
            break
    assert ip.set_dhcp_linklocal_all(INTERFACE_FOR_TEST)
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if interface['connectionid'] == INTERFACE_FOR_TEST:
            assert interface['ipv4']['requestmode'] == 'dhcp_linklocal'
            break

@pytest.mark.destructive_test
def test_dhcp_static(ip, grains):
    if False:
        print('Hello World!')
    '\n    Test cases:\n        - dhcp -> static\n        - static -> dhcp\n    '
    if grains['lsb_distrib_id'] == 'nilrt':
        raise pytest.skip.Exception('Test is just for newer nilrt distros', _use_item_location=True)
    assert ip.set_dhcp_linklocal_all(INTERFACE_FOR_TEST)
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if interface['connectionid'] == INTERFACE_FOR_TEST:
            assert interface['ipv4']['requestmode'] == 'dhcp_linklocal'
            break
    assert ip.set_static_all(INTERFACE_FOR_TEST, '192.168.1.125', '255.255.255.0', '192.168.1.1', '8.8.8.8 8.8.8.4')
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if interface['connectionid'] == INTERFACE_FOR_TEST:
            assert interface['ipv4']['requestmode'] == 'static'
            assert interface['ipv4']['address'] == '192.168.1.125'
            assert interface['ipv4']['netmask'] == '255.255.255.0'
            assert '8.8.8.4' in interface['ipv4']['dns']
            assert '8.8.8.8' in interface['ipv4']['dns']
            break
    assert ip.set_dhcp_linklocal_all(INTERFACE_FOR_TEST)
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if interface['connectionid'] == INTERFACE_FOR_TEST:
            assert interface['ipv4']['requestmode'] == 'dhcp_linklocal'
            break

@pytest.mark.destructive_test
def test_static_disable(ip, grains):
    if False:
        while True:
            i = 10
    '\n    Test cases:\n        - static -> disable\n        - disable -> static\n    '
    if grains['lsb_distrib_id'] == 'nilrt':
        raise pytest.skip.Exception('Test is just for newer nilrt distros', _use_item_location=True)
    assert ip.set_static_all(INTERFACE_FOR_TEST, '192.168.1.125', '255.255.255.0', '192.168.1.1', '8.8.8.8')
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if interface['connectionid'] == INTERFACE_FOR_TEST:
            assert interface['ipv4']['requestmode'] == 'static'
            assert interface['ipv4']['address'] == '192.168.1.125'
            assert interface['ipv4']['netmask'] == '255.255.255.0'
            assert interface['ipv4']['dns'] == ['8.8.8.8']
            break
    assert ip.disable(INTERFACE_FOR_TEST)
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if interface['connectionid'] == INTERFACE_FOR_TEST:
            assert interface['ipv4']['requestmode'] == 'disabled'
            break
    assert ip.set_static_all(INTERFACE_FOR_TEST, '192.168.1.125', '255.255.255.0', '192.168.1.1')
    info = ip.get_interfaces_details(timeout=300)
    for interface in info['interfaces']:
        if interface['connectionid'] == INTERFACE_FOR_TEST:
            assert interface['ipv4']['requestmode'] == 'static'
            assert interface['ipv4']['address'] == '192.168.1.125'
            assert interface['ipv4']['netmask'] == '255.255.255.0'
            assert interface['ipv4']['dns'] == []
            break