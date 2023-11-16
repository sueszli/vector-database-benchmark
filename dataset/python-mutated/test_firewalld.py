"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>
"""
import pytest
import salt.modules.firewalld as firewalld
from tests.support.helpers import dedent
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {firewalld: {}}

def test_version():
    if False:
        while True:
            i = 10
    '\n    Test for Return version from firewall-cmd\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value=2):
        assert firewalld.version() == 2

def test_default_zone():
    if False:
        i = 10
        return i + 15
    '\n    Test for Print default zone for connections and interfaces\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='A'):
        assert firewalld.default_zone() == 'A'

def test_list_zones():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for List everything added for or enabled in all zones\n    '
    firewall_cmd_ret = dedent('            nm-shared\n              target: ACCEPT\n              icmp-block-inversion: no\n              interfaces:\n              sources:\n              services: dhcp dns ssh\n              ports:\n              protocols: icmp ipv6-icmp\n              masquerade: no\n              forward-ports:\n              source-ports:\n              icmp-blocks:\n              rich rules:\n            \trule priority="32767" reject\n\n            public\n              target: default\n              icmp-block-inversion: no\n              interfaces:\n              sources:\n              services: cockpit dhcpv6-client ssh\n              ports:\n              protocols:\n              masquerade: no\n              forward-ports:\n              source-ports:\n              icmp-blocks:\n              rich rules:\n            ')
    ret = {'nm-shared': {'forward-ports': [''], 'icmp-block-inversion': ['no'], 'icmp-blocks': [''], 'interfaces': [''], 'masquerade': ['no'], 'ports': [''], 'protocols': ['icmp ipv6-icmp'], 'rich rules': ['', 'rule priority="32767" reject'], 'services': ['dhcp dns ssh'], 'source-ports': [''], 'sources': [''], 'target': ['ACCEPT']}, 'public': {'forward-ports': [''], 'icmp-block-inversion': ['no'], 'icmp-blocks': [''], 'interfaces': [''], 'masquerade': ['no'], 'ports': [''], 'protocols': [''], 'rich rules': [''], 'services': ['cockpit dhcpv6-client ssh'], 'source-ports': [''], 'sources': [''], 'target': ['default']}}
    with patch.object(firewalld, '__firewall_cmd', return_value=firewall_cmd_ret):
        assert firewalld.list_zones() == ret

def test_list_zones_empty_response():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test list_zones if firewall-cmd call returns nothing\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value=''):
        assert firewalld.list_zones() == {}

def test_get_zones():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Print predefined zones\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='A'):
        assert firewalld.get_zones() == ['A']

def test_get_services():
    if False:
        return 10
    '\n    Test for Print predefined services\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='A'):
        assert firewalld.get_services() == ['A']

def test_get_icmp_types():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Print predefined icmptypes\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='A'):
        assert firewalld.get_icmp_types() == ['A']

def test_new_zone():
    if False:
        while True:
            i = 10
    '\n    Test for Add a new zone\n    '
    with patch.object(firewalld, '__mgmt', return_value='success'):
        mock = MagicMock(return_value='A')
        with patch.object(firewalld, '__firewall_cmd', mock):
            assert firewalld.new_zone('zone') == 'A'
    with patch.object(firewalld, '__mgmt', return_value='A'):
        assert firewalld.new_zone('zone') == 'A'
    with patch.object(firewalld, '__mgmt', return_value='A'):
        assert firewalld.new_zone('zone', False) == 'A'

def test_delete_zone():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Delete an existing zone\n    '
    with patch.object(firewalld, '__mgmt', return_value='success'):
        with patch.object(firewalld, '__firewall_cmd', return_value='A'):
            assert firewalld.delete_zone('zone') == 'A'
    with patch.object(firewalld, '__mgmt', return_value='A'):
        assert firewalld.delete_zone('zone') == 'A'
    mock = MagicMock(return_value='A')
    with patch.object(firewalld, '__mgmt', return_value='A'):
        assert firewalld.delete_zone('zone', False) == 'A'

def test_set_default_zone():
    if False:
        while True:
            i = 10
    '\n    Test for Set default zone\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='A'):
        assert firewalld.set_default_zone('zone') == 'A'

def test_new_service():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Add a new service\n    '
    with patch.object(firewalld, '__mgmt', return_value='success'):
        mock = MagicMock(return_value='A')
        with patch.object(firewalld, '__firewall_cmd', return_value='A'):
            assert firewalld.new_service('zone') == 'A'
    with patch.object(firewalld, '__mgmt', return_value='A'):
        assert firewalld.new_service('zone') == 'A'
    with patch.object(firewalld, '__mgmt', return_value='A'):
        assert firewalld.new_service('zone', False) == 'A'

def test_delete_service():
    if False:
        while True:
            i = 10
    '\n    Test for Delete an existing service\n    '
    with patch.object(firewalld, '__mgmt', return_value='success'):
        mock = MagicMock(return_value='A')
        with patch.object(firewalld, '__firewall_cmd', return_value='A'):
            assert firewalld.delete_service('name') == 'A'
    with patch.object(firewalld, '__mgmt', return_value='A'):
        assert firewalld.delete_service('name') == 'A'
    with patch.object(firewalld, '__mgmt', return_value='A'):
        assert firewalld.delete_service('name', False) == 'A'

def test_list_all():
    if False:
        print('Hello World!')
    '\n    Test for List everything added for or enabled in a zone\n    '
    firewall_cmd_ret = dedent('        public\n          target: default\n          icmp-block-inversion: no\n          interfaces: eth0\n          sources:\n          services: cockpit dhcpv6-client ssh\n          ports:\n          protocols:\n          masquerade: no\n          forward-ports:\n          source-ports:\n          icmp-blocks:\n          rich rules:\n        ')
    ret = {'public': {'forward-ports': [''], 'icmp-block-inversion': ['no'], 'icmp-blocks': [''], 'interfaces': ['eth0'], 'masquerade': ['no'], 'ports': [''], 'protocols': [''], 'rich rules': [''], 'services': ['cockpit dhcpv6-client ssh'], 'source-ports': [''], 'sources': [''], 'target': ['default']}}
    with patch.object(firewalld, '__firewall_cmd', return_value=firewall_cmd_ret):
        assert firewalld.list_all() == ret

def test_list_all_empty_response():
    if False:
        print('Hello World!')
    '\n    Test list_all if firewall-cmd call returns nothing\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value=''):
        assert firewalld.list_all() == {}

def test_list_services():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for List services added for zone as a space separated list.\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value=''):
        assert firewalld.list_services() == []

def test_add_service():
    if False:
        i = 10
        return i + 15
    '\n    Test for Add a service for zone\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value=''):
        assert firewalld.add_service('name') == ''

def test_remove_service():
    if False:
        return 10
    '\n    Test for Remove a service from zone\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value=''):
        assert firewalld.remove_service('name') == ''

def test_add_masquerade():
    if False:
        i = 10
        return i + 15
    '\n    Test for adding masquerade\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='success'):
        assert firewalld.add_masquerade('name') == 'success'

def test_remove_masquerade():
    if False:
        while True:
            i = 10
    '\n    Test for removing masquerade\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='success'):
        assert firewalld.remove_masquerade('name') == 'success'

def test_add_port():
    if False:
        return 10
    '\n    Test adding a port to a specific zone\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='success'):
        assert firewalld.add_port('zone', '80/tcp') == 'success'

def test_remove_port():
    if False:
        i = 10
        return i + 15
    '\n    Test removing a port from a specific zone\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='success'):
        assert firewalld.remove_port('zone', '80/tcp') == 'success'

def test_list_ports():
    if False:
        return 10
    '\n    Test listing ports within a zone\n    '
    ret = '22/tcp 53/udp 53/tcp'
    exp = ['22/tcp', '53/udp', '53/tcp']
    with patch.object(firewalld, '__firewall_cmd', return_value=ret):
        assert firewalld.list_ports('zone') == exp

def test_add_port_fwd():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test adding port forwarding on a zone\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='success'):
        assert firewalld.add_port_fwd('zone', '22', '2222', 'tcp') == 'success'

def test_remove_port_fwd():
    if False:
        while True:
            i = 10
    '\n    Test removing port forwarding on a zone\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='success'):
        assert firewalld.remove_port_fwd('zone', '22', '2222', 'tcp') == 'success'

def test_list_port_fwd():
    if False:
        i = 10
        return i + 15
    '\n    Test listing all port forwarding for a zone\n    '
    ret = 'port=23:proto=tcp:toport=8080:toaddr=\nport=80:proto=tcp:toport=443:toaddr='
    exp = [{'Destination address': '', 'Destination port': '8080', 'Protocol': 'tcp', 'Source port': '23'}, {'Destination address': '', 'Destination port': '443', 'Protocol': 'tcp', 'Source port': '80'}]
    with patch.object(firewalld, '__firewall_cmd', return_value=ret):
        assert firewalld.list_port_fwd('zone') == exp

def test_block_icmp():
    if False:
        return 10
    '\n    Test ICMP block\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='success'):
        with patch.object(firewalld, 'get_icmp_types', return_value='echo-reply'):
            assert firewalld.block_icmp('zone', 'echo-reply') == 'success'
    with patch.object(firewalld, '__firewall_cmd'):
        assert not firewalld.block_icmp('zone', 'echo-reply')

def test_allow_icmp():
    if False:
        while True:
            i = 10
    '\n    Test ICMP allow\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='success'):
        with patch.object(firewalld, 'get_icmp_types', return_value='echo-reply'):
            assert firewalld.allow_icmp('zone', 'echo-reply') == 'success'
    with patch.object(firewalld, '__firewall_cmd', return_value='success'):
        assert not firewalld.allow_icmp('zone', 'echo-reply')

def test_list_icmp_block():
    if False:
        while True:
            i = 10
    '\n    Test ICMP block list\n    '
    ret = 'echo-reply echo-request'
    exp = ['echo-reply', 'echo-request']
    with patch.object(firewalld, '__firewall_cmd', return_value=ret):
        assert firewalld.list_icmp_block('zone') == exp

def test_get_rich_rules():
    if False:
        return 10
    '\n    Test listing rich rules bound to a zone\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value=''):
        assert firewalld.get_rich_rules('zone') == []

def test_add_rich_rule():
    if False:
        i = 10
        return i + 15
    '\n    Test adding a rich rule to a zone\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='success'):
        assert firewalld.add_rich_rule('zone', 'rule family="ipv4" source address="1.2.3.4" accept') == 'success'

def test_remove_rich_rule():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test removing a rich rule to a zone\n    '
    with patch.object(firewalld, '__firewall_cmd', return_value='success'):
        assert firewalld.remove_rich_rule('zone', 'rule family="ipv4" source address="1.2.3.4" accept') == 'success'