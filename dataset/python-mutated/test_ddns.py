"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>
"""
import textwrap
import pytest
import salt.modules.ddns as ddns
import salt.utils.json
from tests.support.mock import MagicMock, mock_open, patch
try:
    import dns.query
    import dns.tsigkeyring
    HAS_DNS = True
except ImportError:
    HAS_DNS = False
pytestmark = [pytest.mark.skipif(HAS_DNS is False, reason='dnspython libs not installed.')]

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {ddns: {}}

def test_add_host():
    if False:
        while True:
            i = 10
    '\n    Test cases for Add, replace, or update the A\n    and PTR (reverse) records for a host.\n    '
    with patch('salt.modules.ddns.update') as ddns_update:
        ddns_update.return_value = False
        assert not ddns.add_host(zone='A', name='B', ttl=1, ip='172.27.0.0')
        ddns_update.return_value = True
        assert ddns.add_host(zone='A', name='B', ttl=1, ip='172.27.0.0')

def test_delete_host():
    if False:
        return 10
    '\n    Tests for delete the forward and reverse records for a host.\n    '
    with patch('salt.modules.ddns.delete') as ddns_delete:
        ddns_delete.return_value = False
        with patch.object(dns.query, 'udp') as mock:
            mock.answer = [{'address': 'localhost'}]
            assert not ddns.delete_host(zone='A', name='B')

def test_update():
    if False:
        print('Hello World!')
    '\n    Test to add, replace, or update a DNS record.\n    '
    mock_request = textwrap.dedent('        id 29380\n        opcode QUERY\n        rcode NOERROR\n        flags RD\n        ;QUESTION\n        name.zone. IN AAAA\n        ;ANSWER\n        ;AUTHORITY\n        ;ADDITIONAL')
    mock_rdtype = 28

    class MockRrset:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.items = [{'address': 'localhost'}]
            self.ttl = 2

    class MockAnswer:

        def __init__(self, *args, **kwargs):
            if False:
                return 10
            self.answer = [MockRrset()]

        def rcode(self):
            if False:
                for i in range(10):
                    print('nop')
            return 0

    def mock_udp_query(*args, **kwargs):
        if False:
            while True:
                i = 10
        return MockAnswer
    with patch.object(dns.message, 'make_query', MagicMock(return_value=mock_request)):
        with patch.object(dns.query, 'udp', mock_udp_query()):
            with patch.object(dns.rdatatype, 'from_text', MagicMock(return_value=mock_rdtype)):
                with patch.object(ddns, '_get_keyring', return_value=None):
                    with patch.object(ddns, '_config', return_value=None):
                        assert ddns.update('zone', 'name', 1, 'AAAA', '::1')

def test_delete():
    if False:
        i = 10
        return i + 15
    '\n    Test to delete a DNS record.\n    '
    file_data = salt.utils.json.dumps({'A': 'B'})

    class MockAnswer:

        def __init__(self, *args, **kwargs):
            if False:
                return 10
            self.answer = [{'address': 'localhost'}]

        def rcode(self):
            if False:
                i = 10
                return i + 15
            return 0

    def mock_udp_query(*args, **kwargs):
        if False:
            print('Hello World!')
        return MockAnswer
    with patch.object(dns.query, 'udp', mock_udp_query()):
        with patch('salt.utils.files.fopen', mock_open(read_data=file_data), create=True):
            with patch.object(dns.tsigkeyring, 'from_text', return_value=True):
                with patch.object(ddns, '_get_keyring', return_value=None):
                    with patch.object(ddns, '_config', return_value=None):
                        assert ddns.delete(zone='A', name='B')