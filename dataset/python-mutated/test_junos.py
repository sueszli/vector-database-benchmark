import io
import pytest
import salt.proxy.junos as junos
from tests.support.mock import ANY, patch
try:
    import jxmlease
    from jnpr.junos.device import Device
    from jnpr.junos.exception import ConnectError
    HAS_JUNOS = True
except ImportError:
    HAS_JUNOS = False
pytestmark = [pytest.mark.skipif(not HAS_JUNOS, reason='The junos-eznc and jxmlease modules are required')]

@pytest.fixture
def opts():
    if False:
        for i in range(10):
            print('nop')
    return {'proxy': {'username': 'xxxx', 'password]': 'xxx', 'host': 'junos', 'port': '960'}}

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {junos: {'__pillar__': {}}}

def test_init(opts):
    if False:
        return 10
    with patch('ncclient.manager.connect') as mock_connect:
        junos.init(opts)
        assert junos.thisproxy.get('initialized')
        mock_connect.assert_called_with(allow_agent=True, device_params={'name': 'junos', 'local': False, 'use_filter': False}, host='junos', hostkey_verify=False, key_filename=None, look_for_keys=True, password=None, port='960', sock_fd=None, ssh_config=ANY, timeout=30, username='xxxx')

def test_init_err(opts):
    if False:
        i = 10
        return i + 15
    with patch('ncclient.manager.connect') as mock_connect:
        mock_connect.side_effect = ConnectError
        junos.init(opts)
        assert not junos.thisproxy.get('initialized')

def test_alive(opts):
    if False:
        print('Hello World!')
    with patch('ncclient.manager.connect') as mock_connect:
        junos.init(opts)
        junos.thisproxy['conn']._conn._session._buffer = io.BytesIO()
        assert junos.alive(opts)
        assert junos.thisproxy.get('initialized')