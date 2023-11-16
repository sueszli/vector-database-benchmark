"""
    :codeauthor: :email:`Anthony Shaw <anthonyshaw@apache.org>`
"""
import pytest
import salt.proxy.napalm as napalm_proxy
import tests.support.napalm as napalm_test_support
from tests.support.mock import MagicMock, patch

@pytest.fixture
def test_opts():
    if False:
        i = 10
        return i + 15
    return {'proxytype': 'napalm', 'driver': 'junos', 'host': 'core05.nrt02', 'id': 'core05.nrt02'}

def mock_get_device(opts, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return {'DRIVER': napalm_test_support.MockNapalmDevice(), 'UP': True}

@pytest.fixture
def configure_loader_modules(test_opts):
    if False:
        i = 10
        return i + 15
    module_globals = {'__salt__': {'config.option': MagicMock(return_value={'test': {'driver': 'test', 'key': '2orgk34kgk34g'}}), '__context__': {}}}
    module_globals['napalm_base'] = MagicMock()
    with patch('salt.utils.napalm.get_device', mock_get_device):
        yield {napalm_proxy: module_globals}

def test_init(test_opts):
    if False:
        for i in range(10):
            print('nop')
    ret = napalm_proxy.init(test_opts)
    assert ret is True

def test_alive(test_opts):
    if False:
        return 10
    napalm_proxy.init(test_opts)
    ret = napalm_proxy.alive(test_opts)
    assert ret is True

def test_ping(test_opts):
    if False:
        while True:
            i = 10
    mock_context = {'napalm_device': {'opts': {'proxytype': 'napalm', 'driver': 'junos', 'host': 'core05.nrt02', 'id': 'core05.nrt02'}, 'id': 'core05.nrt02', 'network_device': {'DRIVER': '', 'UP': True}, 'details': {'initialized': True}}}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.ping()
        assert ret is True
    mock_context = {'napalm_device': {'opts': {'proxytype': 'napalm', 'driver': 'junos', 'host': 'core05.nrt02', 'id': 'core05.nrt02'}, 'id': 'core05.nrt02', 'network_device': {'DRIVER': '', 'UP': False}, 'details': {'initialized': True}}}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.ping()
        assert not ret
    mock_context = {'napalm_device': {'opts': {'proxytype': 'napalm', 'driver': 'junos', 'host': 'core05.nrt02', 'id': 'core05.nrt02'}, 'id': 'core05.nrt02', 'network_device': {'DRIVER': ''}, 'details': {'initialized': True}}}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.ping()
        assert not ret
    mock_context = {'napalm_device': {'opts': {'proxytype': 'napalm', 'driver': 'junos', 'host': 'core05.nrt02', 'id': 'core05.nrt02'}, 'id': 'core05.nrt02', 'details': {'initialized': True}}}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.ping()
        assert not ret
    mock_context = {}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.ping()
        assert not ret

def test_initialized(test_opts):
    if False:
        while True:
            i = 10
    mock_context = {'napalm_device': {'opts': {'proxytype': 'napalm', 'driver': 'junos', 'host': 'core05.nrt02', 'id': 'core05.nrt02'}, 'id': 'core05.nrt02', 'network_device': {'DRIVER': '', 'UP': True}, 'details': {'initialized': True}}}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.initialized()
        assert ret is True
    mock_context = {'napalm_device': {'opts': {'proxytype': 'napalm', 'driver': 'junos', 'host': 'core05.nrt02', 'id': 'core05.nrt02'}, 'id': 'core05.nrt02', 'network_device': {'DRIVER': '', 'UP': False}, 'details': {'initialized': False}}}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.initialized()
        assert not ret
    mock_context = {'napalm_device': {'opts': {'proxytype': 'napalm', 'driver': 'junos', 'host': 'core05.nrt02', 'id': 'core05.nrt02'}, 'id': 'core05.nrt02', 'network_device': {'DRIVER': ''}, 'details': {}}}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.initialized()
        assert not ret
    mock_context = {'napalm_device': {'opts': {'proxytype': 'napalm', 'driver': 'junos', 'host': 'core05.nrt02', 'id': 'core05.nrt02'}, 'id': 'core05.nrt02'}}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.initialized()
        assert not ret
    mock_context = {}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.initialized()
        assert not ret

def test_get_device(test_opts):
    if False:
        while True:
            i = 10
    mock_context = {'napalm_device': {'opts': {'proxytype': 'napalm', 'driver': 'junos', 'host': 'core05.nrt02', 'id': 'core05.nrt02'}, 'id': 'core05.nrt02', 'network_device': {'DRIVER': '', 'UP': True}, 'details': {'initialized': True}}}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.get_device()
        assert ret == {'DRIVER': '', 'UP': True}
    mock_context = {'napalm_device': {'opts': {'proxytype': 'napalm', 'driver': 'junos', 'host': 'core05.nrt02', 'id': 'core05.nrt02'}, 'id': 'core05.nrt02', 'details': {'initialized': True}}}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.get_device()
        assert not ret
    mock_context = {}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.get_device()
        assert not ret

def test_get_grains(test_opts):
    if False:
        print('Hello World!')
    napalm_proxy.init(test_opts)
    ret = napalm_proxy.get_grains()
    assert ret['out'] == napalm_test_support.TEST_FACTS.copy()

def test_grains_refresh(test_opts):
    if False:
        print('Hello World!')
    napalm_proxy.init(test_opts)
    ret = napalm_proxy.grains_refresh()
    assert ret['out'] == napalm_test_support.TEST_FACTS.copy()

def test_fns():
    if False:
        for i in range(10):
            print('nop')
    ret = napalm_proxy.fns()
    assert 'details' in ret.keys()

def test_shutdown(test_opts):
    if False:
        for i in range(10):
            print('nop')
    napalm_proxy.init(test_opts)
    ret = napalm_proxy.shutdown(test_opts)
    assert ret is True

def test_call(test_opts):
    if False:
        while True:
            i = 10
    mock_context = {'napalm_device': {'opts': {'proxytype': 'napalm', 'driver': 'junos', 'host': 'core05.nrt02', 'id': 'core05.nrt02'}, 'id': 'core05.nrt02', 'network_device': mock_get_device(test_opts), 'details': {'initialized': True}}}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.call('get_arp_table')
        assert ret['out'] == napalm_test_support.TEST_ARP_TABLE.copy()
    mock_context = {}
    with patch.dict(napalm_proxy.__context__, mock_context):
        ret = napalm_proxy.call('get_arp_table')
        assert ret == {'result': False, 'comment': 'Not initialised yet', 'out': None}