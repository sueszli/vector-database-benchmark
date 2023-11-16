import logging
import pytest
import salt.beacons.network_settings as network_settings
from tests.support.mock import MagicMock, patch
try:
    from pyroute2 import NDB
    from pyroute2.ndb.compat import ipdb_interfaces_view
    HAS_NDB = True
except ImportError:
    HAS_NDB = False
try:
    from pyroute2 import IPDB
    HAS_IPDB = True
except ImportError:
    HAS_IPDB = False
log = logging.getLogger(__name__)

class MockIPClass:

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.args = args
        self.kwargs = kwargs

    def by_name(self):
        if False:
            for i in range(10):
                print('nop')
        return {}

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {network_settings: {'__context__': {}, '__salt__': {}}}

def test_non_list_config():
    if False:
        print('Hello World!')
    config = {}
    ret = network_settings.validate(config)
    assert ret == (False, 'Configuration for network_settings beacon must be a list.')

def test_empty_config():
    if False:
        return 10
    config = [{}]
    ret = network_settings.validate(config)
    assert ret == (True, 'Valid beacon configuration')

def test_interface():
    if False:
        print('Hello World!')
    config = [{'interfaces': {'enp14s0u1u2': {'promiscuity': None}}}]
    LAST_STATS = network_settings._copy_interfaces_info({'enp14s0u1u2': {'family': '0', 'promiscuity': '0', 'group': '0'}})
    NEW_STATS = network_settings._copy_interfaces_info({'enp14s0u1u2': {'family': '0', 'promiscuity': '1', 'group': '0'}})
    ret = network_settings.validate(config)
    assert ret == (True, 'Valid beacon configuration')
    with patch.object(network_settings, 'LAST_STATS', {}), patch.object(network_settings, 'IP', MockIPClass), patch('salt.beacons.network_settings._copy_interfaces_info', MagicMock(side_effect=[LAST_STATS, NEW_STATS])):
        ret = network_settings.beacon(config)
        assert ret == []
        ret = network_settings.beacon(config)
        _expected = [{'interface': 'enp14s0u1u2', 'tag': 'enp14s0u1u2', 'change': {'promiscuity': '1'}}]
        assert ret == _expected

def test_interface_no_change():
    if False:
        while True:
            i = 10
    config = [{'interfaces': {'enp14s0u1u2': {'promiscuity': None}}}]
    LAST_STATS = network_settings._copy_interfaces_info({'enp14s0u1u2': {'family': '0', 'promiscuity': '0', 'group': '0'}})
    NEW_STATS = network_settings._copy_interfaces_info({'enp14s0u1u2': {'family': '0', 'promiscuity': '0', 'group': '0'}})
    ret = network_settings.validate(config)
    assert ret == (True, 'Valid beacon configuration')
    with patch.object(network_settings, 'LAST_STATS', {}), patch.object(network_settings, 'IP', MockIPClass), patch('salt.beacons.network_settings._copy_interfaces_info', MagicMock(side_effect=[LAST_STATS, NEW_STATS])):
        ret = network_settings.beacon(config)
        assert ret == []
        ret = network_settings.beacon(config)
        assert ret == []

def test_wildcard_interface():
    if False:
        return 10
    config = [{'interfaces': {'en*': {'promiscuity': None}}}]
    LAST_STATS = network_settings._copy_interfaces_info({'enp14s0u1u2': {'family': '0', 'promiscuity': '0', 'group': '0'}})
    NEW_STATS = network_settings._copy_interfaces_info({'enp14s0u1u2': {'family': '0', 'promiscuity': '1', 'group': '0'}})
    ret = network_settings.validate(config)
    assert ret == (True, 'Valid beacon configuration')
    with patch.object(network_settings, 'LAST_STATS', {}), patch.object(network_settings, 'IP', MockIPClass), patch('salt.beacons.network_settings._copy_interfaces_info', MagicMock(side_effect=[LAST_STATS, NEW_STATS])):
        ret = network_settings.beacon(config)
        assert ret == []
        ret = network_settings.beacon(config)
        _expected = [{'interface': 'enp14s0u1u2', 'tag': 'enp14s0u1u2', 'change': {'promiscuity': '1'}}]
        assert ret == _expected

@pytest.mark.skipif(HAS_IPDB is False, reason='pyroute2.IPDB not available, skipping')
def test_interface_dict_fields_old():
    if False:
        for i in range(10):
            print('nop')
    with IPDB() as ipdb:
        for attr in network_settings.ATTRS:
            assert attr in ipdb.interfaces[1]

@pytest.mark.skipif(HAS_NDB is False, reason='pyroute2.ndb.compat not yet available, skipping')
def test_interface_dict_fields_new():
    if False:
        i = 10
        return i + 15
    with NDB() as ndb:
        view = ipdb_interfaces_view(ndb)
        for attr in network_settings.ATTRS:
            assert attr in view['lo']