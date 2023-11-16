import pytest
import salt.utils.minions
import salt.utils.network
from tests.support.mock import patch

def test_connected_ids():
    if False:
        while True:
            i = 10
    '\n    test ckminion connected_ids when\n    local_port_tcp returns 127.0.0.1\n    '
    opts = {'publish_port': 4505, 'detect_remote_minions': False, 'minion_data_cache': True}
    minion = 'minion'
    ips = {'203.0.113.1', '203.0.113.2', '127.0.0.1'}
    mdata = {'grains': {'ipv4': ips, 'ipv6': []}}
    patch_net = patch('salt.utils.network.local_port_tcp', return_value=ips)
    patch_list = patch('salt.cache.Cache.list', return_value=[minion])
    patch_fetch = patch('salt.cache.Cache.fetch', return_value=mdata)
    ckminions = salt.utils.minions.CkMinions(opts)
    with patch_net, patch_list, patch_fetch:
        ret = ckminions.connected_ids()
        assert ret == {minion}

def test_connected_ids_remote_minions():
    if False:
        for i in range(10):
            print('nop')
    '\n    test ckminion connected_ids when\n    detect_remote_minions is set\n    '
    opts = {'publish_port': 4505, 'detect_remote_minions': True, 'remote_minions_port': 22, 'minion_data_cache': True}
    minion = 'minion'
    minion2 = 'minion2'
    minion2_ip = '192.168.2.10'
    minion_ips = {'203.0.113.1', '203.0.113.2', '127.0.0.1'}
    mdata = {'grains': {'ipv4': minion_ips, 'ipv6': []}}
    mdata2 = {'grains': {'ipv4': [minion2_ip], 'ipv6': []}}
    patch_net = patch('salt.utils.network.local_port_tcp', return_value=minion_ips)
    patch_remote_net = patch('salt.utils.network.remote_port_tcp', return_value={minion2_ip})
    patch_list = patch('salt.cache.Cache.list', return_value=[minion, minion2])
    patch_fetch = patch('salt.cache.Cache.fetch', side_effect=[mdata, mdata2])
    ckminions = salt.utils.minions.CkMinions(opts)
    with patch_net, patch_list, patch_fetch, patch_remote_net:
        ret = ckminions.connected_ids()
        assert ret == {minion2, minion}

def test_validate_tgt_returns_true_when_no_valid_minions_have_been_found():
    if False:
        print('Hello World!')
    '\n    CKMinions is only able to check against minions the master knows about. If\n    no minion keys have been accepted it will return True.\n    '
    ckminions = salt.utils.minions.CkMinions(opts={})
    with patch('salt.utils.minions.CkMinions.check_minions', autospec=True, return_value={}):
        result = ckminions.validate_tgt('fnord', 'fnord', 'fnord', minions=[])
        assert result is True

@pytest.mark.parametrize('valid_minions, target_minions', [(['one', 'two', 'three'], ['one', 'two', 'five']), (['one'], ['one', 'two']), (['one', 'two', 'three', 'four'], ['five'])])
def test_validate_tgt_should_return_false_when_minions_have_minions_not_in_valid_minions(valid_minions, target_minions):
    if False:
        return 10
    ckminions = salt.utils.minions.CkMinions(opts={})
    with patch('salt.utils.minions.CkMinions.check_minions', autospec=True, return_value={'minions': valid_minions}):
        result = ckminions.validate_tgt('fnord', 'fnord', 'fnord', minions=target_minions)
        assert result is False

@pytest.mark.parametrize('valid_minions, target_minions', [(['one', 'two', 'three', 'five'], ['one', 'two', 'five']), (['one'], ['one']), (['one', 'two', 'three', 'four', 'five'], ['five'])])
def test_validate_tgt_should_return_true_when_all_minions_are_found_in_valid_minions(valid_minions, target_minions):
    if False:
        i = 10
        return i + 15
    ckminions = salt.utils.minions.CkMinions(opts={})
    with patch('salt.utils.minions.CkMinions.check_minions', autospec=True, return_value={'minions': valid_minions}):
        result = ckminions.validate_tgt('fnord', 'fnord', 'fnord', minions=target_minions)
        assert result is True