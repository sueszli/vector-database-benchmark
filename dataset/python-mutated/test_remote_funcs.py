import pytest
import salt.config
import salt.daemons.masterapi as masterapi
import salt.utils.platform
from tests.support.mock import MagicMock, patch
pytestmark = [pytest.mark.slow_test]

class FakeCache:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.data = {}

    def store(self, bank, key, value):
        if False:
            i = 10
            return i + 15
        self.data[bank, key] = value

    def fetch(self, bank, key):
        if False:
            while True:
                i = 10
        return self.data[bank, key]

@pytest.fixture
def funcs(temp_salt_master):
    if False:
        while True:
            i = 10
    opts = temp_salt_master.config.copy()
    salt.cache.MemCache.data.clear()
    funcs = masterapi.RemoteFuncs(opts)
    funcs.cache = FakeCache()
    return funcs

def test_mine_get(funcs, tgt_type_key='tgt_type'):
    if False:
        i = 10
        return i + 15
    '\n    Asserts that ``mine_get`` gives the expected results.\n\n    Actually this only tests that:\n\n    - the correct check minions method is called\n    - the correct cache key is subsequently used\n    '
    funcs.cache.store('minions/webserver', 'mine', dict(ip_addr='2001:db8::1:3'))
    with patch('salt.utils.minions.CkMinions._check_compound_minions', MagicMock(return_value=dict(minions=['webserver'], missing=[]))):
        ret = funcs._mine_get({'id': 'requester_minion', 'tgt': 'G@roles:web', 'fun': 'ip_addr', tgt_type_key: 'compound'})
    assert ret == dict(webserver='2001:db8::1:3')

def test_mine_get_pre_nitrogen_compat(funcs):
    if False:
        return 10
    '\n    Asserts that pre-Nitrogen API key ``expr_form`` is still accepted.\n\n    This is what minions before Nitrogen would issue.\n    '
    test_mine_get(funcs, tgt_type_key='expr_form')

def test_mine_get_dict_str(funcs, tgt_type_key='tgt_type'):
    if False:
        print('Hello World!')
    '\n    Asserts that ``mine_get`` gives the expected results when request\n    is a comma-separated list.\n\n    Actually this only tests that:\n\n    - the correct check minions method is called\n    - the correct cache key is subsequently used\n    '
    funcs.cache.store('minions/webserver', 'mine', dict(ip_addr='2001:db8::1:3', ip4_addr='127.0.0.1'))
    with patch('salt.utils.minions.CkMinions._check_compound_minions', MagicMock(return_value=dict(minions=['webserver'], missing=[]))):
        ret = funcs._mine_get({'id': 'requester_minion', 'tgt': 'G@roles:web', 'fun': 'ip_addr,ip4_addr', tgt_type_key: 'compound'})
    assert ret == dict(ip_addr=dict(webserver='2001:db8::1:3'), ip4_addr=dict(webserver='127.0.0.1'))

def test_mine_get_dict_list(funcs, tgt_type_key='tgt_type'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Asserts that ``mine_get`` gives the expected results when request\n    is a list.\n\n    Actually this only tests that:\n\n    - the correct check minions method is called\n    - the correct cache key is subsequently used\n    '
    funcs.cache.store('minions/webserver', 'mine', dict(ip_addr='2001:db8::1:3', ip4_addr='127.0.0.1'))
    with patch('salt.utils.minions.CkMinions._check_compound_minions', MagicMock(return_value=dict(minions=['webserver'], missing=[]))):
        ret = funcs._mine_get({'id': 'requester_minion', 'tgt': 'G@roles:web', 'fun': ['ip_addr', 'ip4_addr'], tgt_type_key: 'compound'})
    assert ret == dict(ip_addr=dict(webserver='2001:db8::1:3'), ip4_addr=dict(webserver='127.0.0.1'))

def test_mine_get_acl_allowed(funcs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Asserts that ``mine_get`` gives the expected results when this is allowed\n    in the client-side ACL that was stored in the mine data.\n    '
    funcs.cache.store('minions/webserver', 'mine', {'ip_addr': {salt.utils.mine.MINE_ITEM_ACL_DATA: '2001:db8::1:4', salt.utils.mine.MINE_ITEM_ACL_ID: salt.utils.mine.MINE_ITEM_ACL_VERSION, 'allow_tgt': 'requester_minion', 'allow_tgt_type': 'glob'}})
    with patch('salt.utils.minions.CkMinions._check_glob_minions', MagicMock(return_value={'minions': ['requester_minion'], 'missing': []})), patch('salt.utils.minions.CkMinions._check_compound_minions', MagicMock(return_value={'minions': ['webserver'], 'missing': []})):
        ret = funcs._mine_get({'id': 'requester_minion', 'tgt': 'anything', 'tgt_type': 'compound', 'fun': ['ip_addr']})
    assert ret == {'ip_addr': {'webserver': '2001:db8::1:4'}}

def test_mine_get_acl_rejected(funcs):
    if False:
        return 10
    "\n    Asserts that ``mine_get`` gives the expected results when this is rejected\n    in the client-side ACL that was stored in the mine data. This results in\n    no data being sent back (just as if the entry wouldn't exist).\n    "
    funcs.cache.store('minions/webserver', 'mine', {'ip_addr': {salt.utils.mine.MINE_ITEM_ACL_DATA: '2001:db8::1:4', salt.utils.mine.MINE_ITEM_ACL_ID: salt.utils.mine.MINE_ITEM_ACL_VERSION, 'allow_tgt': 'not_requester_minion', 'allow_tgt_type': 'glob'}})
    with patch('salt.utils.minions.CkMinions._check_glob_minions', MagicMock(return_value={'minions': ['not_requester_minion'], 'missing': []})), patch('salt.utils.minions.CkMinions._check_compound_minions', MagicMock(return_value={'minions': ['webserver'], 'missing': []})):
        ret = funcs._mine_get({'id': 'requester_minion', 'tgt': 'anything', 'tgt_type': 'compound', 'fun': ['ip_addr']})
    assert ret == {}