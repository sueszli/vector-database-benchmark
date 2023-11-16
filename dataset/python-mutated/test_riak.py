"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.riak
"""
import pytest
import salt.modules.riak as riak
from tests.support.mock import patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {riak: {}}

def test_start():
    if False:
        print('Hello World!')
    '\n    Test for start Riak\n    '
    with patch.object(riak, '__execute_cmd', return_value={'retcode': 0, 'stdout': 'success'}):
        assert riak.start() == {'success': True, 'comment': 'success'}

def test_stop():
    if False:
        return 10
    '\n    Test for stop Riak\n    '
    with patch.object(riak, '__execute_cmd', return_value={'retcode': 0, 'stdout': 'success'}):
        assert riak.stop() == {'success': True, 'comment': 'success'}

def test_cluster_join():
    if False:
        return 10
    '\n    Test for Join a Riak cluster\n    '
    with patch.object(riak, '__execute_cmd', return_value={'retcode': 0, 'stdout': 'success'}):
        assert riak.cluster_join('A', 'B') == {'success': True, 'comment': 'success'}

def test_cluster_leave():
    if False:
        i = 10
        return i + 15
    '\n    Test for leaving a Riak cluster\n    '
    with patch.object(riak, '__execute_cmd', return_value={'retcode': 0, 'stdout': 'success'}):
        assert riak.cluster_leave('A', 'B') == {'success': True, 'comment': 'success'}

def test_cluster_plan():
    if False:
        return 10
    '\n    Test for Review Cluster Plan\n    '
    with patch.object(riak, '__execute_cmd', return_value={'retcode': 0, 'stdout': 'success'}):
        assert riak.cluster_plan()

def test_cluster_commit():
    if False:
        i = 10
        return i + 15
    '\n    Test for Commit Cluster Changes\n    '
    with patch.object(riak, '__execute_cmd', return_value={'retcode': 0, 'stdout': 'success'}):
        assert riak.cluster_commit() == {'success': True, 'comment': 'success'}

def test_member_status():
    if False:
        print('Hello World!')
    '\n    Test for Get cluster member status\n    '
    with patch.object(riak, '__execute_cmd', return_value={'stdout': 'A:a/B:b\nC:c/D:d'}):
        assert riak.member_status() == {'membership': {}, 'summary': {'A': 'a', 'C': 'c', 'B': 'b', 'D': 'd', 'Exiting': 0, 'Down': 0, 'Valid': 0, 'Leaving': 0, 'Joining': 0}}

def test_status():
    if False:
        print('Hello World!')
    '\n    Test status information\n    '
    ret = {'stdout': 'vnode_map_update_time_95 : 0\nvnode_map_update_time_99 : 0'}
    with patch.object(riak, '__execute_cmd', return_value=ret):
        assert riak.status() == {'vnode_map_update_time_95': '0', 'vnode_map_update_time_99': '0'}

def test_test():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the Riak test\n    '
    with patch.object(riak, '__execute_cmd', return_value={'retcode': 0, 'stdout': 'success'}):
        assert riak.test() == {'success': True, 'comment': 'success'}

def test_services():
    if False:
        i = 10
        return i + 15
    '\n    Test Riak Service List\n    '
    with patch.object(riak, '__execute_cmd', return_value={'stdout': '[a,b,c]'}):
        assert riak.services() == ['a', 'b', 'c']