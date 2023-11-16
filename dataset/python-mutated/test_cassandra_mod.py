"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>

    Test cases for salt.modules.cassandra_mod
"""
import pytest
import salt.modules.cassandra_mod as cassandra
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {cassandra: {}}

def test_compactionstats():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Return compactionstats info\n    '
    mock = MagicMock(return_value='A')
    with patch.object(cassandra, '_nodetool', mock):
        assert cassandra.compactionstats() == 'A'

def test_version():
    if False:
        while True:
            i = 10
    '\n    Test for Return the cassandra version\n    '
    mock = MagicMock(return_value='A')
    with patch.object(cassandra, '_nodetool', mock):
        assert cassandra.version() == 'A'

def test_netstats():
    if False:
        return 10
    '\n    Test for Return netstats info\n    '
    mock = MagicMock(return_value='A')
    with patch.object(cassandra, '_nodetool', mock):
        assert cassandra.netstats() == 'A'

def test_tpstats():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Return tpstats info\n    '
    mock = MagicMock(return_value='A')
    with patch.object(cassandra, '_nodetool', mock):
        assert cassandra.tpstats() == 'A'

def test_info():
    if False:
        return 10
    '\n    Test for Return cassandra node info\n    '
    mock = MagicMock(return_value='A')
    with patch.object(cassandra, '_nodetool', mock):
        assert cassandra.info() == 'A'

def test_ring():
    if False:
        return 10
    '\n    Test for Return ring info\n    '
    mock = MagicMock(return_value='A')
    with patch.object(cassandra, '_nodetool', mock):
        assert cassandra.ring() == 'A'

def test_keyspaces():
    if False:
        while True:
            i = 10
    '\n    Test for Return existing keyspaces\n    '
    mock_keyspaces = ['A', 'B', 'C', 'D']

    class MockSystemManager:

        def list_keyspaces(self):
            if False:
                print('Hello World!')
            return mock_keyspaces
    mock_sys_mgr = MagicMock(return_value=MockSystemManager())
    with patch.object(cassandra, '_sys_mgr', mock_sys_mgr):
        assert cassandra.keyspaces() == mock_keyspaces

def test_column_families():
    if False:
        while True:
            i = 10
    '\n    Test for Return existing column families for all keyspaces\n    '
    mock_keyspaces = ['A', 'B']

    class MockSystemManager:

        def list_keyspaces(self):
            if False:
                i = 10
                return i + 15
            return mock_keyspaces

        def get_keyspace_column_families(self, keyspace):
            if False:
                return 10
            if keyspace == 'A':
                return {'a': 'saltines', 'b': 'biscuits'}
            if keyspace == 'B':
                return {'c': 'cheese', 'd': 'crackers'}
    mock_sys_mgr = MagicMock(return_value=MockSystemManager())
    with patch.object(cassandra, '_sys_mgr', mock_sys_mgr):
        assert cassandra.column_families('Z') is None
        assert cassandra.column_families('A') == ['a', 'b']
        assert cassandra.column_families() == {'A': ['a', 'b'], 'B': ['c', 'd']}

def test_column_family_definition():
    if False:
        while True:
            i = 10
    '\n    Test for Return a dictionary of column family definitions for the given\n    keyspace/column_family\n    '

    class MockSystemManager:

        def get_keyspace_column_families(self, keyspace):
            if False:
                for i in range(10):
                    print('nop')
            if keyspace == 'A':
                return {'a': object, 'b': object}
            if keyspace == 'B':
                raise Exception
    mock_sys_mgr = MagicMock(return_value=MockSystemManager())
    with patch.object(cassandra, '_sys_mgr', mock_sys_mgr):
        assert cassandra.column_family_definition('A', 'a') == vars(object)
        assert cassandra.column_family_definition('B', 'a') is None