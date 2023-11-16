"""
Test case for the cassandra_cql module
"""
import logging
import ssl
import pytest
import salt.modules.cassandra_cql as cassandra_cql
from salt.exceptions import CommandExecutionError
from tests.support.mock import MagicMock, patch
log = logging.getLogger(__name__)
pytestmark = pytest.mark.skipif(not cassandra_cql.HAS_DRIVER, reason='Cassandra CQL driver not loaded')

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {cassandra_cql: {}}

def test_cql_query(caplog):
    if False:
        print('Hello World!')
    '\n    Test salt.modules.cassandra_cql.cql_query function\n    '
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock = MagicMock(return_value=(mock_session, mock_client))
    query = 'query'
    with patch.object(cassandra_cql, '_connect', mock):
        query_result = cassandra_cql.cql_query(query)
    assert query_result == []
    query = {'5.0.1': 'query1', '5.0.0': 'query2'}
    mock_version = MagicMock(return_value='5.0.1')
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock = MagicMock(return_value=(mock_session, mock_client))
    with patch.object(cassandra_cql, 'version', mock_version):
        with patch.object(cassandra_cql, '_connect', mock):
            query_result = cassandra_cql.cql_query(query)
    assert query_result == []

def test_cql_query_with_prepare(caplog):
    if False:
        print('Hello World!')
    '\n    Test salt.modules.cassandra_cql.cql_query_with_prepare function\n    '
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock = MagicMock(return_value=(mock_session, mock_client))
    query = 'query'
    statement_args = {'arg1': 'test'}
    mock_context = MagicMock(return_value={'cassandra_cql_prepared': {'statement_name': query}})
    with patch.object(cassandra_cql, '__context__', mock_context):
        with patch.object(cassandra_cql, '_connect', mock):
            query_result = cassandra_cql.cql_query_with_prepare(query, 'statement_name', statement_args)
    assert query_result == []

def test_version(caplog):
    if False:
        print('Hello World!')
    '\n    Test salt.modules.cassandra_cql.version function\n    '
    mock_cql_query = MagicMock(return_value=[{'release_version': '5.0.1'}])
    with patch.object(cassandra_cql, 'cql_query', mock_cql_query):
        version = cassandra_cql.version()
    assert version == '5.0.1'
    mock_cql_query = MagicMock(side_effect=CommandExecutionError)
    with pytest.raises(CommandExecutionError) as err:
        with patch.object(cassandra_cql, 'cql_query', mock_cql_query):
            version = cassandra_cql.version()
    assert '{}'.format(err.value) == ''
    assert 'Could not get Cassandra version.' in caplog.text
    for record in caplog.records:
        assert record.levelname == 'CRITICAL'

def test_info():
    if False:
        print('Hello World!')
    '\n    Test salt.modules.cassandra_cql.info function\n    '
    expected = {'result': 'info'}
    mock_cql_query = MagicMock(return_value=expected)
    with patch.object(cassandra_cql, 'cql_query', mock_cql_query):
        info = cassandra_cql.info()
    assert info == expected

def test_list_keyspaces():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test salt.modules.cassandra_cql.list_keyspaces function\n    '
    expected = [{'keyspace_name': 'name1'}, {'keyspace_name': 'name2'}]
    mock_cql_query = MagicMock(return_value=expected)
    with patch.object(cassandra_cql, 'cql_query', mock_cql_query):
        keyspaces = cassandra_cql.list_keyspaces()
    assert keyspaces == expected

def test_list_column_families():
    if False:
        i = 10
        return i + 15
    '\n    Test salt.modules.cassandra_cql.list_column_families function\n    '
    expected = [{'colum_name': 'column1'}, {'column_name': 'column2'}]
    mock_cql_query = MagicMock(return_value=expected)
    with patch.object(cassandra_cql, 'cql_query', mock_cql_query):
        columns = cassandra_cql.list_column_families()
    assert columns == expected

def test_keyspace_exists():
    if False:
        i = 10
        return i + 15
    '\n    Test salt.modules.cassandra_cql.keyspace_exists function\n    '
    expected = 'keyspace'
    mock_cql_query = MagicMock(return_value=expected)
    with patch.object(cassandra_cql, 'cql_query', mock_cql_query):
        exists = cassandra_cql.keyspace_exists('keyspace')
    assert exists == bool(expected)
    expected = []
    mock_cql_query = MagicMock(return_value=expected)
    with patch.object(cassandra_cql, 'cql_query', mock_cql_query):
        exists = cassandra_cql.keyspace_exists('keyspace')
    assert exists == bool(expected)

def test_create_keyspace():
    if False:
        while True:
            i = 10
    '\n    Test salt.modules.cassandra_cql.create_keyspace function\n    '
    expected = None
    mock_cql_query = MagicMock(return_value=expected)
    with patch.object(cassandra_cql, 'cql_query', mock_cql_query):
        result = cassandra_cql.create_keyspace('keyspace')
    assert result == expected

def test_drop_keyspace():
    if False:
        i = 10
        return i + 15
    '\n    Test salt.modules.cassandra_cql.drop_keyspace function\n    '
    expected = True
    mock_cql_query = MagicMock(return_value=expected)
    with patch.object(cassandra_cql, 'cql_query', mock_cql_query):
        result = cassandra_cql.drop_keyspace('keyspace')
    assert result == expected

def test_list_users():
    if False:
        return 10
    '\n    Test salt.modules.cassandra_cql.list_users function\n    '
    expected = [{'name': 'user1', 'super': True}, {'name': 'user2', 'super': False}]
    mock_cql_query = MagicMock(return_value=expected)
    with patch.object(cassandra_cql, 'cql_query', mock_cql_query):
        result = cassandra_cql.list_users()
    assert result == expected

def test_create_user():
    if False:
        print('Hello World!')
    '\n    Test salt.modules.cassandra_cql.create_user function\n    '
    expected = True
    mock_cql_query = MagicMock(return_value=expected)
    with patch.object(cassandra_cql, 'cql_query', mock_cql_query):
        result = cassandra_cql.create_user('user', 'password')
    assert result == expected

def test_list_permissions():
    if False:
        i = 10
        return i + 15
    '\n    Test salt.modules.cassandra_cql.list_permissions function\n    '
    expected = [{'permission': 'ALTER', 'resource': '<keyspace one>', 'role': 'user1', 'username': 'user1'}]
    mock_cql_query = MagicMock(return_value=expected)
    with patch.object(cassandra_cql, 'cql_query', mock_cql_query):
        result = cassandra_cql.list_permissions(username='user1', resource='one')
    assert result == expected

def test_grant_permission():
    if False:
        i = 10
        return i + 15
    '\n    Test salt.modules.cassandra_cql.grant_permission function\n    '
    expected = True
    mock_cql_query = MagicMock(return_value=expected)
    with patch.object(cassandra_cql, 'cql_query', mock_cql_query):
        result = cassandra_cql.grant_permission(username='user1', resource='one')
    assert result == expected

def test_returns_opts_if_specified():
    if False:
        print('Hello World!')
    '\n    If ssl options are present then check that they are parsed and returned\n    '
    options = MagicMock(return_value={'cluster': ['192.168.50.10', '192.168.50.11', '192.168.50.12'], 'port': 9000, 'ssl_options': {'ca_certs': '/etc/ssl/certs/ca-bundle.trust.crt', 'ssl_version': 'PROTOCOL_TLSv1'}, 'username': 'cas_admin'})
    with patch.dict(cassandra_cql.__salt__, {'config.option': options}):
        assert cassandra_cql._get_ssl_opts() == {'ca_certs': '/etc/ssl/certs/ca-bundle.trust.crt', 'ssl_version': ssl.PROTOCOL_TLSv1}

def test_invalid_protocol_version():
    if False:
        i = 10
        return i + 15
    '\n    Check that the protocol version is imported only if it isvalid\n    '
    options = MagicMock(return_value={'cluster': ['192.168.50.10', '192.168.50.11', '192.168.50.12'], 'port': 9000, 'ssl_options': {'ca_certs': '/etc/ssl/certs/ca-bundle.trust.crt', 'ssl_version': 'Invalid'}, 'username': 'cas_admin'})
    with patch.dict(cassandra_cql.__salt__, {'config.option': options}):
        with pytest.raises(CommandExecutionError):
            cassandra_cql._get_ssl_opts()

def test_unspecified_opts():
    if False:
        return 10
    "\n    Check that it returns None when ssl opts aren't specified\n    "
    with patch.dict(cassandra_cql.__salt__, {'config.option': MagicMock(return_value={})}):
        assert cassandra_cql._get_ssl_opts() is None

def test_valid_asynchronous_args():
    if False:
        print('Hello World!')
    mock_execute = MagicMock(return_value={})
    mock_execute_async = MagicMock(return_value={})
    mock_context = {'cassandra_cql_returner_cluster': MagicMock(return_value={}), 'cassandra_cql_returner_session': MagicMock(execute=mock_execute, execute_async=mock_execute_async, prepare=lambda _: MagicMock(bind=lambda _: None), row_factory=None), 'cassandra_cql_prepared': {}}
    with patch.dict(cassandra_cql.__context__, mock_context):
        cassandra_cql.cql_query_with_prepare('SELECT now() from system.local;', 'select_now', [], asynchronous=True)
        mock_execute_async.assert_called_once()

def test_valid_async_args():
    if False:
        return 10
    mock_execute = MagicMock(return_value={})
    mock_execute_async = MagicMock(return_value={})
    mock_context = {'cassandra_cql_returner_cluster': MagicMock(return_value={}), 'cassandra_cql_returner_session': MagicMock(execute=mock_execute, execute_async=mock_execute_async, prepare=lambda _: MagicMock(bind=lambda _: None), row_factory=None), 'cassandra_cql_prepared': {}}
    kwargs = {'async': True}
    with patch.dict(cassandra_cql.__context__, mock_context):
        cassandra_cql.cql_query_with_prepare('SELECT now() from system.local;', 'select_now', [], **kwargs)
        mock_execute_async.assert_called_once()