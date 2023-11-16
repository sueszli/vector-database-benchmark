from unittest.mock import Mock, call, patch
import pytest
from celery import states
from celery.backends import cosmosdbsql
from celery.backends.cosmosdbsql import CosmosDBSQLBackend
from celery.exceptions import ImproperlyConfigured
MODULE_TO_MOCK = 'celery.backends.cosmosdbsql'
pytest.importorskip('pydocumentdb')

class test_DocumentDBBackend:

    def setup_method(self):
        if False:
            return 10
        self.url = 'cosmosdbsql://:key@endpoint'
        self.backend = CosmosDBSQLBackend(app=self.app, url=self.url)

    def test_missing_third_party_sdk(self):
        if False:
            i = 10
            return i + 15
        pydocumentdb = cosmosdbsql.pydocumentdb
        try:
            cosmosdbsql.pydocumentdb = None
            with pytest.raises(ImproperlyConfigured):
                CosmosDBSQLBackend(app=self.app, url=self.url)
        finally:
            cosmosdbsql.pydocumentdb = pydocumentdb

    def test_bad_connection_url(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ImproperlyConfigured):
            CosmosDBSQLBackend._parse_url('cosmosdbsql://:key@')
        with pytest.raises(ImproperlyConfigured):
            CosmosDBSQLBackend._parse_url('cosmosdbsql://:@host')
        with pytest.raises(ImproperlyConfigured):
            CosmosDBSQLBackend._parse_url('cosmosdbsql://corrupted')

    def test_default_connection_url(self):
        if False:
            print('Hello World!')
        (endpoint, password) = CosmosDBSQLBackend._parse_url('cosmosdbsql://:key@host')
        assert password == 'key'
        assert endpoint == 'https://host:443'
        (endpoint, password) = CosmosDBSQLBackend._parse_url('cosmosdbsql://:key@host:443')
        assert password == 'key'
        assert endpoint == 'https://host:443'
        (endpoint, password) = CosmosDBSQLBackend._parse_url('cosmosdbsql://:key@host:8080')
        assert password == 'key'
        assert endpoint == 'http://host:8080'

    def test_bad_partition_key(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError):
            CosmosDBSQLBackend._get_partition_key('')
        with pytest.raises(ValueError):
            CosmosDBSQLBackend._get_partition_key('   ')
        with pytest.raises(ValueError):
            CosmosDBSQLBackend._get_partition_key(None)

    def test_bad_consistency_level(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ImproperlyConfigured):
            CosmosDBSQLBackend(app=self.app, url=self.url, consistency_level='DoesNotExist')

    @patch(MODULE_TO_MOCK + '.DocumentClient')
    def test_create_client(self, mock_factory):
        if False:
            for i in range(10):
                print('nop')
        mock_instance = Mock()
        mock_factory.return_value = mock_instance
        backend = CosmosDBSQLBackend(app=self.app, url=self.url)
        assert mock_instance.CreateDatabase.call_count == 0
        assert mock_instance.CreateCollection.call_count == 0
        assert backend._client is not None
        assert mock_instance.CreateDatabase.call_count == 1
        assert mock_instance.CreateCollection.call_count == 1
        assert backend._client is not None
        assert mock_instance.CreateDatabase.call_count == 1
        assert mock_instance.CreateCollection.call_count == 1

    @patch(MODULE_TO_MOCK + '.CosmosDBSQLBackend._client')
    def test_get(self, mock_client):
        if False:
            print('Hello World!')
        self.backend.get(b'mykey')
        mock_client.ReadDocument.assert_has_calls([call('dbs/celerydb/colls/celerycol/docs/mykey', {'partitionKey': 'mykey'}), call().get('value')])

    @patch(MODULE_TO_MOCK + '.CosmosDBSQLBackend._client')
    def test_get_missing(self, mock_client):
        if False:
            i = 10
            return i + 15
        mock_client.ReadDocument.side_effect = cosmosdbsql.HTTPFailure(cosmosdbsql.ERROR_NOT_FOUND)
        assert self.backend.get(b'mykey') is None

    @patch(MODULE_TO_MOCK + '.CosmosDBSQLBackend._client')
    def test_set(self, mock_client):
        if False:
            while True:
                i = 10
        self.backend._set_with_state(b'mykey', 'myvalue', states.SUCCESS)
        mock_client.CreateDocument.assert_called_once_with('dbs/celerydb/colls/celerycol', {'id': 'mykey', 'value': 'myvalue'}, {'partitionKey': 'mykey'})

    @patch(MODULE_TO_MOCK + '.CosmosDBSQLBackend._client')
    def test_mget(self, mock_client):
        if False:
            return 10
        keys = [b'mykey1', b'mykey2']
        self.backend.mget(keys)
        mock_client.ReadDocument.assert_has_calls([call('dbs/celerydb/colls/celerycol/docs/mykey1', {'partitionKey': 'mykey1'}), call().get('value'), call('dbs/celerydb/colls/celerycol/docs/mykey2', {'partitionKey': 'mykey2'}), call().get('value')])

    @patch(MODULE_TO_MOCK + '.CosmosDBSQLBackend._client')
    def test_delete(self, mock_client):
        if False:
            print('Hello World!')
        self.backend.delete(b'mykey')
        mock_client.DeleteDocument.assert_called_once_with('dbs/celerydb/colls/celerycol/docs/mykey', {'partitionKey': 'mykey'})