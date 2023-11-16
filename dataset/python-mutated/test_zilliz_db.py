import os
from unittest import mock
from unittest.mock import Mock, patch
import pytest
from embedchain.config import ZillizDBConfig
from embedchain.vectordb.zilliz import ZillizVectorDB

class TestZillizVectorDBConfig:

    @mock.patch.dict(os.environ, {'ZILLIZ_CLOUD_URI': 'mocked_uri', 'ZILLIZ_CLOUD_TOKEN': 'mocked_token'})
    def test_init_with_uri_and_token(self):
        if False:
            i = 10
            return i + 15
        '\n        Test if the `ZillizVectorDBConfig` instance is initialized with the correct uri and token values.\n        '
        expected_uri = 'mocked_uri'
        expected_token = 'mocked_token'
        db_config = ZillizDBConfig()
        assert db_config.uri == expected_uri
        assert db_config.token == expected_token

    @mock.patch.dict(os.environ, {'ZILLIZ_CLOUD_URI': 'mocked_uri', 'ZILLIZ_CLOUD_TOKEN': 'mocked_token'})
    def test_init_without_uri(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if the `ZillizVectorDBConfig` instance throws an error when no URI found.\n        '
        try:
            del os.environ['ZILLIZ_CLOUD_URI']
        except KeyError:
            pass
        with pytest.raises(AttributeError):
            ZillizDBConfig()

    @mock.patch.dict(os.environ, {'ZILLIZ_CLOUD_URI': 'mocked_uri', 'ZILLIZ_CLOUD_TOKEN': 'mocked_token'})
    def test_init_without_token(self):
        if False:
            return 10
        '\n        Test if the `ZillizVectorDBConfig` instance throws an error when no Token found.\n        '
        try:
            del os.environ['ZILLIZ_CLOUD_TOKEN']
        except KeyError:
            pass
        with pytest.raises(AttributeError):
            ZillizDBConfig()

class TestZillizVectorDB:

    @pytest.fixture
    @mock.patch.dict(os.environ, {'ZILLIZ_CLOUD_URI': 'mocked_uri', 'ZILLIZ_CLOUD_TOKEN': 'mocked_token'})
    def mock_config(self, mocker):
        if False:
            i = 10
            return i + 15
        return mocker.Mock(spec=ZillizDBConfig())

    @patch('embedchain.vectordb.zilliz.MilvusClient', autospec=True)
    @patch('embedchain.vectordb.zilliz.connections.connect', autospec=True)
    def test_zilliz_vector_db_setup(self, mock_connect, mock_client, mock_config):
        if False:
            return 10
        '\n        Test if the `ZillizVectorDB` instance is initialized with the correct uri and token values.\n        '
        ZillizVectorDB(config=mock_config)
        mock_client.assert_called_once_with(uri=mock_config.uri, token=mock_config.token)
        mock_connect.assert_called_once_with(uri=mock_config.uri, token=mock_config.token)

class TestZillizDBCollection:

    @pytest.fixture
    @mock.patch.dict(os.environ, {'ZILLIZ_CLOUD_URI': 'mocked_uri', 'ZILLIZ_CLOUD_TOKEN': 'mocked_token'})
    def mock_config(self, mocker):
        if False:
            while True:
                i = 10
        return mocker.Mock(spec=ZillizDBConfig())

    @pytest.fixture
    def mock_embedder(self, mocker):
        if False:
            i = 10
            return i + 15
        return mocker.Mock()

    @mock.patch.dict(os.environ, {'ZILLIZ_CLOUD_URI': 'mocked_uri', 'ZILLIZ_CLOUD_TOKEN': 'mocked_token'})
    def test_init_with_default_collection(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if the `ZillizVectorDB` instance is initialized with the correct default collection name.\n        '
        db_config = ZillizDBConfig()
        assert db_config.collection_name == 'embedchain_store'

    @mock.patch.dict(os.environ, {'ZILLIZ_CLOUD_URI': 'mocked_uri', 'ZILLIZ_CLOUD_TOKEN': 'mocked_token'})
    def test_init_with_custom_collection(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if the `ZillizVectorDB` instance is initialized with the correct custom collection name.\n        '
        expected_collection = 'test_collection'
        db_config = ZillizDBConfig(collection_name='test_collection')
        assert db_config.collection_name == expected_collection

    @patch('embedchain.vectordb.zilliz.MilvusClient', autospec=True)
    @patch('embedchain.vectordb.zilliz.connections', autospec=True)
    def test_query_with_skip_embedding(self, mock_connect, mock_client, mock_config):
        if False:
            print('Hello World!')
        '\n        Test if the `ZillizVectorDB` instance is takes in the query with skip_embeddings.\n        '
        zilliz_db = ZillizVectorDB(config=mock_config)
        zilliz_db.collection = Mock(is_empty=False)
        assert zilliz_db.client == mock_client()
        with patch.object(zilliz_db.client, 'search') as mock_search:
            mock_search.return_value = [[{'entity': {'text': 'result_doc', 'url': 'url_1', 'doc_id': 'doc_id_1'}}]]
            query_result = zilliz_db.query(input_query=['query_text'], n_results=1, where={}, skip_embedding=True)
            mock_search.assert_called_with(collection_name=mock_config.collection_name, data=['query_text'], limit=1, output_fields=['text', 'url', 'doc_id'])
            assert query_result == ['result_doc']
            query_result_with_citations = zilliz_db.query(input_query=['query_text'], n_results=1, where={}, skip_embedding=True, citations=True)
            mock_search.assert_called_with(collection_name=mock_config.collection_name, data=['query_text'], limit=1, output_fields=['text', 'url', 'doc_id'])
            assert query_result_with_citations == [('result_doc', 'url_1', 'doc_id_1')]

    @patch('embedchain.vectordb.zilliz.MilvusClient', autospec=True)
    @patch('embedchain.vectordb.zilliz.connections', autospec=True)
    def test_query_without_skip_embedding(self, mock_connect, mock_client, mock_embedder, mock_config):
        if False:
            while True:
                i = 10
        '\n        Test if the `ZillizVectorDB` instance is takes in the query without skip_embeddings.\n        '
        zilliz_db = ZillizVectorDB(config=mock_config)
        zilliz_db.embedder = mock_embedder
        zilliz_db.collection = Mock(is_empty=False)
        assert zilliz_db.client == mock_client()
        with patch.object(zilliz_db.client, 'search') as mock_search:
            mock_embedder.embedding_fn.return_value = ['query_vector']
            mock_search.return_value = [[{'entity': {'text': 'result_doc', 'url': 'url_1', 'doc_id': 'doc_id_1'}}]]
            query_result = zilliz_db.query(input_query=['query_text'], n_results=1, where={}, skip_embedding=False)
            mock_search.assert_called_with(collection_name=mock_config.collection_name, data=['query_vector'], limit=1, output_fields=['text', 'url', 'doc_id'])
            assert query_result == ['result_doc']
            query_result_with_citations = zilliz_db.query(input_query=['query_text'], n_results=1, where={}, skip_embedding=False, citations=True)
            mock_search.assert_called_with(collection_name=mock_config.collection_name, data=['query_vector'], limit=1, output_fields=['text', 'url', 'doc_id'])
            assert query_result_with_citations == [('result_doc', 'url_1', 'doc_id_1')]