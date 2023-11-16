import os
import unittest
from unittest.mock import patch
from embedchain import App
from embedchain.config import AppConfig, ElasticsearchDBConfig
from embedchain.embedder.gpt4all import GPT4AllEmbedder
from embedchain.vectordb.elasticsearch import ElasticsearchDB

class TestEsDB(unittest.TestCase):

    @patch('embedchain.vectordb.elasticsearch.Elasticsearch')
    def test_setUp(self, mock_client):
        if False:
            for i in range(10):
                print('nop')
        self.db = ElasticsearchDB(config=ElasticsearchDBConfig(es_url='https://localhost:9200'))
        self.vector_dim = 384
        app_config = AppConfig(collection_name=False, collect_metrics=False)
        self.app = App(config=app_config, db=self.db)
        self.assertEqual(self.db.client, mock_client.return_value)

    @patch('embedchain.vectordb.elasticsearch.Elasticsearch')
    def test_query(self, mock_client):
        if False:
            while True:
                i = 10
        self.db = ElasticsearchDB(config=ElasticsearchDBConfig(es_url='https://localhost:9200'))
        app_config = AppConfig(collection_name=False, collect_metrics=False)
        self.app = App(config=app_config, db=self.db, embedder=GPT4AllEmbedder())
        self.assertEqual(self.db.client, mock_client.return_value)
        embeddings = [[1, 2, 3], [4, 5, 6]]
        documents = ['This is a document.', 'This is another document.']
        metadatas = [{'url': 'url_1', 'doc_id': 'doc_id_1'}, {'url': 'url_2', 'doc_id': 'doc_id_2'}]
        ids = ['doc_1', 'doc_2']
        self.db.add(embeddings, documents, metadatas, ids, skip_embedding=False)
        search_response = {'hits': {'hits': [{'_source': {'text': 'This is a document.', 'metadata': {'url': 'url_1', 'doc_id': 'doc_id_1'}}, '_score': 0.9}, {'_source': {'text': 'This is another document.', 'metadata': {'url': 'url_2', 'doc_id': 'doc_id_2'}}, '_score': 0.8}]}}
        mock_client.return_value.search.return_value = search_response
        query = ['This is a document']
        results_without_citations = self.db.query(query, n_results=2, where={}, skip_embedding=False)
        expected_results_without_citations = ['This is a document.', 'This is another document.']
        self.assertEqual(results_without_citations, expected_results_without_citations)
        results_with_citations = self.db.query(query, n_results=2, where={}, skip_embedding=False, citations=True)
        expected_results_with_citations = [('This is a document.', 'url_1', 'doc_id_1'), ('This is another document.', 'url_2', 'doc_id_2')]
        self.assertEqual(results_with_citations, expected_results_with_citations)

    @patch('embedchain.vectordb.elasticsearch.Elasticsearch')
    def test_query_with_skip_embedding(self, mock_client):
        if False:
            return 10
        self.db = ElasticsearchDB(config=ElasticsearchDBConfig(es_url='https://localhost:9200'))
        app_config = AppConfig(collection_name=False, collect_metrics=False)
        self.app = App(config=app_config, db=self.db)
        self.assertEqual(self.db.client, mock_client.return_value)
        embeddings = [[1, 2, 3], [4, 5, 6]]
        documents = ['This is a document.', 'This is another document.']
        metadatas = [{'url': 'url_1', 'doc_id': 'doc_id_1'}, {'url': 'url_2', 'doc_id': 'doc_id_2'}]
        ids = ['doc_1', 'doc_2']
        self.db.add(embeddings, documents, metadatas, ids, skip_embedding=True)
        search_response = {'hits': {'hits': [{'_source': {'text': 'This is a document.', 'metadata': {'url': 'url_1', 'doc_id': 'doc_id_1'}}, '_score': 0.9}, {'_source': {'text': 'This is another document.', 'metadata': {'url': 'url_2', 'doc_id': 'doc_id_2'}}, '_score': 0.8}]}}
        mock_client.return_value.search.return_value = search_response
        query = ['This is a document']
        results = self.db.query(query, n_results=2, where={}, skip_embedding=True)
        self.assertEqual(results, ['This is a document.', 'This is another document.'])

    def test_init_without_url(self):
        if False:
            return 10
        try:
            del os.environ['ELASTICSEARCH_URL']
        except KeyError:
            pass
        with self.assertRaises(AttributeError):
            ElasticsearchDB()

    def test_init_with_invalid_es_config(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            ElasticsearchDB(es_config={'ES_URL': 'some_url', 'valid es_config': False})