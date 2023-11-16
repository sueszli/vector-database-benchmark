from typing import Optional
from unittest.mock import MagicMock
import numpy as np
import pytest
from haystack.document_stores.search_engine import SearchEngineDocumentStore
from haystack.schema import FilterType

@pytest.mark.unit
def test_prepare_hosts():
    if False:
        return 10
    pass

@pytest.mark.document_store
class SearchEngineDocumentStoreTestAbstract:
    """
    This is the base class for any Searchengine Document Store testsuite, it doesn't have the `Test` prefix in the name
    because we want to run its methods only in subclasses.
    """

    @pytest.fixture
    def mocked_get_all_documents_in_index(self, monkeypatch):
        if False:
            print('Hello World!')
        method_mock = MagicMock(return_value=None)
        monkeypatch.setattr(SearchEngineDocumentStore, '_get_all_documents_in_index', method_mock)
        return method_mock
    query = 'test'

    @pytest.mark.integration
    def test___do_bulk(self):
        if False:
            return 10
        pass

    @pytest.mark.integration
    def test___do_scan(self):
        if False:
            i = 10
            return i + 15
        pass

    @pytest.mark.integration
    def test_query_by_embedding(self):
        if False:
            return 10
        pass

    @pytest.mark.integration
    def test_get_meta_values_by_key(self, ds, documents):
        if False:
            while True:
                i = 10
        ds.write_documents(documents)
        result = ds.get_metadata_values_by_key(key='name')
        assert result == [{'count': 3, 'value': 'name_0'}, {'count': 3, 'value': 'name_1'}, {'count': 3, 'value': 'name_2'}]
        result = ds.get_metadata_values_by_key(key='year', filters={'month': ['01']})
        assert result == [{'count': 3, 'value': '2020'}]
        result = ds.get_metadata_values_by_key(key='year', query='Bar')
        assert result == [{'count': 3, 'value': '2021'}]

    @pytest.mark.unit
    def test_query_return_embedding_true(self, mocked_document_store):
        if False:
            print('Hello World!')
        mocked_document_store.return_embedding = True
        mocked_document_store.query(self.query)
        (_, kwargs) = mocked_document_store.client.search.call_args
        assert '_source' not in kwargs

    @pytest.mark.unit
    def test_query_return_embedding_false(self, mocked_document_store):
        if False:
            return 10
        mocked_document_store.return_embedding = False
        mocked_document_store.query(self.query)
        (_, kwargs) = mocked_document_store.client.search.call_args
        assert kwargs['_source'] == {'excludes': ['embedding']}

    @pytest.mark.unit
    def test_query_excluded_meta_data_return_embedding_true(self, mocked_document_store):
        if False:
            print('Hello World!')
        mocked_document_store.return_embedding = True
        mocked_document_store.excluded_meta_data = ['foo', 'embedding']
        mocked_document_store.query(self.query)
        (_, kwargs) = mocked_document_store.client.search.call_args
        assert kwargs['_source'] == {'excludes': ['foo']}

    @pytest.mark.unit
    def test_query_excluded_meta_data_return_embedding_false(self, mocked_document_store):
        if False:
            return 10
        mocked_document_store.return_embedding = False
        mocked_document_store.excluded_meta_data = ['foo']
        mocked_document_store.query(self.query)
        (_, kwargs) = mocked_document_store.client.search.call_args
        assert kwargs['_source'] == {'excludes': ['foo', 'embedding']}

    @pytest.mark.unit
    def test_get_all_documents_return_embedding_true(self, mocked_document_store):
        if False:
            return 10
        mocked_document_store.return_embedding = False
        mocked_document_store.client.search.return_value = {}
        mocked_document_store.get_all_documents(return_embedding=True)
        (_, kwargs) = mocked_document_store.client.search.call_args
        assert '_source' not in kwargs

    @pytest.mark.unit
    def test_get_all_documents_return_embedding_false(self, mocked_document_store):
        if False:
            while True:
                i = 10
        mocked_document_store.return_embedding = True
        mocked_document_store.client.search.return_value = {}
        mocked_document_store.get_all_documents(return_embedding=False)
        (_, kwargs) = mocked_document_store.client.search.call_args
        body = kwargs.get('body', kwargs)
        assert body['_source'] == {'excludes': ['embedding']}

    @pytest.mark.unit
    def test_get_all_documents_excluded_meta_data_has_no_influence(self, mocked_document_store):
        if False:
            while True:
                i = 10
        mocked_document_store.excluded_meta_data = ['foo']
        mocked_document_store.client.search.return_value = {}
        mocked_document_store.get_all_documents(return_embedding=False)
        (_, kwargs) = mocked_document_store.client.search.call_args
        body = kwargs.get('body', kwargs)
        assert body['_source'] == {'excludes': ['embedding']}

    @pytest.mark.unit
    def test_get_document_by_id_return_embedding_true(self, mocked_document_store):
        if False:
            while True:
                i = 10
        mocked_document_store.return_embedding = True
        mocked_document_store.get_document_by_id('123')
        (_, kwargs) = mocked_document_store.client.search.call_args
        assert '_source' not in kwargs

    @pytest.mark.unit
    def test_get_all_labels_legacy_document_id(self, mocked_document_store, mocked_get_all_documents_in_index):
        if False:
            print('Hello World!')
        mocked_get_all_documents_in_index.return_value = [{'_id': '123', '_source': {'query': 'Who made the PDF specification?', 'document': {'content': 'Some content', 'content_type': 'text', 'score': None, 'id': 'fc18c987a8312e72a47fb1524f230bb0', 'meta': {}, 'embedding': [0.1, 0.2, 0.3]}, 'answer': {'answer': 'Adobe Systems', 'type': 'extractive', 'context': 'Some content', 'offsets_in_context': [{'start': 60, 'end': 73}], 'offsets_in_document': [{'start': 60, 'end': 73}], 'document_id': 'fc18c987a8312e72a47fb1524f230bb0', 'meta': {}, 'score': None}, 'is_correct_answer': True, 'is_correct_document': True, 'origin': 'user-feedback', 'pipeline_id': 'some-123'}}]
        labels = mocked_document_store.get_all_labels()
        assert labels[0].answer.document_ids == ['fc18c987a8312e72a47fb1524f230bb0']

    @pytest.mark.unit
    def test_query_batch_req_for_each_batch(self, mocked_document_store):
        if False:
            print('Hello World!')
        mocked_document_store.batch_size = 2
        mocked_document_store.query_batch([self.query] * 3)
        assert mocked_document_store.client.msearch.call_count == 2

    @pytest.mark.unit
    def test_query_by_embedding_batch_req_for_each_batch(self, mocked_document_store):
        if False:
            print('Hello World!')
        mocked_document_store.batch_size = 2
        mocked_document_store.query_by_embedding_batch([np.array([1, 2, 3])] * 3)
        assert mocked_document_store.client.msearch.call_count == 2

    @pytest.mark.integration
    def test_document_with_version_metadata(self, ds: SearchEngineDocumentStore):
        if False:
            print('Hello World!')
        ds.write_documents([{'content': 'test', 'meta': {'version': '2023.1'}}])
        documents = ds.get_all_documents()
        assert documents[0].meta['version'] == '2023.1'

    @pytest.mark.integration
    def test_label_with_version_metadata(self, ds: SearchEngineDocumentStore):
        if False:
            return 10
        ds.write_labels([{'query': 'test', 'document': {'content': 'test'}, 'is_correct_answer': True, 'is_correct_document': True, 'origin': 'gold-label', 'meta': {'version': '2023.1'}, 'answer': None}])
        labels = ds.get_all_labels()
        assert labels[0].meta['version'] == '2023.1'

    @pytest.mark.integration
    @pytest.mark.parametrize('query,filters,result_count', [('tost', {'year': ['2020', '2021', '1990']}, 4), ('test', None, 5), ('test\n', {'year': '2021'}, 3), ('test"', {'year': '2021'}, 3), ('toast', None, 0)])
    def test_custom_query(self, query: str, filters: Optional[FilterType], result_count: int, ds: SearchEngineDocumentStore):
        if False:
            print('Hello World!')
        documents = [{'id': '1', 'content': 'test', 'meta': {'year': '2019'}}, {'id': '2', 'content': 'test', 'meta': {'year': '2020'}}, {'id': '3', 'content': 'test', 'meta': {'year': '2021'}}, {'id': '4', 'content': 'test', 'meta': {'year': '2021'}}, {'id': '5', 'content': 'test', 'meta': {'year': '2021'}}]
        ds.write_documents(documents)
        custom_query = '\n            {\n                "query": {\n                    "bool": {\n                        "must": [{\n                            "multi_match": {\n                                "query": ${query},\n                                "fields": ["content"],\n                                "fuzziness": "AUTO"\n                            }\n                        }],\n                        "filter": ${filters}\n                    }\n                }\n            }\n        '
        results = ds.query(query=query, filters=filters, custom_query=custom_query)
        assert len(results) == result_count

@pytest.mark.document_store
class TestSearchEngineDocumentStore:
    """
    This class tests the concrete methods in SearchEngineDocumentStore
    """

    @pytest.mark.integration
    def test__split_document_list(self):
        if False:
            for i in range(10):
                print('nop')
        pass