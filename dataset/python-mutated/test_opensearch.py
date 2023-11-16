from __future__ import annotations
import pytest
from airflow.providers.opensearch.hooks.opensearch import OpenSearchHook
pytestmark = pytest.mark.db_test
MOCK_SEARCH_RETURN = {'status': 'test'}

class TestOpenSearchHook:

    def test_hook_search(self, mock_hook):
        if False:
            return 10
        self.hook = OpenSearchHook(open_search_conn_id='opensearch_default', log_query=True)
        result = self.hook.search(index_name='testIndex', query={'size': 1, 'query': {'multi_match': {'query': 'test', 'fields': ['testField']}}})
        assert result == MOCK_SEARCH_RETURN

    def test_hook_index(self, mock_hook):
        if False:
            while True:
                i = 10
        self.hook = OpenSearchHook(open_search_conn_id='opensearch_default', log_query=True)
        result = self.hook.index(index_name='test_index', document={'title': 'Monty Python'}, doc_id=3)
        assert result == 3