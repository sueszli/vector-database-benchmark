import pytest
from superset.extensions import cache_manager
from superset.utils.core import backend, DatasourceType
from tests.integration_tests.base_tests import SupersetTestCase

class UtilsCacheManagerTests(SupersetTestCase):

    def test_get_set_explore_form_data_cache(self):
        if False:
            return 10
        key = '12345'
        data = {'foo': 'bar', 'datasource_type': 'query'}
        cache_manager.explore_form_data_cache.set(key, data)
        assert cache_manager.explore_form_data_cache.get(key) == data

    def test_get_same_context_twice(self):
        if False:
            for i in range(10):
                print('nop')
        key = '12345'
        data = {'foo': 'bar', 'datasource_type': 'query'}
        cache_manager.explore_form_data_cache.set(key, data)
        assert cache_manager.explore_form_data_cache.get(key) == data
        assert cache_manager.explore_form_data_cache.get(key) == data

    def test_get_set_explore_form_data_cache_no_datasource_type(self):
        if False:
            i = 10
            return i + 15
        key = '12345'
        data = {'foo': 'bar'}
        cache_manager.explore_form_data_cache.set(key, data)
        assert cache_manager.explore_form_data_cache.get(key) == {'datasource_type': DatasourceType.TABLE, **data}

    def test_get_explore_form_data_cache_invalid_key(self):
        if False:
            i = 10
            return i + 15
        assert cache_manager.explore_form_data_cache.get('foo') == None