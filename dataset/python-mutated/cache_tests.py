"""Unit tests for Superset with caching"""
import json
import pytest
from superset import app, db
from superset.common.db_query_status import QueryStatus
from superset.extensions import cache_manager
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from .base_tests import SupersetTestCase

class TestCache(SupersetTestCase):

    def setUp(self):
        if False:
            return 10
        self.login(username='admin')
        cache_manager.cache.clear()
        cache_manager.data_cache.clear()

    def tearDown(self):
        if False:
            return 10
        cache_manager.cache.clear()
        cache_manager.data_cache.clear()

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_no_data_cache(self):
        if False:
            for i in range(10):
                print('nop')
        data_cache_config = app.config['DATA_CACHE_CONFIG']
        app.config['DATA_CACHE_CONFIG'] = {'CACHE_TYPE': 'NullCache'}
        cache_manager.init_app(app)
        slc = self.get_slice('Top 10 Girl Name Share', db.session)
        json_endpoint = '/superset/explore_json/{}/{}/'.format(slc.datasource_type, slc.datasource_id)
        resp = self.get_json_resp(json_endpoint, {'form_data': json.dumps(slc.viz.form_data)})
        resp_from_cache = self.get_json_resp(json_endpoint, {'form_data': json.dumps(slc.viz.form_data)})
        app.config['DATA_CACHE_CONFIG'] = data_cache_config
        self.assertFalse(resp['is_cached'])
        self.assertFalse(resp_from_cache['is_cached'])

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_slice_data_cache(self):
        if False:
            for i in range(10):
                print('nop')
        data_cache_config = app.config['DATA_CACHE_CONFIG']
        cache_default_timeout = app.config['CACHE_DEFAULT_TIMEOUT']
        app.config['CACHE_DEFAULT_TIMEOUT'] = 100
        app.config['DATA_CACHE_CONFIG'] = {'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 10}
        cache_manager.init_app(app)
        slc = self.get_slice('Top 10 Girl Name Share', db.session)
        json_endpoint = '/superset/explore_json/{}/{}/'.format(slc.datasource_type, slc.datasource_id)
        resp = self.get_json_resp(json_endpoint, {'form_data': json.dumps(slc.viz.form_data)})
        resp_from_cache = self.get_json_resp(json_endpoint, {'form_data': json.dumps(slc.viz.form_data)})
        self.assertFalse(resp['is_cached'])
        self.assertTrue(resp_from_cache['is_cached'])
        self.assertEqual(resp_from_cache['cache_timeout'], 10)
        self.assertEqual(resp_from_cache['status'], QueryStatus.SUCCESS)
        self.assertEqual(resp['data'], resp_from_cache['data'])
        self.assertEqual(resp['query'], resp_from_cache['query'])
        self.assertEqual(cache_manager.data_cache.get(resp_from_cache['cache_key'])['query'], resp_from_cache['query'])
        self.assertIsNone(cache_manager.cache.get(resp_from_cache['cache_key']))
        app.config['DATA_CACHE_CONFIG'] = data_cache_config
        app.config['CACHE_DEFAULT_TIMEOUT'] = cache_default_timeout
        cache_manager.init_app(app)