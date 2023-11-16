from __future__ import annotations
from unittest.mock import call, patch
import pytest
from airflow import models
from airflow.providers.google.common.hooks.discovery_api import GoogleDiscoveryApiHook
from airflow.utils import db
pytestmark = pytest.mark.db_test

class TestGoogleDiscoveryApiHook:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        db.merge_conn(models.Connection(conn_id='google_test', conn_type='google_cloud_platform', host='google', schema='refresh_token', login='client_id', password='client_secret'))

    @patch('airflow.providers.google.common.hooks.discovery_api.build')
    @patch('airflow.providers.google.common.hooks.discovery_api.GoogleDiscoveryApiHook._authorize')
    def test_get_conn(self, mock_authorize, mock_build):
        if False:
            return 10
        google_discovery_api_hook = GoogleDiscoveryApiHook(gcp_conn_id='google_test', api_service_name='youtube', api_version='v2')
        google_discovery_api_hook.get_conn()
        mock_build.assert_called_once_with(serviceName=google_discovery_api_hook.api_service_name, version=google_discovery_api_hook.api_version, http=mock_authorize.return_value, cache_discovery=False)

    @patch('airflow.providers.google.common.hooks.discovery_api.getattr')
    @patch('airflow.providers.google.common.hooks.discovery_api.GoogleDiscoveryApiHook.get_conn')
    def test_query(self, mock_get_conn, mock_getattr):
        if False:
            for i in range(10):
                print('nop')
        google_discovery_api_hook = GoogleDiscoveryApiHook(gcp_conn_id='google_test', api_service_name='analyticsreporting', api_version='v4')
        endpoint = 'analyticsreporting.reports.batchGet'
        data = {'body': {'reportRequests': [{'viewId': '180628393', 'dateRanges': [{'startDate': '7daysAgo', 'endDate': 'today'}], 'metrics': [{'expression': 'ga:sessions'}], 'dimensions': [{'name': 'ga:country'}]}]}}
        num_retries = 1
        google_discovery_api_hook.query(endpoint, data, num_retries=num_retries)
        google_api_endpoint_name_parts = endpoint.split('.')
        mock_getattr.assert_has_calls([call(mock_get_conn.return_value, google_api_endpoint_name_parts[1]), call()(), call(mock_getattr.return_value.return_value, google_api_endpoint_name_parts[2]), call()(**data), call()().execute(num_retries=num_retries)])

    @patch('airflow.providers.google.common.hooks.discovery_api.getattr')
    @patch('airflow.providers.google.common.hooks.discovery_api.GoogleDiscoveryApiHook.get_conn')
    def test_query_with_pagination(self, mock_get_conn, mock_getattr):
        if False:
            return 10
        google_api_conn_client_sub_call = mock_getattr.return_value.return_value
        mock_getattr.return_value.side_effect = [google_api_conn_client_sub_call, google_api_conn_client_sub_call, google_api_conn_client_sub_call, google_api_conn_client_sub_call, google_api_conn_client_sub_call, None]
        google_discovery_api_hook = GoogleDiscoveryApiHook(gcp_conn_id='google_test', api_service_name='analyticsreporting', api_version='v4')
        endpoint = 'analyticsreporting.reports.batchGet'
        data = {'body': {'reportRequests': [{'viewId': '180628393', 'dateRanges': [{'startDate': '7daysAgo', 'endDate': 'today'}], 'metrics': [{'expression': 'ga:sessions'}], 'dimensions': [{'name': 'ga:country'}]}]}}
        num_retries = 1
        google_discovery_api_hook.query(endpoint, data, paginate=True, num_retries=num_retries)
        api_endpoint_name_parts = endpoint.split('.')
        google_api_conn_client = mock_get_conn.return_value
        mock_getattr.assert_has_calls([call(google_api_conn_client, api_endpoint_name_parts[1]), call()(), call(google_api_conn_client_sub_call, api_endpoint_name_parts[2]), call()(**data), call()().__bool__(), call()().execute(num_retries=num_retries), call(google_api_conn_client, api_endpoint_name_parts[1]), call()(), call(google_api_conn_client_sub_call, api_endpoint_name_parts[2] + '_next'), call()(google_api_conn_client_sub_call, google_api_conn_client_sub_call.execute.return_value), call()().__bool__(), call()().execute(num_retries=num_retries), call(google_api_conn_client, api_endpoint_name_parts[1]), call()(), call(google_api_conn_client_sub_call, api_endpoint_name_parts[2] + '_next'), call()(google_api_conn_client_sub_call, google_api_conn_client_sub_call.execute.return_value)])