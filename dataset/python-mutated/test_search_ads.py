from __future__ import annotations
from unittest import mock
import pytest
from airflow.providers.google.marketing_platform.sensors.search_ads import GoogleSearchAdsReportSensor
pytestmark = pytest.mark.db_test
API_VERSION = 'api_version'
GCP_CONN_ID = 'google_cloud_default'

class TestSearchAdsReportSensor:

    @mock.patch('airflow.providers.google.marketing_platform.sensors.search_ads.GoogleSearchAdsHook')
    @mock.patch('airflow.providers.google.marketing_platform.sensors.search_ads.BaseSensorOperator')
    def test_poke(self, mock_base_op, hook_mock):
        if False:
            for i in range(10):
                print('nop')
        report_id = 'REPORT_ID'
        op = GoogleSearchAdsReportSensor(report_id=report_id, api_version=API_VERSION, task_id='test_task')
        op.poke(context=None)
        hook_mock.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, delegate_to=None, api_version=API_VERSION, impersonation_chain=None)
        hook_mock.return_value.get.assert_called_once_with(report_id=report_id)