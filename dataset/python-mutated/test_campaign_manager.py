from __future__ import annotations
from unittest import mock
from airflow.providers.google.marketing_platform.sensors.campaign_manager import GoogleCampaignManagerReportSensor
API_VERSION = 'v4'
GCP_CONN_ID = 'google_cloud_default'

class TestGoogleCampaignManagerDeleteReportOperator:

    @mock.patch('airflow.providers.google.marketing_platform.sensors.campaign_manager.GoogleCampaignManagerHook')
    @mock.patch('airflow.providers.google.marketing_platform.sensors.campaign_manager.BaseSensorOperator')
    def test_execute(self, mock_base_op, hook_mock):
        if False:
            while True:
                i = 10
        profile_id = 'PROFILE_ID'
        report_id = 'REPORT_ID'
        file_id = 'FILE_ID'
        hook_mock.return_value.get_report.return_value = {'status': 'REPORT_AVAILABLE'}
        op = GoogleCampaignManagerReportSensor(profile_id=profile_id, report_id=report_id, file_id=file_id, api_version=API_VERSION, task_id='test_task')
        result = op.poke(context=None)
        hook_mock.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, delegate_to=None, api_version=API_VERSION, impersonation_chain=None)
        hook_mock.return_value.get_report.assert_called_once_with(profile_id=profile_id, report_id=report_id, file_id=file_id)
        assert result