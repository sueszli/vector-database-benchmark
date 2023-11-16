from __future__ import annotations
from unittest import mock
from airflow.providers.google.cloud.transfers.calendar_to_gcs import GoogleCalendarToGCSOperator
API_VERSION = 'v3'
CALENDAR_ID = '1234567890'
EVENT = {'summary': 'Calendar Test Event', 'description': 'A chance to test creating an event from airflow.', 'start': {'dateTime': '2021-12-28T09:00:00-07:00', 'timeZone': 'America/Los_Angeles'}, 'end': {'dateTime': '2021-12-28T17:00:00-07:00', 'timeZone': 'America/Los_Angeles'}}
BUCKET = 'destination_bucket'
PATH = 'path/to/reports'
GCP_CONN_ID = 'test'
IMPERSONATION_CHAIN = ['ACCOUNT_1', 'ACCOUNT_2', 'ACCOUNT_3']

class TestGoogleCalendarToGCSOperator:

    @mock.patch('airflow.providers.google.cloud.transfers.calendar_to_gcs.GCSHook')
    @mock.patch('airflow.providers.google.cloud.transfers.calendar_to_gcs.NamedTemporaryFile')
    def test_upload_data(self, mock_tempfile, mock_gcs_hook):
        if False:
            print('Hello World!')
        filename = 'file://97g23r'
        file_handle = mock.MagicMock()
        mock_tempfile.return_value.__enter__.return_value = file_handle
        mock_tempfile.return_value.__enter__.return_value.name = filename
        expected_dest_file = f'{PATH}/{CALENDAR_ID}.json'
        op = GoogleCalendarToGCSOperator(api_version=API_VERSION, task_id='test_task', calendar_id=CALENDAR_ID, destination_bucket=BUCKET, destination_path=PATH, gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        result = op._upload_data(events=[EVENT])
        file_handle.flush.assert_called_once_with()
        mock_gcs_hook.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        assert result == expected_dest_file

    @mock.patch('airflow.providers.google.cloud.transfers.calendar_to_gcs.GoogleCalendarHook')
    @mock.patch('airflow.providers.google.cloud.transfers.calendar_to_gcs.GoogleCalendarToGCSOperator._upload_data')
    def test_execute(self, mock_upload_data, mock_calendar_hook):
        if False:
            i = 10
            return i + 15
        context = {}
        data = [EVENT]
        mock_upload_data.side_effect = PATH
        mock_calendar_hook.return_value.get_events.return_value = data
        op = GoogleCalendarToGCSOperator(api_version=API_VERSION, task_id='test_task', calendar_id=CALENDAR_ID, destination_bucket=BUCKET, destination_path=PATH, gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        op.execute(context)
        mock_calendar_hook.assert_called_once_with(api_version=API_VERSION, gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        mock_calendar_hook.return_value.get_events.assert_called_once_with(calendar_id=CALENDAR_ID, i_cal_uid=None, max_attendees=None, max_results=None, order_by=None, private_extended_property=None, q=None, shared_extended_property=None, show_deleted=None, show_hidden_invitation=None, single_events=None, sync_token=None, time_max=None, time_min=None, time_zone=None, updated_min=None)
        mock_upload_data.assert_called_once_with(data)