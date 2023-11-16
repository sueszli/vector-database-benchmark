from __future__ import annotations
from unittest import mock
from airflow.providers.google.suite.transfers.gcs_to_sheets import GCSToGoogleSheetsOperator
GCP_CONN_ID = 'test'
IMPERSONATION_CHAIN = ['ACCOUNT_1', 'ACCOUNT_2', 'ACCOUNT_3']
SPREADSHEET_ID = '1234567890'
VALUES = [[1, 2, 3]]
BUCKET = 'destination_bucket'
PATH = 'path/to/reports'

class TestGCSToGoogleSheets:

    @mock.patch('airflow.providers.google.suite.transfers.gcs_to_sheets.GCSHook')
    @mock.patch('airflow.providers.google.suite.transfers.gcs_to_sheets.GSheetsHook')
    @mock.patch('airflow.providers.google.suite.transfers.gcs_to_sheets.NamedTemporaryFile')
    @mock.patch('airflow.providers.google.suite.transfers.gcs_to_sheets.csv.reader')
    def test_execute(self, mock_reader, mock_tempfile, mock_sheet_hook, mock_gcs_hook):
        if False:
            return 10
        filename = 'file://97g23r'
        file_handle = mock.MagicMock()
        mock_tempfile.return_value.__enter__.return_value = file_handle
        mock_tempfile.return_value.__enter__.return_value.name = filename
        mock_reader.return_value = VALUES
        op = GCSToGoogleSheetsOperator(task_id='test_task', spreadsheet_id=SPREADSHEET_ID, bucket_name=BUCKET, object_name=PATH, gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        op.execute(None)
        mock_sheet_hook.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        mock_gcs_hook.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        mock_gcs_hook.return_value.download.assert_called_once_with(bucket_name=BUCKET, object_name=PATH, filename=filename)
        mock_reader.assert_called_once_with(file_handle)
        mock_sheet_hook.return_value.update_values.assert_called_once_with(spreadsheet_id=SPREADSHEET_ID, range_='Sheet1', values=VALUES)