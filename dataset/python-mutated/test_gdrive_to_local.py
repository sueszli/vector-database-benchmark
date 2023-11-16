from __future__ import annotations
from tempfile import NamedTemporaryFile
from unittest import mock
from airflow.providers.google.cloud.transfers.gdrive_to_local import GoogleDriveToLocalOperator
TASK_ID = 'test-drive-to-local-operator'
FOLDER_ID = '1234567890qwerty'
FILE_NAME = 'file.pdf'
GCP_CONN_ID = 'google_cloud_default'

class TestGoogleDriveToLocalOperator:

    @mock.patch('airflow.providers.google.cloud.transfers.gdrive_to_local.GoogleDriveHook')
    def test_execute(self, hook_mock):
        if False:
            print('Hello World!')
        with NamedTemporaryFile('wb') as temp_file:
            op = GoogleDriveToLocalOperator(task_id=TASK_ID, folder_id=FOLDER_ID, file_name=FILE_NAME, gcp_conn_id=GCP_CONN_ID, output_file=temp_file.name)
            meta = {'id': '123xyz'}
            hook_mock.return_value.get_file_id.return_value = meta
            op.execute(context=None)
            hook_mock.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=None)
            hook_mock.return_value.download_file.assert_called_once_with(file_id=meta['id'], file_handle=mock.ANY)