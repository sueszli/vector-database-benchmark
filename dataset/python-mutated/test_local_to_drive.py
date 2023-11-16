from __future__ import annotations
from pathlib import Path
from unittest import mock
from airflow.providers.google.suite.transfers.local_to_drive import LocalFilesystemToGoogleDriveOperator
GCP_CONN_ID = 'test'
DRIVE_FOLDER = Path('test_folder')
LOCAL_PATHS = [Path('test1'), Path('test2')]
REMOTE_FILE_IDS = ['rtest1', 'rtest2']

class TestLocalFilesystemToGoogleDriveOperator:

    @mock.patch('airflow.providers.google.suite.transfers.local_to_drive.GoogleDriveHook')
    def test_execute(self, mock_hook):
        if False:
            i = 10
            return i + 15
        context = {}
        mock_hook.return_value.upload_file.return_value = REMOTE_FILE_IDS
        op = LocalFilesystemToGoogleDriveOperator(task_id='test_task', local_paths=LOCAL_PATHS, drive_folder=DRIVE_FOLDER, gcp_conn_id=GCP_CONN_ID, folder_id='some_folder_id')
        op.execute(context)
        calls = [mock.call(local_location='test1', remote_location='test_folder/test1', chunk_size=100 * 1024 * 1024, resumable=False, folder_id='some_folder_id', show_full_target_path=True), mock.call(local_location='test2', remote_location='test_folder/test2', chunk_size=100 * 1024 * 1024, resumable=False, folder_id='some_folder_id', show_full_target_path=True)]
        mock_hook.return_value.upload_file.assert_has_calls(calls)