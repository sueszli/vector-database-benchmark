from __future__ import annotations
import os
from unittest import mock
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.google.cloud.transfers.gcs_to_sftp import GCSToSFTPOperator
TASK_ID = 'test-gcs-to-sftp-operator'
GCP_CONN_ID = 'GCP_CONN_ID'
SFTP_CONN_ID = 'SFTP_CONN_ID'
IMPERSONATION_CHAIN = ['ACCOUNT_1', 'ACCOUNT_2', 'ACCOUNT_3']
TEST_BUCKET = 'test-bucket'
DESTINATION_SFTP = 'destination_path'

class TestGoogleCloudStorageToSFTPOperator:

    @pytest.mark.parametrize('source_object, target_object, keep_directory_structure', [('folder/test_object.txt', 'folder/test_object.txt', True), ('folder/subfolder/test_object.txt', 'folder/subfolder/test_object.txt', True), ('folder/test_object.txt', 'test_object.txt', False), ('folder/subfolder/test_object.txt', 'test_object.txt', False)])
    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_sftp.GCSHook')
    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_sftp.SFTPHook')
    def test_execute_copy_single_file(self, sftp_hook_mock, gcs_hook_mock, source_object, target_object, keep_directory_structure):
        if False:
            for i in range(10):
                print('nop')
        task = GCSToSFTPOperator(task_id=TASK_ID, source_bucket=TEST_BUCKET, source_object=source_object, destination_path=DESTINATION_SFTP, keep_directory_structure=keep_directory_structure, move_object=False, gcp_conn_id=GCP_CONN_ID, sftp_conn_id=SFTP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        task.execute({})
        gcs_hook_mock.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        sftp_hook_mock.assert_called_once_with(SFTP_CONN_ID)
        gcs_hook_mock.return_value.download.assert_called_with(bucket_name=TEST_BUCKET, object_name=source_object, filename=mock.ANY)
        sftp_hook_mock.return_value.store_file.assert_called_with(os.path.join(DESTINATION_SFTP, target_object), mock.ANY)
        gcs_hook_mock.return_value.delete.assert_not_called()

    @pytest.mark.parametrize('source_object, target_object, keep_directory_structure', [('folder/test_object.txt', 'folder/test_object.txt', True), ('folder/subfolder/test_object.txt', 'folder/subfolder/test_object.txt', True), ('folder/test_object.txt', 'test_object.txt', False), ('folder/subfolder/test_object.txt', 'test_object.txt', False)])
    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_sftp.GCSHook')
    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_sftp.SFTPHook')
    def test_execute_move_single_file(self, sftp_hook_mock, gcs_hook_mock, source_object, target_object, keep_directory_structure):
        if False:
            return 10
        task = GCSToSFTPOperator(task_id=TASK_ID, source_bucket=TEST_BUCKET, source_object=source_object, destination_path=DESTINATION_SFTP, keep_directory_structure=keep_directory_structure, move_object=True, gcp_conn_id=GCP_CONN_ID, sftp_conn_id=SFTP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        task.execute(None)
        gcs_hook_mock.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        sftp_hook_mock.assert_called_once_with(SFTP_CONN_ID)
        gcs_hook_mock.return_value.download.assert_called_with(bucket_name=TEST_BUCKET, object_name=source_object, filename=mock.ANY)
        sftp_hook_mock.return_value.store_file.assert_called_with(os.path.join(DESTINATION_SFTP, target_object), mock.ANY)
        gcs_hook_mock.return_value.delete.assert_called_once_with(TEST_BUCKET, source_object)

    @pytest.mark.parametrize('source_object, prefix, delimiter, gcs_files_list, target_objects, keep_directory_structure', [('folder/test_object*.txt', 'folder/test_object', '.txt', ['folder/test_object/file1.txt', 'folder/test_object/file2.txt'], ['test_object/file1.txt', 'test_object/file2.txt'], False), ('folder/test_object/*', 'folder/test_object/', '', ['folder/test_object/file1.txt', 'folder/test_object/file2.txt'], ['file1.txt', 'file2.txt'], False), ('folder/test_object*.txt', 'folder/test_object', '.txt', ['folder/test_object/file1.txt', 'folder/test_object/file2.txt'], ['folder/test_object/file1.txt', 'folder/test_object/file2.txt'], True), ('folder/test_object/*', 'folder/test_object/', '', ['folder/test_object/file1.txt', 'folder/test_object/file2.txt'], ['folder/test_object/file1.txt', 'folder/test_object/file2.txt'], True)])
    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_sftp.GCSHook')
    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_sftp.SFTPHook')
    def test_execute_copy_with_wildcard(self, sftp_hook_mock, gcs_hook_mock, source_object, prefix, delimiter, gcs_files_list, target_objects, keep_directory_structure):
        if False:
            while True:
                i = 10
        gcs_hook_mock.return_value.list.return_value = gcs_files_list
        operator = GCSToSFTPOperator(task_id=TASK_ID, source_bucket=TEST_BUCKET, source_object=source_object, destination_path=DESTINATION_SFTP, keep_directory_structure=keep_directory_structure, move_object=False, gcp_conn_id=GCP_CONN_ID, sftp_conn_id=SFTP_CONN_ID)
        operator.execute(None)
        gcs_hook_mock.return_value.list.assert_called_with(TEST_BUCKET, delimiter=delimiter, prefix=prefix)
        gcs_hook_mock.return_value.download.assert_has_calls([mock.call(bucket_name=TEST_BUCKET, object_name=gcs_file, filename=mock.ANY) for gcs_file in gcs_files_list])
        sftp_hook_mock.return_value.store_file.assert_has_calls([mock.call(os.path.join(DESTINATION_SFTP, target_object), mock.ANY) for target_object in target_objects])
        gcs_hook_mock.return_value.delete.assert_not_called()

    @pytest.mark.parametrize('source_object, prefix, delimiter, gcs_files_list, target_objects, keep_directory_structure', [('folder/test_object*.txt', 'folder/test_object', '.txt', ['folder/test_object/file1.txt', 'folder/test_object/file2.txt'], ['test_object/file1.txt', 'test_object/file2.txt'], False), ('folder/test_object/*', 'folder/test_object/', '', ['folder/test_object/file1.txt', 'folder/test_object/file2.txt'], ['file1.txt', 'file2.txt'], False), ('folder/test_object*.txt', 'folder/test_object', '.txt', ['folder/test_object/file1.txt', 'folder/test_object/file2.txt'], ['folder/test_object/file1.txt', 'folder/test_object/file2.txt'], True), ('folder/test_object/*', 'folder/test_object/', '', ['folder/test_object/file1.txt', 'folder/test_object/file2.txt'], ['folder/test_object/file1.txt', 'folder/test_object/file2.txt'], True)])
    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_sftp.GCSHook')
    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_sftp.SFTPHook')
    def test_execute_move_with_wildcard(self, sftp_hook_mock, gcs_hook_mock, source_object, prefix, delimiter, gcs_files_list, target_objects, keep_directory_structure):
        if False:
            print('Hello World!')
        gcs_hook_mock.return_value.list.return_value = gcs_files_list
        operator = GCSToSFTPOperator(task_id=TASK_ID, source_bucket=TEST_BUCKET, source_object=source_object, destination_path=DESTINATION_SFTP, keep_directory_structure=keep_directory_structure, move_object=True, gcp_conn_id=GCP_CONN_ID, sftp_conn_id=SFTP_CONN_ID)
        operator.execute(None)
        gcs_hook_mock.return_value.list.assert_called_with(TEST_BUCKET, delimiter=delimiter, prefix=prefix)
        gcs_hook_mock.return_value.download.assert_has_calls([mock.call(bucket_name=TEST_BUCKET, object_name=gcs_file, filename=mock.ANY) for gcs_file in gcs_files_list])
        sftp_hook_mock.return_value.store_file.assert_has_calls([mock.call(os.path.join(DESTINATION_SFTP, target_object), mock.ANY) for target_object in target_objects])
        gcs_hook_mock.return_value.delete.assert_has_calls([mock.call(TEST_BUCKET, gcs_file) for gcs_file in gcs_files_list])

    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_sftp.GCSHook')
    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_sftp.SFTPHook')
    def test_execute_more_than_one_wildcard_exception(self, sftp_hook_mock, gcs_hook_mock):
        if False:
            for i in range(10):
                print('nop')
        operator = GCSToSFTPOperator(task_id=TASK_ID, source_bucket=TEST_BUCKET, source_object='csv/*/test_*.csv', destination_path=DESTINATION_SFTP, move_object=False, gcp_conn_id=GCP_CONN_ID, sftp_conn_id=SFTP_CONN_ID)
        with pytest.raises(AirflowException):
            operator.execute(None)