from __future__ import annotations
from unittest import mock
from airflow.providers.amazon.aws.transfers.s3_to_ftp import S3ToFTPOperator
TASK_ID = 'test_s3_to_ftp'
BUCKET = 'test-s3-bucket'
S3_KEY = 'test/test_1_file.csv'
FTP_PATH = '/tmp/remote_path.txt'
AWS_CONN_ID = 'aws_default'
FTP_CONN_ID = 'ftp_default'

class TestS3ToFTPOperator:

    @mock.patch('airflow.providers.ftp.hooks.ftp.FTPHook.store_file')
    @mock.patch('airflow.providers.amazon.aws.hooks.s3.S3Hook.get_key')
    @mock.patch('airflow.providers.amazon.aws.transfers.s3_to_ftp.NamedTemporaryFile')
    def test_execute(self, mock_local_tmp_file, mock_s3_hook_get_key, mock_ftp_hook_store_file):
        if False:
            return 10
        operator = S3ToFTPOperator(task_id=TASK_ID, s3_bucket=BUCKET, s3_key=S3_KEY, ftp_path=FTP_PATH)
        operator.execute(None)
        mock_s3_hook_get_key.assert_called_once_with(operator.s3_key, operator.s3_bucket)
        mock_local_tmp_file_value = mock_local_tmp_file.return_value.__enter__.return_value
        mock_s3_hook_get_key.return_value.download_fileobj.assert_called_once_with(mock_local_tmp_file_value)
        mock_ftp_hook_store_file.assert_called_once_with(operator.ftp_path, mock_local_tmp_file_value.name)