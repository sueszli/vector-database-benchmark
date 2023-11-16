from __future__ import annotations
from unittest import mock
from unittest.mock import MagicMock
import pytest
from airflow.exceptions import AirflowException
from airflow.models.xcom import MAX_XCOM_SIZE
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
TASK_ID = 'test-gcs-operator'
TEST_BUCKET = 'test-bucket'
TEST_PROJECT = 'test-project'
DELIMITER = '.csv'
PREFIX = 'TEST'
MOCK_FILES = ['TEST1.csv', 'TEST2.csv', 'TEST3.csv']
TEST_OBJECT = 'dir1/test-object'
LOCAL_FILE_PATH = '/home/airflow/gcp/test-object'
XCOM_KEY = 'some_xkom_key'
FILE_CONTENT_STR = 'some file content'
FILE_CONTENT_BYTES_UTF8 = b'some file content'
FILE_CONTENT_BYTES_UTF16 = b'\xff\xfes\x00o\x00m\x00e\x00 \x00f\x00i\x00l\x00e\x00 \x00c\x00o\x00n\x00t\x00e\x00n\x00t\x00'

class TestGoogleCloudStorageDownloadOperator:

    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_local.GCSHook')
    def test_execute(self, mock_hook):
        if False:
            while True:
                i = 10
        operator = GCSToLocalFilesystemOperator(task_id=TASK_ID, bucket=TEST_BUCKET, object_name=TEST_OBJECT, filename=LOCAL_FILE_PATH)
        operator.execute(None)
        mock_hook.return_value.download.assert_called_once_with(bucket_name=TEST_BUCKET, object_name=TEST_OBJECT, filename=LOCAL_FILE_PATH)

    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_local.GCSHook')
    def test_size_lt_max_xcom_size(self, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        operator = GCSToLocalFilesystemOperator(task_id=TASK_ID, bucket=TEST_BUCKET, object_name=TEST_OBJECT, store_to_xcom_key=XCOM_KEY)
        context = {'ti': MagicMock()}
        mock_hook.return_value.download.return_value = FILE_CONTENT_BYTES_UTF8
        mock_hook.return_value.get_size.return_value = MAX_XCOM_SIZE - 1
        operator.execute(context=context)
        mock_hook.return_value.get_size.assert_called_once_with(bucket_name=TEST_BUCKET, object_name=TEST_OBJECT)
        mock_hook.return_value.download.assert_called_once_with(bucket_name=TEST_BUCKET, object_name=TEST_OBJECT)
        context['ti'].xcom_push.assert_called_once_with(key=XCOM_KEY, value=FILE_CONTENT_STR)

    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_local.GCSHook')
    def test_size_gt_max_xcom_size(self, mock_hook):
        if False:
            i = 10
            return i + 15
        operator = GCSToLocalFilesystemOperator(task_id=TASK_ID, bucket=TEST_BUCKET, object_name=TEST_OBJECT, store_to_xcom_key=XCOM_KEY)
        context = {'ti': MagicMock()}
        mock_hook.return_value.download.return_value = FILE_CONTENT_BYTES_UTF8
        mock_hook.return_value.get_size.return_value = MAX_XCOM_SIZE + 1
        with pytest.raises(AirflowException, match='file is too large'):
            operator.execute(context=context)

    @mock.patch('airflow.providers.google.cloud.transfers.gcs_to_local.GCSHook')
    def test_xcom_encoding(self, mock_hook):
        if False:
            while True:
                i = 10
        operator = GCSToLocalFilesystemOperator(task_id=TASK_ID, bucket=TEST_BUCKET, object_name=TEST_OBJECT, store_to_xcom_key=XCOM_KEY, file_encoding='utf-16')
        context = {'ti': MagicMock()}
        mock_hook.return_value.download.return_value = FILE_CONTENT_BYTES_UTF16
        mock_hook.return_value.get_size.return_value = MAX_XCOM_SIZE - 1
        operator.execute(context=context)
        mock_hook.return_value.get_size.assert_called_once_with(bucket_name=TEST_BUCKET, object_name=TEST_OBJECT)
        mock_hook.return_value.download.assert_called_once_with(bucket_name=TEST_BUCKET, object_name=TEST_OBJECT)
        context['ti'].xcom_push.assert_called_once_with(key=XCOM_KEY, value=FILE_CONTENT_STR)