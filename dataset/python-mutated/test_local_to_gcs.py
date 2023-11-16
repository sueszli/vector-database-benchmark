from __future__ import annotations
import datetime
import os
from glob import glob
from unittest import mock
import pytest
from airflow.models.dag import DAG
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
pytestmark = pytest.mark.db_test

class TestFileToGcsOperator:
    _config = {'bucket': 'dummy', 'mime_type': 'application/octet-stream', 'gzip': False}

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        args = {'owner': 'airflow', 'start_date': datetime.datetime(2017, 1, 1)}
        self.dag = DAG('test_dag_id', default_args=args)
        self.testfile1 = '/tmp/fake1.csv'
        with open(self.testfile1, 'wb') as f:
            f.write(b'x' * 393216)
        self.testfile2 = '/tmp/fake2.csv'
        with open(self.testfile2, 'wb') as f:
            f.write(b'x' * 393216)
        self.testfiles = [self.testfile1, self.testfile2]

    def teardown_method(self):
        if False:
            print('Hello World!')
        os.remove(self.testfile1)
        os.remove(self.testfile2)

    def test_init(self):
        if False:
            i = 10
            return i + 15
        operator = LocalFilesystemToGCSOperator(task_id='file_to_gcs_operator', dag=self.dag, src=self.testfile1, dst='test/test1.csv', **self._config)
        assert operator.src == self.testfile1
        assert operator.dst == 'test/test1.csv'
        assert operator.bucket == self._config['bucket']
        assert operator.mime_type == self._config['mime_type']
        assert operator.gzip == self._config['gzip']

    @mock.patch('airflow.providers.google.cloud.transfers.local_to_gcs.GCSHook', autospec=True)
    def test_execute(self, mock_hook):
        if False:
            while True:
                i = 10
        mock_instance = mock_hook.return_value
        operator = LocalFilesystemToGCSOperator(task_id='gcs_to_file_sensor', dag=self.dag, src=self.testfile1, dst='test/test1.csv', **self._config)
        operator.execute(None)
        mock_instance.upload.assert_called_once_with(bucket_name=self._config['bucket'], filename=self.testfile1, gzip=self._config['gzip'], mime_type=self._config['mime_type'], object_name='test/test1.csv')

    @pytest.mark.db_test
    def test_execute_with_empty_src(self):
        if False:
            i = 10
            return i + 15
        operator = LocalFilesystemToGCSOperator(task_id='local_to_sensor', dag=self.dag, src='no_file.txt', dst='test/no_file.txt', **self._config)
        with pytest.raises(FileNotFoundError):
            operator.execute(None)

    @mock.patch('airflow.providers.google.cloud.transfers.local_to_gcs.GCSHook', autospec=True)
    def test_execute_multiple(self, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        mock_instance = mock_hook.return_value
        operator = LocalFilesystemToGCSOperator(task_id='gcs_to_file_sensor', dag=self.dag, src=self.testfiles, dst='test/', **self._config)
        operator.execute(None)
        files_objects = zip(self.testfiles, ['test/' + os.path.basename(testfile) for testfile in self.testfiles])
        calls = [mock.call(bucket_name=self._config['bucket'], filename=filepath, gzip=self._config['gzip'], mime_type=self._config['mime_type'], object_name=object_name) for (filepath, object_name) in files_objects]
        mock_instance.upload.assert_has_calls(calls)

    @mock.patch('airflow.providers.google.cloud.transfers.local_to_gcs.GCSHook', autospec=True)
    def test_execute_wildcard(self, mock_hook):
        if False:
            while True:
                i = 10
        mock_instance = mock_hook.return_value
        operator = LocalFilesystemToGCSOperator(task_id='gcs_to_file_sensor', dag=self.dag, src='/tmp/fake*.csv', dst='test/', **self._config)
        operator.execute(None)
        object_names = ['test/' + os.path.basename(fp) for fp in glob('/tmp/fake*.csv')]
        files_objects = zip(glob('/tmp/fake*.csv'), object_names)
        calls = [mock.call(bucket_name=self._config['bucket'], filename=filepath, gzip=self._config['gzip'], mime_type=self._config['mime_type'], object_name=object_name) for (filepath, object_name) in files_objects]
        mock_instance.upload.assert_has_calls(calls)

    @mock.patch('airflow.providers.google.cloud.transfers.local_to_gcs.GCSHook', autospec=True)
    def test_execute_negative(self, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        mock_instance = mock_hook.return_value
        operator = LocalFilesystemToGCSOperator(task_id='gcs_to_file_sensor', dag=self.dag, src='/tmp/fake*.csv', dst='test/test1.csv', **self._config)
        print(glob('/tmp/fake*.csv'))
        with pytest.raises(ValueError):
            operator.execute(None)
        mock_instance.assert_not_called()