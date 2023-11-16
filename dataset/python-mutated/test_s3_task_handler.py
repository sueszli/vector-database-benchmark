from __future__ import annotations
import contextlib
import copy
import os
from unittest import mock
import boto3
import moto
import pytest
from botocore.exceptions import ClientError
from airflow.models import DAG, DagRun, TaskInstance
from airflow.operators.empty import EmptyOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.log.s3_task_handler import S3TaskHandler
from airflow.utils.session import create_session
from airflow.utils.state import State, TaskInstanceState
from airflow.utils.timezone import datetime
from tests.test_utils.config import conf_vars

@pytest.fixture(autouse=True, scope='module')
def s3mock():
    if False:
        for i in range(10):
            print('nop')
    with moto.mock_s3():
        yield

@pytest.mark.db_test
class TestS3TaskHandler:

    @conf_vars({('logging', 'remote_log_conn_id'): 'aws_default'})
    @pytest.fixture(autouse=True)
    def setup_tests(self, create_log_template, tmp_path_factory):
        if False:
            while True:
                i = 10
        self.remote_log_base = 's3://bucket/remote/log/location'
        self.remote_log_location = 's3://bucket/remote/log/location/1.log'
        self.remote_log_key = 'remote/log/location/1.log'
        self.local_log_location = str(tmp_path_factory.mktemp('local-s3-log-location'))
        create_log_template('{try_number}.log')
        self.s3_task_handler = S3TaskHandler(self.local_log_location, self.remote_log_base)
        assert self.s3_task_handler.hook is not None
        date = datetime(2016, 1, 1)
        self.dag = DAG('dag_for_testing_s3_task_handler', start_date=date)
        task = EmptyOperator(task_id='task_for_testing_s3_log_handler', dag=self.dag)
        dag_run = DagRun(dag_id=self.dag.dag_id, execution_date=date, run_id='test', run_type='manual')
        with create_session() as session:
            session.add(dag_run)
            session.commit()
            session.refresh(dag_run)
        self.ti = TaskInstance(task=task, run_id=dag_run.run_id)
        self.ti.dag_run = dag_run
        self.ti.try_number = 1
        self.ti.state = State.RUNNING
        self.conn = boto3.client('s3')
        moto.moto_api._internal.models.moto_api_backend.reset()
        self.conn.create_bucket(Bucket='bucket')
        yield
        self.dag.clear()
        with create_session() as session:
            session.query(DagRun).delete()
        if self.s3_task_handler.handler:
            with contextlib.suppress(Exception):
                os.remove(self.s3_task_handler.handler.baseFilename)

    def test_hook(self):
        if False:
            while True:
                i = 10
        assert isinstance(self.s3_task_handler.hook, S3Hook)
        assert self.s3_task_handler.hook.transfer_config.use_threads is False

    def test_log_exists(self):
        if False:
            while True:
                i = 10
        self.conn.put_object(Bucket='bucket', Key=self.remote_log_key, Body=b'')
        assert self.s3_task_handler.s3_log_exists(self.remote_log_location)

    def test_log_exists_none(self):
        if False:
            print('Hello World!')
        assert not self.s3_task_handler.s3_log_exists(self.remote_log_location)

    def test_log_exists_raises(self):
        if False:
            i = 10
            return i + 15
        assert not self.s3_task_handler.s3_log_exists('s3://nonexistentbucket/foo')

    def test_log_exists_no_hook(self):
        if False:
            i = 10
            return i + 15
        with mock.patch('airflow.providers.amazon.aws.hooks.s3.S3Hook') as mock_hook:
            mock_hook.side_effect = Exception('Failed to connect')
            with pytest.raises(Exception):
                self.s3_task_handler.s3_log_exists(self.remote_log_location)

    def test_set_context_raw(self):
        if False:
            return 10
        self.ti.raw = True
        mock_open = mock.mock_open()
        with mock.patch('airflow.providers.amazon.aws.log.s3_task_handler.open', mock_open):
            self.s3_task_handler.set_context(self.ti)
        assert not self.s3_task_handler.upload_on_close
        mock_open.assert_not_called()

    def test_set_context_not_raw(self):
        if False:
            return 10
        mock_open = mock.mock_open()
        with mock.patch('airflow.providers.amazon.aws.log.s3_task_handler.open', mock_open):
            self.s3_task_handler.set_context(self.ti)
        assert self.s3_task_handler.upload_on_close
        mock_open.assert_called_once_with(os.path.join(self.local_log_location, '1.log'), 'w')
        mock_open().write.assert_not_called()

    def test_read(self):
        if False:
            while True:
                i = 10
        self.conn.put_object(Bucket='bucket', Key=self.remote_log_key, Body=b'Log line\n')
        ti = copy.copy(self.ti)
        ti.state = TaskInstanceState.SUCCESS
        (log, metadata) = self.s3_task_handler.read(ti)
        actual = log[0][0][-1]
        expected = '*** Found logs in s3:\n***   * s3://bucket/remote/log/location/1.log\nLog line'
        assert actual == expected
        assert metadata == [{'end_of_log': True, 'log_pos': 8}]

    def test_read_when_s3_log_missing(self):
        if False:
            for i in range(10):
                print('nop')
        ti = copy.copy(self.ti)
        ti.state = TaskInstanceState.SUCCESS
        self.s3_task_handler._read_from_logs_server = mock.Mock(return_value=([], []))
        (log, metadata) = self.s3_task_handler.read(ti)
        assert 1 == len(log)
        assert len(log) == len(metadata)
        actual = log[0][0][-1]
        expected = '*** No logs found on s3 for ti=<TaskInstance: dag_for_testing_s3_task_handler.task_for_testing_s3_log_handler test [success]>\n'
        assert actual == expected
        assert {'end_of_log': True, 'log_pos': 0} == metadata[0]

    def test_read_when_s3_log_missing_and_log_pos_missing_pre_26(self):
        if False:
            for i in range(10):
                print('nop')
        ti = copy.copy(self.ti)
        ti.state = TaskInstanceState.SUCCESS
        with mock.patch('airflow.providers.amazon.aws.log.s3_task_handler.hasattr', return_value=False):
            (log, metadata) = self.s3_task_handler.read(ti)
        assert 1 == len(log)
        assert log[0][0][-1].startswith('*** Falling back to local log')

    def test_read_when_s3_log_missing_and_log_pos_zero_pre_26(self):
        if False:
            print('Hello World!')
        ti = copy.copy(self.ti)
        ti.state = TaskInstanceState.SUCCESS
        with mock.patch('airflow.providers.amazon.aws.log.s3_task_handler.hasattr', return_value=False):
            (log, metadata) = self.s3_task_handler.read(ti, metadata={'log_pos': 0})
        assert 1 == len(log)
        assert log[0][0][-1].startswith('*** Falling back to local log')

    def test_read_when_s3_log_missing_and_log_pos_over_zero_pre_26(self):
        if False:
            return 10
        ti = copy.copy(self.ti)
        ti.state = TaskInstanceState.SUCCESS
        with mock.patch('airflow.providers.amazon.aws.log.s3_task_handler.hasattr', return_value=False):
            (log, metadata) = self.s3_task_handler.read(ti, metadata={'log_pos': 1})
        assert 1 == len(log)
        assert not log[0][0][-1].startswith('*** Falling back to local log')

    def test_s3_read_when_log_missing(self):
        if False:
            while True:
                i = 10
        handler = self.s3_task_handler
        url = 's3://bucket/foo'
        with mock.patch.object(handler.log, 'error') as mock_error:
            result = handler.s3_read(url, return_error=True)
            msg = f'Could not read logs from {url} with error: An error occurred (404) when calling the HeadObject operation: Not Found'
            assert result == msg
            mock_error.assert_called_once_with(msg, exc_info=True)

    def test_read_raises_return_error(self):
        if False:
            i = 10
            return i + 15
        handler = self.s3_task_handler
        url = 's3://nonexistentbucket/foo'
        with mock.patch.object(handler.log, 'error') as mock_error:
            result = handler.s3_read(url, return_error=True)
            msg = f'Could not read logs from {url} with error: An error occurred (NoSuchBucket) when calling the HeadObject operation: The specified bucket does not exist'
            assert result == msg
            mock_error.assert_called_once_with(msg, exc_info=True)

    def test_write(self):
        if False:
            return 10
        with mock.patch.object(self.s3_task_handler.log, 'error') as mock_error:
            self.s3_task_handler.s3_write('text', self.remote_log_location)
            mock_error.assert_not_called()
        body = boto3.resource('s3').Object('bucket', self.remote_log_key).get()['Body'].read()
        assert body == b'text'

    def test_write_existing(self):
        if False:
            i = 10
            return i + 15
        self.conn.put_object(Bucket='bucket', Key=self.remote_log_key, Body=b'previous ')
        self.s3_task_handler.s3_write('text', self.remote_log_location)
        body = boto3.resource('s3').Object('bucket', self.remote_log_key).get()['Body'].read()
        assert body == b'previous \ntext'

    def test_write_raises(self):
        if False:
            i = 10
            return i + 15
        handler = self.s3_task_handler
        url = 's3://nonexistentbucket/foo'
        with mock.patch.object(handler.log, 'error') as mock_error:
            handler.s3_write('text', url)
            mock_error.assert_called_once_with('Could not write logs to %s', url, exc_info=True)

    def test_close(self):
        if False:
            return 10
        self.s3_task_handler.set_context(self.ti)
        assert self.s3_task_handler.upload_on_close
        self.s3_task_handler.close()
        boto3.resource('s3').Object('bucket', self.remote_log_key).get()

    def test_close_no_upload(self):
        if False:
            for i in range(10):
                print('nop')
        self.ti.raw = True
        self.s3_task_handler.set_context(self.ti)
        assert not self.s3_task_handler.upload_on_close
        self.s3_task_handler.close()
        with pytest.raises(ClientError):
            boto3.resource('s3').Object('bucket', self.remote_log_key).get()

    @pytest.mark.parametrize('delete_local_copy, expected_existence_of_local_copy, airflow_version', [(True, False, '2.6.0'), (False, True, '2.6.0'), (True, True, '2.5.0'), (False, True, '2.5.0')])
    def test_close_with_delete_local_logs_conf(self, delete_local_copy, expected_existence_of_local_copy, airflow_version):
        if False:
            for i in range(10):
                print('nop')
        with conf_vars({('logging', 'delete_local_logs'): str(delete_local_copy)}), mock.patch('airflow.version.version', airflow_version):
            handler = S3TaskHandler(self.local_log_location, self.remote_log_base)
        handler.log.info('test')
        handler.set_context(self.ti)
        assert handler.upload_on_close
        handler.close()
        assert os.path.exists(handler.handler.baseFilename) == expected_existence_of_local_copy