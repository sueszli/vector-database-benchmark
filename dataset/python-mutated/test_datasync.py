from __future__ import annotations
from unittest import mock
import boto3
import pytest
from moto import mock_datasync
from airflow.exceptions import AirflowException
from airflow.models import DAG, DagRun, TaskInstance
from airflow.providers.amazon.aws.hooks.datasync import DataSyncHook
from airflow.providers.amazon.aws.operators.datasync import DataSyncOperator
from airflow.utils import timezone
from airflow.utils.timezone import datetime
TEST_DAG_ID = 'unit_tests'
DEFAULT_DATE = datetime(2018, 1, 1)
SOURCE_HOST_NAME = 'airflow.host'
SOURCE_SUBDIR = 'airflow_subdir'
DESTINATION_BUCKET_NAME = 'airflow_bucket'
SOURCE_LOCATION_URI = f'smb://{SOURCE_HOST_NAME}/{SOURCE_SUBDIR}'
DESTINATION_LOCATION_URI = f's3://{DESTINATION_BUCKET_NAME}'
DESTINATION_LOCATION_ARN = f'arn:aws:s3:::{DESTINATION_BUCKET_NAME}'
CREATE_TASK_KWARGS = {'Options': {'VerifyMode': 'NONE', 'Atime': 'NONE'}}
UPDATE_TASK_KWARGS = {'Options': {'VerifyMode': 'BEST_EFFORT', 'Atime': 'NONE'}}
MOCK_DATA = {'task_id': 'test_datasync_task_operator', 'create_task_id': 'test_datasync_create_task_operator', 'get_task_id': 'test_datasync_get_tasks_operator', 'update_task_id': 'test_datasync_update_task_operator', 'delete_task_id': 'test_datasync_delete_task_operator', 'source_location_uri': SOURCE_LOCATION_URI, 'destination_location_uri': DESTINATION_LOCATION_URI, 'create_task_kwargs': CREATE_TASK_KWARGS, 'update_task_kwargs': UPDATE_TASK_KWARGS, 'create_source_location_kwargs': {'Subdirectory': SOURCE_SUBDIR, 'ServerHostname': SOURCE_HOST_NAME, 'User': 'airflow', 'Password': 'airflow_password', 'AgentArns': ['some_agent']}, 'create_destination_location_kwargs': {'S3BucketArn': DESTINATION_LOCATION_ARN, 'S3Config': {'BucketAccessRoleArn': 'myrole'}}}

@mock_datasync
@mock.patch.object(DataSyncHook, 'get_conn')
class DataSyncTestCaseBase:

    def setup_method(self, method):
        if False:
            return 10
        args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
        self.dag = DAG(TEST_DAG_ID + 'test_schedule_dag_once', default_args=args, schedule='@once')
        self.client = boto3.client('datasync', region_name='us-east-1')
        self.datasync = None
        self.source_location_arn = self.client.create_location_smb(**MOCK_DATA['create_source_location_kwargs'])['LocationArn']
        self.destination_location_arn = self.client.create_location_s3(**MOCK_DATA['create_destination_location_kwargs'])['LocationArn']
        self.task_arn = self.client.create_task(SourceLocationArn=self.source_location_arn, DestinationLocationArn=self.destination_location_arn)['TaskArn']

    def teardown_method(self, method):
        if False:
            for i in range(10):
                print('nop')
        tasks = self.client.list_tasks()
        for task in tasks['Tasks']:
            self.client.delete_task(TaskArn=task['TaskArn'])
        locations = self.client.list_locations()
        for location in locations['Locations']:
            self.client.delete_location(LocationArn=location['LocationArn'])
        self.client = None

@mock_datasync
@mock.patch.object(DataSyncHook, 'get_conn')
class TestDataSyncOperatorCreate(DataSyncTestCaseBase):

    def set_up_operator(self, task_id='test_datasync_create_task_operator', task_arn=None, source_location_uri=SOURCE_LOCATION_URI, destination_location_uri=DESTINATION_LOCATION_URI, allow_random_location_choice=False):
        if False:
            i = 10
            return i + 15
        self.datasync = DataSyncOperator(task_id=task_id, dag=self.dag, task_arn=task_arn, source_location_uri=source_location_uri, destination_location_uri=destination_location_uri, create_task_kwargs={'Options': {'VerifyMode': 'NONE', 'Atime': 'NONE'}}, create_source_location_kwargs={'Subdirectory': SOURCE_SUBDIR, 'ServerHostname': SOURCE_HOST_NAME, 'User': 'airflow', 'Password': 'airflow_password', 'AgentArns': ['some_agent']}, create_destination_location_kwargs={'S3BucketArn': DESTINATION_LOCATION_ARN, 'S3Config': {'BucketAccessRoleArn': 'myrole'}}, allow_random_location_choice=allow_random_location_choice, wait_interval_seconds=0)

    def test_init(self, mock_get_conn):
        if False:
            return 10
        self.set_up_operator()
        assert self.datasync.task_id == MOCK_DATA['create_task_id']
        assert self.datasync.aws_conn_id == 'aws_default'
        assert not self.datasync.allow_random_task_choice
        assert not self.datasync.task_execution_kwargs
        assert self.datasync.source_location_uri == MOCK_DATA['source_location_uri']
        assert self.datasync.destination_location_uri == MOCK_DATA['destination_location_uri']
        assert self.datasync.create_task_kwargs == MOCK_DATA['create_task_kwargs']
        assert self.datasync.create_source_location_kwargs == MOCK_DATA['create_source_location_kwargs']
        assert self.datasync.create_destination_location_kwargs == MOCK_DATA['create_destination_location_kwargs']
        assert not self.datasync.allow_random_location_choice
        mock_get_conn.assert_not_called()

    def test_init_fails(self, mock_get_conn):
        if False:
            for i in range(10):
                print('nop')
        mock_get_conn.return_value = self.client
        with pytest.raises(AirflowException):
            self.set_up_operator(source_location_uri=None)
        with pytest.raises(AirflowException):
            self.set_up_operator(destination_location_uri=None)
        with pytest.raises(AirflowException):
            self.set_up_operator(source_location_uri=None, destination_location_uri=None)
        mock_get_conn.assert_not_called()

    def test_create_task(self, mock_get_conn):
        if False:
            i = 10
            return i + 15
        mock_get_conn.return_value = self.client
        self.set_up_operator()
        tasks = self.client.list_tasks()
        for task in tasks['Tasks']:
            self.client.delete_task(TaskArn=task['TaskArn'])
        tasks = self.client.list_tasks()
        assert len(tasks['Tasks']) == 0
        locations = self.client.list_locations()
        assert len(locations['Locations']) == 2
        result = self.datasync.execute(None)
        assert result is not None
        task_arn = result['TaskArn']
        tasks = self.client.list_tasks()
        assert len(tasks['Tasks']) == 1
        locations = self.client.list_locations()
        assert len(locations['Locations']) == 2
        task = self.client.describe_task(TaskArn=task_arn)
        assert task['Options'] == CREATE_TASK_KWARGS['Options']
        mock_get_conn.assert_called()

    def test_create_task_and_location(self, mock_get_conn):
        if False:
            for i in range(10):
                print('nop')
        mock_get_conn.return_value = self.client
        self.set_up_operator()
        tasks = self.client.list_tasks()
        for task in tasks['Tasks']:
            self.client.delete_task(TaskArn=task['TaskArn'])
        locations = self.client.list_locations()
        for location in locations['Locations']:
            self.client.delete_location(LocationArn=location['LocationArn'])
        tasks = self.client.list_tasks()
        assert len(tasks['Tasks']) == 0
        locations = self.client.list_locations()
        assert len(locations['Locations']) == 0
        result = self.datasync.execute(None)
        assert result is not None
        tasks = self.client.list_tasks()
        assert len(tasks['Tasks']) == 1
        locations = self.client.list_locations()
        assert len(locations['Locations']) == 2
        mock_get_conn.assert_called()

    def test_dont_create_task(self, mock_get_conn):
        if False:
            print('Hello World!')
        mock_get_conn.return_value = self.client
        tasks = self.client.list_tasks()
        tasks_before = len(tasks['Tasks'])
        self.set_up_operator(task_arn=self.task_arn)
        self.datasync.execute(None)
        tasks = self.client.list_tasks()
        tasks_after = len(tasks['Tasks'])
        assert tasks_before == tasks_after
        mock_get_conn.assert_called()

    def test_create_task_many_locations(self, mock_get_conn):
        if False:
            for i in range(10):
                print('nop')
        mock_get_conn.return_value = self.client
        tasks = self.client.list_tasks()
        for task in tasks['Tasks']:
            self.client.delete_task(TaskArn=task['TaskArn'])
        self.client.create_location_smb(**MOCK_DATA['create_source_location_kwargs'])
        self.set_up_operator(task_id='datasync_task1')
        with pytest.raises(AirflowException):
            self.datasync.execute(None)
        tasks = self.client.list_tasks()
        for task in tasks['Tasks']:
            self.client.delete_task(TaskArn=task['TaskArn'])
        self.set_up_operator(task_id='datasync_task2', allow_random_location_choice=True)
        self.datasync.execute(None)
        mock_get_conn.assert_called()

    def test_execute_specific_task(self, mock_get_conn):
        if False:
            print('Hello World!')
        mock_get_conn.return_value = self.client
        task_arn = self.client.create_task(SourceLocationArn=self.source_location_arn, DestinationLocationArn=self.destination_location_arn)['TaskArn']
        self.set_up_operator(task_arn=task_arn)
        result = self.datasync.execute(None)
        assert result['TaskArn'] == task_arn
        assert self.datasync.task_arn == task_arn
        mock_get_conn.assert_called()

    @pytest.mark.db_test
    def test_return_value(self, mock_get_conn):
        if False:
            print('Hello World!')
        'Test we return the right value -- that will get put in to XCom by the execution engine'
        mock_get_conn.return_value = self.client
        self.set_up_operator()
        dag_run = DagRun(dag_id=self.dag.dag_id, execution_date=timezone.utcnow(), run_id='test')
        ti = TaskInstance(task=self.datasync)
        ti.dag_run = dag_run
        assert self.datasync.execute(ti.get_template_context()) is not None
        mock_get_conn.assert_called()

@mock_datasync
@mock.patch.object(DataSyncHook, 'get_conn')
class TestDataSyncOperatorGetTasks(DataSyncTestCaseBase):

    def set_up_operator(self, task_id='test_datasync_get_tasks_operator', task_arn=None, source_location_uri=SOURCE_LOCATION_URI, destination_location_uri=DESTINATION_LOCATION_URI, allow_random_task_choice=False):
        if False:
            while True:
                i = 10
        self.datasync = DataSyncOperator(task_id=task_id, dag=self.dag, task_arn=task_arn, source_location_uri=source_location_uri, destination_location_uri=destination_location_uri, create_source_location_kwargs=MOCK_DATA['create_source_location_kwargs'], create_destination_location_kwargs=MOCK_DATA['create_destination_location_kwargs'], create_task_kwargs=MOCK_DATA['create_task_kwargs'], allow_random_task_choice=allow_random_task_choice, wait_interval_seconds=0)

    def test_init(self, mock_get_conn):
        if False:
            while True:
                i = 10
        self.set_up_operator()
        assert self.datasync.task_id == MOCK_DATA['get_task_id']
        assert self.datasync.aws_conn_id == 'aws_default'
        assert not self.datasync.allow_random_location_choice
        assert self.datasync.source_location_uri == MOCK_DATA['source_location_uri']
        assert self.datasync.destination_location_uri == MOCK_DATA['destination_location_uri']
        assert not self.datasync.allow_random_task_choice
        mock_get_conn.assert_not_called()

    def test_init_fails(self, mock_get_conn):
        if False:
            return 10
        mock_get_conn.return_value = self.client
        with pytest.raises(AirflowException):
            self.set_up_operator(source_location_uri=None)
        with pytest.raises(AirflowException):
            self.set_up_operator(destination_location_uri=None)
        with pytest.raises(AirflowException):
            self.set_up_operator(source_location_uri=None, destination_location_uri=None)
        mock_get_conn.assert_not_called()

    def test_get_no_location(self, mock_get_conn):
        if False:
            print('Hello World!')
        mock_get_conn.return_value = self.client
        self.set_up_operator()
        locations = self.client.list_locations()
        for location in locations['Locations']:
            self.client.delete_location(LocationArn=location['LocationArn'])
        locations = self.client.list_locations()
        assert len(locations['Locations']) == 0
        result = self.datasync.execute(None)
        assert result is not None
        locations = self.client.list_locations()
        assert result is not None
        assert len(locations) == 2
        mock_get_conn.assert_called()

    def test_get_no_tasks2(self, mock_get_conn):
        if False:
            i = 10
            return i + 15
        mock_get_conn.return_value = self.client
        self.set_up_operator()
        tasks = self.client.list_tasks()
        for task in tasks['Tasks']:
            self.client.delete_task(TaskArn=task['TaskArn'])
        tasks = self.client.list_tasks()
        assert len(tasks['Tasks']) == 0
        result = self.datasync.execute(None)
        assert result is not None
        mock_get_conn.assert_called()

    def test_get_one_task(self, mock_get_conn):
        if False:
            for i in range(10):
                print('nop')
        mock_get_conn.return_value = self.client
        self.set_up_operator()
        assert self.datasync.task_arn is None
        tasks = self.client.list_tasks()
        assert len(tasks['Tasks']) == 1
        locations = self.client.list_locations()
        assert len(locations['Locations']) == 2
        result = self.datasync.execute(None)
        assert result is not None
        task_arn = result['TaskArn']
        assert task_arn is not None
        assert task_arn
        assert task_arn == self.task_arn
        tasks = self.client.list_tasks()
        assert len(tasks['Tasks']) == 1
        locations = self.client.list_locations()
        assert len(locations['Locations']) == 2
        mock_get_conn.assert_called()

    def test_get_many_tasks(self, mock_get_conn):
        if False:
            while True:
                i = 10
        mock_get_conn.return_value = self.client
        self.set_up_operator(task_id='datasync_task1')
        self.client.create_task(SourceLocationArn=self.source_location_arn, DestinationLocationArn=self.destination_location_arn)
        tasks = self.client.list_tasks()
        assert len(tasks['Tasks']) == 2
        locations = self.client.list_locations()
        assert len(locations['Locations']) == 2
        with pytest.raises(AirflowException):
            self.datasync.execute(None)
        tasks = self.client.list_tasks()
        assert len(tasks['Tasks']) == 2
        locations = self.client.list_locations()
        assert len(locations['Locations']) == 2
        self.set_up_operator(task_id='datasync_task2', task_arn=self.task_arn, allow_random_task_choice=True)
        self.datasync.execute(None)
        mock_get_conn.assert_called()

    def test_execute_specific_task(self, mock_get_conn):
        if False:
            while True:
                i = 10
        mock_get_conn.return_value = self.client
        task_arn = self.client.create_task(SourceLocationArn=self.source_location_arn, DestinationLocationArn=self.destination_location_arn)['TaskArn']
        self.set_up_operator(task_arn=task_arn)
        result = self.datasync.execute(None)
        assert result['TaskArn'] == task_arn
        assert self.datasync.task_arn == task_arn
        mock_get_conn.assert_called()

    @pytest.mark.db_test
    def test_return_value(self, mock_get_conn):
        if False:
            print('Hello World!')
        'Test we return the right value -- that will get put in to XCom by the execution engine'
        mock_get_conn.return_value = self.client
        self.set_up_operator()
        dag_run = DagRun(dag_id=self.dag.dag_id, execution_date=timezone.utcnow(), run_id='test')
        ti = TaskInstance(task=self.datasync)
        ti.dag_run = dag_run
        result = self.datasync.execute(ti.get_template_context())
        assert result['TaskArn'] == self.task_arn
        mock_get_conn.assert_called()

@mock_datasync
@mock.patch.object(DataSyncHook, 'get_conn')
class TestDataSyncOperatorUpdate(DataSyncTestCaseBase):

    def set_up_operator(self, task_id='test_datasync_update_task_operator', task_arn='self', update_task_kwargs='default'):
        if False:
            print('Hello World!')
        if task_arn == 'self':
            task_arn = self.task_arn
        if update_task_kwargs == 'default':
            update_task_kwargs = {'Options': {'VerifyMode': 'BEST_EFFORT', 'Atime': 'NONE'}}
        self.datasync = DataSyncOperator(task_id=task_id, dag=self.dag, task_arn=task_arn, update_task_kwargs=update_task_kwargs, wait_interval_seconds=0)

    def test_init(self, mock_get_conn):
        if False:
            while True:
                i = 10
        self.set_up_operator()
        assert self.datasync.task_id == MOCK_DATA['update_task_id']
        assert self.datasync.aws_conn_id == 'aws_default'
        assert self.datasync.task_arn == self.task_arn
        assert self.datasync.update_task_kwargs == MOCK_DATA['update_task_kwargs']
        mock_get_conn.assert_not_called()

    def test_init_fails(self, mock_get_conn):
        if False:
            return 10
        mock_get_conn.return_value = self.client
        with pytest.raises(AirflowException):
            self.set_up_operator(task_arn=None)
        mock_get_conn.assert_not_called()

    def test_update_task(self, mock_get_conn):
        if False:
            i = 10
            return i + 15
        mock_get_conn.return_value = self.client
        self.set_up_operator()
        task = self.client.describe_task(TaskArn=self.task_arn)
        assert 'Options' not in task
        result = self.datasync.execute(None)
        assert result is not None
        assert result['TaskArn'] == self.task_arn
        assert self.datasync.task_arn is not None
        task = self.client.describe_task(TaskArn=self.task_arn)
        assert task['Options'] == UPDATE_TASK_KWARGS['Options']
        mock_get_conn.assert_called()

    def test_execute_specific_task(self, mock_get_conn):
        if False:
            return 10
        mock_get_conn.return_value = self.client
        task_arn = self.client.create_task(SourceLocationArn=self.source_location_arn, DestinationLocationArn=self.destination_location_arn)['TaskArn']
        self.set_up_operator(task_arn=task_arn)
        result = self.datasync.execute(None)
        assert result['TaskArn'] == task_arn
        assert self.datasync.task_arn == task_arn
        mock_get_conn.assert_called()

    @pytest.mark.db_test
    def test_return_value(self, mock_get_conn):
        if False:
            return 10
        'Test we return the right value -- that will get put in to XCom by the execution engine'
        mock_get_conn.return_value = self.client
        self.set_up_operator()
        dag_run = DagRun(dag_id=self.dag.dag_id, execution_date=timezone.utcnow(), run_id='test')
        ti = TaskInstance(task=self.datasync)
        ti.dag_run = dag_run
        result = self.datasync.execute(ti.get_template_context())
        assert result['TaskArn'] == self.task_arn
        mock_get_conn.assert_called()

@mock_datasync
@mock.patch.object(DataSyncHook, 'get_conn')
class TestDataSyncOperator(DataSyncTestCaseBase):

    def set_up_operator(self, task_id='test_datasync_task_operator', task_arn='self', wait_for_completion=True):
        if False:
            while True:
                i = 10
        if task_arn == 'self':
            task_arn = self.task_arn
        self.datasync = DataSyncOperator(task_id=task_id, dag=self.dag, wait_interval_seconds=0, wait_for_completion=wait_for_completion, task_arn=task_arn)

    def test_init(self, mock_get_conn):
        if False:
            for i in range(10):
                print('nop')
        self.set_up_operator()
        assert self.datasync.task_id == MOCK_DATA['task_id']
        assert self.datasync.aws_conn_id == 'aws_default'
        assert self.datasync.wait_interval_seconds == 0
        assert self.datasync.task_arn == self.task_arn
        mock_get_conn.assert_not_called()

    def test_init_fails(self, mock_get_conn):
        if False:
            for i in range(10):
                print('nop')
        mock_get_conn.return_value = self.client
        with pytest.raises(AirflowException):
            self.set_up_operator(task_arn=None)
        mock_get_conn.assert_not_called()

    def test_execute_task(self, mock_get_conn):
        if False:
            return 10
        mock_get_conn.return_value = self.client
        self.set_up_operator()
        assert self.datasync.task_arn == self.task_arn
        tasks = self.client.list_tasks()
        len_tasks_before = len(tasks['Tasks'])
        locations = self.client.list_locations()
        len_locations_before = len(locations['Locations'])
        result = self.datasync.execute(None)
        assert result is not None
        task_execution_arn = result['TaskExecutionArn']
        assert task_execution_arn is not None
        tasks = self.client.list_tasks()
        assert len(tasks['Tasks']) == len_tasks_before
        locations = self.client.list_locations()
        assert len(locations['Locations']) == len_locations_before
        task_execution = self.client.describe_task_execution(TaskExecutionArn=task_execution_arn)
        assert task_execution['Status'] == 'SUCCESS'
        task_execution_arn = task_execution['TaskExecutionArn']
        assert '/'.join(task_execution_arn.split('/')[:2]) == self.task_arn
        mock_get_conn.assert_called()

    @mock.patch.object(DataSyncHook, 'wait_for_task_execution')
    def test_execute_task_without_wait_for_completion(self, mock_wait, mock_get_conn):
        if False:
            print('Hello World!')
        self.set_up_operator(wait_for_completion=False)
        result = self.datasync.execute(None)
        assert result is not None
        task_execution_arn = result['TaskExecutionArn']
        assert task_execution_arn is not None
        mock_wait.assert_not_called()

    @mock.patch.object(DataSyncHook, 'wait_for_task_execution')
    def test_failed_task(self, mock_wait, mock_get_conn):
        if False:
            return 10
        mock_get_conn.return_value = self.client
        mock_wait.return_value = False
        self.set_up_operator()
        with pytest.raises(AirflowException):
            self.datasync.execute(None)
        mock_get_conn.assert_called()

    @mock.patch.object(DataSyncHook, 'wait_for_task_execution')
    def test_killed_task(self, mock_wait, mock_get_conn):
        if False:
            for i in range(10):
                print('nop')
        mock_get_conn.return_value = self.client

        def kill_task(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            self.datasync.on_kill()
            return True
        mock_wait.side_effect = kill_task
        self.set_up_operator()
        result = self.datasync.execute(None)
        assert result is not None
        task_execution_arn = result['TaskExecutionArn']
        assert task_execution_arn is not None
        task = self.client.describe_task(TaskArn=self.task_arn)
        assert task['Status'] == 'AVAILABLE'
        task_execution = self.client.describe_task_execution(TaskExecutionArn=task_execution_arn)
        assert task_execution['Status'] == 'ERROR'
        mock_get_conn.assert_called()

    def test_execute_specific_task(self, mock_get_conn):
        if False:
            i = 10
            return i + 15
        mock_get_conn.return_value = self.client
        task_arn = self.client.create_task(SourceLocationArn=self.source_location_arn, DestinationLocationArn=self.destination_location_arn)['TaskArn']
        self.set_up_operator(task_arn=task_arn)
        result = self.datasync.execute(None)
        assert result['TaskArn'] == task_arn
        assert self.datasync.task_arn == task_arn
        mock_get_conn.assert_called()

    @pytest.mark.db_test
    def test_return_value(self, mock_get_conn):
        if False:
            print('Hello World!')
        'Test we return the right value -- that will get put in to XCom by the execution engine'
        mock_get_conn.return_value = self.client
        self.set_up_operator()
        dag_run = DagRun(dag_id=self.dag.dag_id, execution_date=timezone.utcnow(), run_id='test')
        ti = TaskInstance(task=self.datasync)
        ti.dag_run = dag_run
        assert self.datasync.execute(ti.get_template_context()) is not None
        mock_get_conn.assert_called()

@mock_datasync
@mock.patch.object(DataSyncHook, 'get_conn')
class TestDataSyncOperatorDelete(DataSyncTestCaseBase):

    def set_up_operator(self, task_id='test_datasync_delete_task_operator', task_arn='self'):
        if False:
            print('Hello World!')
        if task_arn == 'self':
            task_arn = self.task_arn
        self.datasync = DataSyncOperator(task_id=task_id, dag=self.dag, task_arn=task_arn, delete_task_after_execution=True, wait_interval_seconds=0)

    def test_init(self, mock_get_conn):
        if False:
            while True:
                i = 10
        self.set_up_operator()
        assert self.datasync.task_id == MOCK_DATA['delete_task_id']
        assert self.datasync.aws_conn_id == 'aws_default'
        assert self.datasync.task_arn == self.task_arn
        mock_get_conn.assert_not_called()

    def test_init_fails(self, mock_get_conn):
        if False:
            while True:
                i = 10
        mock_get_conn.return_value = self.client
        with pytest.raises(AirflowException):
            self.set_up_operator(task_arn=None)
        mock_get_conn.assert_not_called()

    def test_delete_task(self, mock_get_conn):
        if False:
            while True:
                i = 10
        mock_get_conn.return_value = self.client
        self.set_up_operator()
        tasks = self.client.list_tasks()
        assert len(tasks['Tasks']) == 1
        locations = self.client.list_locations()
        assert len(locations['Locations']) == 2
        result = self.datasync.execute(None)
        assert result is not None
        assert result['TaskArn'] == self.task_arn
        tasks = self.client.list_tasks()
        assert len(tasks['Tasks']) == 0
        locations = self.client.list_locations()
        assert len(locations['Locations']) == 2
        mock_get_conn.assert_called()

    def test_execute_specific_task(self, mock_get_conn):
        if False:
            i = 10
            return i + 15
        mock_get_conn.return_value = self.client
        task_arn = self.client.create_task(SourceLocationArn=self.source_location_arn, DestinationLocationArn=self.destination_location_arn)['TaskArn']
        self.set_up_operator(task_arn=task_arn)
        result = self.datasync.execute(None)
        assert result['TaskArn'] == task_arn
        assert self.datasync.task_arn == task_arn
        mock_get_conn.assert_called()

    @pytest.mark.db_test
    def test_return_value(self, mock_get_conn):
        if False:
            i = 10
            return i + 15
        'Test we return the right value -- that will get put in to XCom by the execution engine'
        mock_get_conn.return_value = self.client
        self.set_up_operator()
        dag_run = DagRun(dag_id=self.dag.dag_id, execution_date=timezone.utcnow(), run_id='test')
        ti = TaskInstance(task=self.datasync)
        ti.dag_run = dag_run
        result = self.datasync.execute(ti.get_template_context())
        assert result['TaskArn'] == self.task_arn
        mock_get_conn.assert_called()