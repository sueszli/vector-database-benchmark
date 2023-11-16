from __future__ import annotations
from unittest import mock
import pytest
from botocore.waiter import Waiter
from airflow.exceptions import AirflowException
from airflow.providers.amazon.aws.hooks.emr import EmrHook
from airflow.providers.amazon.aws.operators.emr import EmrStartNotebookExecutionOperator, EmrStopNotebookExecutionOperator
from tests.providers.amazon.aws.utils.test_waiter import assert_expected_waiter_type
PARAMS = {'EditorId': 'test_editor', 'RelativePath': 'test_relative_path', 'ServiceRole': 'test_role', 'NotebookExecutionName': 'test_name', 'NotebookParams': 'test_params', 'NotebookInstanceSecurityGroupId': 'test_notebook_instance_security_group_id', 'Tags': [{'test_key': 'test_value'}], 'ExecutionEngine': {'Id': 'test_cluster_id', 'Type': 'EMR', 'MasterInstanceSecurityGroupId': 'test_master_instance_security_group_id'}}

class TestEmrStartNotebookExecutionOperator:

    @mock.patch('botocore.waiter.get_service_module_name', return_value='emr')
    @mock.patch.object(EmrHook, 'conn')
    @mock.patch.object(Waiter, 'wait')
    def test_start_notebook_execution_wait_for_completion(self, mock_waiter, mock_conn, _):
        if False:
            return 10
        test_execution_id = 'test-execution-id'
        mock_conn.start_notebook_execution.return_value = {'NotebookExecutionId': test_execution_id, 'ResponseMetadata': {'HTTPStatusCode': 200}}
        op = EmrStartNotebookExecutionOperator(task_id='test-id', editor_id=PARAMS['EditorId'], relative_path=PARAMS['RelativePath'], cluster_id=PARAMS['ExecutionEngine']['Id'], service_role=PARAMS['ServiceRole'], notebook_execution_name=PARAMS['NotebookExecutionName'], notebook_params=PARAMS['NotebookParams'], notebook_instance_security_group_id=PARAMS['NotebookInstanceSecurityGroupId'], master_instance_security_group_id=PARAMS['ExecutionEngine']['MasterInstanceSecurityGroupId'], tags=PARAMS['Tags'], wait_for_completion=True)
        op_response = op.execute(None)
        mock_conn.start_notebook_execution.assert_called_once_with(**PARAMS)
        mock_waiter.assert_called_once_with(mock.ANY, NotebookExecutionId=test_execution_id, WaiterConfig=mock.ANY)
        assert_expected_waiter_type(mock_waiter, 'notebook_running')
        assert op_response == test_execution_id

    @mock.patch('airflow.providers.amazon.aws.hooks.emr.EmrHook.conn')
    def test_start_notebook_execution_no_wait_for_completion(self, mock_conn):
        if False:
            return 10
        test_execution_id = 'test-execution-id'
        mock_conn.start_notebook_execution.return_value = {'NotebookExecutionId': test_execution_id, 'ResponseMetadata': {'HTTPStatusCode': 200}}
        op = EmrStartNotebookExecutionOperator(task_id='test-id', editor_id=PARAMS['EditorId'], relative_path=PARAMS['RelativePath'], cluster_id=PARAMS['ExecutionEngine']['Id'], service_role=PARAMS['ServiceRole'], notebook_execution_name=PARAMS['NotebookExecutionName'], notebook_params=PARAMS['NotebookParams'], notebook_instance_security_group_id=PARAMS['NotebookInstanceSecurityGroupId'], master_instance_security_group_id=PARAMS['ExecutionEngine']['MasterInstanceSecurityGroupId'], tags=PARAMS['Tags'])
        op_response = op.execute(None)
        mock_conn.start_notebook_execution.assert_called_once_with(**PARAMS)
        assert op.wait_for_completion is False
        assert not mock_conn.describe_notebook_execution.called
        assert op_response == test_execution_id

    @mock.patch('airflow.providers.amazon.aws.hooks.emr.EmrHook.conn')
    def test_start_notebook_execution_http_code_fail(self, mock_conn):
        if False:
            while True:
                i = 10
        test_execution_id = 'test-execution-id'
        mock_conn.start_notebook_execution.return_value = {'NotebookExecutionId': test_execution_id, 'ResponseMetadata': {'HTTPStatusCode': 400}}
        op = EmrStartNotebookExecutionOperator(task_id='test-id', editor_id=PARAMS['EditorId'], relative_path=PARAMS['RelativePath'], cluster_id=PARAMS['ExecutionEngine']['Id'], service_role=PARAMS['ServiceRole'], notebook_execution_name=PARAMS['NotebookExecutionName'], notebook_params=PARAMS['NotebookParams'], notebook_instance_security_group_id=PARAMS['NotebookInstanceSecurityGroupId'], master_instance_security_group_id=PARAMS['ExecutionEngine']['MasterInstanceSecurityGroupId'], tags=PARAMS['Tags'])
        with pytest.raises(AirflowException, match='Starting notebook execution failed:'):
            op.execute(None)
        mock_conn.start_notebook_execution.assert_called_once_with(**PARAMS)

    @mock.patch('botocore.waiter.get_service_module_name', return_value='emr')
    @mock.patch('time.sleep', return_value=None)
    @mock.patch.object(EmrHook, 'conn')
    @mock.patch.object(Waiter, 'wait')
    def test_start_notebook_execution_wait_for_completion_multiple_attempts(self, mock_waiter, mock_conn, *_):
        if False:
            print('Hello World!')
        test_execution_id = 'test-execution-id'
        mock_conn.start_notebook_execution.return_value = {'NotebookExecutionId': test_execution_id, 'ResponseMetadata': {'HTTPStatusCode': 200}}
        op = EmrStartNotebookExecutionOperator(task_id='test-id', editor_id=PARAMS['EditorId'], relative_path=PARAMS['RelativePath'], cluster_id=PARAMS['ExecutionEngine']['Id'], service_role=PARAMS['ServiceRole'], notebook_execution_name=PARAMS['NotebookExecutionName'], notebook_params=PARAMS['NotebookParams'], notebook_instance_security_group_id=PARAMS['NotebookInstanceSecurityGroupId'], master_instance_security_group_id=PARAMS['ExecutionEngine']['MasterInstanceSecurityGroupId'], tags=PARAMS['Tags'], wait_for_completion=True)
        op_response = op.execute(None)
        mock_conn.start_notebook_execution.assert_called_once_with(**PARAMS)
        mock_waiter.assert_called_once_with(mock.ANY, NotebookExecutionId=test_execution_id, WaiterConfig=mock.ANY)
        assert_expected_waiter_type(mock_waiter, 'notebook_running')
        assert op_response == test_execution_id

    @mock.patch('botocore.waiter.get_service_module_name', return_value='emr')
    @mock.patch.object(EmrHook, 'conn')
    @mock.patch.object(Waiter, 'wait')
    def test_start_notebook_execution_wait_for_completion_fail_state(self, mock_waiter, mock_conn, _):
        if False:
            while True:
                i = 10
        test_execution_id = 'test-execution-id'
        mock_conn.start_notebook_execution.return_value = {'NotebookExecutionId': test_execution_id, 'ResponseMetadata': {'HTTPStatusCode': 200}}
        mock_conn.describe_notebook_execution.return_value = {'NotebookExecution': {'Status': 'FAILED'}}
        op = EmrStartNotebookExecutionOperator(task_id='test-id', editor_id=PARAMS['EditorId'], relative_path=PARAMS['RelativePath'], cluster_id=PARAMS['ExecutionEngine']['Id'], service_role=PARAMS['ServiceRole'], notebook_execution_name=PARAMS['NotebookExecutionName'], notebook_params=PARAMS['NotebookParams'], notebook_instance_security_group_id=PARAMS['NotebookInstanceSecurityGroupId'], master_instance_security_group_id=PARAMS['ExecutionEngine']['MasterInstanceSecurityGroupId'], tags=PARAMS['Tags'], wait_for_completion=True)
        with pytest.raises(AirflowException, match='Notebook Execution reached failure state FAILED\\.'):
            op.execute(None)
        mock_waiter.assert_called_once_with(mock.ANY, NotebookExecutionId=test_execution_id, WaiterConfig=mock.ANY)
        assert_expected_waiter_type(mock_waiter, 'notebook_running')
        mock_conn.start_notebook_execution.assert_called_once_with(**PARAMS)

class TestStopEmrNotebookExecutionOperator:

    @mock.patch('airflow.providers.amazon.aws.hooks.emr.EmrHook.conn')
    def test_stop_notebook_execution(self, mock_conn):
        if False:
            print('Hello World!')
        mock_conn.stop_notebook_execution.return_value = None
        test_execution_id = 'test-execution-id'
        op = EmrStopNotebookExecutionOperator(task_id='test-id', notebook_execution_id=test_execution_id)
        op.execute(None)
        mock_conn.stop_notebook_execution.assert_called_once_with(NotebookExecutionId=test_execution_id)
        assert not mock_conn.describe_notebook_execution.called

    @mock.patch('botocore.waiter.get_service_module_name', return_value='emr')
    @mock.patch.object(EmrHook, 'conn')
    @mock.patch.object(Waiter, 'wait')
    def test_stop_notebook_execution_wait_for_completion(self, mock_waiter, mock_conn, _):
        if False:
            while True:
                i = 10
        mock_conn.stop_notebook_execution.return_value = None
        test_execution_id = 'test-execution-id'
        op = EmrStopNotebookExecutionOperator(task_id='test-id', notebook_execution_id=test_execution_id, wait_for_completion=True)
        op.execute(None)
        mock_conn.stop_notebook_execution.assert_called_once_with(NotebookExecutionId=test_execution_id)
        mock_waiter.assert_called_once_with(mock.ANY, NotebookExecutionId=test_execution_id, WaiterConfig=mock.ANY)
        assert_expected_waiter_type(mock_waiter, 'notebook_stopped')

    @mock.patch('botocore.waiter.get_service_module_name', return_value='emr')
    @mock.patch.object(EmrHook, 'conn')
    @mock.patch.object(Waiter, 'wait')
    def test_stop_notebook_execution_wait_for_completion_fail_state(self, mock_waiter, mock_conn, _):
        if False:
            print('Hello World!')
        mock_conn.stop_notebook_execution.return_value = None
        test_execution_id = 'test-execution-id'
        op = EmrStopNotebookExecutionOperator(task_id='test-id', notebook_execution_id=test_execution_id, wait_for_completion=True)
        op.execute(None)
        mock_conn.stop_notebook_execution.assert_called_once_with(NotebookExecutionId=test_execution_id)
        mock_waiter.assert_called_once_with(mock.ANY, NotebookExecutionId=test_execution_id, WaiterConfig=mock.ANY)
        assert_expected_waiter_type(mock_waiter, 'notebook_stopped')

    @mock.patch('botocore.waiter.get_service_module_name', return_value='emr')
    @mock.patch('time.sleep', return_value=None)
    @mock.patch.object(Waiter, 'wait')
    @mock.patch.object(EmrHook, 'conn')
    def test_stop_notebook_execution_wait_for_completion_multiple_attempts(self, mock_conn, mock_waiter, *_):
        if False:
            i = 10
            return i + 15
        mock_conn.stop_notebook_execution.return_value = None
        test_execution_id = 'test-execution-id'
        op = EmrStopNotebookExecutionOperator(task_id='test-id', notebook_execution_id=test_execution_id, wait_for_completion=True)
        op.execute(None)
        mock_conn.stop_notebook_execution.assert_called_once_with(NotebookExecutionId=test_execution_id)
        mock_waiter.assert_called_once_with(mock.ANY, NotebookExecutionId=test_execution_id, WaiterConfig=mock.ANY)
        assert_expected_waiter_type(mock_waiter, 'notebook_stopped')

    @mock.patch('botocore.waiter.get_service_module_name', return_value='emr')
    @mock.patch.object(Waiter, 'wait')
    @mock.patch.object(EmrHook, 'conn')
    def test_stop_notebook_execution_waiter_config(self, mock_conn, mock_waiter, _):
        if False:
            for i in range(10):
                print('nop')
        test_execution_id = 'test-execution-id'
        countdown = 400
        delay = 12
        op = EmrStopNotebookExecutionOperator(task_id='test-id', notebook_execution_id=test_execution_id, wait_for_completion=True, waiter_countdown=countdown, waiter_check_interval_seconds=delay)
        op.execute(None)
        mock_conn.stop_notebook_execution.assert_called_once_with(NotebookExecutionId=test_execution_id)
        mock_waiter.assert_called_once_with(mock.ANY, NotebookExecutionId=test_execution_id, WaiterConfig={'Delay': delay, 'MaxAttempts': countdown // delay})
        assert_expected_waiter_type(mock_waiter, 'notebook_stopped')