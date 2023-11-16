from __future__ import annotations
from unittest import mock
from unittest.mock import MagicMock
import pytest
from airflow.exceptions import TaskDeferred
from airflow.providers.amazon.aws.hooks.step_function import StepFunctionHook
from airflow.providers.amazon.aws.operators.step_function import StepFunctionGetExecutionOutputOperator, StepFunctionStartExecutionOperator
EXECUTION_ARN = 'arn:aws:states:us-east-1:123456789012:execution:pseudo-state-machine:020f5b16-b1a1-4149-946f-92dd32d97934'
AWS_CONN_ID = 'aws_non_default'
REGION_NAME = 'us-west-2'
STATE_MACHINE_ARN = 'arn:aws:states:us-east-1:000000000000:stateMachine:pseudo-state-machine'
NAME = 'NAME'
INPUT = '{}'

class TestStepFunctionGetExecutionOutputOperator:
    TASK_ID = 'step_function_get_execution_output'

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.mock_context = MagicMock()

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        operator = StepFunctionGetExecutionOutputOperator(task_id=self.TASK_ID, execution_arn=EXECUTION_ARN, aws_conn_id=AWS_CONN_ID, region_name=REGION_NAME)
        assert self.TASK_ID == operator.task_id
        assert EXECUTION_ARN == operator.execution_arn
        assert AWS_CONN_ID == operator.aws_conn_id
        assert REGION_NAME == operator.region_name

    @mock.patch('airflow.providers.amazon.aws.operators.step_function.StepFunctionHook')
    @pytest.mark.parametrize('response', ['output', 'error'])
    def test_execute(self, mock_hook, response):
        if False:
            for i in range(10):
                print('nop')
        hook_response = {response: '{}'}
        hook_instance = mock_hook.return_value
        hook_instance.describe_execution.return_value = hook_response
        operator = StepFunctionGetExecutionOutputOperator(task_id=self.TASK_ID, execution_arn=EXECUTION_ARN, aws_conn_id=AWS_CONN_ID, region_name=REGION_NAME)
        result = operator.execute(self.mock_context)
        assert {} == result

class TestStepFunctionStartExecutionOperator:
    TASK_ID = 'step_function_start_execution_task'

    def setup_method(self):
        if False:
            print('Hello World!')
        self.mock_context = MagicMock()

    def test_init(self):
        if False:
            i = 10
            return i + 15
        operator = StepFunctionStartExecutionOperator(task_id=self.TASK_ID, state_machine_arn=STATE_MACHINE_ARN, name=NAME, state_machine_input=INPUT, aws_conn_id=AWS_CONN_ID, region_name=REGION_NAME)
        assert self.TASK_ID == operator.task_id
        assert STATE_MACHINE_ARN == operator.state_machine_arn
        assert NAME == operator.name
        assert INPUT == operator.input
        assert AWS_CONN_ID == operator.aws_conn_id
        assert REGION_NAME == operator.region_name

    @mock.patch('airflow.providers.amazon.aws.operators.step_function.StepFunctionHook')
    def test_execute(self, mock_hook):
        if False:
            return 10
        hook_response = 'arn:aws:states:us-east-1:123456789012:execution:pseudo-state-machine:020f5b16-b1a1-4149-946f-92dd32d97934'
        hook_instance = mock_hook.return_value
        hook_instance.start_execution.return_value = hook_response
        operator = StepFunctionStartExecutionOperator(task_id=self.TASK_ID, state_machine_arn=STATE_MACHINE_ARN, name=NAME, state_machine_input=INPUT, aws_conn_id=AWS_CONN_ID, region_name=REGION_NAME)
        result = operator.execute(self.mock_context)
        assert hook_response == result

    @mock.patch.object(StepFunctionHook, 'start_execution')
    def test_step_function_start_execution_deferrable(self, mock_start_execution):
        if False:
            return 10
        mock_start_execution.return_value = 'test-execution-arn'
        operator = StepFunctionStartExecutionOperator(task_id=self.TASK_ID, state_machine_arn=STATE_MACHINE_ARN, name=NAME, state_machine_input=INPUT, aws_conn_id=AWS_CONN_ID, region_name=REGION_NAME, deferrable=True)
        with pytest.raises(TaskDeferred):
            operator.execute(None)