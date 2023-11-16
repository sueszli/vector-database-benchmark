from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from airflow.exceptions import TaskDeferred
from airflow.providers.amazon.aws.operators.emr import EmrTerminateJobFlowOperator
from airflow.providers.amazon.aws.triggers.emr import EmrTerminateJobFlowTrigger
TERMINATE_SUCCESS_RETURN = {'ResponseMetadata': {'HTTPStatusCode': 200}}

@pytest.fixture
def mocked_hook_client():
    if False:
        i = 10
        return i + 15
    with patch('airflow.providers.amazon.aws.hooks.emr.EmrHook.conn') as m:
        yield m

class TestEmrTerminateJobFlowOperator:

    def test_execute_terminates_the_job_flow_and_does_not_error(self, mocked_hook_client):
        if False:
            while True:
                i = 10
        mocked_hook_client.terminate_job_flows.return_value = TERMINATE_SUCCESS_RETURN
        operator = EmrTerminateJobFlowOperator(task_id='test_task', job_flow_id='j-8989898989', aws_conn_id='aws_default')
        operator.execute(MagicMock())

    def test_create_job_flow_deferrable(self, mocked_hook_client):
        if False:
            return 10
        mocked_hook_client.terminate_job_flows.return_value = TERMINATE_SUCCESS_RETURN
        operator = EmrTerminateJobFlowOperator(task_id='test_task', job_flow_id='j-8989898989', aws_conn_id='aws_default', deferrable=True)
        with pytest.raises(TaskDeferred) as exc:
            operator.execute(MagicMock())
        assert isinstance(exc.value.trigger, EmrTerminateJobFlowTrigger), 'Trigger is not a EmrTerminateJobFlowTrigger'