from __future__ import annotations
from typing import TYPE_CHECKING
from unittest import mock
import pytest
from airflow.exceptions import TaskDeferred
from airflow.providers.amazon.aws.hooks.sagemaker import SageMakerHook
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerStartPipelineOperator, SageMakerStopPipelineOperator
from airflow.providers.amazon.aws.triggers.sagemaker import SageMakerPipelineTrigger
if TYPE_CHECKING:
    from unittest.mock import MagicMock

class TestSageMakerStartPipelineOperator:

    @mock.patch.object(SageMakerHook, 'start_pipeline')
    @mock.patch.object(SageMakerHook, 'check_status')
    def test_execute(self, check_status, start_pipeline):
        if False:
            i = 10
            return i + 15
        op = SageMakerStartPipelineOperator(task_id='test_sagemaker_operator', pipeline_name='my_pipeline', display_name='test_disp_name', pipeline_params={'is_a_test': 'yes'}, wait_for_completion=True, check_interval=12, verbose=False)
        op.execute({})
        start_pipeline.assert_called_once_with(pipeline_name='my_pipeline', display_name='test_disp_name', pipeline_params={'is_a_test': 'yes'})
        check_status.assert_called_once()

    @mock.patch.object(SageMakerHook, 'start_pipeline')
    def test_defer(self, start_mock):
        if False:
            print('Hello World!')
        op = SageMakerStartPipelineOperator(task_id='test_sagemaker_operator', pipeline_name='my_pipeline', deferrable=True)
        with pytest.raises(TaskDeferred) as defer:
            op.execute({})
        assert isinstance(defer.value.trigger, SageMakerPipelineTrigger)
        assert defer.value.trigger.waiter_type == SageMakerPipelineTrigger.Type.COMPLETE

class TestSageMakerStopPipelineOperator:

    @mock.patch.object(SageMakerHook, 'stop_pipeline')
    def test_execute(self, stop_pipeline):
        if False:
            print('Hello World!')
        op = SageMakerStopPipelineOperator(task_id='test_sagemaker_operator', pipeline_exec_arn='pipeline_arn')
        op.execute({})
        stop_pipeline.assert_called_once_with(pipeline_exec_arn='pipeline_arn', fail_if_not_running=False)

    @mock.patch.object(SageMakerHook, 'stop_pipeline')
    def test_defer(self, stop_mock: MagicMock):
        if False:
            for i in range(10):
                print('nop')
        stop_mock.return_value = 'Stopping'
        op = SageMakerStopPipelineOperator(task_id='test_sagemaker_operator', pipeline_exec_arn='my_pipeline_arn', deferrable=True)
        with pytest.raises(TaskDeferred) as defer:
            op.execute({})
        assert isinstance(defer.value.trigger, SageMakerPipelineTrigger)
        assert defer.value.trigger.waiter_type == SageMakerPipelineTrigger.Type.STOPPED