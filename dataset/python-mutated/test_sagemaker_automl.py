from __future__ import annotations
from unittest import mock
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.amazon.aws.hooks.sagemaker import SageMakerHook
from airflow.providers.amazon.aws.sensors.sagemaker import SageMakerAutoMLSensor

class TestSageMakerAutoMLSensor:

    @staticmethod
    def get_response_with_state(state: str):
        if False:
            for i in range(10):
                print('nop')
        states = {'Completed', 'InProgress', 'Failed', 'Stopped', 'Stopping'}
        assert state in states
        return {'AutoMLJobStatus': state, 'AutoMLJobSecondaryStatus': 'Starting', 'ResponseMetadata': {'HTTPStatusCode': 200}}

    @mock.patch.object(SageMakerHook, '_describe_auto_ml_job')
    def test_sensor_with_failure(self, mock_describe):
        if False:
            while True:
                i = 10
        mock_describe.return_value = self.get_response_with_state('Failed')
        sensor = SageMakerAutoMLSensor(job_name='job_job', task_id='test_task')
        with pytest.raises(AirflowException):
            sensor.execute(None)
        mock_describe.assert_called_once_with('job_job')

    @mock.patch.object(SageMakerHook, '_describe_auto_ml_job')
    def test_sensor(self, mock_describe):
        if False:
            while True:
                i = 10
        mock_describe.side_effect = [self.get_response_with_state('InProgress'), self.get_response_with_state('Stopping'), self.get_response_with_state('Stopped')]
        sensor = SageMakerAutoMLSensor(job_name='job_job', task_id='test_task', poke_interval=0)
        sensor.execute(None)
        assert mock_describe.call_count == 3