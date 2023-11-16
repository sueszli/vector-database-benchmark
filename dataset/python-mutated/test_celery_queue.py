from __future__ import annotations
from unittest.mock import patch
import pytest
from airflow.exceptions import AirflowSkipException
from airflow.providers.celery.sensors.celery_queue import CeleryQueueSensor

class TestCeleryQueueSensor:

    def setup_method(self):
        if False:
            i = 10
            return i + 15

        class TestCeleryqueueSensor(CeleryQueueSensor):

            def _check_task_id(self, context):
                if False:
                    while True:
                        i = 10
                return True
        self.sensor = TestCeleryqueueSensor

    @patch('celery.app.control.Inspect')
    def test_poke_success(self, mock_inspect):
        if False:
            while True:
                i = 10
        mock_inspect_result = mock_inspect.return_value
        mock_inspect_result.reserved.return_value = {'test_queue': []}
        mock_inspect_result.scheduled.return_value = {'test_queue': []}
        mock_inspect_result.active.return_value = {'test_queue': []}
        test_sensor = self.sensor(celery_queue='test_queue', task_id='test-task')
        assert test_sensor.poke(None)

    @patch('celery.app.control.Inspect')
    def test_poke_fail(self, mock_inspect):
        if False:
            while True:
                i = 10
        mock_inspect_result = mock_inspect.return_value
        mock_inspect_result.reserved.return_value = {'test_queue': []}
        mock_inspect_result.scheduled.return_value = {'test_queue': []}
        mock_inspect_result.active.return_value = {'test_queue': ['task']}
        test_sensor = self.sensor(celery_queue='test_queue', task_id='test-task')
        assert not test_sensor.poke(None)

    @pytest.mark.parametrize('soft_fail, expected_exception', ((False, KeyError), (True, AirflowSkipException)))
    @patch('celery.app.control.Inspect')
    def test_poke_fail_with_exception(self, mock_inspect, soft_fail, expected_exception):
        if False:
            return 10
        mock_inspect_result = mock_inspect.return_value
        mock_inspect_result.reserved.return_value = {}
        mock_inspect_result.scheduled.return_value = {}
        mock_inspect_result.active.return_value = {}
        with pytest.raises(expected_exception):
            test_sensor = self.sensor(celery_queue='test_queue', task_id='test-task', soft_fail=soft_fail)
            test_sensor.poke(None)

    @patch('celery.app.control.Inspect')
    def test_poke_success_with_taskid(self, mock_inspect):
        if False:
            print('Hello World!')
        test_sensor = self.sensor(celery_queue='test_queue', task_id='test-task', target_task_id='target-task')
        assert test_sensor.poke(None)