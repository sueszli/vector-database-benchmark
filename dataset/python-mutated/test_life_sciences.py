"""Tests for Google Life Sciences Run Pipeline operator """
from __future__ import annotations
from unittest import mock
from airflow.providers.google.cloud.operators.life_sciences import LifeSciencesRunPipelineOperator
TEST_BODY = {'pipeline': {'actions': [{}], 'resources': {}, 'environment': {}, 'timeout': '3.5s'}}
TEST_OPERATION = {'name': 'operation-name', 'metadata': {'@type': 'anytype'}, 'done': True, 'response': 'response'}
TEST_PROJECT_ID = 'life-science-project-id'
TEST_LOCATION = 'test-location'

class TestLifeSciencesRunPipelineOperator:

    @mock.patch('airflow.providers.google.cloud.operators.life_sciences.LifeSciencesHook')
    def test_executes(self, mock_hook):
        if False:
            while True:
                i = 10
        mock_instance = mock_hook.return_value
        mock_instance.run_pipeline.return_value = TEST_OPERATION
        operator = LifeSciencesRunPipelineOperator(task_id='task-id', body=TEST_BODY, location=TEST_LOCATION, project_id=TEST_PROJECT_ID)
        context = mock.MagicMock()
        result = operator.execute(context=context)
        assert result == TEST_OPERATION

    @mock.patch('airflow.providers.google.cloud.operators.life_sciences.LifeSciencesHook')
    def test_executes_without_project_id(self, mock_hook):
        if False:
            return 10
        mock_instance = mock_hook.return_value
        mock_instance.run_pipeline.return_value = TEST_OPERATION
        operator = LifeSciencesRunPipelineOperator(task_id='task-id', body=TEST_BODY, location=TEST_LOCATION)
        context = mock.MagicMock()
        result = operator.execute(context=context)
        assert result == TEST_OPERATION