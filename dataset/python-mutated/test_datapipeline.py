from __future__ import annotations
from unittest import mock
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.google.cloud.operators.datapipeline import CreateDataPipelineOperator, RunDataPipelineOperator
TASK_ID = 'test-datapipeline-operators'
TEST_BODY = {'name': 'projects/test-datapipeline-operators/locations/test-location/pipelines/test-pipeline', 'type': 'PIPELINE_TYPE_BATCH', 'workload': {'dataflowFlexTemplateRequest': {'launchParameter': {'containerSpecGcsPath': 'gs://dataflow-templates-us-central1/latest/Word_Count_metadata', 'jobName': 'test-job', 'environment': {'tempLocation': 'test-temp-location'}, 'parameters': {'inputFile': 'gs://dataflow-samples/shakespeare/kinglear.txt', 'output': 'gs://test/output/my_output'}}, 'projectId': 'test-project-id', 'location': 'test-location'}}}
TEST_LOCATION = 'test-location'
TEST_PROJECTID = 'test-project-id'
TEST_GCP_CONN_ID = 'test_gcp_conn_id'
TEST_DATA_PIPELINE_NAME = 'test_data_pipeline_name'

class TestCreateDataPipelineOperator:

    @pytest.fixture
    def create_operator(self):
        if False:
            print('Hello World!')
        '\n        Creates a mock create datapipeline operator to be used in testing.\n        '
        return CreateDataPipelineOperator(task_id='test_create_datapipeline', body=TEST_BODY, project_id=TEST_PROJECTID, location=TEST_LOCATION, gcp_conn_id=TEST_GCP_CONN_ID)

    @mock.patch('airflow.providers.google.cloud.operators.datapipeline.DataPipelineHook')
    def test_execute(self, mock_hook, create_operator):
        if False:
            while True:
                i = 10
        '\n        Test that the execute function creates and calls the DataPipeline hook with the correct parameters\n        '
        create_operator.execute(mock.MagicMock())
        mock_hook.assert_called_once_with(gcp_conn_id='test_gcp_conn_id', impersonation_chain=None)
        mock_hook.return_value.create_data_pipeline.assert_called_once_with(project_id=TEST_PROJECTID, body=TEST_BODY, location=TEST_LOCATION)

    def test_body_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that if the operator is not passed a Request Body, an AirflowException is raised\n        '
        init_kwargs = {'task_id': 'test_create_datapipeline', 'body': {}, 'project_id': TEST_PROJECTID, 'location': TEST_LOCATION, 'gcp_conn_id': TEST_GCP_CONN_ID}
        with pytest.raises(AirflowException):
            CreateDataPipelineOperator(**init_kwargs).execute(mock.MagicMock())

    def test_projectid_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that if the operator is not passed a Project ID, an AirflowException is raised\n        '
        init_kwargs = {'task_id': 'test_create_datapipeline', 'body': TEST_BODY, 'project_id': None, 'location': TEST_LOCATION, 'gcp_conn_id': TEST_GCP_CONN_ID}
        with pytest.raises(AirflowException):
            CreateDataPipelineOperator(**init_kwargs).execute(mock.MagicMock())

    def test_location_invalid(self):
        if False:
            print('Hello World!')
        '\n        Test that if the operator is not passed a location, an AirflowException is raised\n        '
        init_kwargs = {'task_id': 'test_create_datapipeline', 'body': TEST_BODY, 'project_id': TEST_PROJECTID, 'location': None, 'gcp_conn_id': TEST_GCP_CONN_ID}
        with pytest.raises(AirflowException):
            CreateDataPipelineOperator(**init_kwargs).execute(mock.MagicMock())

    def test_response_invalid(self):
        if False:
            return 10
        '\n        Test that if the Response Body contains an error message, an AirflowException is raised\n        '
        init_kwargs = {'task_id': 'test_create_datapipeline', 'body': {'error': 'Testing that AirflowException is raised'}, 'project_id': TEST_PROJECTID, 'location': TEST_LOCATION, 'gcp_conn_id': TEST_GCP_CONN_ID}
        with pytest.raises(AirflowException):
            CreateDataPipelineOperator(**init_kwargs).execute(mock.MagicMock())

class TestRunDataPipelineOperator:

    @pytest.fixture
    def run_operator(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a RunDataPipelineOperator instance with test data\n        '
        return RunDataPipelineOperator(task_id=TASK_ID, data_pipeline_name=TEST_DATA_PIPELINE_NAME, project_id=TEST_PROJECTID, location=TEST_LOCATION, gcp_conn_id=TEST_GCP_CONN_ID)

    @mock.patch('airflow.providers.google.cloud.operators.datapipeline.DataPipelineHook')
    def test_execute(self, data_pipeline_hook_mock, run_operator):
        if False:
            while True:
                i = 10
        '\n        Test Run Operator execute with correct parameters\n        '
        run_operator.execute(mock.MagicMock())
        data_pipeline_hook_mock.assert_called_once_with(gcp_conn_id=TEST_GCP_CONN_ID)
        data_pipeline_hook_mock.return_value.run_data_pipeline.assert_called_once_with(data_pipeline_name=TEST_DATA_PIPELINE_NAME, project_id=TEST_PROJECTID, location=TEST_LOCATION)

    def test_invalid_data_pipeline_name(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that AirflowException is raised if Run Operator is not given a data pipeline name.\n        '
        init_kwargs = {'task_id': TASK_ID, 'data_pipeline_name': None, 'project_id': TEST_PROJECTID, 'location': TEST_LOCATION, 'gcp_conn_id': TEST_GCP_CONN_ID}
        with pytest.raises(AirflowException):
            RunDataPipelineOperator(**init_kwargs).execute(mock.MagicMock())

    def test_invalid_project_id(self):
        if False:
            while True:
                i = 10
        '\n        Test that AirflowException is raised if Run Operator is not given a project ID.\n        '
        init_kwargs = {'task_id': TASK_ID, 'data_pipeline_name': TEST_DATA_PIPELINE_NAME, 'project_id': None, 'location': TEST_LOCATION, 'gcp_conn_id': TEST_GCP_CONN_ID}
        with pytest.raises(AirflowException):
            RunDataPipelineOperator(**init_kwargs).execute(mock.MagicMock())

    def test_invalid_location(self):
        if False:
            while True:
                i = 10
        '\n        Test that AirflowException is raised if Run Operator is not given a location.\n        '
        init_kwargs = {'task_id': TASK_ID, 'data_pipeline_name': TEST_DATA_PIPELINE_NAME, 'project_id': TEST_PROJECTID, 'location': None, 'gcp_conn_id': TEST_GCP_CONN_ID}
        with pytest.raises(AirflowException):
            RunDataPipelineOperator(**init_kwargs).execute(mock.MagicMock())

    def test_invalid_response(self):
        if False:
            return 10
        '\n        Test that AirflowException is raised if Run Operator fails execution and returns error.\n        '
        init_kwargs = {'task_id': TASK_ID, 'data_pipeline_name': TEST_DATA_PIPELINE_NAME, 'project_id': TEST_PROJECTID, 'location': TEST_LOCATION, 'gcp_conn_id': TEST_GCP_CONN_ID}
        with pytest.raises(AirflowException):
            RunDataPipelineOperator(**init_kwargs).execute(mock.MagicMock()).return_value = {'error': {'message': 'example error'}}