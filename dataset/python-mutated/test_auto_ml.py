from __future__ import annotations
from unittest import mock
import pytest
from google.api_core.gapic_v1.method import DEFAULT
from airflow.providers.google.cloud.hooks.vertex_ai.auto_ml import AutoMLHook
from tests.providers.google.cloud.utils.base_gcp_mock import mock_base_gcp_hook_default_project_id, mock_base_gcp_hook_no_default_project_id
TEST_GCP_CONN_ID: str = 'test-gcp-conn-id'
TEST_REGION: str = 'test-region'
TEST_PROJECT_ID: str = 'test-project-id'
TEST_PIPELINE_JOB: dict = {}
TEST_PIPELINE_JOB_ID: str = 'test-pipeline-job-id'
TEST_TRAINING_PIPELINE: dict = {}
TEST_TRAINING_PIPELINE_NAME: str = 'test-training-pipeline'
BASE_STRING = 'airflow.providers.google.common.hooks.base_google.{}'
CUSTOM_JOB_STRING = 'airflow.providers.google.cloud.hooks.vertex_ai.auto_ml.{}'

class TestAutoMLWithDefaultProjectIdHook:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        with mock.patch(BASE_STRING.format('GoogleBaseHook.__init__'), new=mock_base_gcp_hook_default_project_id):
            self.hook = AutoMLHook(gcp_conn_id=TEST_GCP_CONN_ID)

    @mock.patch(CUSTOM_JOB_STRING.format('AutoMLHook.get_pipeline_service_client'))
    def test_delete_training_pipeline(self, mock_client) -> None:
        if False:
            print('Hello World!')
        self.hook.delete_training_pipeline(project_id=TEST_PROJECT_ID, region=TEST_REGION, training_pipeline=TEST_TRAINING_PIPELINE_NAME)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.delete_training_pipeline.assert_called_once_with(request=dict(name=mock_client.return_value.training_pipeline_path.return_value), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.training_pipeline_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION, TEST_TRAINING_PIPELINE_NAME)

    @mock.patch(CUSTOM_JOB_STRING.format('AutoMLHook.get_pipeline_service_client'))
    def test_get_training_pipeline(self, mock_client) -> None:
        if False:
            return 10
        self.hook.get_training_pipeline(project_id=TEST_PROJECT_ID, region=TEST_REGION, training_pipeline=TEST_TRAINING_PIPELINE_NAME)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.get_training_pipeline.assert_called_once_with(request=dict(name=mock_client.return_value.training_pipeline_path.return_value), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.training_pipeline_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION, TEST_TRAINING_PIPELINE_NAME)

    @mock.patch(CUSTOM_JOB_STRING.format('AutoMLHook.get_pipeline_service_client'))
    def test_list_training_pipelines(self, mock_client) -> None:
        if False:
            return 10
        self.hook.list_training_pipelines(project_id=TEST_PROJECT_ID, region=TEST_REGION)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.list_training_pipelines.assert_called_once_with(request=dict(parent=mock_client.return_value.common_location_path.return_value, page_size=None, page_token=None, filter=None, read_mask=None), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.common_location_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION)

class TestAutoMLWithoutDefaultProjectIdHook:

    def test_delegate_to_runtime_error(self):
        if False:
            while True:
                i = 10
        with pytest.raises(RuntimeError):
            AutoMLHook(gcp_conn_id=TEST_GCP_CONN_ID, delegate_to='delegate_to')

    def setup_method(self):
        if False:
            print('Hello World!')
        with mock.patch(BASE_STRING.format('GoogleBaseHook.__init__'), new=mock_base_gcp_hook_no_default_project_id):
            self.hook = AutoMLHook(gcp_conn_id=TEST_GCP_CONN_ID)

    @mock.patch(CUSTOM_JOB_STRING.format('AutoMLHook.get_pipeline_service_client'))
    def test_delete_training_pipeline(self, mock_client) -> None:
        if False:
            i = 10
            return i + 15
        self.hook.delete_training_pipeline(project_id=TEST_PROJECT_ID, region=TEST_REGION, training_pipeline=TEST_TRAINING_PIPELINE_NAME)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.delete_training_pipeline.assert_called_once_with(request=dict(name=mock_client.return_value.training_pipeline_path.return_value), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.training_pipeline_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION, TEST_TRAINING_PIPELINE_NAME)

    @mock.patch(CUSTOM_JOB_STRING.format('AutoMLHook.get_pipeline_service_client'))
    def test_get_training_pipeline(self, mock_client) -> None:
        if False:
            return 10
        self.hook.get_training_pipeline(project_id=TEST_PROJECT_ID, region=TEST_REGION, training_pipeline=TEST_TRAINING_PIPELINE_NAME)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.get_training_pipeline.assert_called_once_with(request=dict(name=mock_client.return_value.training_pipeline_path.return_value), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.training_pipeline_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION, TEST_TRAINING_PIPELINE_NAME)

    @mock.patch(CUSTOM_JOB_STRING.format('AutoMLHook.get_pipeline_service_client'))
    def test_list_training_pipelines(self, mock_client) -> None:
        if False:
            return 10
        self.hook.list_training_pipelines(project_id=TEST_PROJECT_ID, region=TEST_REGION)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.list_training_pipelines.assert_called_once_with(request=dict(parent=mock_client.return_value.common_location_path.return_value, page_size=None, page_token=None, filter=None, read_mask=None), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.common_location_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION)