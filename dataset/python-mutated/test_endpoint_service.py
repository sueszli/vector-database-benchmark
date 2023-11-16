from __future__ import annotations
from unittest import mock
import pytest
from google.api_core.gapic_v1.method import DEFAULT
from airflow.providers.google.cloud.hooks.vertex_ai.endpoint_service import EndpointServiceHook
from tests.providers.google.cloud.utils.base_gcp_mock import mock_base_gcp_hook_default_project_id, mock_base_gcp_hook_no_default_project_id
TEST_GCP_CONN_ID: str = 'test-gcp-conn-id'
TEST_REGION: str = 'test-region'
TEST_PROJECT_ID: str = 'test-project-id'
TEST_ENDPOINT: dict = {}
TEST_ENDPOINT_ID: str = '1234567890'
TEST_ENDPOINT_NAME: str = 'test_endpoint_name'
TEST_DEPLOYED_MODEL: dict = {}
TEST_DEPLOYED_MODEL_ID: str = 'test-deployed-model-id'
TEST_TRAFFIC_SPLIT: dict = {}
TEST_UPDATE_MASK: dict = {}
BASE_STRING = 'airflow.providers.google.common.hooks.base_google.{}'
ENDPOINT_SERVICE_STRING = 'airflow.providers.google.cloud.hooks.vertex_ai.endpoint_service.{}'

class TestEndpointServiceWithDefaultProjectIdHook:

    def test_delegate_to_runtime_error(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(RuntimeError):
            EndpointServiceHook(gcp_conn_id=TEST_GCP_CONN_ID, delegate_to='delegate_to')

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch(BASE_STRING.format('GoogleBaseHook.__init__'), new=mock_base_gcp_hook_default_project_id):
            self.hook = EndpointServiceHook(gcp_conn_id=TEST_GCP_CONN_ID)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_create_endpoint(self, mock_client) -> None:
        if False:
            return 10
        self.hook.create_endpoint(project_id=TEST_PROJECT_ID, region=TEST_REGION, endpoint=TEST_ENDPOINT, endpoint_id=TEST_ENDPOINT_ID)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.create_endpoint.assert_called_once_with(request=dict(parent=mock_client.return_value.common_location_path.return_value, endpoint=TEST_ENDPOINT, endpoint_id=TEST_ENDPOINT_ID), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.common_location_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_delete_endpoint(self, mock_client) -> None:
        if False:
            i = 10
            return i + 15
        self.hook.delete_endpoint(project_id=TEST_PROJECT_ID, region=TEST_REGION, endpoint=TEST_ENDPOINT_NAME)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.delete_endpoint.assert_called_once_with(request=dict(name=mock_client.return_value.endpoint_path.return_value), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.endpoint_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION, TEST_ENDPOINT_NAME)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_deploy_model(self, mock_client) -> None:
        if False:
            print('Hello World!')
        self.hook.deploy_model(project_id=TEST_PROJECT_ID, region=TEST_REGION, endpoint=TEST_ENDPOINT_NAME, deployed_model=TEST_DEPLOYED_MODEL, traffic_split=TEST_TRAFFIC_SPLIT)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.deploy_model.assert_called_once_with(request=dict(endpoint=mock_client.return_value.endpoint_path.return_value, deployed_model=TEST_DEPLOYED_MODEL, traffic_split=TEST_TRAFFIC_SPLIT), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.endpoint_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION, TEST_ENDPOINT_NAME)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_get_endpoint(self, mock_client) -> None:
        if False:
            while True:
                i = 10
        self.hook.get_endpoint(project_id=TEST_PROJECT_ID, region=TEST_REGION, endpoint=TEST_ENDPOINT_NAME)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.get_endpoint.assert_called_once_with(request=dict(name=mock_client.return_value.endpoint_path.return_value), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.endpoint_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION, TEST_ENDPOINT_NAME)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_list_endpoints(self, mock_client) -> None:
        if False:
            return 10
        self.hook.list_endpoints(project_id=TEST_PROJECT_ID, region=TEST_REGION)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.list_endpoints.assert_called_once_with(request=dict(parent=mock_client.return_value.common_location_path.return_value, filter=None, page_size=None, page_token=None, read_mask=None, order_by=None), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.common_location_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_undeploy_model(self, mock_client) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.hook.undeploy_model(project_id=TEST_PROJECT_ID, region=TEST_REGION, endpoint=TEST_ENDPOINT_NAME, deployed_model_id=TEST_DEPLOYED_MODEL_ID, traffic_split=TEST_TRAFFIC_SPLIT)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.undeploy_model.assert_called_once_with(request=dict(endpoint=mock_client.return_value.endpoint_path.return_value, deployed_model_id=TEST_DEPLOYED_MODEL_ID, traffic_split=TEST_TRAFFIC_SPLIT), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.endpoint_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION, TEST_ENDPOINT_NAME)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_update_endpoint(self, mock_client) -> None:
        if False:
            i = 10
            return i + 15
        self.hook.update_endpoint(project_id=TEST_PROJECT_ID, region=TEST_REGION, endpoint_id=TEST_ENDPOINT_NAME, endpoint=TEST_ENDPOINT, update_mask=TEST_UPDATE_MASK)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.update_endpoint.assert_called_once_with(request=dict(endpoint=TEST_ENDPOINT, update_mask=TEST_UPDATE_MASK), metadata=(), retry=DEFAULT, timeout=None)

class TestEndpointServiceWithoutDefaultProjectIdHook:

    def setup_method(self):
        if False:
            while True:
                i = 10
        with mock.patch(BASE_STRING.format('GoogleBaseHook.__init__'), new=mock_base_gcp_hook_no_default_project_id):
            self.hook = EndpointServiceHook(gcp_conn_id=TEST_GCP_CONN_ID)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_create_endpoint(self, mock_client) -> None:
        if False:
            i = 10
            return i + 15
        self.hook.create_endpoint(project_id=TEST_PROJECT_ID, region=TEST_REGION, endpoint=TEST_ENDPOINT, endpoint_id=TEST_ENDPOINT_ID)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.create_endpoint.assert_called_once_with(request=dict(parent=mock_client.return_value.common_location_path.return_value, endpoint=TEST_ENDPOINT, endpoint_id=TEST_ENDPOINT_ID), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.common_location_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_delete_endpoint(self, mock_client) -> None:
        if False:
            print('Hello World!')
        self.hook.delete_endpoint(project_id=TEST_PROJECT_ID, region=TEST_REGION, endpoint=TEST_ENDPOINT_NAME)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.delete_endpoint.assert_called_once_with(request=dict(name=mock_client.return_value.endpoint_path.return_value), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.endpoint_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION, TEST_ENDPOINT_NAME)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_deploy_model(self, mock_client) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.hook.deploy_model(project_id=TEST_PROJECT_ID, region=TEST_REGION, endpoint=TEST_ENDPOINT_NAME, deployed_model=TEST_DEPLOYED_MODEL, traffic_split=TEST_TRAFFIC_SPLIT)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.deploy_model.assert_called_once_with(request=dict(endpoint=mock_client.return_value.endpoint_path.return_value, deployed_model=TEST_DEPLOYED_MODEL, traffic_split=TEST_TRAFFIC_SPLIT), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.endpoint_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION, TEST_ENDPOINT_NAME)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_get_endpoint(self, mock_client) -> None:
        if False:
            i = 10
            return i + 15
        self.hook.get_endpoint(project_id=TEST_PROJECT_ID, region=TEST_REGION, endpoint=TEST_ENDPOINT_NAME)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.get_endpoint.assert_called_once_with(request=dict(name=mock_client.return_value.endpoint_path.return_value), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.endpoint_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION, TEST_ENDPOINT_NAME)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_list_endpoints(self, mock_client) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.hook.list_endpoints(project_id=TEST_PROJECT_ID, region=TEST_REGION)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.list_endpoints.assert_called_once_with(request=dict(parent=mock_client.return_value.common_location_path.return_value, filter=None, page_size=None, page_token=None, read_mask=None, order_by=None), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.common_location_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_undeploy_model(self, mock_client) -> None:
        if False:
            while True:
                i = 10
        self.hook.undeploy_model(project_id=TEST_PROJECT_ID, region=TEST_REGION, endpoint=TEST_ENDPOINT_NAME, deployed_model_id=TEST_DEPLOYED_MODEL_ID, traffic_split=TEST_TRAFFIC_SPLIT)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.undeploy_model.assert_called_once_with(request=dict(endpoint=mock_client.return_value.endpoint_path.return_value, deployed_model_id=TEST_DEPLOYED_MODEL_ID, traffic_split=TEST_TRAFFIC_SPLIT), metadata=(), retry=DEFAULT, timeout=None)
        mock_client.return_value.endpoint_path.assert_called_once_with(TEST_PROJECT_ID, TEST_REGION, TEST_ENDPOINT_NAME)

    @mock.patch(ENDPOINT_SERVICE_STRING.format('EndpointServiceHook.get_endpoint_service_client'))
    def test_update_endpoint(self, mock_client) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.hook.update_endpoint(project_id=TEST_PROJECT_ID, region=TEST_REGION, endpoint_id=TEST_ENDPOINT_NAME, endpoint=TEST_ENDPOINT, update_mask=TEST_UPDATE_MASK)
        mock_client.assert_called_once_with(TEST_REGION)
        mock_client.return_value.update_endpoint.assert_called_once_with(request=dict(endpoint=TEST_ENDPOINT, update_mask=TEST_UPDATE_MASK), metadata=(), retry=DEFAULT, timeout=None)