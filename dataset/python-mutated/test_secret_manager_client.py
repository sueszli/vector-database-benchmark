from __future__ import annotations
from unittest import mock
from google.api_core.exceptions import NotFound, PermissionDenied
from google.cloud.secretmanager_v1.types import AccessSecretVersionResponse
from airflow.providers.google.cloud._internal_client.secret_manager_client import _SecretManagerClient
from airflow.providers.google.common.consts import CLIENT_INFO
INTERNAL_CLIENT_MODULE = 'airflow.providers.google.cloud._internal_client.secret_manager_client'
INTERNAL_COMMON_MODULE = 'airflow.providers.google.common.consts'

class TestSecretManagerClient:

    @mock.patch(INTERNAL_CLIENT_MODULE + '.SecretManagerServiceClient')
    def test_auth(self, mock_secrets_client):
        if False:
            for i in range(10):
                print('nop')
        mock_secrets_client.return_value = mock.MagicMock()
        secrets_client = _SecretManagerClient(credentials='credentials')
        _ = secrets_client.client
        mock_secrets_client.assert_called_with(credentials='credentials', client_info=CLIENT_INFO)

    @mock.patch(INTERNAL_CLIENT_MODULE + '.SecretManagerServiceClient')
    def test_get_non_existing_key(self, mock_secrets_client):
        if False:
            while True:
                i = 10
        mock_client = mock.MagicMock()
        mock_secrets_client.return_value = mock_client
        mock_client.secret_version_path.return_value = 'full-path'
        mock_client.access_secret_version.side_effect = NotFound('test-msg')
        secrets_client = _SecretManagerClient(credentials='credentials')
        secret = secrets_client.get_secret(secret_id='missing', project_id='project_id')
        mock_client.secret_version_path.assert_called_once_with('project_id', 'missing', 'latest')
        assert secret is None
        mock_client.access_secret_version.assert_called_once_with(request={'name': 'full-path'})

    @mock.patch(INTERNAL_CLIENT_MODULE + '.SecretManagerServiceClient')
    def test_get_no_permissions(self, mock_secrets_client):
        if False:
            return 10
        mock_client = mock.MagicMock()
        mock_secrets_client.return_value = mock_client
        mock_client.secret_version_path.return_value = 'full-path'
        mock_client.access_secret_version.side_effect = PermissionDenied('test-msg')
        secrets_client = _SecretManagerClient(credentials='credentials')
        secret = secrets_client.get_secret(secret_id='missing', project_id='project_id')
        mock_client.secret_version_path.assert_called_once_with('project_id', 'missing', 'latest')
        assert secret is None
        mock_client.access_secret_version.assert_called_once_with(request={'name': 'full-path'})

    @mock.patch(INTERNAL_CLIENT_MODULE + '.SecretManagerServiceClient')
    def test_get_invalid_id(self, mock_secrets_client):
        if False:
            print('Hello World!')
        mock_client = mock.MagicMock()
        mock_secrets_client.return_value = mock_client
        mock_client.secret_version_path.return_value = 'full-path'
        mock_client.access_secret_version.side_effect = PermissionDenied('test-msg')
        secrets_client = _SecretManagerClient(credentials='credentials')
        secret = secrets_client.get_secret(secret_id='not.allow', project_id='project_id')
        mock_client.secret_version_path.assert_called_once_with('project_id', 'not.allow', 'latest')
        assert secret is None
        mock_client.access_secret_version.assert_called_once_with(request={'name': 'full-path'})

    @mock.patch(INTERNAL_CLIENT_MODULE + '.SecretManagerServiceClient')
    def test_get_existing_key(self, mock_secrets_client):
        if False:
            for i in range(10):
                print('nop')
        mock_client = mock.MagicMock()
        mock_secrets_client.return_value = mock_client
        mock_client.secret_version_path.return_value = 'full-path'
        test_response = AccessSecretVersionResponse()
        test_response.payload.data = b'result'
        mock_client.access_secret_version.return_value = test_response
        secrets_client = _SecretManagerClient(credentials='credentials')
        secret = secrets_client.get_secret(secret_id='existing', project_id='project_id')
        mock_client.secret_version_path.assert_called_once_with('project_id', 'existing', 'latest')
        assert 'result' == secret
        mock_client.access_secret_version.assert_called_once_with(request={'name': 'full-path'})

    @mock.patch(INTERNAL_CLIENT_MODULE + '.SecretManagerServiceClient')
    def test_get_existing_key_with_version(self, mock_secrets_client):
        if False:
            i = 10
            return i + 15
        mock_client = mock.MagicMock()
        mock_secrets_client.return_value = mock_client
        mock_client.secret_version_path.return_value = 'full-path'
        test_response = AccessSecretVersionResponse()
        test_response.payload.data = b'result'
        mock_client.access_secret_version.return_value = test_response
        secrets_client = _SecretManagerClient(credentials='credentials')
        secret = secrets_client.get_secret(secret_id='existing', project_id='project_id', secret_version='test-version')
        mock_client.secret_version_path.assert_called_once_with('project_id', 'existing', 'test-version')
        assert 'result' == secret
        mock_client.access_secret_version.assert_called_once_with(request={'name': 'full-path'})