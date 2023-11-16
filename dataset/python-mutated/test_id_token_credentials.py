from __future__ import annotations
import json
import os
from unittest import mock
import pytest
from google.auth import exceptions
from google.auth.environment_vars import CREDENTIALS
from airflow.providers.google.common.utils.id_token_credentials import IDTokenCredentialsAdapter, get_default_id_token_credentials

class TestIDTokenCredentialsAdapter:

    def test_should_use_id_token_from_parent_credentials(self):
        if False:
            for i in range(10):
                print('nop')
        parent_credentials = mock.MagicMock()
        type(parent_credentials).id_token = mock.PropertyMock(side_effect=['ID_TOKEN1', 'ID_TOKEN2'])
        creds = IDTokenCredentialsAdapter(credentials=parent_credentials)
        assert creds.token == 'ID_TOKEN1'
        request_adapter = mock.MagicMock()
        creds.refresh(request_adapter)
        assert creds.token == 'ID_TOKEN2'

class TestGetDefaultIdTokenCredentials:

    @mock.patch.dict('os.environ')
    @mock.patch('google.auth._cloud_sdk.get_application_default_credentials_path', return_value='/tmp/INVALID_PATH.json')
    @mock.patch('google.auth.compute_engine._metadata.ping', return_value=False)
    def test_should_raise_exception(self, mock_metadata_ping, mock_gcloud_sdk_path):
        if False:
            for i in range(10):
                print('nop')
        if CREDENTIALS in os.environ:
            del os.environ[CREDENTIALS]
        with pytest.raises(exceptions.DefaultCredentialsError, match='Please set GOOGLE_APPLICATION_CREDENTIALS'):
            get_default_id_token_credentials(target_audience='example.org')

    @mock.patch.dict('os.environ')
    @mock.patch('google.auth._cloud_sdk.get_application_default_credentials_path', return_value='/tmp/INVALID_PATH.json')
    @mock.patch('google.auth.compute_engine._metadata.ping', return_value=True)
    @mock.patch('google.auth.compute_engine.IDTokenCredentials')
    def test_should_support_metadata_credentials(self, credentials, mock_metadata_ping, mock_gcloud_sdk_path):
        if False:
            for i in range(10):
                print('nop')
        if CREDENTIALS in os.environ:
            del os.environ[CREDENTIALS]
        assert credentials.return_value == get_default_id_token_credentials(target_audience='example.org')

    @mock.patch.dict('os.environ')
    @mock.patch('airflow.providers.google.common.utils.id_token_credentials.open', mock.mock_open(read_data=json.dumps({'client_id': 'CLIENT_ID', 'client_secret': 'CLIENT_SECRET', 'refresh_token': 'REFRESH_TOKEN', 'type': 'authorized_user'})))
    @mock.patch('google.auth._cloud_sdk.get_application_default_credentials_path', return_value=__file__)
    def test_should_support_user_credentials_from_gcloud(self, mock_gcloud_sdk_path):
        if False:
            for i in range(10):
                print('nop')
        if CREDENTIALS in os.environ:
            del os.environ[CREDENTIALS]
        credentials = get_default_id_token_credentials(target_audience='example.org')
        assert isinstance(credentials, IDTokenCredentialsAdapter)
        assert credentials.credentials.client_secret == 'CLIENT_SECRET'

    @mock.patch.dict('os.environ')
    @mock.patch('airflow.providers.google.common.utils.id_token_credentials.open', mock.mock_open(read_data=json.dumps({'type': 'service_account', 'project_id': 'PROJECT_ID', 'private_key_id': 'PRIVATE_KEY_ID', 'private_key': 'PRIVATE_KEY', 'client_email': 'CLIENT_EMAIL', 'client_id': 'CLIENT_ID', 'auth_uri': 'https://accounts.google.com/o/oauth2/auth', 'token_uri': 'https://oauth2.googleapis.com/token', 'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs', 'client_x509_cert_url': 'https://www.googleapis.com/robot/v1/metadata/x509/CERT'})))
    @mock.patch('google.auth._service_account_info.from_dict', return_value='SIGNER')
    @mock.patch('google.auth._cloud_sdk.get_application_default_credentials_path', return_value=__file__)
    def test_should_support_service_account_from_gcloud(self, mock_gcloud_sdk_path, mock_from_dict):
        if False:
            print('Hello World!')
        if CREDENTIALS in os.environ:
            del os.environ[CREDENTIALS]
        credentials = get_default_id_token_credentials(target_audience='example.org')
        assert credentials.service_account_email == 'CLIENT_EMAIL'

    @mock.patch.dict('os.environ')
    @mock.patch('airflow.providers.google.common.utils.id_token_credentials.open', mock.mock_open(read_data=json.dumps({'type': 'service_account', 'project_id': 'PROJECT_ID', 'private_key_id': 'PRIVATE_KEY_ID', 'private_key': 'PRIVATE_KEY', 'client_email': 'CLIENT_EMAIL', 'client_id': 'CLIENT_ID', 'auth_uri': 'https://accounts.google.com/o/oauth2/auth', 'token_uri': 'https://oauth2.googleapis.com/token', 'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs', 'client_x509_cert_url': 'https://www.googleapis.com/robot/v1/metadata/x509/CERT'})))
    @mock.patch('google.auth._service_account_info.from_dict', return_value='SIGNER')
    def test_should_support_service_account_from_env(self, mock_gcloud_sdk_path):
        if False:
            return 10
        os.environ[CREDENTIALS] = __file__
        credentials = get_default_id_token_credentials(target_audience='example.org')
        assert credentials.service_account_email == 'CLIENT_EMAIL'