from __future__ import annotations
import textwrap
from unittest.mock import patch
import pytest
from airflow.security import permissions
from tests.test_utils.api_connexion_utils import assert_401, create_user, delete_user
from tests.test_utils.config import conf_vars
pytestmark = pytest.mark.db_test
MOCK_CONF = {'core': {'parallelism': '1024'}, 'smtp': {'smtp_host': 'localhost', 'smtp_mail_from': 'airflow@example.com'}}
MOCK_CONF_WITH_SENSITIVE_VALUE = {'core': {'parallelism': '1024', 'sql_alchemy_conn': 'mock_conn'}, 'smtp': {'smtp_host': 'localhost', 'smtp_mail_from': 'airflow@example.com'}}

@pytest.fixture(scope='module')
def configured_app(minimal_app_for_api):
    if False:
        for i in range(10):
            print('nop')
    app = minimal_app_for_api
    create_user(app, username='test', role_name='Test', permissions=[(permissions.ACTION_CAN_READ, permissions.RESOURCE_CONFIG)])
    create_user(app, username='test_no_permissions', role_name='TestNoPermissions')
    with conf_vars({('webserver', 'expose_config'): 'True'}):
        yield minimal_app_for_api
    delete_user(app, username='test')
    delete_user(app, username='test_no_permissions')

class TestGetConfig:

    @pytest.fixture(autouse=True)
    def setup_attrs(self, configured_app) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.app = configured_app
        self.client = self.app.test_client()

    @patch('airflow.api_connexion.endpoints.config_endpoint.conf.as_dict', return_value=MOCK_CONF)
    def test_should_respond_200_text_plain(self, mock_as_dict):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get('/api/v1/config', headers={'Accept': 'text/plain'}, environ_overrides={'REMOTE_USER': 'test'})
        mock_as_dict.assert_called_with(display_source=False, display_sensitive=True)
        assert response.status_code == 200
        expected = textwrap.dedent('        [core]\n        parallelism = 1024\n\n        [smtp]\n        smtp_host = localhost\n        smtp_mail_from = airflow@example.com\n        ')
        assert expected == response.data.decode()

    @patch('airflow.api_connexion.endpoints.config_endpoint.conf.as_dict', return_value=MOCK_CONF)
    @conf_vars({('webserver', 'expose_config'): 'non-sensitive-only'})
    def test_should_respond_200_text_plain_with_non_sensitive_only(self, mock_as_dict):
        if False:
            print('Hello World!')
        response = self.client.get('/api/v1/config', headers={'Accept': 'text/plain'}, environ_overrides={'REMOTE_USER': 'test'})
        mock_as_dict.assert_called_with(display_source=False, display_sensitive=False)
        assert response.status_code == 200
        expected = textwrap.dedent('        [core]\n        parallelism = 1024\n\n        [smtp]\n        smtp_host = localhost\n        smtp_mail_from = airflow@example.com\n        ')
        assert expected == response.data.decode()

    @patch('airflow.api_connexion.endpoints.config_endpoint.conf.as_dict', return_value=MOCK_CONF)
    def test_should_respond_200_application_json(self, mock_as_dict):
        if False:
            return 10
        response = self.client.get('/api/v1/config', headers={'Accept': 'application/json'}, environ_overrides={'REMOTE_USER': 'test'})
        mock_as_dict.assert_called_with(display_source=False, display_sensitive=True)
        assert response.status_code == 200
        expected = {'sections': [{'name': 'core', 'options': [{'key': 'parallelism', 'value': '1024'}]}, {'name': 'smtp', 'options': [{'key': 'smtp_host', 'value': 'localhost'}, {'key': 'smtp_mail_from', 'value': 'airflow@example.com'}]}]}
        assert expected == response.json

    @patch('airflow.api_connexion.endpoints.config_endpoint.conf.as_dict', return_value=MOCK_CONF)
    def test_should_respond_200_single_section_as_text_plain(self, mock_as_dict):
        if False:
            print('Hello World!')
        response = self.client.get('/api/v1/config?section=smtp', headers={'Accept': 'text/plain'}, environ_overrides={'REMOTE_USER': 'test'})
        mock_as_dict.assert_called_with(display_source=False, display_sensitive=True)
        assert response.status_code == 200
        expected = textwrap.dedent('        [smtp]\n        smtp_host = localhost\n        smtp_mail_from = airflow@example.com\n        ')
        assert expected == response.data.decode()

    @patch('airflow.api_connexion.endpoints.config_endpoint.conf.as_dict', return_value=MOCK_CONF)
    def test_should_respond_200_single_section_as_json(self, mock_as_dict):
        if False:
            return 10
        response = self.client.get('/api/v1/config?section=smtp', headers={'Accept': 'application/json'}, environ_overrides={'REMOTE_USER': 'test'})
        mock_as_dict.assert_called_with(display_source=False, display_sensitive=True)
        assert response.status_code == 200
        expected = {'sections': [{'name': 'smtp', 'options': [{'key': 'smtp_host', 'value': 'localhost'}, {'key': 'smtp_mail_from', 'value': 'airflow@example.com'}]}]}
        assert expected == response.json

    @patch('airflow.api_connexion.endpoints.config_endpoint.conf.as_dict', return_value=MOCK_CONF)
    def test_should_respond_404_when_section_not_exist(self, mock_as_dict):
        if False:
            print('Hello World!')
        response = self.client.get('/api/v1/config?section=smtp1', headers={'Accept': 'application/json'}, environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 404
        assert 'section=smtp1 not found.' in response.json['detail']

    @patch('airflow.api_connexion.endpoints.config_endpoint.conf.as_dict', return_value=MOCK_CONF)
    def test_should_respond_406(self, mock_as_dict):
        if False:
            return 10
        response = self.client.get('/api/v1/config', headers={'Accept': 'application/octet-stream'}, environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 406

    def test_should_raises_401_unauthenticated(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get('/api/v1/config', headers={'Accept': 'application/json'})
        assert_401(response)

    def test_should_raises_403_unauthorized(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get('/api/v1/config', headers={'Accept': 'application/json'}, environ_overrides={'REMOTE_USER': 'test_no_permissions'})
        assert response.status_code == 403

    @conf_vars({('webserver', 'expose_config'): 'False'})
    def test_should_respond_403_when_expose_config_off(self):
        if False:
            return 10
        response = self.client.get('/api/v1/config', headers={'Accept': 'application/json'}, environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 403
        assert 'chose not to expose' in response.json['detail']

class TestGetValue:

    @pytest.fixture(autouse=True)
    def setup_attrs(self, configured_app) -> None:
        if False:
            print('Hello World!')
        self.app = configured_app
        self.client = self.app.test_client()

    @patch('airflow.api_connexion.endpoints.config_endpoint.conf.as_dict', return_value=MOCK_CONF)
    def test_should_respond_200_text_plain(self, mock_as_dict):
        if False:
            return 10
        response = self.client.get('/api/v1/config/section/smtp/option/smtp_mail_from', headers={'Accept': 'text/plain'}, environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        expected = textwrap.dedent('        [smtp]\n        smtp_mail_from = airflow@example.com\n        ')
        assert expected == response.data.decode()

    @patch('airflow.api_connexion.endpoints.config_endpoint.conf.as_dict', return_value=MOCK_CONF_WITH_SENSITIVE_VALUE)
    @conf_vars({('webserver', 'expose_config'): 'non-sensitive-only'})
    @pytest.mark.parametrize('section, option', [('core', 'sql_alchemy_conn'), ('core', 'SQL_ALCHEMY_CONN'), ('corE', 'sql_alchemy_conn'), ('CORE', 'sql_alchemy_conn')])
    def test_should_respond_200_text_plain_with_non_sensitive_only(self, mock_as_dict, section, option):
        if False:
            while True:
                i = 10
        response = self.client.get(f'/api/v1/config/section/{section}/option/{option}', headers={'Accept': 'text/plain'}, environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        expected = textwrap.dedent(f'        [{section}]\n        {option} = < hidden >\n        ')
        assert expected == response.data.decode()

    @patch('airflow.api_connexion.endpoints.config_endpoint.conf.as_dict', return_value=MOCK_CONF)
    def test_should_respond_200_application_json(self, mock_as_dict):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get('/api/v1/config/section/smtp/option/smtp_mail_from', headers={'Accept': 'application/json'}, environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        expected = {'sections': [{'name': 'smtp', 'options': [{'key': 'smtp_mail_from', 'value': 'airflow@example.com'}]}]}
        assert expected == response.json

    @patch('airflow.api_connexion.endpoints.config_endpoint.conf.as_dict', return_value=MOCK_CONF)
    def test_should_respond_404_when_option_not_exist(self, mock_as_dict):
        if False:
            while True:
                i = 10
        response = self.client.get('/api/v1/config/section/smtp/option/smtp_mail_from1', headers={'Accept': 'application/json'}, environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 404
        assert 'The option [smtp/smtp_mail_from1] is not found in config.' in response.json['detail']

    @patch('airflow.api_connexion.endpoints.config_endpoint.conf.as_dict', return_value=MOCK_CONF)
    def test_should_respond_406(self, mock_as_dict):
        if False:
            return 10
        response = self.client.get('/api/v1/config/section/smtp/option/smtp_mail_from', headers={'Accept': 'application/octet-stream'}, environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 406

    def test_should_raises_401_unauthenticated(self):
        if False:
            return 10
        response = self.client.get('/api/v1/config/section/smtp/option/smtp_mail_from', headers={'Accept': 'application/json'})
        assert_401(response)

    def test_should_raises_403_unauthorized(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get('/api/v1/config/section/smtp/option/smtp_mail_from', headers={'Accept': 'application/json'}, environ_overrides={'REMOTE_USER': 'test_no_permissions'})
        assert response.status_code == 403

    @conf_vars({('webserver', 'expose_config'): 'False'})
    def test_should_respond_403_when_expose_config_off(self):
        if False:
            while True:
                i = 10
        response = self.client.get('/api/v1/config/section/smtp/option/smtp_mail_from', headers={'Accept': 'application/json'}, environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 403
        assert 'chose not to expose' in response.json['detail']