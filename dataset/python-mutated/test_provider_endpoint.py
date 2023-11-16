from __future__ import annotations
from unittest import mock
import pytest
from airflow.providers_manager import ProviderInfo
from airflow.security import permissions
from tests.test_utils.api_connexion_utils import create_user, delete_user
pytestmark = pytest.mark.db_test
MOCK_PROVIDERS = {'apache-airflow-providers-amazon': ProviderInfo('1.0.0', {'package-name': 'apache-airflow-providers-amazon', 'name': 'Amazon', 'description': '`Amazon Web Services (AWS) <https://aws.amazon.com/>`__.\n', 'versions': ['1.0.0']}, 'package'), 'apache-airflow-providers-apache-cassandra': ProviderInfo('1.0.0', {'package-name': 'apache-airflow-providers-apache-cassandra', 'name': 'Apache Cassandra', 'description': '`Apache Cassandra <http://cassandra.apache.org/>`__.\n', 'versions': ['1.0.0']}, 'package')}

@pytest.fixture(scope='module')
def configured_app(minimal_app_for_api):
    if False:
        for i in range(10):
            print('nop')
    app = minimal_app_for_api
    create_user(app, username='test', role_name='Test', permissions=[(permissions.ACTION_CAN_READ, permissions.RESOURCE_PROVIDER)])
    create_user(app, username='test_no_permissions', role_name='TestNoPermissions')
    yield app
    delete_user(app, username='test')
    delete_user(app, username='test_no_permissions')

class TestBaseProviderEndpoint:

    @pytest.fixture(autouse=True)
    def setup_attrs(self, configured_app) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.app = configured_app
        self.client = self.app.test_client()

class TestGetProviders(TestBaseProviderEndpoint):

    @mock.patch('airflow.providers_manager.ProvidersManager.providers', new_callable=mock.PropertyMock, return_value={})
    def test_response_200_empty_list(self, mock_providers):
        if False:
            return 10
        response = self.client.get('/api/v1/providers', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert response.json == {'providers': [], 'total_entries': 0}

    @mock.patch('airflow.providers_manager.ProvidersManager.providers', new_callable=mock.PropertyMock, return_value=MOCK_PROVIDERS)
    def test_response_200(self, mock_providers):
        if False:
            return 10
        response = self.client.get('/api/v1/providers', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert response.json == {'providers': [{'description': 'Amazon Web Services (AWS) https://aws.amazon.com/', 'package_name': 'apache-airflow-providers-amazon', 'version': '1.0.0'}, {'description': 'Apache Cassandra http://cassandra.apache.org/', 'package_name': 'apache-airflow-providers-apache-cassandra', 'version': '1.0.0'}], 'total_entries': 2}

    def test_should_raises_401_unauthenticated(self):
        if False:
            while True:
                i = 10
        response = self.client.get('/api/v1/providers')
        assert response.status_code == 401

    def test_should_raise_403_forbidden(self):
        if False:
            print('Hello World!')
        response = self.client.get('/api/v1/providers', environ_overrides={'REMOTE_USER': 'test_no_permissions'})
        assert response.status_code == 403