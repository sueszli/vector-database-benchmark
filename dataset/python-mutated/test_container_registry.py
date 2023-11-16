from __future__ import annotations
from unittest import mock
import pytest
from airflow.models import Connection
from airflow.providers.microsoft.azure.hooks.container_registry import AzureContainerRegistryHook

class TestAzureContainerRegistryHook:

    @pytest.mark.parametrize('mocked_connection', [Connection(conn_id='azure_container_registry', conn_type='azure_container_registry', login='myuser', password='password', host='test.cr')], indirect=True)
    def test_get_conn(self, mocked_connection):
        if False:
            while True:
                i = 10
        hook = AzureContainerRegistryHook(conn_id=mocked_connection.conn_id)
        assert hook.connection is not None
        assert hook.connection.username == 'myuser'
        assert hook.connection.password == 'password'
        assert hook.connection.server == 'test.cr'

    @pytest.mark.parametrize('mocked_connection', [Connection(conn_id='azure_container_registry', conn_type='azure_container_registry', login='myuser', password='', host='test.cr', extra={'subscription_id': 'subscription_id', 'resource_group': 'resource_group'})], indirect=True)
    @mock.patch('airflow.providers.microsoft.azure.hooks.container_registry.ContainerRegistryManagementClient')
    @mock.patch('airflow.providers.microsoft.azure.hooks.container_registry.get_sync_default_azure_credential')
    def test_get_conn_with_default_azure_credential(self, mocked_default_azure_credential, mocked_client, mocked_connection):
        if False:
            return 10
        mocked_client.return_value.registries.list_credentials.return_value.as_dict.return_value = {'username': 'myuser', 'passwords': [{'name': 'password', 'value': 'password'}]}
        hook = AzureContainerRegistryHook(conn_id=mocked_connection.conn_id)
        assert hook.connection is not None
        assert hook.connection.username == 'myuser'
        assert hook.connection.password == 'password'
        assert hook.connection.server == 'test.cr'
        assert mocked_default_azure_credential.called_with(managed_identity_client_id=None, workload_identity_tenant_id=None)