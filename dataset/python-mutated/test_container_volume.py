from __future__ import annotations
from unittest import mock
import pytest
from airflow.models import Connection
from airflow.providers.microsoft.azure.hooks.container_volume import AzureContainerVolumeHook

class TestAzureContainerVolumeHook:

    @pytest.mark.parametrize('mocked_connection', [Connection(conn_id='azure_container_test_connection', conn_type='wasb', login='login', password='key')], indirect=True)
    def test_get_file_volume(self, mocked_connection):
        if False:
            for i in range(10):
                print('nop')
        hook = AzureContainerVolumeHook(azure_container_volume_conn_id=mocked_connection.conn_id)
        volume = hook.get_file_volume(mount_name='mount', share_name='share', storage_account_name='storage', read_only=True)
        assert volume is not None
        assert volume.name == 'mount'
        assert volume.azure_file.share_name == 'share'
        assert volume.azure_file.storage_account_key == 'key'
        assert volume.azure_file.storage_account_name == 'storage'
        assert volume.azure_file.read_only is True

    @pytest.mark.parametrize('mocked_connection', [Connection(conn_id='azure_container_test_connection_connection_string', conn_type='wasb', login='login', password='key', extra={'connection_string': 'a=b;AccountKey=1'})], indirect=True)
    def test_get_file_volume_connection_string(self, mocked_connection):
        if False:
            print('Hello World!')
        hook = AzureContainerVolumeHook(azure_container_volume_conn_id=mocked_connection.conn_id)
        volume = hook.get_file_volume(mount_name='mount', share_name='share', storage_account_name='storage', read_only=True)
        assert volume is not None
        assert volume.name == 'mount'
        assert volume.azure_file.share_name == 'share'
        assert volume.azure_file.storage_account_key == '1'
        assert volume.azure_file.storage_account_name == 'storage'
        assert volume.azure_file.read_only is True

    @pytest.mark.parametrize('mocked_connection', [Connection(conn_id='azure_container_volume_test_default_azure-credential', conn_type='wasb', login='', password='', extra={'subscription_id': 'subscription_id', 'resource_group': 'resource_group'})], indirect=True)
    @mock.patch('airflow.providers.microsoft.azure.hooks.container_volume.StorageManagementClient')
    @mock.patch('airflow.providers.microsoft.azure.hooks.container_volume.get_sync_default_azure_credential')
    def test_get_file_volume_default_azure_credential(self, mocked_default_azure_credential, mocked_client, mocked_connection):
        if False:
            for i in range(10):
                print('nop')
        mocked_client.return_value.storage_accounts.list_keys.return_value.as_dict.return_value = {'keys': [{'key_name': 'key1', 'value': 'value', 'permissions': 'FULL', 'creation_time': '2023-07-13T16:16:10.474107Z'}]}
        hook = AzureContainerVolumeHook(azure_container_volume_conn_id=mocked_connection.conn_id)
        volume = hook.get_file_volume(mount_name='mount', share_name='share', storage_account_name='storage', read_only=True)
        assert volume is not None
        assert volume.name == 'mount'
        assert volume.azure_file.share_name == 'share'
        assert volume.azure_file.storage_account_key == 'value'
        assert volume.azure_file.storage_account_name == 'storage'
        assert volume.azure_file.read_only is True
        assert mocked_default_azure_credential.called_with(managed_identity_client_id=None, workload_identity_tenant_id=None)