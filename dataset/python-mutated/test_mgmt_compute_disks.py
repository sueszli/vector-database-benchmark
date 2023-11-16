import os
import unittest
import pytest
import azure.mgmt.compute
from azure.profiles import ProfileDefinition
from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer, recorded_by_proxy
AZURE_LOCATION = 'eastus'

class TestMgmtComputeMultiVersion(AzureMgmtRecordedTestCase):

    def setup_method(self, method):
        if False:
            print('Hello World!')
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.compute.ComputeManagementClient)
        self.mgmt_client.profile = ProfileDefinition({self.mgmt_client._PROFILE_TAG: {None: '2019-07-01', 'availability_sets': '2019-07-01', 'dedicated_host_groups': '2019-07-01', 'dedicated_hosts': '2019-07-01', 'disk_encryption_sets': '2019-11-01', 'disks': '2019-03-01', 'images': '2019-07-01', 'log_analytics': '2019-07-01', 'operations': '2019-07-01', 'proximity_placement_groups': '2019-07-01', 'resource_skus': '2019-04-01', 'snapshots': '2019-11-01', 'usage': '2019-07-01', 'virtual_machine_extension_images': '2019-07-01', 'virtual_machine_extensions': '2019-07-01', 'virtual_machine_images': '2019-07-01', 'virtual_machine_run_commands': '2019-07-01', 'virtual_machine_scale_set_extensions': '2019-07-01', 'virtual_machine_scale_set_rolling_upgrades': '2019-07-01', 'virtual_machine_scale_set_vm_extensions': '2019-07-01', 'virtual_machine_scale_set_vms': '2019-07-01', 'virtual_machine_scale_sets': '2019-07-01', 'virtual_machine_sizes': '2019-07-01', 'virtual_machines': '2019-07-01'}}, self.mgmt_client._PROFILE_TAG + ' test')

    @pytest.mark.skipif(os.getenv('AZURE_TEST_RUN_LIVE') not in ('true', 'yes'), reason='only run live test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_disks_multi(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        DISK_NAME = self.get_resource_name('disknamex')
        BODY = {'location': 'eastus', 'creation_data': {'create_option': 'Empty'}, 'disk_size_gb': '200'}
        result = self.mgmt_client.disks.begin_create_or_update(resource_group.name, DISK_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.disks.get(resource_group.name, DISK_NAME)
        result = self.mgmt_client.disks.list_by_resource_group(resource_group.name)
        result = self.mgmt_client.disks.list()
        BODY = {'disk_size_gb': '200'}
        result = self.mgmt_client.disks.begin_update(resource_group.name, DISK_NAME, BODY)
        result = result.result()
        BODY = {'access': 'Read', 'duration_in_seconds': '1800'}
        result = self.mgmt_client.disks.begin_grant_access(resource_group.name, DISK_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.disks.begin_revoke_access(resource_group.name, DISK_NAME)
        result = result.result()
        result = self.mgmt_client.disks.begin_delete(resource_group.name, DISK_NAME)
        result = result.result()

class TestMgmtCompute(AzureMgmtRecordedTestCase):

    def setup_method(self, method):
        if False:
            return 10
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.compute.ComputeManagementClient)
        if self.is_live:
            from azure.mgmt.keyvault import KeyVaultManagementClient
            self.keyvault_client = self.create_mgmt_client(KeyVaultManagementClient)

    def create_key(self, group_name, location, key_vault, tenant_id, object_id):
        if False:
            while True:
                i = 10
        if self.is_live:
            result = self.keyvault_client.vaults.begin_create_or_update(group_name, key_vault, {'location': location, 'properties': {'sku': {'family': 'A', 'name': 'standard'}, 'tenant_id': tenant_id, 'access_policies': [{'tenant_id': tenant_id, 'object_id': object_id, 'permissions': {'keys': ['encrypt', 'decrypt', 'wrapKey', 'unwrapKey', 'sign', 'verify', 'get', 'list', 'create', 'update', 'import', 'delete', 'backup', 'restore', 'recover', 'purge']}}], 'enabled_for_disk_encryption': True}}).result()
            vault_url = result.properties.vault_uri
            vault_id = result.id
            from azure.keyvault.keys import KeyClient
            credentials = self.settings.get_azure_core_credentials()
            key_client = KeyClient(vault_url, credentials)
            from dateutil import parser as date_parse
            expires_on = date_parse.parse('2050-02-02T08:00:00.000Z')
            key = key_client.create_key('testkey', 'RSA', size=2048, expires_on=expires_on)
            return (vault_id, key.id)
        else:
            return ('000', '000')

    @unittest.skip('The KEY_VAULT_NAME need artificially generated,skip for now')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_disk_encryption(self, resource_group):
        if False:
            print('Hello World!')
        SUBSCRIPTION_ID = self.get_settings_value('SUBSCRIPTION_ID')
        TENANT_ID = self.settings.TENANT_ID
        CLIENT_OID = self.settings.CLIENT_OID if self.is_live else '000'
        RESOURCE_GROUP = resource_group.name
        KEY_VAULT_NAME = self.get_resource_name('keyvaultxmmkyxy')
        DISK_ENCRYPTION_SET_NAME = self.get_resource_name('diskencryptionset')
        (VAULT_ID, KEY_URI) = self.create_key(RESOURCE_GROUP, AZURE_LOCATION, KEY_VAULT_NAME, TENANT_ID, CLIENT_OID)
        BODY = {'location': 'eastus', 'identity': {'type': 'SystemAssigned'}, 'active_key': {'source_vault': {'id': VAULT_ID}, 'key_url': KEY_URI}}
        result = self.mgmt_client.disk_encryption_sets.begin_create_or_update(resource_group.name, DISK_ENCRYPTION_SET_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.disk_encryption_sets.get(resource_group.name, DISK_ENCRYPTION_SET_NAME)
        result = self.mgmt_client.disk_encryption_sets.list_by_resource_group(resource_group.name)
        result = self.mgmt_client.disk_encryption_sets.list()
        BODY = {'active_key': {'source_vault': {'id': VAULT_ID}, 'key_url': KEY_URI}, 'tags': {'department': 'Development', 'project': 'Encryption'}}
        result = self.mgmt_client.disk_encryption_sets.begin_update(resource_group.name, DISK_ENCRYPTION_SET_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.disk_encryption_sets.begin_delete(resource_group.name, DISK_ENCRYPTION_SET_NAME)
        result = result.result()

    @pytest.mark.skipif(os.getenv('AZURE_TEST_RUN_LIVE') not in ('true', 'yes'), reason='only run live test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_shot(self, resource_group):
        if False:
            i = 10
            return i + 15
        SUBSCRIPTION_ID = self.get_settings_value('SUBSCRIPTION_ID')
        RESOURCE_GROUP = resource_group.name
        DISK_NAME = self.get_resource_name('disknamex')
        SNAPSHOT_NAME = self.get_resource_name('snapshotx')
        IMAGE_NAME = self.get_resource_name('imagex')
        BODY = {'location': 'eastus', 'creation_data': {'create_option': 'Empty'}, 'disk_size_gb': '200'}
        result = self.mgmt_client.disks.begin_create_or_update(resource_group.name, DISK_NAME, BODY)
        result = result.result()
        BODY = {'location': 'eastus', 'creation_data': {'create_option': 'Copy', 'source_uri': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Compute/disks/' + DISK_NAME}}
        result = self.mgmt_client.snapshots.begin_create_or_update(resource_group.name, SNAPSHOT_NAME, BODY)
        result = result.result()
        BODY = {'location': 'eastus', 'storage_profile': {'os_disk': {'os_type': 'Linux', 'snapshot': {'id': 'subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Compute/snapshots/' + SNAPSHOT_NAME}, 'os_state': 'Generalized'}, 'zone_resilient': False}, 'hyper_v_generation': 'V1'}
        result = self.mgmt_client.images.begin_create_or_update(resource_group.name, IMAGE_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.snapshots.get(resource_group.name, SNAPSHOT_NAME)
        result = self.mgmt_client.images.get(resource_group.name, IMAGE_NAME)
        result = self.mgmt_client.images.list_by_resource_group(resource_group.name)
        result = self.mgmt_client.snapshots.list_by_resource_group(resource_group.name)
        result = self.mgmt_client.images.list()
        result = self.mgmt_client.snapshots.list()
        BODY = {'tags': {'department': 'HR'}}
        result = self.mgmt_client.images.begin_update(resource_group.name, IMAGE_NAME, BODY)
        result = result.result()
        BODY = {'creation_data': {'create_option': 'Copy', 'source_uri': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Compute/disks/' + DISK_NAME}}
        result = self.mgmt_client.snapshots.begin_update(resource_group.name, SNAPSHOT_NAME, BODY)
        result = result.result()
        BODY = {'access': 'Read', 'duration_in_seconds': '1800'}
        result = self.mgmt_client.snapshots.begin_grant_access(resource_group.name, SNAPSHOT_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.snapshots.begin_revoke_access(resource_group.name, SNAPSHOT_NAME)
        result = result.result()
        result = self.mgmt_client.images.begin_delete(resource_group.name, IMAGE_NAME)
        result = result.result()
        result = self.mgmt_client.snapshots.begin_delete(resource_group.name, SNAPSHOT_NAME)
        result = result.result()

    @pytest.mark.skipif(os.getenv('AZURE_TEST_RUN_LIVE') not in ('true', 'yes'), reason='only run live test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_disks(self, resource_group):
        if False:
            print('Hello World!')
        DISK_NAME = self.get_resource_name('disknamex')
        BODY = {'location': 'eastus', 'creation_data': {'create_option': 'Empty'}, 'disk_size_gb': '200'}
        result = self.mgmt_client.disks.begin_create_or_update(resource_group.name, DISK_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.disks.get(resource_group.name, DISK_NAME)
        result = self.mgmt_client.disks.list_by_resource_group(resource_group.name)
        result = self.mgmt_client.disks.list()
        BODY = {'disk_size_gb': '200'}
        result = self.mgmt_client.disks.begin_update(resource_group.name, DISK_NAME, BODY)
        result = result.result()
        BODY = {'access': 'Read', 'duration_in_seconds': '1800'}
        result = self.mgmt_client.disks.begin_grant_access(resource_group.name, DISK_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.disks.begin_revoke_access(resource_group.name, DISK_NAME)
        result = result.result()
        result = self.mgmt_client.disks.begin_delete(resource_group.name, DISK_NAME)
        result = result.result()