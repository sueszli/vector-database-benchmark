import os
import datetime as dt
import unittest
import pytest
import azure.mgmt.compute
from azure.core.exceptions import HttpResponseError
from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer, recorded_by_proxy
AZURE_LOCATION = 'eastus'

class TestMgmtCompute(AzureMgmtRecordedTestCase):

    def setup_method(self, method):
        if False:
            i = 10
            return i + 15
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.compute.ComputeManagementClient)
        from azure.mgmt.storage import StorageManagementClient
        self.storage_client = self.create_mgmt_client(StorageManagementClient)

    def create_sas_uri(self, group_name, location, storage_account_name):
        if False:
            print('Hello World!')
        from azure.mgmt.storage.models import BlobContainer
        from azure.storage.blob import generate_account_sas, AccountSasPermissions, ContainerClient, ResourceTypes
        BODY = {'sku': {'name': 'Standard_GRS'}, 'kind': 'StorageV2', 'location': location, 'encryption': {'services': {'file': {'key_type': 'Account', 'enabled': True}, 'blob': {'key_type': 'Account', 'enabled': True}}, 'key_source': 'Microsoft.Storage'}, 'tags': {'key1': 'value1', 'key2': 'value2'}}
        result = self.storage_client.storage_accounts.begin_create(group_name, storage_account_name, BODY)
        storage_account = result.result()
        keys = self.storage_client.storage_accounts.list_keys(group_name, storage_account_name).keys
        sas_token = generate_account_sas(account_name=storage_account_name, account_key=keys[0].value, resource_types=ResourceTypes(object=True), permission=AccountSasPermissions(read=True, list=True), start=dt.datetime.now() - dt.timedelta(hours=24), expiry=dt.datetime.now() - dt.timedelta(days=8))
        container_client = ContainerClient(storage_account.primary_endpoints.blob.rstrip('/'), credential='?' + sas_token, container_name='foo', blob_name='default')
        return container_client.url

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute(self, resource_group):
        if False:
            print('Hello World!')
        result = self.mgmt_client.operations.list()
        result = self.mgmt_client.usage.list(AZURE_LOCATION)
        result = self.mgmt_client.resource_skus.list()

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_availability_sets(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        AVAILABILITY_SET_NAME = self.get_resource_name('availabilitysets')
        BODY = {'location': 'eastus', 'platform_fault_domain_count': '2', 'platform_update_domain_count': '20'}
        result = self.mgmt_client.availability_sets.create_or_update(resource_group.name, AVAILABILITY_SET_NAME, BODY)
        result = self.mgmt_client.availability_sets.get(resource_group.name, AVAILABILITY_SET_NAME)
        result = self.mgmt_client.availability_sets.list_by_subscription()
        result = self.mgmt_client.availability_sets.list(resource_group.name)
        result = self.mgmt_client.availability_sets.list_available_sizes(resource_group.name, AVAILABILITY_SET_NAME)
        BODY = {'platform_fault_domain_count': '2', 'platform_update_domain_count': '20'}
        result = self.mgmt_client.availability_sets.update(resource_group.name, AVAILABILITY_SET_NAME, BODY)
        resout = self.mgmt_client.availability_sets.delete(resource_group.name, AVAILABILITY_SET_NAME)

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_proximity_placement_groups(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        PROXIMITY_PLACEMENT_GROUP_NAME = self.get_resource_name('proximiityplacementgroups')
        BODY = {'location': 'eastus', 'proximity_placement_group_type': 'Standard'}
        result = self.mgmt_client.proximity_placement_groups.create_or_update(resource_group.name, PROXIMITY_PLACEMENT_GROUP_NAME, BODY)
        result = self.mgmt_client.proximity_placement_groups.get(resource_group.name, PROXIMITY_PLACEMENT_GROUP_NAME)
        result = self.mgmt_client.proximity_placement_groups.list_by_resource_group(resource_group.name)
        result = self.mgmt_client.proximity_placement_groups.list_by_subscription()
        BODY = {'location': 'eastus', 'proximity_placement_group_type': 'Standard'}
        result = self.mgmt_client.proximity_placement_groups.update(resource_group.name, PROXIMITY_PLACEMENT_GROUP_NAME, BODY)
        result = self.mgmt_client.proximity_placement_groups.delete(resource_group.name, PROXIMITY_PLACEMENT_GROUP_NAME)