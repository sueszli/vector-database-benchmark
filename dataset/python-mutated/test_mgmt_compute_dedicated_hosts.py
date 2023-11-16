import unittest
import azure.mgmt.compute
from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer, recorded_by_proxy
AZURE_LOCATION = 'eastus'

class TestMgmtCompute(AzureMgmtRecordedTestCase):

    def setup_method(self, method):
        if False:
            i = 10
            return i + 15
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.compute.ComputeManagementClient)

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_dedicated_hosts(self, resource_group):
        if False:
            print('Hello World!')
        HOST_GROUP_NAME = self.get_resource_name('hostgroup')
        HOST_NAME = self.get_resource_name('hostname')
        BODY = {'location': 'eastus', 'tags': {'department': 'finance'}, 'zones': ['1'], 'platform_fault_domain_count': '3'}
        result = self.mgmt_client.dedicated_host_groups.create_or_update(resource_group.name, HOST_GROUP_NAME, BODY)
        BODY = {'location': 'eastus', 'tags': {'department': 'HR'}, 'platform_fault_domain': '1', 'sku': {'name': 'DSv3-Type1'}}
        result = self.mgmt_client.dedicated_hosts.begin_create_or_update(resource_group.name, HOST_GROUP_NAME, HOST_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.dedicated_host_groups.get(resource_group.name, HOST_GROUP_NAME)
        result = self.mgmt_client.dedicated_hosts.get(resource_group.name, HOST_GROUP_NAME, HOST_NAME)
        result = self.mgmt_client.dedicated_host_groups.list_by_resource_group(resource_group.name)
        result = self.mgmt_client.dedicated_hosts.list_by_host_group(resource_group.name, HOST_GROUP_NAME)
        result = self.mgmt_client.dedicated_host_groups.list_by_subscription()
        BODY = {'tags': {'department': 'finance'}, 'platform_fault_domain_count': '3'}
        result = self.mgmt_client.dedicated_host_groups.update(resource_group.name, HOST_GROUP_NAME, BODY)
        BODY = {'tags': {'department': 'HR'}}
        result = self.mgmt_client.dedicated_hosts.begin_update(resource_group.name, HOST_GROUP_NAME, HOST_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.dedicated_hosts.begin_delete(resource_group.name, HOST_GROUP_NAME, HOST_NAME)
        result = result.result()
        result = self.mgmt_client.dedicated_host_groups.delete(resource_group.name, HOST_GROUP_NAME)