import unittest
import pytest
import time
import azure.mgmt.containerservice
from devtools_testutils import AzureMgmtRecordedTestCase, ResourceGroupPreparer, recorded_by_proxy
AZURE_LOCATION = 'eastus'

class TestMgmtContainerServiceClient(AzureMgmtRecordedTestCase):

    def setup_method(self, method):
        if False:
            return 10
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.containerservice.ContainerServiceClient)

    @pytest.mark.skip('hard to test')
    @ResourceGroupPreparer()
    def test_managed_clusters(self, resource_group):
        if False:
            while True:
                i = 10
        CLIENT_ID = getattr(self.settings, 'CLIENT_ID', '123')
        CLIENT_SECRET = getattr(self.settings, 'CLIENT_SECRET', '123')
        RESOURCE_GROUP = resource_group.name
        RESOURCE_NAME = '7'
        BODY = {'dns_prefix': 'akspythonsdk', 'agent_pool_profiles': [{'name': 'aksagent', 'count': 1, 'vm_size': 'Standard_DS2_v2', 'max_pods': 110, 'min_count': 1, 'max_count': 100, 'os_type': 'Linux', 'type': 'VirtualMachineScaleSets', 'enable_auto_scaling': True, 'mode': 'System'}], 'service_principal_profile': {'client_id': CLIENT_ID, 'secret': CLIENT_SECRET}, 'location': AZURE_LOCATION}
        for i in range(10):
            try:
                result = self.mgmt_client.managed_clusters.begin_create_or_update(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME, parameters=BODY)
                result.result()
            except azure.core.exceptions.ResourceExistsError:
                time.sleep(30)
            else:
                break
        self.mgmt_client.managed_clusters.list_cluster_admin_credentials(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME)
        self.mgmt_client.managed_clusters.list_cluster_user_credentials(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME)
        self.mgmt_client.managed_clusters.get_upgrade_profile(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME)
        self.mgmt_client.managed_clusters.get(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME)
        self.mgmt_client.managed_clusters.list_by_resource_group(resource_group_name=RESOURCE_GROUP)
        self.mgmt_client.managed_clusters.list()
        result.result()
        BODY = {'tags': {'tier': 'testing', 'archv3': ''}}
        result = self.mgmt_client.managed_clusters.begin_update_tags(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME, parameters=BODY)
        result.result()
        self.mgmt_client.managed_clusters.list_cluster_monitoring_user_credentials(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME)
        result = self.mgmt_client.managed_clusters.begin_delete(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME)
        result.result()

    @ResourceGroupPreparer()
    @recorded_by_proxy
    def test_operations(self):
        if False:
            while True:
                i = 10
        result = list(self.mgmt_client.operations.list())
        for item in result:
            print(item.as_dict())

    @pytest.mark.skip('hard to test')
    @ResourceGroupPreparer()
    def test_privateLinkResources(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        CLIENT_ID = getattr(self.settings, 'CLIENT_ID', '123')
        CLIENT_SECRET = getattr(self.settings, 'CLIENT_SECRET', '123')
        RESOURCE_GROUP = resource_group.name
        RESOURCE_NAME = '2'
        BODY = {'dns_prefix': 'akspythonsdk', 'agent_pool_profiles': [{'name': 'aksagent', 'count': 1, 'vm_size': 'Standard_DS2_v2', 'max_pods': 110, 'min_count': 1, 'max_count': 100, 'os_type': 'Linux', 'type': 'VirtualMachineScaleSets', 'enable_auto_scaling': True, 'mode': 'System'}], 'api_server_access_profile': {'enable_private_cluster': True}, 'service_principal_profile': {'client_id': CLIENT_ID, 'secret': CLIENT_SECRET}, 'location': AZURE_LOCATION}
        for i in range(10):
            try:
                result = self.mgmt_client.managed_clusters.begin_create_or_update(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME, parameters=BODY)
                result.result()
            except azure.core.exceptions.ResourceExistsError:
                time.sleep(30)
            else:
                break
        self.mgmt_client.private_link_resources.list(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME)

    @pytest.mark.skip('hard to test')
    @ResourceGroupPreparer()
    def test_resolvePrivateLinkServiceId(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        CLIENT_ID = getattr(self.settings, 'CLIENT_ID', '123')
        CLIENT_SECRET = getattr(self.settings, 'CLIENT_SECRET', '123')
        RESOURCE_GROUP = resource_group.name
        RESOURCE_NAME = '3'
        BODY = {'dns_prefix': 'akspythonsdk', 'agent_pool_profiles': [{'name': 'aksagent', 'count': 1, 'vm_size': 'Standard_DS2_v2', 'max_pods': 110, 'min_count': 1, 'max_count': 100, 'os_type': 'Linux', 'type': 'VirtualMachineScaleSets', 'enable_auto_scaling': True, 'mode': 'System'}], 'api_server_access_profile': {'enable_private_cluster': True}, 'service_principal_profile': {'client_id': CLIENT_ID, 'secret': CLIENT_SECRET}, 'location': AZURE_LOCATION}
        for i in range(10):
            try:
                result = self.mgmt_client.managed_clusters.begin_create_or_update(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME, parameters=BODY)
                result.result()
            except azure.core.exceptions.ResourceExistsError:
                time.sleep(30)
            else:
                break
        BODY = {'name': 'testManagement'}
        self.mgmt_client.resolve_private_link_service_id.post(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME, parameters=BODY)

    @pytest.mark.skip('hard to test')
    @ResourceGroupPreparer()
    def test_agentPools(self, resource_group):
        if False:
            print('Hello World!')
        CLIENT_ID = getattr(self.settings, 'CLIENT_ID', '123')
        CLIENT_SECRET = getattr(self.settings, 'CLIENT_SECRET', '123')
        RESOURCE_GROUP = resource_group.name
        RESOURCE_NAME = '4'
        AGENT_POOL_NAME = 'aksagent'
        MODE = 'System'
        VM_SIZE = 'Standard_DS2_v2'
        BODY = {'dns_prefix': 'akspythonsdk', 'agent_pool_profiles': [{'name': 'aksagent', 'count': 1, 'vm_size': 'Standard_DS2_v2', 'max_pods': 110, 'min_count': 1, 'max_count': 100, 'os_type': 'Linux', 'type': 'VirtualMachineScaleSets', 'enable_auto_scaling': True, 'mode': 'System'}], 'service_principal_profile': {'client_id': CLIENT_ID, 'secret': CLIENT_SECRET}, 'location': AZURE_LOCATION}
        result = self.mgmt_client.managed_clusters.begin_create_or_update(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME, parameters=BODY)
        result.result()
        BODY = {'orchestrator_version': '', 'count': '3', 'vm_size': VM_SIZE, 'os_type': 'Linux', 'type': 'VirtualMachineScaleSets', 'mode': MODE, 'availability_zones': ['1', '2', '3'], 'node_taints': []}
        for i in range(10):
            try:
                result = self.mgmt_client.agent_pools.begin_create_or_update(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME, agent_pool_name=AGENT_POOL_NAME, parameters=BODY)
                result = result.result()
            except azure.core.exceptions.ResourceExistsError:
                time.sleep(30)
            else:
                break
        self.mgmt_client.agent_pools.get(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME, agent_pool_name=AGENT_POOL_NAME)
        self.mgmt_client.agent_pools.get_available_agent_pool_versions(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME)
        self.mgmt_client.agent_pools.list(resource_group_name=RESOURCE_GROUP, resource_name=RESOURCE_NAME)
if __name__ == '__main__':
    unittest.main()