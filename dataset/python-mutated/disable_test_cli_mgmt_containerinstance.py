import unittest
import azure.mgmt.containerinstance
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtContainerInstanceTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(MgmtContainerInstanceTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.containerinstance.ContainerInstanceManagementClient)

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_containerinstance(self, resource_group):
        if False:
            i = 10
            return i + 15
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        TENANT_ID = self.settings.TENANT_ID
        RESOURCE_GROUP = resource_group.name
        CONTAINER_GROUP_NAME = 'myContainerGroup'
        CONTAINER_NAME = 'my-container'
        BODY = {'location': AZURE_LOCATION, 'identity': {'type': 'SystemAssigned, UserAssigned', 'user_assigned_identities': {}}, 'containers': [{'name': 'demo1', 'command': [], 'environment_variables': [], 'image': 'nginx', 'ports': [{'port': '80'}], 'resources': {'requests': {'cpu': '1', 'memory_in_gb': '1.5', 'gpu': {'count': '1', 'sku': 'K80'}}}, 'volume_mounts': [{'name': 'volume1', 'mount_path': '/mnt/volume1', 'read_only': False}, {'name': 'volume2', 'mount_path': '/mnt/volume2', 'read_only': False}, {'name': 'volume3', 'mount_path': '/mnt/volume3', 'read_only': True}]}], 'diagnostics': {'log_analytics': {'workspace_id': 'workspaceid', 'workspace_key': 'workspaceKey', 'log_type': 'ContainerInsights', 'metadata': {'test-key': 'test-metadata-value'}}}, 'network_profile': {'id': 'test-network-profile-id'}, 'dns_config': {'name_servers': ['1.1.1.1'], 'search_domains': 'cluster.local svc.cluster.local', 'options': 'ndots:2'}, 'image_registry_credentials': [], 'ip_address': {'ports': [{'protocol': 'TCP', 'port': '80'}], 'type': 'Public', 'dns_name_label': 'dnsnamelabel1'}, 'os_type': 'Linux', 'volumes': [{'name': 'volume1', 'azure_file': {'share_name': 'shareName', 'storage_account_name': 'accountName', 'storage_account_key': 'accountKey'}}, {'name': 'volume2'}, {'name': 'volume3', 'secret': {'secret_key1': 'SecretValue1InBase64', 'secret_key2': 'SecretValue2InBase64'}}]}
        BODY = {'location': AZURE_LOCATION, 'identity': {'type': 'SystemAssigned'}, 'containers': [{'name': CONTAINER_NAME, 'command': [], 'environment_variables': [], 'image': 'nginx', 'ports': [{'port': '80'}], 'resources': {'requests': {'cpu': '1', 'memory_in_gb': '1.5', 'gpu': {'count': '1', 'sku': 'K80'}}}, 'volume_mounts': [{'name': 'empty-volume', 'mount_path': '/mnt/mydir'}]}], 'diagnostics': {'log_analytics': {'workspace_id': 'workspaceid', 'workspace_key': 'workspaceKey'}}, 'os_type': 'Linux', 'restart_policy': 'OnFailure', 'volumes': [{'name': 'empty-volume', 'empty_dir': {}}]}
        result = self.mgmt_client.container_groups.begin_create_or_update(resource_group_name=RESOURCE_GROUP, container_group_name=CONTAINER_GROUP_NAME, container_group=BODY)
        result = result.result()
        result = self.mgmt_client.containers.list_logs(resource_group_name=RESOURCE_GROUP, container_group_name=CONTAINER_GROUP_NAME, container_name=CONTAINER_NAME, tail='10')
        result = self.mgmt_client.container_groups.get(resource_group_name=RESOURCE_GROUP, container_group_name=CONTAINER_GROUP_NAME)
        result = self.mgmt_client.container_groups.get(resource_group_name=RESOURCE_GROUP, container_group_name=CONTAINER_GROUP_NAME)
        result = self.mgmt_client.container_groups.list_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.location.list_cached_images(location=AZURE_LOCATION)
        result = self.mgmt_client.location.list_capabilities(location=AZURE_LOCATION)
        result = self.mgmt_client.location.list_usage(location=AZURE_LOCATION)
        result = self.mgmt_client.container_groups.list()
        result = self.mgmt_client.operations.list()
        BODY = {'command': '/bin/bash', 'terminal_size': {'rows': '12', 'cols': '12'}}
        result = self.mgmt_client.containers.execute_command(resource_group_name=RESOURCE_GROUP, container_group_name=CONTAINER_GROUP_NAME, container_name=CONTAINER_NAME, container_exec_request=BODY)
        result = self.mgmt_client.container_groups.begin_restart(resource_group_name=RESOURCE_GROUP, container_group_name=CONTAINER_GROUP_NAME)
        result = result.result()
        result = self.mgmt_client.container_groups.begin_start(resource_group_name=RESOURCE_GROUP, container_group_name=CONTAINER_GROUP_NAME)
        result = result.result()
        result = self.mgmt_client.container_groups.stop(resource_group_name=RESOURCE_GROUP, container_group_name=CONTAINER_GROUP_NAME)
        BODY = {'tags': {'tag1key': 'tag1Value', 'tag2key': 'tag2Value'}}
        result = self.mgmt_client.container_groups.update(resource_group_name=RESOURCE_GROUP, container_group_name=CONTAINER_GROUP_NAME, resource=BODY)
        result = self.mgmt_client.container_groups.begin_delete(resource_group_name=RESOURCE_GROUP, container_group_name=CONTAINER_GROUP_NAME)
        result = result.result()
if __name__ == '__main__':
    unittest.main()