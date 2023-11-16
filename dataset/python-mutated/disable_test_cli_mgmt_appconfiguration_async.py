import time
import unittest
import azure.mgmt.appconfiguration
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer
from _aio_testcase import AzureMgmtAsyncTestCase
AZURE_LOCATION = 'eastus'
KEY_UUID = 'test_key_a6af8952-54a6-11e9-b600-2816a84d0309'
LABEL_UUID = '1d7b2b28-549e-11e9-b51c-2816a84d0309'
KEY = 'PYTHON_UNIT_' + KEY_UUID
LABEL = 'test_label1_' + LABEL_UUID
TEST_CONTENT_TYPE = 'test content type'
TEST_VALUE = 'test value'

class MgmtAppConfigurationTest(AzureMgmtAsyncTestCase):

    def setUp(self):
        if False:
            return 10
        super(MgmtAppConfigurationTest, self).setUp()
        from azure.mgmt.appconfiguration.aio import AppConfigurationManagementClient
        self.mgmt_client = self.create_mgmt_aio_client(AppConfigurationManagementClient)
        if self.is_live:
            import azure.mgmt.network
            self.network_client = self.create_mgmt_client(azure.mgmt.network.NetworkManagementClient)

    def create_kv(self, connection_str):
        if False:
            i = 10
            return i + 15
        from azure.appconfiguration import AzureAppConfigurationClient, ConfigurationSetting
        app_config_client = AzureAppConfigurationClient.from_connection_string(connection_str)
        kv = ConfigurationSetting(key=KEY, label=LABEL, value=TEST_VALUE, content_type=TEST_CONTENT_TYPE, tags={'tag1': 'tag1', 'tag2': 'tag2'})
        created_kv = app_config_client.add_configuration_setting(kv)
        return created_kv

    def create_endpoint(self, group_name, vnet_name, sub_net, endpoint_name, conf_store_id):
        if False:
            return 10
        async_vnet_creation = self.network_client.virtual_networks.create_or_update(group_name, vnet_name, {'location': AZURE_LOCATION, 'address_space': {'address_prefixes': ['10.0.0.0/16']}})
        async_vnet_creation.wait()
        async_subnet_creation = self.network_client.subnets.create_or_update(group_name, vnet_name, sub_net, {'address_prefix': '10.0.0.0/24', 'private_link_service_network_policies': 'disabled', 'private_endpoint_network_policies': 'disabled'})
        subnet_info = async_subnet_creation.result()
        BODY = {'location': 'eastus', 'properties': {'privateLinkServiceConnections': [{'name': 'myconnection', 'private_link_service_id': conf_store_id, 'group_ids': ['configurationStores']}], 'subnet': {'id': '/subscriptions/' + self.settings.SUBSCRIPTION_ID + '/resourceGroups/' + group_name + '/providers/Microsoft.Network/virtualNetworks/' + vnet_name + '/subnets/' + sub_net}}}
        result = self.network_client.private_endpoints.create_or_update(group_name, endpoint_name, BODY)
        return result.result()

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_appconfiguration_list_key_values(self, resource_group):
        if False:
            print('Hello World!')
        CONFIGURATION_STORE_NAME = self.get_resource_name('configuration')
        BODY = {'location': 'westus', 'sku': {'name': 'Standard'}, 'tags': {'my_tag': 'myTagValue'}}
        result = self.event_loop.run_until_complete(self.mgmt_client.configuration_stores.begin_create(resource_group.name, CONFIGURATION_STORE_NAME, BODY))
        result = self.event_loop.run_until_complete(result.result())
        keys = self.to_list(self.mgmt_client.configuration_stores.list_keys(resource_group.name, CONFIGURATION_STORE_NAME))
        BODY = {'id': keys[0].id}
        key = self.event_loop.run_until_complete(self.mgmt_client.configuration_stores.regenerate_key(resource_group.name, CONFIGURATION_STORE_NAME, BODY))
        if self.is_live:
            self.create_kv(key.connection_string)
        BODY = {'key': KEY, 'label': LABEL}
        result = self.event_loop.run_until_complete(self.mgmt_client.configuration_stores.list_key_value(resource_group.name, CONFIGURATION_STORE_NAME, BODY))

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_appconfiguration(self, resource_group):
        if False:
            return 10
        SERVICE_NAME = 'myapimrndxyz'
        VNET_NAME = 'vnetnamexxy'
        SUB_NET = 'subnetnamexxy'
        ENDPOINT_NAME = 'endpointxyz'
        CONFIGURATION_STORE_NAME = self.get_resource_name('configuration')
        PRIVATE_ENDPOINT_CONNECTION_NAME = self.get_resource_name('privateendpoint')
        BODY = {'location': 'westus', 'sku': {'name': 'Standard'}, 'tags': {'my_tag': 'myTagValue'}}
        result = self.event_loop.run_until_complete(self.mgmt_client.configuration_stores.begin_create(resource_group.name, CONFIGURATION_STORE_NAME, BODY))
        conf_store = self.event_loop.run_until_complete(result.result())
        if self.is_live:
            endpoint = self.create_endpoint(resource_group.name, VNET_NAME, SUB_NET, ENDPOINT_NAME, conf_store.id)
        conf_store = self.event_loop.run_until_complete(self.mgmt_client.configuration_stores.get(resource_group.name, CONFIGURATION_STORE_NAME))
        PRIVATE_ENDPOINT_CONNECTION_NAME = conf_store.private_endpoint_connections[0].name
        private_connection_id = conf_store.private_endpoint_connections[0].id
        BODY = {'id': private_connection_id, 'private_endpoint': {'id': '/subscriptions/' + self.settings.SUBSCRIPTION_ID + '/resourceGroups/' + resource_group.name + '/providers/Microsoft.Network/privateEndpoints/' + ENDPOINT_NAME}, 'private_link_service_connection_state': {'status': 'Approved', 'description': 'Auto-Approved'}}
        result = self.event_loop.run_until_complete(self.mgmt_client.private_endpoint_connections.begin_create_or_update(resource_group.name, CONFIGURATION_STORE_NAME, PRIVATE_ENDPOINT_CONNECTION_NAME, BODY))
        result = self.event_loop.run_until_complete(result.result())
        result = self.event_loop.run_until_complete(self.mgmt_client.private_endpoint_connections.get(resource_group.name, CONFIGURATION_STORE_NAME, PRIVATE_ENDPOINT_CONNECTION_NAME))
        privatelinks = self.to_list(self.mgmt_client.private_link_resources.list_by_configuration_store(resource_group.name, CONFIGURATION_STORE_NAME))
        PRIVATE_LINK_RESOURCE_NAME = privatelinks[0].name
        self.event_loop.run_until_complete(self.mgmt_client.private_link_resources.get(resource_group.name, CONFIGURATION_STORE_NAME, PRIVATE_LINK_RESOURCE_NAME))
        result = self.to_list(self.mgmt_client.private_endpoint_connections.list_by_configuration_store(resource_group.name, CONFIGURATION_STORE_NAME))
        result = self.to_list(self.mgmt_client.operations.list())
        result = self.to_list(self.mgmt_client.configuration_stores.list_by_resource_group(resource_group.name))
        result = self.to_list(self.mgmt_client.configuration_stores.list())
        BODY = {'tags': {'category': 'Marketing'}, 'sku': {'name': 'Standard'}}
        result = self.event_loop.run_until_complete(self.mgmt_client.configuration_stores.begin_update(resource_group.name, CONFIGURATION_STORE_NAME, BODY))
        result = self.event_loop.run_until_complete(result.result())
        BODY = {'name': 'contoso', 'type': 'Microsoft.AppConfiguration/configurationStores'}
        result = self.event_loop.run_until_complete(self.mgmt_client.operations.check_name_availability(BODY))
        BODY = {'name': 'contoso', 'type': 'Microsoft.AppConfiguration/configurationStores'}
        result = self.event_loop.run_until_complete(self.mgmt_client.operations.check_name_availability(BODY))
        result = self.event_loop.run_until_complete(self.mgmt_client.private_endpoint_connections.begin_delete(resource_group.name, CONFIGURATION_STORE_NAME, PRIVATE_ENDPOINT_CONNECTION_NAME))
        result = self.event_loop.run_until_complete(result.result())
        result = self.event_loop.run_until_complete(self.mgmt_client.configuration_stores.begin_delete(resource_group.name, CONFIGURATION_STORE_NAME))
        result = self.event_loop.run_until_complete(result.result())
if __name__ == '__main__':
    unittest.main()