import unittest
import azure.mgmt.containerregistry
from azure.core.exceptions import HttpResponseError
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtRegistryTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(MgmtRegistryTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.containerregistry.ContainerRegistryManagementClient, api_version='2019-12-01-preview')

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_replications(self, resource_group):
        if False:
            print('Hello World!')
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        TENANT_ID = self.settings.TENANT_ID
        RESOURCE_GROUP = resource_group.name
        REGISTRY_NAME = 'myRegistry'
        REPLICATION_NAME = 'myReplication'
        BODY = {'location': 'westus', 'tags': {'key': 'value'}, 'sku': {'name': 'Premium'}, 'admin_user_enabled': True}
        result = self.mgmt_client.registries.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, registry=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'tags': {'key': 'value'}}
        result = self.mgmt_client.replications.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, replication_name=REPLICATION_NAME, replication=BODY)
        result = result.result()
        result = self.mgmt_client.replications.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, replication_name=REPLICATION_NAME)
        result = self.mgmt_client.replications.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        BODY = {'tags': {'key': 'value'}}
        result = self.mgmt_client.replications.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, replication_name=REPLICATION_NAME, replication_update_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.replications.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, replication_name=REPLICATION_NAME)
        result = result.result()
        result = self.mgmt_client.registries.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = result.result()

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_webhooks(self, resource_group):
        if False:
            print('Hello World!')
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        TENANT_ID = self.settings.TENANT_ID
        RESOURCE_GROUP = resource_group.name
        REGISTRY_NAME = 'myRegistry'
        WEBHOOK_NAME = 'myWebhook'
        BODY = {'location': AZURE_LOCATION, 'tags': {'key': 'value'}, 'sku': {'name': 'Standard'}, 'admin_user_enabled': True}
        result = self.mgmt_client.registries.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, registry=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'service_uri': 'http://www.microsoft.com', 'status': 'enabled', 'actions': ['push']}
        result = self.mgmt_client.webhooks.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME, webhook_create_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.webhooks.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME)
        result = self.mgmt_client.webhooks.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.webhooks.get_callback_config(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME)
        result = self.mgmt_client.webhooks.list_events(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME)
        result = self.mgmt_client.webhooks.ping(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME)
        BODY = {'location': AZURE_LOCATION, 'service_uri': 'http://www.microsoft.com', 'status': 'enabled', 'actions': ['push']}
        result = self.mgmt_client.webhooks.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME, webhook_update_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.webhooks.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME)
        result = result.result()
        result = self.mgmt_client.registries.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = result.result()

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_agent_pools(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        RESOURCE_GROUP = resource_group.name
        REGISTRY_NAME = 'myRegistry'
        AGENT_POOL_NAME = 'myagentpool'
        BODY = {'location': AZURE_LOCATION, 'tags': {'key': 'value'}, 'sku': {'name': 'Premium'}, 'admin_user_enabled': False}
        result = self.mgmt_client.registries.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, registry=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'tags': {'key': 'value'}, 'count': '1', 'tier': 'S1', 'os': 'Linux'}
        result = self.mgmt_client.agent_pools.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, agent_pool_name=AGENT_POOL_NAME, agent_pool=BODY)
        result = result.result()
        result = self.mgmt_client.agent_pools.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, agent_pool_name=AGENT_POOL_NAME)
        result = self.mgmt_client.agent_pools.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.agent_pools.get_queue_status(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, agent_pool_name=AGENT_POOL_NAME)
        BODY = {'count': '1'}
        result = self.mgmt_client.agent_pools.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, agent_pool_name=AGENT_POOL_NAME, update_parameters=BODY)
        result = result.result()
        try:
            result = self.mgmt_client.agent_pools.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, agent_pool_name=AGENT_POOL_NAME)
            result = result.result()
        except HttpResponseError as e:
            if not str(e).startswith('(ResourceNotFound)'):
                raise e
        result = self.mgmt_client.registries.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = result.result()

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_scope_maps_and_tokens(self, resource_group):
        if False:
            i = 10
            return i + 15
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        RESOURCE_GROUP = resource_group.name
        REGISTRY_NAME = 'myRegistry'
        SCOPE_MAP_NAME = 'myScopeMap'
        TOKEN_NAME = 'myToken'
        BODY = {'location': AZURE_LOCATION, 'tags': {'key': 'value'}, 'sku': {'name': 'Premium'}, 'admin_user_enabled': False}
        result = self.mgmt_client.registries.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, registry=BODY)
        result = result.result()
        BODY = {'description': 'Developer Scopes', 'actions': ['repositories/foo/content/read', 'repositories/foo/content/delete']}
        result = self.mgmt_client.scope_maps.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, scope_map_name=SCOPE_MAP_NAME, scope_map_create_parameters=BODY)
        result = result.result()
        BODY = {'scope_map_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.ContainerRegistry/registries/' + REGISTRY_NAME + '/scopeMaps/' + SCOPE_MAP_NAME, 'status': 'enabled'}
        result = self.mgmt_client.tokens.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, token_name=TOKEN_NAME, token_create_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.tokens.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, token_name=TOKEN_NAME)
        result = self.mgmt_client.scope_maps.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, scope_map_name=SCOPE_MAP_NAME)
        result = self.mgmt_client.tokens.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.scope_maps.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        BODY = {'description': 'Developer Scopes', 'actions': ['repositories/foo/content/read', 'repositories/foo/content/delete']}
        result = self.mgmt_client.scope_maps.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, scope_map_name=SCOPE_MAP_NAME, scope_map_update_parameters=BODY)
        result = result.result()
        BODY = {'scope_map_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.ContainerRegistry/registries/' + REGISTRY_NAME + '/scopeMaps/' + SCOPE_MAP_NAME}
        result = self.mgmt_client.tokens.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, token_name=TOKEN_NAME, token_update_parameters=BODY)
        result = result.result()
        BODY = {'token_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.ContainerRegistry/registries/' + REGISTRY_NAME + '/tokens/' + TOKEN_NAME, 'expiry': '2020-12-31T15:59:59.0707808Z'}
        result = self.mgmt_client.registries.begin_generate_credentials(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, generate_credentials_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.tokens.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, token_name=TOKEN_NAME)
        result = result.result()
        result = self.mgmt_client.scope_maps.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, scope_map_name=SCOPE_MAP_NAME)
        result = result.result()
        result = self.mgmt_client.registries.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = result.result()

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_registries(self, resource_group):
        if False:
            i = 10
            return i + 15
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        RESOURCE_GROUP = resource_group.name
        REGISTRY_NAME = 'myRegistry'
        TOKEN_NAME = 'myToken'
        BODY = {'location': AZURE_LOCATION, 'tags': {'key': 'value'}, 'sku': {'name': 'Standard'}, 'admin_user_enabled': True}
        result = self.mgmt_client.registries.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, registry=BODY)
        result = result.result()
        result = self.mgmt_client.registries.list_private_link_resources(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.registries.list_usages(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.registries.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.registries.list_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.registries.list()
        result = self.mgmt_client.registries.get_build_source_upload_url(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        BODY = {'name': 'password'}
        result = self.mgmt_client.registries.regenerate_credential(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, regenerate_credential_parameters=BODY)
        result = self.mgmt_client.registries.list_credentials(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        BODY = {'source': {'resource_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.ContainerRegistry/registries/' + REGISTRY_NAME, 'source_image': 'sourceRepository@sha256:0000000000000000000000000000000000000000000000000000000000000000'}, 'target_tags': ['targetRepository:targetTag'], 'untagged_target_repositories': ['targetRepository1'], 'mode': 'Force'}
        BODY = {'type': 'EncodedTaskRunRequest', 'values': [{'name': 'mytestargument', 'value': 'mytestvalue', 'is_secret': False}, {'name': 'mysecrettestargument', 'value': 'mysecrettestvalue', 'is_secret': True}], 'platform': {'os': 'Linux'}, 'agent_configuration': {'cpu': '2'}, 'encoded_task_content': 'c3RlcHM6Cnt7IGlmIFZhbHVlcy5lbnZpcm9ubWVudCA9PSAncHJvZCcgfX0KICAtIHJ1bjogcHJvZCBzZXR1cAp7eyBlbHNlIGlmIFZhbHVlcy5lbnZpcm9ubWVudCA9PSAnc3RhZ2luZycgfX0KICAtIHJ1bjogc3RhZ2luZyBzZXR1cAp7eyBlbHNlIH19CiAgLSBydW46IGRlZmF1bHQgc2V0dXAKe3sgZW5kIH19CgogIC0gcnVuOiBidWlsZCAtdCBGYW5jeVRoaW5nOnt7LlZhbHVlcy5lbnZpcm9ubWVudH19LXt7LlZhbHVlcy52ZXJzaW9ufX0gLgoKcHVzaDogWydGYW5jeVRoaW5nOnt7LlZhbHVlcy5lbnZpcm9ubWVudH19LXt7LlZhbHVlcy52ZXJzaW9ufX0nXQ==', 'encoded_values_content': 'ZW52aXJvbm1lbnQ6IHByb2QKdmVyc2lvbjogMQ=='}
        result = self.mgmt_client.registries.begin_schedule_run(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_request=BODY)
        result = result.result()
        BODY = {'tags': {'key': 'value'}, 'sku': {'name': 'Standard'}, 'admin_user_enabled': True}
        result = self.mgmt_client.registries.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, registry_update_parameters=BODY)
        result = result.result()
        BODY = {'name': 'myRegistry', 'type': 'Microsoft.ContainerRegistry/registries'}
        result = self.mgmt_client.registries.check_name_availability(registry_name_check_request=BODY)
        result = self.mgmt_client.operations.list()
        result = self.mgmt_client.registries.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = result.result()