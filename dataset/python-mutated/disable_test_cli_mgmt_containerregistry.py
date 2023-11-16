import unittest
import azure.mgmt.containerregistry
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtContainerRegistryTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(MgmtContainerRegistryTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.containerregistry.ContainerRegistryManagementClient)

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_containerregistry(self, resource_group):
        if False:
            return 10
        UNIQUE = resource_group.name[-4:]
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        TENANT_ID = self.settings.TENANT_ID
        RESOURCE_GROUP = resource_group.name
        REGISTRY_NAME = 'myRegistry' + UNIQUE
        EXPORT_PIPELINE_NAME = 'myExportPipeline'
        IMPORT_PIPELINE_NAME = 'myImportPipeline'
        PIPELINE_RUN_NAME = 'myPipelineRun'
        PRIVATE_ENDPOINT_CONNECTION_NAME = 'myPrivateEndpointConnection'
        REPLICATION_NAME = 'myReplication'
        WEBHOOK_NAME = 'myWebhook'
        AGENT_POOL_NAME = 'myAgentPool'
        RUN_ID = 'myRunId'
        TASK_RUN_NAME = 'myTaskRun'
        TASK_NAME = 'myTask'
        SCOPE_MAP_NAME = 'myScopeMap'
        TOKEN_NAME = 'myToken'
        BODY = {'location': AZURE_LOCATION, 'tags': {'key': 'value'}, 'sku': {'name': 'Standard'}, 'admin_user_enabled': True}
        result = self.mgmt_client.registries.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, registry=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'identity': {'type': 'SystemAssigned'}, 'tags': {'testkey': 'value'}, 'status': 'Enabled', 'platform': {'os': 'Linux', 'architecture': 'amd64'}, 'agent_configuration': {'cpu': '2'}, 'step': {'type': 'Docker', 'image_names': ['azurerest:testtag'], 'docker_file_path': 'src/DockerFile', 'context_path': 'src', 'is_push_enabled': True, 'no_cache': False, 'arguments': [{'name': 'mytestargument', 'value': 'mytestvalue', 'is_secret': False}, {'name': 'mysecrettestargument', 'value': 'mysecrettestvalue', 'is_secret': True}]}, 'trigger': {'timer_triggers': [{'name': 'myTimerTrigger', 'schedule': '30 9 * * 1-5'}], 'source_triggers': [{'name': 'mySourceTrigger', 'source_repository': {'source_control_type': 'Github', 'repository_url': 'https://github.com/Azure/azure-rest-api-specs', 'branch': 'master', 'source_control_auth_properties': {'token_type': 'PAT', 'token': 'xxxxx'}}, 'source_trigger_events': ['commit']}], 'base_image_trigger': {'name': 'myBaseImageTrigger', 'base_image_trigger_type': 'Runtime'}}}
        BODY = {'location': AZURE_LOCATION, 'identity': {'type': 'SystemAssigned, UserAssigned', 'user_assigned_identities': {}}, 'tags': {'testkey': 'value'}, 'status': 'Enabled', 'platform': {'os': 'Linux', 'architecture': 'amd64'}, 'agent_configuration': {'cpu': '2'}, 'step': {'type': 'Docker', 'image_names': ['azurerest:testtag'], 'docker_file_path': 'src/DockerFile', 'context_path': 'src', 'is_push_enabled': True, 'no_cache': False, 'arguments': [{'name': 'mytestargument', 'value': 'mytestvalue', 'is_secret': False}, {'name': 'mysecrettestargument', 'value': 'mysecrettestvalue', 'is_secret': True}]}, 'trigger': {'timer_triggers': [{'name': 'myTimerTrigger', 'schedule': '30 9 * * 1-5'}], 'source_triggers': [{'name': 'mySourceTrigger', 'source_repository': {'source_control_type': 'Github', 'repository_url': 'https://github.com/Azure/azure-rest-api-specs', 'branch': 'master', 'source_control_auth_properties': {'token_type': 'PAT', 'token': 'xxxxx'}}, 'source_trigger_events': ['commit']}], 'base_image_trigger': {'name': 'myBaseImageTrigger', 'base_image_trigger_type': 'Runtime', 'update_trigger_endpoint': 'https://user:pass@mycicd.webhook.com?token=foo', 'update_trigger_payload_type': 'Default'}}}
        BODY = {'location': AZURE_LOCATION, 'identity': {'type': 'SystemAssigned'}, 'tags': {'testkey': 'value'}, 'status': 'Enabled', 'platform': {'os': 'Linux', 'architecture': 'amd64'}, 'agent_configuration': {'cpu': '2'}, 'step': {'type': 'Docker', 'image_names': ['azurerest:testtag'], 'docker_file_path': 'src/DockerFile', 'context_path': 'src', 'is_push_enabled': True, 'no_cache': False, 'arguments': [{'name': 'mytestargument', 'value': 'mytestvalue', 'is_secret': False}, {'name': 'mysecrettestargument', 'value': 'mysecrettestvalue', 'is_secret': True}]}, 'trigger': {'timer_triggers': [{'name': 'myTimerTrigger', 'schedule': '30 9 * * 1-5'}], 'source_triggers': [{'name': 'mySourceTrigger', 'source_repository': {'source_control_type': 'Github', 'repository_url': 'https://github.com/Azure/azure-rest-api-specs', 'branch': 'master', 'source_control_auth_properties': {'token_type': 'PAT', 'token': 'xxxxx'}}, 'source_trigger_events': ['commit']}], 'base_image_trigger': {'name': 'myBaseImageTrigger', 'base_image_trigger_type': 'Runtime', 'update_trigger_endpoint': 'https://user:pass@mycicd.webhook.com?token=foo', 'update_trigger_payload_type': 'Token'}}}
        BODY = {'location': AZURE_LOCATION, 'identity': {'type': 'UserAssigned', 'user_assigned_identities': {}}, 'tags': {'testkey': 'value'}, 'status': 'Enabled', 'platform': {'os': 'Linux', 'architecture': 'amd64'}, 'agent_configuration': {'cpu': '2'}, 'step': {'type': 'Docker', 'image_names': ['azurerest:testtag'], 'docker_file_path': 'src/DockerFile', 'context_path': 'src', 'is_push_enabled': True, 'no_cache': False, 'arguments': [{'name': 'mytestargument', 'value': 'mytestvalue', 'is_secret': False}, {'name': 'mysecrettestargument', 'value': 'mysecrettestvalue', 'is_secret': True}]}, 'trigger': {'timer_triggers': [{'name': 'myTimerTrigger', 'schedule': '30 9 * * 1-5'}], 'source_triggers': [{'name': 'mySourceTrigger', 'source_repository': {'source_control_type': 'Github', 'repository_url': 'https://github.com/Azure/azure-rest-api-specs', 'branch': 'master', 'source_control_auth_properties': {'token_type': 'PAT', 'token': 'xxxxx'}}, 'source_trigger_events': ['commit']}], 'base_image_trigger': {'name': 'myBaseImageTrigger', 'base_image_trigger_type': 'Runtime', 'update_trigger_endpoint': 'https://user:pass@mycicd.webhook.com?token=foo', 'update_trigger_payload_type': 'Default'}}}
        BODY = {'scope_map_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.ContainerRegistry/registries/' + REGISTRY_NAME + '/scopeMaps/' + SCOPE_MAP_NAME, 'status': 'disabled', 'credentials': {'certificates': [{'name': 'certificate1', 'encoded_pem_certificate': 'LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUc3akNDQk5hZ0F3SUJBZ0lURmdBQlR3UVpyZGdmdmhxdzBnQUFBQUZQQkRBTkJna3Foa2lHOXcwQkFRc0YKQURDQml6RUxNQWtHQTFVRUJoTUNWVk14RXpBUkJnTlZCQWdUQ2xkaGMyaHBibWQwYjI0eEVEQU9CZ05WQkFjVApCMUpsWkcxdmJtUXhIakFjQmdOVkJBb1RGVTFwWTNKdmMyOW1kQ0JEYjNKd2IzSmhkR2x2YmpFVk1CTUdBMVVFCkN4TU1UV2xqY205emIyWjBJRWxVTVI0d0hBWURWUVFERXhWTmFXTnliM052Wm5RZ1NWUWdWRXhUSUVOQklEUXcKSGhjTk1UZ3dOREV5TWpJek1qUTRXaGNOTWpBd05ERXlNakl6TWpRNFdqQTVNVGN3TlFZRFZRUURFeTV6WlhKMgphV05sWTJ4cFpXNTBZMlZ5ZEMxd1lYSjBibVZ5TG0xaGJtRm5aVzFsYm5RdVlYcDFjbVV1WTI5dE1JSUJJakFOCkJna3Foa2lHOXcwQkFRRUZBQU9DQVE4QU1JSUJDZ0tDQVFFQTBSYjdJcHpxMmR4emhhbVpyS1ZDakMzeTQyYlYKUnNIY2pCUTFuSDBHZ1puUDhXeDZDSE1mWThybkVJQzRLeVRRYkJXVzhnNXlmc3NSQ0ZXbFpxYjR6SkRXS0pmTgpGSmNMUm9LNnhwTktZYVZVTkVlT25IdUxHYTM0ZlA0VjBFRjZybzdvbkRLME5zanhjY1dZVzRNVXVzc0xrQS94CkUrM2RwU1REdk1KcjJoWUpsVnFDcVR6blQvbmZaVUZzQUVEQnp5MUpOOHZiZDlIR2czc2Myd0x4dk95cFJOc0gKT1V3V2pmN2xzWWZleEVlcWkzY29EeHc2alpLVWEyVkdsUnBpTkowMjhBQitYSi9TU1FVNVBsd0JBbU9TT3ovRApGY0NKdGpPZlBqU1NKckFIQVV3SHU3RzlSV05JTFBwYU9zQ1J5eitETE5zNGpvNlEvUUg4d1lManJRSURBUUFCCm80SUNtakNDQXBZd0N3WURWUjBQQkFRREFnU3dNQjBHQTFVZEpRUVdNQlFHQ0NzR0FRVUZCd01DQmdnckJnRUYKQlFjREFUQWRCZ05WSFE0RUZnUVVlbEdkVVJrZzJoSFFOWEQ4WUc4L3drdjJVT0F3SHdZRFZSMGpCQmd3Rm9BVQplbnVNd2Mvbm9Nb2MxR3Y2KytFend3OGFvcDB3Z2F3R0ExVWRId1NCcERDQm9UQ0JucUNCbTZDQm1JWkxhSFIwCmNEb3ZMMjF6WTNKc0xtMXBZM0p2YzI5bWRDNWpiMjB2Y0d0cEwyMXpZMjl5Y0M5amNtd3ZUV2xqY205emIyWjAKSlRJd1NWUWxNakJVVEZNbE1qQkRRU1V5TURRdVkzSnNoa2xvZEhSd09pOHZZM0pzTG0xcFkzSnZjMjltZEM1agpiMjB2Y0d0cEwyMXpZMjl5Y0M5amNtd3ZUV2xqY205emIyWjBKVEl3U1ZRbE1qQlVURk1sTWpCRFFTVXlNRFF1ClkzSnNNSUdGQmdnckJnRUZCUWNCQVFSNU1IY3dVUVlJS3dZQkJRVUhNQUtHUldoMGRIQTZMeTkzZDNjdWJXbGoKY205emIyWjBMbU52YlM5d2Eya3ZiWE5qYjNKd0wwMXBZM0p2YzI5bWRDVXlNRWxVSlRJd1ZFeFRKVEl3UTBFbApNakEwTG1OeWREQWlCZ2dyQmdFRkJRY3dBWVlXYUhSMGNEb3ZMMjlqYzNBdWJYTnZZM053TG1OdmJUQStCZ2tyCkJnRUVBWUkzRlFjRU1UQXZCaWNyQmdFRUFZSTNGUWlIMm9aMWcrN1pBWUxKaFJ1QnRaNWhoZlRyWUlGZGhOTGYKUW9Mbmszb0NBV1FDQVIwd1RRWURWUjBnQkVZd1JEQkNCZ2tyQmdFRUFZSTNLZ0V3TlRBekJnZ3JCZ0VGQlFjQwpBUlluYUhSMGNEb3ZMM2QzZHk1dGFXTnliM052Wm5RdVkyOXRMM0JyYVM5dGMyTnZjbkF2WTNCek1DY0dDU3NHCkFRUUJnamNWQ2dRYU1CZ3dDZ1lJS3dZQkJRVUhBd0l3Q2dZSUt3WUJCUVVIQXdFd09RWURWUjBSQkRJd01JSXUKYzJWeWRtbGpaV05zYVdWdWRHTmxjblF0Y0dGeWRHNWxjaTV0WVc1aFoyVnRaVzUwTG1GNmRYSmxMbU52YlRBTgpCZ2txaGtpRzl3MEJBUXNGQUFPQ0FnRUFIVXIzbk1vdUI5WWdDUlRWYndUTllIS2RkWGJkSW1GUXNDYys4T1g1CjE5c0N6dFFSR05iSXEwVW1Ba01MbFVvWTIxckh4ZXdxU2hWczFhL2RwaFh5Tk1pcUdaU2QzU1BtYzZscitqUFQKNXVEREs0MUlWeXN0K2VUNlpyazFvcCtMVmdkeS9EU2lyNzVqcWZFY016bS82bU8rNnFNeWRLTWtVYmM5K3JHVwphUkpUcjRWUUdIRmEwNEIwZVZpNUd4MG9pL2RpZDNSaXg2aXJMMjFJSGEwYjN6c1hzZHpHU0R2K3hqL2Q2S0l4Ckdrd2FhYmZvU1NoQnFqaFNlQ0VyZXFlb1RpYjljdGw0MGRVdUp3THl4bjhHS2N6K3AvMEJUOEIxU3lYK01OQ2wKY0pkMjVtMjhLajY2TGUxOEVyeFlJYXZJVGVGa3Y2eGZjdkEvcHladDdPaU41QTlGQk1IUmpQK1kyZ2tvdjMrcQpISFRUZG4xNnlRajduNit3YlFHNGVleXc0YisyQkRLcUxNVFU2ZmlSQ3ZPM2FPZVBLSFVNN3R4b1FidWl6Z3NzCkNiMzl3QnJOTEZsMkJLQ1RkSCtkSU9oZVJiSkZvbmlwOGRPOUVFZWdSSG9lQW54ZUlYTFBrdXMzTzEvZjRhNkIKWHQ3RG5BUm8xSzJmeEp3VXRaU2MvR3dFSjU5NzlnRXlEa3pDZEVsLzdpWE9QZXVjTXhlM2xVM2pweUtsNERUaApjSkJqQytqNGpLWTFrK1U4b040aGdqYnJISUx6Vnd2eU15OU5KS290U3BMSjQxeHdPOHlGangxalFTT3Bxc0N1ClFhUFUvTjhSZ0hxWjBGTkFzS3dNUmZ6WmdXanRCNzRzYUVEdk5jVmNuNFhCQnFNSG0ydHo2Uzk3d3kxZGt0cTgKSE5BPQotLS0tLUVORCBDRVJUSUZJQ0FURS0tLS0tCg=='}]}}
        BODY = {'location': AZURE_LOCATION, 'tags': {'key': 'value'}, 'service_uri': 'http://myservice.com', 'custom_headers': {'authorization': 'Basic 000000000000000000000000000000000000000000000000000'}, 'status': 'enabled', 'scope': 'myRepository', 'actions': ['push']}
        result = self.mgmt_client.webhooks.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME, webhook_create_parameters=BODY)
        result = result.result()
        BODY = {'force_update_tag': 'test', 'run_request': {'type': 'EncodedTaskRunRequest', 'encoded_task_content': 'c3RlcHM6IAogIC0gY21kOiB7eyAuVmFsdWVzLmNvbW1hbmQgfX0K', 'encoded_values_content': 'Y29tbWFuZDogYmFzaCBlY2hvIHt7LlJ1bi5SZWdpc3RyeX19Cg==', 'values': [], 'platform': {'os': 'Linux', 'architecture': 'amd64'}}}
        result = self.mgmt_client.task_runs.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_run_name=TASK_RUN_NAME, task_run=BODY)
        result = result.result()
        BODY = {'description': 'Developer Scopes', 'actions': ['repositories/myrepository/contentWrite', 'repositories/myrepository/delete']}
        result = self.mgmt_client.scope_maps.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, scope_map_name=SCOPE_MAP_NAME, scope_map_create_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'tags': {'key': 'value'}, 'count': '1', 'tier': 'S1', 'os': 'Linux'}
        result = self.mgmt_client.agent_pools.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, agent_pool_name=AGENT_POOL_NAME, agent_pool=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'tags': {'key': 'value'}}
        result = self.mgmt_client.replications.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, replication_name=REPLICATION_NAME, replication=BODY)
        result = result.result()
        BODY = {'request': {'pipeline_resource_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.ContainerRegistry/registries/' + REGISTRY_NAME + '/importPipelines/' + IMPORT_PIPELINE_NAME, 'source': {'type': 'AzureStorageBlob', 'name': 'myblob.tar.gz'}, 'catalog_digest': 'sha256@'}, 'force_update_tag': '2020-03-04T17:23:21.9261521+00:00'}
        result = self.mgmt_client.pipeline_runs.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, pipeline_run_name=PIPELINE_RUN_NAME, pipeline_run_create_parameters=BODY)
        result = result.result()
        BODY = {'request': {'pipeline_resource_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.ContainerRegistry/registries/' + REGISTRY_NAME + '/exportPipelines/' + EXPORT_PIPELINE_NAME, 'target': {'type': 'AzureStorageBlob', 'name': 'myblob.tar.gz'}, 'artifacts': ['sourceRepository/hello-world', 'sourceRepository2@sha256:00000000000000000000000000000000000']}}
        result = self.mgmt_client.pipeline_runs.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, pipeline_run_name=PIPELINE_RUN_NAME, pipeline_run_create_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'identity': {'type': 'SystemAssigned'}, 'target': {'type': 'AzureStorageBlobContainer', 'uri': 'https://accountname.blob.core.windows.net/containername', 'key_vault_uri': 'https://myvault.vault.azure.net/secrets/acrexportsas'}, 'options': ['OverwriteBlobs']}
        result = self.mgmt_client.export_pipelines.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, export_pipeline_name=EXPORT_PIPELINE_NAME, export_pipeline_create_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'identity': {'type': 'UserAssigned', 'user_assigned_identities': {}}, 'source': {'type': 'AzureStorageBlobContainer', 'uri': 'https://accountname.blob.core.windows.net/containername', 'key_vault_uri': 'https://myvault.vault.azure.net/secrets/acrimportsas'}, 'options': ['OverwriteTags', 'DeleteSourceBlobOnSuccess', 'ContinueOnErrors']}
        result = self.mgmt_client.import_pipelines.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, import_pipeline_name=IMPORT_PIPELINE_NAME, import_pipeline_create_parameters=BODY)
        result = result.result()
        BODY = {'private_link_service_connection_state': {'status': 'Approved', 'description': 'Auto-Approved'}}
        result = self.mgmt_client.private_endpoint_connections.begin_create_or_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, private_endpoint_connection_name=PRIVATE_ENDPOINT_CONNECTION_NAME, private_endpoint_connection=BODY)
        result = result.result()
        result = self.mgmt_client.private_endpoint_connections.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, private_endpoint_connection_name=PRIVATE_ENDPOINT_CONNECTION_NAME)
        result = self.mgmt_client.import_pipelines.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, import_pipeline_name=IMPORT_PIPELINE_NAME)
        result = self.mgmt_client.export_pipelines.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, export_pipeline_name=EXPORT_PIPELINE_NAME)
        result = self.mgmt_client.pipeline_runs.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, pipeline_run_name=PIPELINE_RUN_NAME)
        result = self.mgmt_client.replications.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, replication_name=REPLICATION_NAME)
        result = self.mgmt_client.agent_pools.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, agent_pool_name=AGENT_POOL_NAME)
        result = self.mgmt_client.scope_maps.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, scope_map_name=SCOPE_MAP_NAME)
        result = self.mgmt_client.task_runs.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_run_name=TASK_RUN_NAME)
        result = self.mgmt_client.webhooks.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME)
        result = self.mgmt_client.private_endpoint_connections.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.registries.list_private_link_resources(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.runs.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_id=RUN_ID)
        result = self.mgmt_client.import_pipelines.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.export_pipelines.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.pipeline_runs.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.replications.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.registries.list_usages(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.agent_pools.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.scope_maps.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.task_runs.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.webhooks.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.tokens.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.tasks.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.runs.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, top='10')
        result = self.mgmt_client.registries.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.registries.list_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.registries.list()
        result = self.mgmt_client.operations.list()
        result = self.mgmt_client.agent_pools.get_queue_status(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, agent_pool_name=AGENT_POOL_NAME)
        result = self.mgmt_client.webhooks.get_callback_config(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME)
        result = self.mgmt_client.task_runs.get_details(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_run_name=TASK_RUN_NAME)
        result = self.mgmt_client.webhooks.list_events(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME)
        BODY = {'tags': {'key': 'value'}}
        result = self.mgmt_client.replications.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, replication_name=REPLICATION_NAME, replication_update_parameters=BODY)
        result = result.result()
        BODY = {'count': '1'}
        result = self.mgmt_client.agent_pools.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, agent_pool_name=AGENT_POOL_NAME, update_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.webhooks.ping(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME)
        result = self.mgmt_client.runs.get_log_sas_url(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_id=RUN_ID)
        BODY = {'description': 'Developer Scopes', 'actions': ['repositories/myrepository/contentWrite', 'repositories/myrepository/contentRead']}
        result = self.mgmt_client.scope_maps.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, scope_map_name=SCOPE_MAP_NAME, scope_map_update_parameters=BODY)
        result = result.result()
        BODY = {'force_update_tag': 'test', 'run_request': {'type': 'EncodedTaskRunRequest', 'encoded_task_content': 'c3RlcHM6IAogIC0gY21kOiB7eyAuVmFsdWVzLmNvbW1hbmQgfX0K', 'encoded_values_content': 'Y29tbWFuZDogYmFzaCBlY2hvIHt7LlJ1bi5SZWdpc3RyeX19Cg==', 'values': [], 'platform': {'os': 'Linux', 'architecture': 'amd64'}, 'is_archive_enabled': True}}
        result = self.mgmt_client.task_runs.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_run_name=TASK_RUN_NAME, update_parameters=BODY)
        result = result.result()
        BODY = {'tags': {'key': 'value'}, 'service_uri': 'http://myservice.com', 'custom_headers': {'authorization': 'Basic 000000000000000000000000000000000000000000000000000'}, 'status': 'enabled', 'scope': 'myRepository', 'actions': ['push']}
        result = self.mgmt_client.webhooks.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME, webhook_update_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.registries.get_build_source_upload_url(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.runs.begin_cancel(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_id=RUN_ID)
        result = result.result()
        BODY = {'scope_map_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.ContainerRegistry/registries/' + REGISTRY_NAME + '/scopeMaps/' + SCOPE_MAP_NAME, 'credentials': {'certificates': [{'name': 'certificate1', 'encoded_pem_certificate': 'LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUc3akNDQk5hZ0F3SUJBZ0lURmdBQlR3UVpyZGdmdmhxdzBnQUFBQUZQQkRBTkJna3Foa2lHOXcwQkFRc0YKQURDQml6RUxNQWtHQTFVRUJoTUNWVk14RXpBUkJnTlZCQWdUQ2xkaGMyaHBibWQwYjI0eEVEQU9CZ05WQkFjVApCMUpsWkcxdmJtUXhIakFjQmdOVkJBb1RGVTFwWTNKdmMyOW1kQ0JEYjNKd2IzSmhkR2x2YmpFVk1CTUdBMVVFCkN4TU1UV2xqY205emIyWjBJRWxVTVI0d0hBWURWUVFERXhWTmFXTnliM052Wm5RZ1NWUWdWRXhUSUVOQklEUXcKSGhjTk1UZ3dOREV5TWpJek1qUTRXaGNOTWpBd05ERXlNakl6TWpRNFdqQTVNVGN3TlFZRFZRUURFeTV6WlhKMgphV05sWTJ4cFpXNTBZMlZ5ZEMxd1lYSjBibVZ5TG0xaGJtRm5aVzFsYm5RdVlYcDFjbVV1WTI5dE1JSUJJakFOCkJna3Foa2lHOXcwQkFRRUZBQU9DQVE4QU1JSUJDZ0tDQVFFQTBSYjdJcHpxMmR4emhhbVpyS1ZDakMzeTQyYlYKUnNIY2pCUTFuSDBHZ1puUDhXeDZDSE1mWThybkVJQzRLeVRRYkJXVzhnNXlmc3NSQ0ZXbFpxYjR6SkRXS0pmTgpGSmNMUm9LNnhwTktZYVZVTkVlT25IdUxHYTM0ZlA0VjBFRjZybzdvbkRLME5zanhjY1dZVzRNVXVzc0xrQS94CkUrM2RwU1REdk1KcjJoWUpsVnFDcVR6blQvbmZaVUZzQUVEQnp5MUpOOHZiZDlIR2czc2Myd0x4dk95cFJOc0gKT1V3V2pmN2xzWWZleEVlcWkzY29EeHc2alpLVWEyVkdsUnBpTkowMjhBQitYSi9TU1FVNVBsd0JBbU9TT3ovRApGY0NKdGpPZlBqU1NKckFIQVV3SHU3RzlSV05JTFBwYU9zQ1J5eitETE5zNGpvNlEvUUg4d1lManJRSURBUUFCCm80SUNtakNDQXBZd0N3WURWUjBQQkFRREFnU3dNQjBHQTFVZEpRUVdNQlFHQ0NzR0FRVUZCd01DQmdnckJnRUYKQlFjREFUQWRCZ05WSFE0RUZnUVVlbEdkVVJrZzJoSFFOWEQ4WUc4L3drdjJVT0F3SHdZRFZSMGpCQmd3Rm9BVQplbnVNd2Mvbm9Nb2MxR3Y2KytFend3OGFvcDB3Z2F3R0ExVWRId1NCcERDQm9UQ0JucUNCbTZDQm1JWkxhSFIwCmNEb3ZMMjF6WTNKc0xtMXBZM0p2YzI5bWRDNWpiMjB2Y0d0cEwyMXpZMjl5Y0M5amNtd3ZUV2xqY205emIyWjAKSlRJd1NWUWxNakJVVEZNbE1qQkRRU1V5TURRdVkzSnNoa2xvZEhSd09pOHZZM0pzTG0xcFkzSnZjMjltZEM1agpiMjB2Y0d0cEwyMXpZMjl5Y0M5amNtd3ZUV2xqY205emIyWjBKVEl3U1ZRbE1qQlVURk1sTWpCRFFTVXlNRFF1ClkzSnNNSUdGQmdnckJnRUZCUWNCQVFSNU1IY3dVUVlJS3dZQkJRVUhNQUtHUldoMGRIQTZMeTkzZDNjdWJXbGoKY205emIyWjBMbU52YlM5d2Eya3ZiWE5qYjNKd0wwMXBZM0p2YzI5bWRDVXlNRWxVSlRJd1ZFeFRKVEl3UTBFbApNakEwTG1OeWREQWlCZ2dyQmdFRkJRY3dBWVlXYUhSMGNEb3ZMMjlqYzNBdWJYTnZZM053TG1OdmJUQStCZ2tyCkJnRUVBWUkzRlFjRU1UQXZCaWNyQmdFRUFZSTNGUWlIMm9aMWcrN1pBWUxKaFJ1QnRaNWhoZlRyWUlGZGhOTGYKUW9Mbmszb0NBV1FDQVIwd1RRWURWUjBnQkVZd1JEQkNCZ2tyQmdFRUFZSTNLZ0V3TlRBekJnZ3JCZ0VGQlFjQwpBUlluYUhSMGNEb3ZMM2QzZHk1dGFXTnliM052Wm5RdVkyOXRMM0JyYVM5dGMyTnZjbkF2WTNCek1DY0dDU3NHCkFRUUJnamNWQ2dRYU1CZ3dDZ1lJS3dZQkJRVUhBd0l3Q2dZSUt3WUJCUVVIQXdFd09RWURWUjBSQkRJd01JSXUKYzJWeWRtbGpaV05zYVdWdWRHTmxjblF0Y0dGeWRHNWxjaTV0WVc1aFoyVnRaVzUwTG1GNmRYSmxMbU52YlRBTgpCZ2txaGtpRzl3MEJBUXNGQUFPQ0FnRUFIVXIzbk1vdUI5WWdDUlRWYndUTllIS2RkWGJkSW1GUXNDYys4T1g1CjE5c0N6dFFSR05iSXEwVW1Ba01MbFVvWTIxckh4ZXdxU2hWczFhL2RwaFh5Tk1pcUdaU2QzU1BtYzZscitqUFQKNXVEREs0MUlWeXN0K2VUNlpyazFvcCtMVmdkeS9EU2lyNzVqcWZFY016bS82bU8rNnFNeWRLTWtVYmM5K3JHVwphUkpUcjRWUUdIRmEwNEIwZVZpNUd4MG9pL2RpZDNSaXg2aXJMMjFJSGEwYjN6c1hzZHpHU0R2K3hqL2Q2S0l4Ckdrd2FhYmZvU1NoQnFqaFNlQ0VyZXFlb1RpYjljdGw0MGRVdUp3THl4bjhHS2N6K3AvMEJUOEIxU3lYK01OQ2wKY0pkMjVtMjhLajY2TGUxOEVyeFlJYXZJVGVGa3Y2eGZjdkEvcHladDdPaU41QTlGQk1IUmpQK1kyZ2tvdjMrcQpISFRUZG4xNnlRajduNit3YlFHNGVleXc0YisyQkRLcUxNVFU2ZmlSQ3ZPM2FPZVBLSFVNN3R4b1FidWl6Z3NzCkNiMzl3QnJOTEZsMkJLQ1RkSCtkSU9oZVJiSkZvbmlwOGRPOUVFZWdSSG9lQW54ZUlYTFBrdXMzTzEvZjRhNkIKWHQ3RG5BUm8xSzJmeEp3VXRaU2MvR3dFSjU5NzlnRXlEa3pDZEVsLzdpWE9QZXVjTXhlM2xVM2pweUtsNERUaApjSkJqQytqNGpLWTFrK1U4b040aGdqYnJISUx6Vnd2eU15OU5KS290U3BMSjQxeHdPOHlGangxalFTT3Bxc0N1ClFhUFUvTjhSZ0hxWjBGTkFzS3dNUmZ6WmdXanRCNzRzYUVEdk5jVmNuNFhCQnFNSG0ydHo2Uzk3d3kxZGt0cTgKSE5BPQotLS0tLUVORCBDRVJUSUZJQ0FURS0tLS0tCg=='}]}}
        BODY = {'tags': {'testkey': 'value'}, 'status': 'Enabled', 'agent_configuration': {'cpu': '3'}, 'step': {'type': 'Docker', 'image_names': ['azurerest:testtag1'], 'docker_file_path': 'src/DockerFile'}, 'trigger': {'source_triggers': [{'name': 'mySourceTrigger', 'source_repository': {'source_control_auth_properties': {'token_type': 'PAT', 'token': 'xxxxx'}}, 'source_trigger_events': ['commit']}]}, 'credentials': {'custom_registries': {'myregistry.azurecr.io': {'user_name': {'type': 'Opaque', 'value': 'username'}, 'password': {'type': 'Vaultsecret', 'value': 'https://myacbvault.vault.azure.net/secrets/password'}, 'identity': '[system]'}}}}
        BODY = {'tags': {'testkey': 'value'}, 'status': 'Enabled', 'agent_configuration': {'cpu': '3'}, 'step': {'type': 'Docker', 'image_names': ['azurerest:testtag1'], 'docker_file_path': 'src/DockerFile'}, 'trigger': {'source_triggers': [{'name': 'mySourceTrigger', 'source_repository': {'source_control_auth_properties': {'token_type': 'PAT', 'token': 'xxxxx'}}, 'source_trigger_events': ['commit']}]}, 'credentials': {'custom_registries': {'myregistry.azurecr.io': {'user_name': {'type': 'Vaultsecret', 'value': 'https://myacbvault.vault.azure.net/secrets/username'}, 'password': {'type': 'Vaultsecret', 'value': 'https://myacbvault.vault.azure.net/secrets/password'}, 'identity': '[system]'}}}}
        BODY = {'tags': {'testkey': 'value'}, 'status': 'Enabled', 'agent_configuration': {'cpu': '3'}, 'step': {'type': 'Docker', 'image_names': ['azurerest:testtag1'], 'docker_file_path': 'src/DockerFile'}, 'trigger': {'source_triggers': [{'name': 'mySourceTrigger', 'source_repository': {'source_control_auth_properties': {'token_type': 'PAT', 'token': 'xxxxx'}}, 'source_trigger_events': ['commit']}]}, 'credentials': {'custom_registries': {'myregistry.azurecr.io': {'identity': '[system]'}}}}
        BODY = {'tags': {'testkey': 'value'}, 'status': 'Enabled', 'agent_configuration': {'cpu': '3'}, 'step': {'type': 'Docker', 'image_names': ['azurerest:testtag1'], 'docker_file_path': 'src/DockerFile'}, 'trigger': {'source_triggers': [{'name': 'mySourceTrigger', 'source_repository': {'source_control_auth_properties': {'token_type': 'PAT', 'token': 'xxxxx'}}, 'source_trigger_events': ['commit']}]}, 'credentials': {'custom_registries': {'myregistry.azurecr.io': {'user_name': {'type': 'Opaque', 'value': 'username'}, 'password': {'type': 'Opaque', 'value': '***'}}}}}
        BODY = {'name': 'password'}
        result = self.mgmt_client.registries.regenerate_credential(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, regenerate_credential_parameters=BODY)
        BODY = {'token_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.ContainerRegistry/registries/' + REGISTRY_NAME + '/tokens/' + TOKEN_NAME, 'expiry': '2020-12-31T15:59:59.0707808Z'}
        result = self.mgmt_client.registries.begin_generate_credentials(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, generate_credentials_parameters=BODY)
        result = result.result()
        BODY = {'is_archive_enabled': True}
        result = self.mgmt_client.runs.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_id=RUN_ID, run_update_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.registries.list_credentials(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        BODY = {'source': {'resource_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.ContainerRegistry/registries/' + REGISTRY_NAME, 'source_image': 'sourceRepository@sha256:0000000000000000000000000000000000000000000000000000000000000000'}, 'target_tags': ['targetRepository:targetTag'], 'untagged_target_repositories': ['targetRepository1'], 'mode': 'Force'}
        result = self.mgmt_client.registries.begin_import_image(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, parameters=BODY)
        result = result.result()
        BODY = {'source': {'resource_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.ContainerRegistry/registries/' + REGISTRY_NAME, 'source_image': 'sourceRepository:sourceTag'}, 'target_tags': ['targetRepository:targetTag'], 'untagged_target_repositories': ['targetRepository1'], 'mode': 'Force'}
        result = self.mgmt_client.registries.begin_import_image(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, parameters=BODY)
        result = result.result()
        BODY = {'type': 'EncodedTaskRunRequest', 'values': [{'name': 'mytestargument', 'value': 'mytestvalue', 'is_secret': False}, {'name': 'mysecrettestargument', 'value': 'mysecrettestvalue', 'is_secret': True}], 'platform': {'os': 'Linux'}, 'agent_configuration': {'cpu': '2'}, 'encoded_task_content': 'c3RlcHM6Cnt7IGlmIFZhbHVlcy5lbnZpcm9ubWVudCA9PSAncHJvZCcgfX0KICAtIHJ1bjogcHJvZCBzZXR1cAp7eyBlbHNlIGlmIFZhbHVlcy5lbnZpcm9ubWVudCA9PSAnc3RhZ2luZycgfX0KICAtIHJ1bjogc3RhZ2luZyBzZXR1cAp7eyBlbHNlIH19CiAgLSBydW46IGRlZmF1bHQgc2V0dXAKe3sgZW5kIH19CgogIC0gcnVuOiBidWlsZCAtdCBGYW5jeVRoaW5nOnt7LlZhbHVlcy5lbnZpcm9ubWVudH19LXt7LlZhbHVlcy52ZXJzaW9ufX0gLgoKcHVzaDogWydGYW5jeVRoaW5nOnt7LlZhbHVlcy5lbnZpcm9ubWVudH19LXt7LlZhbHVlcy52ZXJzaW9ufX0nXQ==', 'encoded_values_content': 'ZW52aXJvbm1lbnQ6IHByb2QKdmVyc2lvbjogMQ=='}
        result = self.mgmt_client.registries.begin_schedule_run(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_request=BODY)
        result = result.result()
        BODY = {'type': 'DockerBuildRequest', 'is_archive_enabled': True, 'image_names': ['azurerest:testtag'], 'no_cache': True, 'source_location': 'https://myaccount.blob.core.windows.net/sascontainer/source.zip?sv=2015-04-05&st=2015-04-29T22%3A18%3A26Z&se=2015-04-30T02%3A23%3A26Z&sr=b&sp=rw&sip=168.1.5.60-168.1.5.70&spr=https&sig=Z%2FRHIX5Xcg0Mq2rqI3OlWTjEg2tYkboXr1P9ZUXDtkk%3D', 'arguments': [{'name': 'mytestargument', 'value': 'mytestvalue', 'is_secret': False}, {'name': 'mysecrettestargument', 'value': 'mysecrettestvalue', 'is_secret': True}], 'is_push_enabled': True, 'platform': {'os': 'Linux', 'architecture': 'amd64'}, 'agent_configuration': {'cpu': '2'}, 'docker_file_path': 'DockerFile'}
        result = self.mgmt_client.registries.begin_schedule_run(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_request=BODY)
        result = result.result()
        BODY = {'type': 'FileTaskRunRequest', 'values': [{'name': 'mytestargument', 'value': 'mytestvalue', 'is_secret': False}, {'name': 'mysecrettestargument', 'value': 'mysecrettestvalue', 'is_secret': True}], 'platform': {'os': 'Linux'}, 'task_file_path': 'acb.yaml', 'credentials': {'source_registry': {'login_mode': 'Default'}, 'custom_registries': {'myregistry.azurecr.io': {'user_name': {'type': 'Opaque', 'value': 'reg1'}, 'password': {'type': 'Opaque', 'value': '***'}}}}}
        result = self.mgmt_client.registries.begin_schedule_run(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_request=BODY)
        result = result.result()
        BODY = {'type': 'DockerBuildRequest', 'is_archive_enabled': True, 'image_names': ['azurerest:testtag'], 'no_cache': True, 'source_location': 'https://myaccount.blob.core.windows.net/sascontainer/source.zip?sv=2015-04-05&st=2015-04-29T22%3A18%3A26Z&se=2015-04-30T02%3A23%3A26Z&sr=b&sp=rw&sip=168.1.5.60-168.1.5.70&spr=https&sig=Z%2FRHIX5Xcg0Mq2rqI3OlWTjEg2tYkboXr1P9ZUXDtkk%3D', 'arguments': [{'name': 'mytestargument', 'value': 'mytestvalue', 'is_secret': False}, {'name': 'mysecrettestargument', 'value': 'mysecrettestvalue', 'is_secret': True}], 'is_push_enabled': True, 'platform': {'os': 'Linux', 'architecture': 'amd64'}, 'agent_configuration': {'cpu': '2'}, 'docker_file_path': 'DockerFile', 'target': 'stage1', 'credentials': {'source_registry': {'login_mode': 'Default'}, 'custom_registries': {'myregistry.azurecr.io': {'user_name': {'type': 'Opaque', 'value': 'reg1'}, 'password': {'type': 'Opaque', 'value': '***'}}, 'myregistry2.azurecr.io': {'user_name': {'type': 'Opaque', 'value': 'reg2'}, 'password': {'type': 'Opaque', 'value': '***'}}}}}
        result = self.mgmt_client.registries.begin_schedule_run(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_request=BODY)
        result = result.result()
        BODY = {'type': 'FileTaskRunRequest', 'source_location': 'https://myaccount.blob.core.windows.net/sascontainer/source.zip?sv=2015-04-05&st=2015-04-29T22%3A18%3A26Z&se=2015-04-30T02%3A23%3A26Z&sr=b&sp=rw&sip=168.1.5.60-168.1.5.70&spr=https&sig=Z%2FRHIX5Xcg0Mq2rqI3OlWTjEg2tYkboXr1P9ZUXDtkk%3D', 'values': [{'name': 'mytestargument', 'value': 'mytestvalue', 'is_secret': False}, {'name': 'mysecrettestargument', 'value': 'mysecrettestvalue', 'is_secret': True}], 'platform': {'os': 'Linux'}, 'agent_configuration': {'cpu': '2'}, 'task_file_path': 'acb.yaml', 'values_file_path': 'prod-values.yaml'}
        result = self.mgmt_client.registries.begin_schedule_run(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_request=BODY)
        result = result.result()
        BODY = {'source': {'registry_uri': 'registry.hub.docker.com', 'source_image': 'library/hello-world'}, 'target_tags': ['targetRepository:targetTag'], 'untagged_target_repositories': ['targetRepository1'], 'mode': 'Force'}
        result = self.mgmt_client.registries.begin_import_image(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, parameters=BODY)
        result = result.result()
        BODY = {'type': 'TaskRunRequest', 'override_task_step_properties': {'file': 'overriddenDockerfile', 'target': 'build', 'arguments': [{'name': 'mytestargument', 'value': 'mytestvalue', 'is_secret': False}, {'name': 'mysecrettestargument', 'value': 'mysecrettestvalue', 'is_secret': True}], 'values': [{'name': 'mytestname', 'value': 'mytestvalue', 'is_secret': False}, {'name': 'mysecrettestname', 'value': 'mysecrettestvalue', 'is_secret': True}], 'update_trigger_token': 'aGVsbG8gd29ybGQ='}, 'task_id': 'myTask'}
        result = self.mgmt_client.registries.begin_schedule_run(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_request=BODY)
        result = result.result()
        BODY = {'tags': {'key': 'value'}, 'sku': {'name': 'Standard'}, 'admin_user_enabled': True}
        result = self.mgmt_client.registries.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, registry_update_parameters=BODY)
        result = result.result()
        BODY = {'name': 'myRegistry', 'type': 'Microsoft.ContainerRegistry/registries'}
        result = self.mgmt_client.registries.check_name_availability(registry_name_check_request=BODY)
        BODY = {'name': 'myRegistry', 'type': 'Microsoft.ContainerRegistry/registries'}
        result = self.mgmt_client.registries.check_name_availability(registry_name_check_request=BODY)
        result = self.mgmt_client.private_endpoint_connections.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, private_endpoint_connection_name=PRIVATE_ENDPOINT_CONNECTION_NAME)
        result = result.result()
        result = self.mgmt_client.export_pipelines.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, export_pipeline_name=EXPORT_PIPELINE_NAME)
        result = result.result()
        result = self.mgmt_client.import_pipelines.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, import_pipeline_name=IMPORT_PIPELINE_NAME)
        result = result.result()
        result = self.mgmt_client.pipeline_runs.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, pipeline_run_name=PIPELINE_RUN_NAME)
        result = result.result()
        result = self.mgmt_client.replications.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, replication_name=REPLICATION_NAME)
        result = result.result()
        result = self.mgmt_client.agent_pools.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, agent_pool_name=AGENT_POOL_NAME)
        result = result.result()
        result = self.mgmt_client.scope_maps.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, scope_map_name=SCOPE_MAP_NAME)
        result = result.result()
        result = self.mgmt_client.task_runs.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_run_name=TASK_RUN_NAME)
        result = result.result()
        result = self.mgmt_client.webhooks.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, webhook_name=WEBHOOK_NAME)
        result = result.result()
        result = self.mgmt_client.registries.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = result.result()
if __name__ == '__main__':
    unittest.main()