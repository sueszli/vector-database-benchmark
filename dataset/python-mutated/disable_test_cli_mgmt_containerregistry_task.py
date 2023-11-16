import unittest
import azure.mgmt.containerregistry
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtRegistryTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(MgmtRegistryTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.containerregistry.ContainerRegistryManagementClient, api_version='2019-12-01-preview')

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_pipelines(self, resource_group):
        if False:
            while True:
                i = 10
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        RESOURCE_GROUP = resource_group.name
        REGISTRY_NAME = 'myRegistry'
        PIPELINE_RUN_NAME = 'myPipelineRun'
        IMPORT_PIPELINE_NAME = 'myImportPipeline'
        EXPORT_PIPELINE_NAME = 'myExportPipeline'
        BODY = {'location': AZURE_LOCATION, 'tags': {'key': 'value'}, 'sku': {'name': 'Premium'}, 'admin_user_enabled': False}
        result = self.mgmt_client.registries.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, registry=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'identity': {'type': 'SystemAssigned'}, 'source': {'type': 'AzureStorageBlobContainer', 'uri': 'https://accountname.blob.core.windows.net/containername', 'key_vault_uri': 'https://myvault.vault.azure.net/secrets/acrimportsas'}, 'options': ['OverwriteTags', 'DeleteSourceBlobOnSuccess', 'ContinueOnErrors']}
        result = self.mgmt_client.import_pipelines.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, import_pipeline_name=IMPORT_PIPELINE_NAME, import_pipeline_create_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'identity': {'type': 'SystemAssigned'}, 'target': {'type': 'AzureStorageBlobContainer', 'uri': 'https://accountname.blob.core.windows.net/containername', 'key_vault_uri': 'https://myvault.vault.azure.net/secrets/acrexportsas'}, 'options': ['OverwriteBlobs']}
        result = self.mgmt_client.export_pipelines.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, export_pipeline_name=EXPORT_PIPELINE_NAME, export_pipeline_create_parameters=BODY)
        result = result.result()
        BODY = {'request': {'pipeline_resource_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.ContainerRegistry/registries/' + REGISTRY_NAME + '/importPipelines/' + IMPORT_PIPELINE_NAME, 'source': {'type': 'AzureStorageBlob', 'name': 'myblob.tar.gz'}, 'catalog_digest': 'sha256@'}, 'force_update_tag': '2020-03-04T17:23:21.9261521+00:00'}
        result = self.mgmt_client.import_pipelines.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, import_pipeline_name=IMPORT_PIPELINE_NAME)
        result = self.mgmt_client.export_pipelines.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, export_pipeline_name=EXPORT_PIPELINE_NAME)
        result = self.mgmt_client.pipeline_runs.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.import_pipelines.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.export_pipelines.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.import_pipelines.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, import_pipeline_name=IMPORT_PIPELINE_NAME)
        result = result.result()
        result = self.mgmt_client.export_pipelines.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, export_pipeline_name=EXPORT_PIPELINE_NAME)
        result = result.result()
        result = self.mgmt_client.registries.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = result.result()

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_task_run(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        RESOURCE_GROUP = resource_group.name
        REGISTRY_NAME = 'myRegistry'
        TASK_RUN_NAME = 'myTaskRun'
        BODY = {'location': AZURE_LOCATION, 'tags': {'key': 'value'}, 'sku': {'name': 'Premium'}, 'admin_user_enabled': False}
        result = self.mgmt_client.registries.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, registry=BODY)
        result = result.result()
        BODY = {'force_update_tag': 'test', 'run_request': {'type': 'DockerBuildRequest', 'image_names': ['testtaskrun:v1'], 'is_push_enabled': True, 'no_cache': False, 'docker_file_path': 'Dockerfile', 'platform': {'os': 'linux', 'architecture': 'amd64'}, 'source_location': 'https://github.com/Azure-Samples/acr-build-helloworld-node.git', 'is_archive_enabled': True}}
        result = self.mgmt_client.task_runs.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_run_name=TASK_RUN_NAME, task_run=BODY)
        result = result.result()
        result = self.mgmt_client.task_runs.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_run_name=TASK_RUN_NAME)
        RUN_ID = result.run_result.run_id
        result = self.mgmt_client.runs.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_id=RUN_ID)
        result = self.mgmt_client.task_runs.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.runs.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, top='10')
        result = self.mgmt_client.task_runs.get_details(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_run_name=TASK_RUN_NAME)
        result = self.mgmt_client.runs.get_log_sas_url(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_id=RUN_ID)
        result = self.mgmt_client.runs.begin_cancel(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_id=RUN_ID)
        result = result.result()
        BODY = {'force_update_tag': 'test', 'run_request': {'type': 'DockerBuildRequest', 'image_names': ['testtaskrun:v1'], 'is_push_enabled': True, 'no_cache': False, 'docker_file_path': 'Dockerfile', 'platform': {'os': 'linux', 'architecture': 'amd64'}, 'source_location': 'https://github.com/Azure-Samples/acr-build-helloworld-node.git', 'is_archive_enabled': True}}
        result = self.mgmt_client.task_runs.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_run_name=TASK_RUN_NAME, update_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.runs.begin_cancel(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_id=RUN_ID)
        result = result.result()
        result = self.mgmt_client.task_runs.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_run_name=TASK_RUN_NAME)
        result = result.result()
        BODY = {'is_archive_enabled': True}
        result = self.mgmt_client.runs.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, run_id=RUN_ID, run_update_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.registries.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = result.result()

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_tasks(self, resource_group):
        if False:
            while True:
                i = 10
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        REGISTRY_NAME = 'myRegistry'
        RESOURCE_GROUP = resource_group.name
        TASK_RUN_NAME = 'myTaskRun'
        TASK_NAME = 'myTask'
        BODY = {'location': AZURE_LOCATION, 'tags': {'key': 'value'}, 'sku': {'name': 'Standard'}, 'admin_user_enabled': True}
        result = self.mgmt_client.registries.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, registry=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'tags': {'testkey': 'value'}, 'status': 'Enabled', 'platform': {'os': 'Linux', 'architecture': 'amd64'}, 'agent_configuration': {'cpu': '2'}, 'step': {'type': 'Docker', 'context_path': 'https://github.com/SteveLasker/node-helloworld', 'image_names': ['testtask:v1'], 'docker_file_path': 'DockerFile', 'is_push_enabled': True, 'no_cache': False}, 'trigger': {'base_image_trigger': {'name': 'myBaseImageTrigger', 'base_image_trigger_type': 'Runtime', 'update_trigger_payload_type': 'Default', 'status': 'Enabled'}}}
        result = self.mgmt_client.tasks.begin_create(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_name=TASK_NAME, task_create_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.tasks.get(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_name=TASK_NAME)
        result = self.mgmt_client.tasks.list(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = self.mgmt_client.tasks.get_details(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_name=TASK_NAME)
        BODY = {'location': AZURE_LOCATION, 'tags': {'testkey': 'value'}, 'status': 'Enabled', 'platform': {'os': 'Linux', 'architecture': 'amd64'}, 'agent_configuration': {'cpu': '2'}, 'step': {'type': 'Docker', 'context_path': 'https://github.com/SteveLasker/node-helloworld', 'image_names': ['testtask:v1'], 'docker_file_path': 'DockerFile', 'is_push_enabled': True, 'no_cache': False}, 'trigger': {'base_image_trigger': {'name': 'myBaseImageTrigger', 'base_image_trigger_type': 'Runtime', 'update_trigger_payload_type': 'Default', 'status': 'Enabled'}}}
        result = self.mgmt_client.tasks.begin_update(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_name=TASK_NAME, task_update_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.tasks.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME, task_name=TASK_NAME)
        result = result.result()
        result = self.mgmt_client.registries.begin_delete(resource_group_name=RESOURCE_GROUP, registry_name=REGISTRY_NAME)
        result = result.result()