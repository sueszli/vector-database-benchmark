import unittest
import azure.mgmt.appplatform
from devtools_testutils import AzureMgmtTestCase, ResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtAppPlatformTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            return 10
        super(MgmtAppPlatformTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.appplatform.AppPlatformManagementClient)

    @unittest.skip('skip test')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_appplatform(self, resource_group):
        if False:
            while True:
                i = 10
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        TENANT_ID = self.settings.TENANT_ID
        RESOURCE_GROUP = resource_group.name
        SERVICE_NAME = 'myservice'
        LOCATION = 'myLocation'
        APP_NAME = 'myapp'
        BINDING_NAME = 'mybinding'
        DATABASE_ACCOUNT_NAME = 'myDatabaseAccount'
        CERTIFICATE_NAME = 'myCertificate'
        DOMAIN_NAME = 'myDomain'
        DEPLOYMENT_NAME = 'mydeployment'
        BODY = {'properties': {'config_server_properties': {'config_server': {'git_property': {'uri': 'https://github.com/fake-user/fake-repository.git', 'label': 'master', 'search_paths': ['/']}}}, 'trace': {'enabled': True, 'app_insight_instrumentation_key': '00000000-0000-0000-0000-000000000000'}}, 'tags': {'key1': 'value1'}, 'location': 'eastus'}
        result = self.mgmt_client.services.create_or_update(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME, resource=BODY)
        result = result.result()
        PROPERTIES = {'public': True, 'active_deployment_name': 'mydeployment1', 'fqdn': 'myapp.mydomain.com', 'https_only': False, 'temporary_disk': {'size_in_gb': '2', 'mount_path': '/mytemporarydisk'}, 'persistent_disk': {'size_in_gb': '2', 'mount_path': '/mypersistentdisk'}}
        result = self.mgmt_client.apps.create_or_update(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME, app_name=APP_NAME, properties=PROPERTIES, location='eastus')
        result = result.result()
        PROPERTIES = {'vault_uri': 'https://myvault.vault.azure.net', 'key_vault_cert_name': 'mycert', 'cert_version': '08a219d06d874795a96db47e06fbb01e'}
        PROPERTIES = {'thumbprint': '934367bf1c97033f877db0f15cb1b586957d3133', 'app_name': 'myapp', 'cert_name': 'mycert'}
        PROPERTIES = {'resource_name': 'my-cosmosdb-1', 'resource_type': 'Microsoft.DocumentDB', 'resource_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.DocumentDB/databaseAccounts/' + DATABASE_ACCOUNT_NAME + '', 'key': 'xxxx', 'binding_parameters': {'database_name': 'db1', 'api_type': 'SQL'}}
        PROPERTIES = {'source': {'type': 'Source', 'relative_path': 'resources/a172cedcae47474b615c54d510a5d84a8dea3032e958587430b413538be3f333-2019082605-e3095339-1723-44b7-8b5e-31b1003978bc', 'version': '1.0', 'artifact_selector': 'sub-module-1'}, 'deployment_settings': {'cpu': '1', 'memory_in_gb': '3', 'jvm_options': '-Xms1G -Xmx3G', 'instance_count': '1', 'environment_variables': {'env': 'test'}, 'runtime_version': 'Java_8'}}
        result = self.mgmt_client.deployments.list(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME, app_name=APP_NAME)
        result = self.mgmt_client.bindings.list(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME, app_name=APP_NAME)
        result = self.mgmt_client.apps.get(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME, app_name=APP_NAME)
        result = self.mgmt_client.deployments.list_cluster_all_deployments(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME)
        result = self.mgmt_client.apps.list(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME)
        result = self.mgmt_client.services.get(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME)
        result = self.mgmt_client.services.list(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.services.list_by_subscription()
        result = self.mgmt_client.operations.list()
        PROPERTIES = {'source': {'type': 'Source', 'relative_path': 'resources/a172cedcae47474b615c54d510a5d84a8dea3032e958587430b413538be3f333-2019082605-e3095339-1723-44b7-8b5e-31b1003978bc', 'version': '1.0', 'artifact_selector': 'sub-module-1'}}
        PROPERTIES = {'key': 'xxxx', 'binding_parameters': {'database_name': 'db1', 'api_type': 'SQL'}}
        PROPERTIES = {'thumbprint': '934367bf1c97033f877db0f15cb1b586957d3133', 'app_name': 'myapp', 'cert_name': 'mycert'}
        result = self.mgmt_client.apps.get_resource_upload_url(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME, app_name=APP_NAME)
        PROPERTIES = {'public': True, 'active_deployment_name': 'mydeployment1', 'fqdn': 'myapp.mydomain.com', 'https_only': False, 'temporary_disk': {'size_in_gb': '2', 'mount_path': '/mytemporarydisk'}, 'persistent_disk': {'size_in_gb': '2', 'mount_path': '/mypersistentdisk'}}
        result = self.mgmt_client.services.disable_test_endpoint(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME)
        result = self.mgmt_client.services.enable_test_endpoint(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME)
        result = self.mgmt_client.services.regenerate_test_key(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME, key_type='Primary')
        result = self.mgmt_client.services.list_test_keys(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME)
        BODY = {'properties': {'config_server_properties': {'config_server': {'git_property': {'uri': 'https://github.com/fake-user/fake-repository.git', 'label': 'master', 'search_paths': ['/']}}}, 'trace': {'enabled': True, 'app_insight_instrumentation_key': '00000000-0000-0000-0000-000000000000'}}, 'location': 'eastus', 'tags': {'key1': 'value1'}}
        result = self.mgmt_client.services.check_name_availability(azure_location=AZURE_LOCATION, type='Microsoft.AppPlatform/Spring', name='myservice')
        result = self.mgmt_client.apps.delete(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME, app_name=APP_NAME)
        result = self.mgmt_client.services.delete(resource_group_name=RESOURCE_GROUP, service_name=SERVICE_NAME)
        result = result.result()
if __name__ == '__main__':
    unittest.main()