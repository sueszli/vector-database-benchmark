import unittest
import azure.mgmt.web
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer
AZURE_LOCATION = 'eastus2'

class MgmtWebSiteTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(MgmtWebSiteTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.web.WebSiteManagementClient)

    @unittest.skip('skip temporarily')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_static_site(self, resource_group):
        if False:
            while True:
                i = 10
        GITHUB_TOKEN = self.settings.GITHUB_TOKEN if self.is_live else 'xxx'
        TENANT_ID = self.settings.TENANT_ID
        RESOURCE_GROUP = resource_group.name
        NAME = 'myname'
        PR_ID = '1'
        DOMAIN_NAME = 'mydomain'
        BODY = {'location': AZURE_LOCATION, 'sku': {'name': 'Free'}, 'repository_url': 'https://github.com/00Kai0/html-docs-hello-world', 'branch': 'master', 'repository_token': GITHUB_TOKEN, 'build_properties': {'app_location': 'app', 'api_location': 'api', 'app_artifact_location': 'build'}}
        result = self.mgmt_client.static_sites.create_or_update_static_site(resource_group_name=RESOURCE_GROUP, name=NAME, static_site_envelope=BODY)
        BODY = {'properties': {'setting1': 'someval', 'setting2': 'someval2'}}
        BODY = {'setting1': 'someval', 'setting2': 'someval2'}
        result = self.mgmt_client.static_sites.list_static_site_build_functions(resource_group_name=RESOURCE_GROUP, name=NAME, pr_id=PR_ID)
        result = self.mgmt_client.static_sites.list_static_site_custom_domains(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.static_sites.list_static_site_functions(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.static_sites.get_static_site_builds(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.static_sites.get_static_site(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.static_sites.get_static_sites_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.static_sites.list()
        BODY = {'roles': 'contributor'}
        BODY = {'domain': 'happy-sea-15afae3e.azurestaticwebsites.net', 'provider': 'aad', 'user_details': 'username', 'roles': 'admin,contributor', 'num_hours_to_expiration': '1'}
        result = self.mgmt_client.static_sites.list_static_site_secrets(resource_group_name=RESOURCE_GROUP, name=NAME)
        BODY = {'should_update_repository': True, 'repository_token': GITHUB_TOKEN}
        result = self.mgmt_client.static_sites.reset_static_site_api_key(resource_group_name=RESOURCE_GROUP, name=NAME, reset_properties_envelope=BODY)
        result = self.mgmt_client.static_sites.detach_static_site(resource_group_name=RESOURCE_GROUP, name=NAME)
        BODY = {}
        result = self.mgmt_client.static_sites.delete_static_site(resource_group_name=RESOURCE_GROUP, name=NAME)