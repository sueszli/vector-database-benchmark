import unittest
import azure.mgmt.web
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtWebSiteTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(MgmtWebSiteTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.web.WebSiteManagementClient)

    @unittest.skip('skip temporarily')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_app_service_plan(self, resource_group):
        if False:
            return 10
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        TENANT_ID = self.settings.TENANT_ID
        RESOURCE_GROUP = resource_group.name
        NAME = 'myname'
        BODY = {'kind': 'app', 'location': AZURE_LOCATION, 'sku': {'name': 'P1', 'tier': 'Premium', 'size': 'P1', 'family': 'P', 'capacity': '1'}}
        result = self.mgmt_client.app_service_plans.begin_create_or_update(resource_group_name=RESOURCE_GROUP, name=NAME, app_service_plan=BODY)
        result = result.result()
        result = self.mgmt_client.app_service_plans.get(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.app_service_plans.list_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.app_service_plans.list()
        BODY = {'kind': 'app'}
        result = self.mgmt_client.app_service_plans.update(resource_group_name=RESOURCE_GROUP, name=NAME, app_service_plan=BODY)
        result = self.mgmt_client.app_service_plans.delete(resource_group_name=RESOURCE_GROUP, name=NAME)