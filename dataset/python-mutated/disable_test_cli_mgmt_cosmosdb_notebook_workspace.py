import unittest
import azure.mgmt.cosmosdb
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer, ResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtCosmosDBTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(MgmtCosmosDBTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.cosmosdb.CosmosDBManagementClient)

    @unittest.skip('hard to test')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_notebook_workspace(self, resource_group):
        if False:
            print('Hello World!')
        RESOURCE_GROUP = resource_group.name
        ACCOUNT_NAME = 'myaccountxxyyzzz'
        DATABASE_NAME = 'myDatabase'
        NOTEBOOK_WORKSPACE_NAME = 'myNotebookWorkspace'
        BODY = {'location': AZURE_LOCATION, 'kind': 'GlobalDocumentDB', 'database_account_offer_type': 'Standard', 'locations': [{'location_name': 'eastus', 'is_zone_redundant': False, 'failover_priority': '0'}], 'api_properties': {}}
        result = self.mgmt_client.database_accounts.begin_create_or_update(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, create_update_parameters=BODY)
        result = result.result()
        BODY = {}
        result = self.mgmt_client.notebook_workspaces.begin_create_or_update(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, notebook_workspace_name=NOTEBOOK_WORKSPACE_NAME, notebook_create_update_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.notebook_workspaces.get(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, notebook_workspace_name=NOTEBOOK_WORKSPACE_NAME)
        result = self.mgmt_client.notebook_workspaces.list_by_database_account(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = self.mgmt_client.notebook_workspaces.begin_regenerate_auth_token(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, notebook_workspace_name=NOTEBOOK_WORKSPACE_NAME)
        result = result.result()
        result = self.mgmt_client.notebook_workspaces.list_connection_info(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, notebook_workspace_name=NOTEBOOK_WORKSPACE_NAME)
        result = self.mgmt_client.notebook_workspaces.begin_start(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, notebook_workspace_name=NOTEBOOK_WORKSPACE_NAME)
        result = result.result()
        result = self.mgmt_client.notebook_workspaces.begin_delete(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, notebook_workspace_name=NOTEBOOK_WORKSPACE_NAME)
        result = result.result()
        result = self.mgmt_client.database_accounts.begin_delete(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = result.result()