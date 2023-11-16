import unittest
import azure.mgmt.cosmosdb
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer, ResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtCosmosDBTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(MgmtCosmosDBTest, self).setUp()
        self.re_replacer.register_pattern_pair('"primaryMasterKey":".{88}"', '"primaryMasterKey":"FakeKey"')
        self.re_replacer.register_pattern_pair('"secondaryMasterKey":".{88}"', '"secondaryMasterKey":"FakeKey"')
        self.re_replacer.register_pattern_pair('"primaryReadonlyMasterKey":".{88}"', '"primaryReadonlyMasterKey":"FakeKey"')
        self.re_replacer.register_pattern_pair('"secondaryReadonlyMasterKey":".{88}"', '"secondaryReadonlyMasterKey":"FakeKey"')
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.cosmosdb.CosmosDBManagementClient)

    @unittest.skip('hard to test')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_dbaccount(self, resource_group):
        if False:
            return 10
        RESOURCE_GROUP = resource_group.name
        ACCOUNT_NAME = 'myaccountxxyyzzz'
        result = self.mgmt_client.operations.list()
        BODY = {'location': AZURE_LOCATION, 'database_account_offer_type': 'Standard', 'locations': [{'failover_priority': '2', 'location_name': 'southcentralus', 'is_zone_redundant': False}, {'location_name': 'eastus', 'failover_priority': '1'}, {'location_name': 'westus', 'failover_priority': '0'}]}
        result = self.mgmt_client.database_accounts.begin_create_or_update(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, create_update_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.database_accounts.list_metric_definitions(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = self.mgmt_client.database_accounts.get_read_only_keys(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = self.mgmt_client.database_accounts.list_metrics(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, filter="$filter=(name.value eq 'Total Requests') and timeGrain eq duration'PT5M' and startTime eq '2017-11-19T23:53:55.2780000Z' and endTime eq '2017-11-20T00:13:55.2780000Z")
        result = self.mgmt_client.database_accounts.list_usages(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, filter="$filter=name.value eq 'Storage'")
        result = self.mgmt_client.database_accounts.get(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = self.mgmt_client.database_accounts.list_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.database_accounts.list()
        BODY = {'failover_policies': [{'location_name': 'eastus', 'failover_priority': '0'}, {'location_name': 'westus', 'failover_priority': '1'}, {'failover_priority': '2', 'location_name': 'southcentralus'}]}
        result = self.mgmt_client.database_accounts.begin_failover_priority_change(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, failover_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.database_accounts.list_connection_strings(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = self.mgmt_client.database_accounts.list_connection_strings(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        BODY = {'region': 'eastus'}
        BODY = {'region': 'eastus'}
        BODY = {'key_kind': 'primary'}
        result = self.mgmt_client.database_accounts.begin_regenerate_key(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, key_to_regenerate=BODY)
        result = result.result()
        result = self.mgmt_client.database_accounts.list_read_only_keys(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = self.mgmt_client.database_accounts.list_keys(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        BODY = {'tags': {'dept': 'finance'}}
        result = self.mgmt_client.database_accounts.begin_update(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, update_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.database_accounts.check_name_exists(account_name=ACCOUNT_NAME)
        result = self.mgmt_client.database_accounts.begin_delete(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = result.result()