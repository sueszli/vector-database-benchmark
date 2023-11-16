import unittest
import azure.mgmt.cosmosdb
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer, ResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtCosmosDBTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(MgmtCosmosDBTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.cosmosdb.CosmosDBManagementClient)

    @unittest.skip('hard to test')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_cassandra_resource(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        RESOURCE_GROUP = resource_group.name
        ACCOUNT_NAME = 'myaccountxxyyzzz'
        DATABASE_NAME = 'myDatabase'
        KEYSPACE_NAME = 'myKeyspace'
        TABLE_NAME = 'myTable'
        BODY = {'location': AZURE_LOCATION, 'kind': 'GlobalDocumentDB', 'database_account_offer_type': 'Standard', 'locations': [{'location_name': 'eastus', 'is_zone_redundant': False, 'failover_priority': '0'}], 'capabilities': [{'name': 'EnableCassandra'}], 'api_properties': {}}
        result = self.mgmt_client.database_accounts.begin_create_or_update(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, create_update_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': KEYSPACE_NAME}, 'options': {'throughput': '2000'}}
        result = self.mgmt_client.cassandra_resources.begin_create_update_cassandra_keyspace(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME, create_update_cassandra_keyspace_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': TABLE_NAME, 'default_ttl': '100', 'schema': {'columns': [{'name': 'columnA', 'type': 'Ascii'}], 'partition_keys': [{'name': 'columnA'}]}}, 'options': {'throughput': '2000'}}
        result = self.mgmt_client.cassandra_resources.begin_create_update_cassandra_table(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME, table_name=TABLE_NAME, create_update_cassandra_table_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'throughput': '400'}}
        result = self.mgmt_client.cassandra_resources.begin_update_cassandra_keyspace_throughput(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME, update_throughput_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'throughput': '400'}}
        result = self.mgmt_client.cassandra_resources.begin_update_cassandra_table_throughput(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME, table_name=TABLE_NAME, update_throughput_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.cassandra_resources.get_cassandra_table_throughput(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME, table_name=TABLE_NAME)
        result = self.mgmt_client.cassandra_resources.get_cassandra_keyspace_throughput(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME)
        result = self.mgmt_client.cassandra_resources.get_cassandra_table(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME, table_name=TABLE_NAME)
        result = self.mgmt_client.cassandra_resources.list_cassandra_tables(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME)
        result = self.mgmt_client.cassandra_resources.get_cassandra_keyspace(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME)
        result = self.mgmt_client.cassandra_resources.list_cassandra_keyspaces(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = self.mgmt_client.cassandra_resources.begin_migrate_cassandra_table_to_autoscale(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME, table_name=TABLE_NAME)
        result = result.result()
        result = self.mgmt_client.cassandra_resources.begin_migrate_cassandra_table_to_manual_throughput(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME, table_name=TABLE_NAME)
        result = result.result()
        result = self.mgmt_client.cassandra_resources.begin_migrate_cassandra_keyspace_to_autoscale(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME)
        result = result.result()
        result = self.mgmt_client.cassandra_resources.begin_migrate_cassandra_keyspace_to_manual_throughput(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME)
        result = result.result()
        result = self.mgmt_client.cassandra_resources.begin_delete_cassandra_table(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME, table_name=TABLE_NAME)
        result = result.result()
        result = self.mgmt_client.cassandra_resources.begin_delete_cassandra_keyspace(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, keyspace_name=KEYSPACE_NAME)
        result = result.result()
        result = self.mgmt_client.database_accounts.begin_delete(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = result.result()