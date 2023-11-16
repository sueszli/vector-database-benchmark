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
    def test_query(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        RESOURCE_GROUP = resource_group.name
        ACCOUNT_NAME = 'myaccountxxyyzzz'
        DATABASE_NAME = 'myDatabase'
        COLLECTION_NAME = 'myCollection'
        REGION = AZURE_LOCATION
        BODY = {'location': AZURE_LOCATION, 'kind': 'MongoDB', 'database_account_offer_type': 'Standard', 'locations': [{'location_name': 'eastus', 'is_zone_redundant': False, 'failover_priority': '0'}], 'api_properties': {}}
        result = self.mgmt_client.database_accounts.begin_create_or_update(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, create_update_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': DATABASE_NAME}, 'options': {'throughput': '2000'}}
        result = self.mgmt_client.mongo_db_resources.begin_create_update_mongo_db_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, create_update_mongo_db_database_parameters=BODY)
        database = result.result()
        DATABASE_RID = database.resource.rid
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': COLLECTION_NAME, 'shard_key': {'theShardKey': 'Hash'}}, 'options': {'throughput': '2000'}}
        result = self.mgmt_client.mongo_db_resources.begin_create_update_mongo_db_collection(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, collection_name=COLLECTION_NAME, create_update_mongo_db_collection_parameters=BODY)
        collection = result.result()
        COLLECTION_RID = collection.resource.rid
        result = self.mgmt_client.database.list_metric_definitions(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_rid=DATABASE_RID)
        result = self.mgmt_client.database.list_metrics(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_rid=DATABASE_RID, filter="$filter=(name.value eq 'Total Requests') and timeGrain eq duration'PT5M' and startTime eq '2017-11-19T23:53:55.2780000Z' and endTime eq '2017-11-20T00:13:55.2780000Z")
        result = self.mgmt_client.database.list_usages(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_rid=DATABASE_RID, filter="$filter=name.value eq 'Storage'")
        result = self.mgmt_client.collection.list_metric_definitions(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_rid=DATABASE_RID, collection_rid=COLLECTION_RID)
        result = self.mgmt_client.collection.list_metrics(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_rid=DATABASE_RID, collection_rid=COLLECTION_RID, filter="$filter=(name.value eq 'Total Requests') and timeGrain eq duration'PT5M' and startTime eq '2017-11-19T23:53:55.2780000Z' and endTime eq '2017-11-20T00:13:55.2780000Z")
        result = self.mgmt_client.collection.list_usages(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_rid=DATABASE_RID, collection_rid=COLLECTION_RID, filter="$filter=name.value eq 'Storage'")
        result = self.mgmt_client.collection_region.list_metrics(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, region=REGION, database_rid=DATABASE_RID, collection_rid=COLLECTION_RID, filter="$filter=(name.value eq 'Total Requests') and timeGrain eq duration'PT5M' and startTime eq '2017-11-19T23:53:55.2780000Z' and endTime eq '2017-11-20T00:13:55.2780000Z")
        result = self.mgmt_client.database_account_region.list_metrics(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, region=REGION, filter="$filter=(name.value eq 'Total Requests') and timeGrain eq duration'PT5M' and startTime eq '2017-11-19T23:53:55.2780000Z' and endTime eq '2017-11-20T00:13:55.2780000Z")
        result = self.mgmt_client.collection_partition_region.list_metrics(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, region=REGION, database_rid=DATABASE_RID, collection_rid=COLLECTION_RID, filter="$filter=(name.value eq 'Max RUs Per Second') and timeGrain eq duration'PT1M' and startTime eq '2017-11-19T23:53:55.2780000Z' and endTime eq '2017-11-20T23:58:55.2780000Z")
        result = self.mgmt_client.collection_partition.list_metrics(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_rid=DATABASE_RID, collection_rid=COLLECTION_RID, filter="$filter=(name.value eq 'Max RUs Per Second') and timeGrain eq duration'PT1M' and startTime eq '2017-11-19T23:53:55.2780000Z' and endTime eq '2017-11-20T23:58:55.2780000Z")
        result = self.mgmt_client.collection_partition.list_usages(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_rid=DATABASE_RID, collection_rid=COLLECTION_RID, filter="$filter=name.value eq 'Partition Storage'")
        result = self.mgmt_client.mongo_db_resources.begin_delete_mongo_db_collection(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, collection_name=COLLECTION_NAME)
        result = result.result()
        result = self.mgmt_client.mongo_db_resources.begin_delete_mongo_db_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME)
        result = result.result()
        result = self.mgmt_client.database_accounts.begin_delete(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = result.result()

    @unittest.skip('hard to test')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_query_partition_key(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        RESOURCE_GROUP = resource_group.name
        ACCOUNT_NAME = 'myaccountxxyyzzz'
        DATABASE_NAME = 'myDatabase'
        GRAPH_NAME = 'myGraph'
        PARTITION_KEY_RANGE_ID = 0
        REGION = AZURE_LOCATION
        BODY = {'location': AZURE_LOCATION, 'kind': 'GlobalDocumentDB', 'database_account_offer_type': 'Standard', 'locations': [{'location_name': 'eastus', 'is_zone_redundant': False, 'failover_priority': '0'}], 'capabilities': [{'name': 'EnableGremlin'}], 'api_properties': {}}
        result = self.mgmt_client.database_accounts.begin_create_or_update(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, create_update_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': DATABASE_NAME}, 'options': {'throughput': '2000'}}
        result = self.mgmt_client.gremlin_resources.begin_create_update_gremlin_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, create_update_gremlin_database_parameters=BODY)
        database = result.result()
        DATABASE_RID = database.resource.rid
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': GRAPH_NAME, 'indexing_policy': {'indexing_mode': 'Consistent', 'automatic': True, 'included_paths': [{'path': '/*', 'indexes': [{'kind': 'Range', 'data_type': 'String', 'precision': '-1'}, {'kind': 'Range', 'data_type': 'Number', 'precision': '-1'}]}], 'excluded_paths': []}, 'partition_key': {'paths': ['/AccountNumber'], 'kind': 'Hash'}, 'default_ttl': '100', 'conflict_resolution_policy': {'mode': 'LastWriterWins', 'conflict_resolution_path': '/path'}}, 'options': {'throughput': '2000'}}
        result = self.mgmt_client.gremlin_resources.begin_create_update_gremlin_graph(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, graph_name=GRAPH_NAME, create_update_gremlin_graph_parameters=BODY)
        collection = result.result()
        COLLECTION_RID = collection.resource.rid
        result = self.mgmt_client.partition_key_range_id_region.list_metrics(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, region=REGION, database_rid=DATABASE_RID, collection_rid=COLLECTION_RID, partition_key_range_id=PARTITION_KEY_RANGE_ID, filter="$filter=(name.value eq 'Max RUs Per Second') and timeGrain eq duration'PT1M' and startTime eq '2017-11-19T23:53:55.2780000Z' and endTime eq '2017-11-20T23:58:55.2780000Z")
        result = self.mgmt_client.partition_key_range_id.list_metrics(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_rid=DATABASE_RID, collection_rid=COLLECTION_RID, partition_key_range_id=PARTITION_KEY_RANGE_ID, filter="$filter=(name.value eq 'Max RUs Per Second') and timeGrain eq duration'PT1M' and startTime eq '2017-11-19T23:53:55.2780000Z' and endTime eq '2017-11-20T23:58:55.2780000Z")
        result = self.mgmt_client.gremlin_resources.begin_delete_gremlin_graph(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, graph_name=GRAPH_NAME)
        result = result.result()
        result = self.mgmt_client.gremlin_resources.begin_delete_gremlin_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME)
        result = result.result()
        result = self.mgmt_client.database_accounts.begin_delete(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = result.result()