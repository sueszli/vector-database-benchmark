import unittest
import azure.mgmt.cosmosdb
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer, ResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtCosmosDBTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(MgmtCosmosDBTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.cosmosdb.CosmosDBManagementClient)

    @unittest.skip('hard to test')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_sql_resource(self, resource_group):
        if False:
            while True:
                i = 10
        RESOURCE_GROUP = resource_group.name
        ACCOUNT_NAME = 'myaccountxxyyzzz'
        DATABASE_NAME = 'myDatabase'
        BODY = {'location': AZURE_LOCATION, 'kind': 'GlobalDocumentDB', 'database_account_offer_type': 'Standard', 'locations': [{'location_name': 'eastus', 'is_zone_redundant': False, 'failover_priority': '0'}], 'api_properties': {}}
        result = self.mgmt_client.database_accounts.begin_create_or_update(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, create_update_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': DATABASE_NAME}, 'options': {'throughput': '2000'}}
        result = self.mgmt_client.sql_resources.begin_create_update_sql_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, create_update_sql_database_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'throughput': '400'}}
        result = self.mgmt_client.sql_resources.begin_update_sql_database_throughput(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, update_throughput_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.sql_resources.get_sql_database_throughput(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME)
        result = self.mgmt_client.sql_resources.get_sql_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME)
        result = self.mgmt_client.sql_resources.list_sql_databases(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = self.mgmt_client.sql_resources.begin_migrate_sql_database_to_autoscale(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME)
        result = result.result()
        result = self.mgmt_client.sql_resources.begin_migrate_sql_database_to_manual_throughput(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME)
        result = result.result()
        result = self.mgmt_client.sql_resources.begin_delete_sql_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME)
        result = result.result()
        result = self.mgmt_client.database_accounts.begin_delete(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = result.result()

    @unittest.skip('hard to test')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_sql_container(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        RESOURCE_GROUP = resource_group.name
        ACCOUNT_NAME = 'myaccountxxyyzzz'
        DATABASE_NAME = 'myDatabase'
        CONTAINER_NAME = 'myContainer'
        BODY = {'location': AZURE_LOCATION, 'kind': 'GlobalDocumentDB', 'database_account_offer_type': 'Standard', 'locations': [{'location_name': 'eastus', 'is_zone_redundant': False, 'failover_priority': '0'}], 'api_properties': {}}
        result = self.mgmt_client.database_accounts.begin_create_or_update(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, create_update_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': DATABASE_NAME}, 'options': {'throughput': 1000}}
        result = self.mgmt_client.sql_resources.begin_create_update_sql_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, create_update_sql_database_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': CONTAINER_NAME, 'indexing_policy': {'indexing_mode': 'Consistent', 'automatic': True, 'included_paths': [{'path': '/*', 'indexes': [{'kind': 'Range', 'data_type': 'String', 'precision': '-1'}, {'kind': 'Range', 'data_type': 'Number', 'precision': '-1'}]}], 'excluded_paths': []}, 'partition_key': {'paths': ['/AccountNumber'], 'kind': 'Hash'}, 'default_ttl': '100', 'unique_key_policy': {'unique_keys': [{'paths': ['/testPath']}]}, 'conflict_resolution_policy': {'mode': 'LastWriterWins', 'conflict_resolution_path': '/path'}}, 'options': {'throughput': '2000'}}
        result = self.mgmt_client.sql_resources.begin_create_update_sql_container(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, create_update_sql_container_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'throughput': '400'}}
        result = self.mgmt_client.sql_resources.begin_update_sql_container_throughput(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, update_throughput_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.sql_resources.get_sql_container_throughput(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME)
        result = self.mgmt_client.sql_resources.get_sql_container(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME)
        result = self.mgmt_client.sql_resources.list_sql_containers(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME)
        result = self.mgmt_client.sql_resources.begin_migrate_sql_container_to_autoscale(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME)
        result = result.result()
        result = self.mgmt_client.sql_resources.begin_migrate_sql_container_to_manual_throughput(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME)
        result = result.result()
        result = self.mgmt_client.sql_resources.begin_delete_sql_container(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME)
        result = result.result()
        result = self.mgmt_client.sql_resources.begin_delete_sql_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME)
        result = result.result()
        result = self.mgmt_client.database_accounts.begin_delete(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = result.result()

    @unittest.skip('hard to test')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_sql_trigger(self, resource_group):
        if False:
            print('Hello World!')
        RESOURCE_GROUP = resource_group.name
        ACCOUNT_NAME = 'myaccountxxyyzzz'
        DATABASE_NAME = 'myDatabase'
        CONTAINER_NAME = 'myContainer'
        TRIGGER_NAME = 'myTrigger'
        BODY = {'location': AZURE_LOCATION, 'kind': 'GlobalDocumentDB', 'database_account_offer_type': 'Standard', 'locations': [{'location_name': 'eastus', 'is_zone_redundant': False, 'failover_priority': '0'}], 'api_properties': {}}
        result = self.mgmt_client.database_accounts.begin_create_or_update(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, create_update_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': DATABASE_NAME}, 'options': {'throughput': 1000}}
        result = self.mgmt_client.sql_resources.begin_create_update_sql_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, create_update_sql_database_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': CONTAINER_NAME, 'indexing_policy': {'indexing_mode': 'Consistent', 'automatic': True, 'included_paths': [{'path': '/*', 'indexes': [{'kind': 'Range', 'data_type': 'String', 'precision': '-1'}, {'kind': 'Range', 'data_type': 'Number', 'precision': '-1'}]}], 'excluded_paths': []}, 'partition_key': {'paths': ['/AccountNumber'], 'kind': 'Hash'}, 'default_ttl': '100', 'unique_key_policy': {'unique_keys': [{'paths': ['/testPath']}]}, 'conflict_resolution_policy': {'mode': 'LastWriterWins', 'conflict_resolution_path': '/path'}}, 'options': {}}
        result = self.mgmt_client.sql_resources.begin_create_update_sql_container(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, create_update_sql_container_parameters=BODY)
        result = result.result()
        BODY = {'resource': {'id': TRIGGER_NAME, 'body': 'body', 'trigger_type': 'Pre', 'trigger_operation': 'All'}, 'options': {}}
        result = self.mgmt_client.sql_resources.begin_create_update_sql_trigger(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, trigger_name=TRIGGER_NAME, create_update_sql_trigger_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.sql_resources.get_sql_trigger(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, trigger_name=TRIGGER_NAME)
        result = self.mgmt_client.sql_resources.list_sql_triggers(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME)
        result = self.mgmt_client.sql_resources.begin_delete_sql_trigger(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, trigger_name=TRIGGER_NAME)
        result = result.result()
        result = self.mgmt_client.sql_resources.begin_delete_sql_container(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME)
        result = result.result()
        result = self.mgmt_client.sql_resources.begin_delete_sql_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME)
        result = result.result()
        result = self.mgmt_client.database_accounts.begin_delete(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = result.result()

    @unittest.skip('hard to test')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_sql_stored_procedure(self, resource_group):
        if False:
            print('Hello World!')
        RESOURCE_GROUP = resource_group.name
        ACCOUNT_NAME = 'myaccountxxyyzzz'
        DATABASE_NAME = 'myDatabase'
        CONTAINER_NAME = 'myContainer'
        STORED_PROCEDURE_NAME = 'myStoredProcedure'
        BODY = {'location': AZURE_LOCATION, 'kind': 'GlobalDocumentDB', 'database_account_offer_type': 'Standard', 'locations': [{'location_name': 'eastus', 'is_zone_redundant': False, 'failover_priority': '0'}], 'api_properties': {}}
        result = self.mgmt_client.database_accounts.begin_create_or_update(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, create_update_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': DATABASE_NAME}, 'options': {'throughput': 1000}}
        result = self.mgmt_client.sql_resources.begin_create_update_sql_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, create_update_sql_database_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': CONTAINER_NAME, 'indexing_policy': {'indexing_mode': 'Consistent', 'automatic': True, 'included_paths': [{'path': '/*', 'indexes': [{'kind': 'Range', 'data_type': 'String', 'precision': '-1'}, {'kind': 'Range', 'data_type': 'Number', 'precision': '-1'}]}], 'excluded_paths': []}, 'partition_key': {'paths': ['/AccountNumber'], 'kind': 'Hash'}, 'default_ttl': '100', 'unique_key_policy': {'unique_keys': [{'paths': ['/testPath']}]}, 'conflict_resolution_policy': {'mode': 'LastWriterWins', 'conflict_resolution_path': '/path'}}, 'options': {}}
        result = self.mgmt_client.sql_resources.begin_create_update_sql_container(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, create_update_sql_container_parameters=BODY)
        result = result.result()
        BODY = {'resource': {'id': STORED_PROCEDURE_NAME, 'body': 'body'}, 'options': {}}
        result = self.mgmt_client.sql_resources.begin_create_update_sql_stored_procedure(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, stored_procedure_name=STORED_PROCEDURE_NAME, create_update_sql_stored_procedure_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.sql_resources.get_sql_stored_procedure(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, stored_procedure_name=STORED_PROCEDURE_NAME)
        result = self.mgmt_client.sql_resources.list_sql_stored_procedures(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME)
        result = self.mgmt_client.sql_resources.begin_delete_sql_stored_procedure(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, stored_procedure_name=STORED_PROCEDURE_NAME)
        result = result.result()
        result = self.mgmt_client.sql_resources.begin_delete_sql_container(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME)
        result = result.result()
        result = self.mgmt_client.sql_resources.begin_delete_sql_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME)
        result = result.result()
        result = self.mgmt_client.database_accounts.begin_delete(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = result.result()

    @unittest.skip('hard to test')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_sql_defined_function(self, resource_group):
        if False:
            print('Hello World!')
        RESOURCE_GROUP = resource_group.name
        ACCOUNT_NAME = 'myaccountxxyyzzz'
        DATABASE_NAME = 'myDatabase'
        CONTAINER_NAME = 'myContainer'
        USER_DEFINED_FUNCTION_NAME = 'myUserDefinedFunction'
        BODY = {'location': AZURE_LOCATION, 'kind': 'GlobalDocumentDB', 'database_account_offer_type': 'Standard', 'locations': [{'location_name': 'eastus', 'is_zone_redundant': False, 'failover_priority': '0'}], 'api_properties': {}}
        result = self.mgmt_client.database_accounts.begin_create_or_update(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, create_update_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': DATABASE_NAME}, 'options': {'throughput': 1000}}
        result = self.mgmt_client.sql_resources.begin_create_update_sql_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, create_update_sql_database_parameters=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'resource': {'id': CONTAINER_NAME, 'indexing_policy': {'indexing_mode': 'Consistent', 'automatic': True, 'included_paths': [{'path': '/*', 'indexes': [{'kind': 'Range', 'data_type': 'String', 'precision': '-1'}, {'kind': 'Range', 'data_type': 'Number', 'precision': '-1'}]}], 'excluded_paths': []}, 'partition_key': {'paths': ['/AccountNumber'], 'kind': 'Hash'}, 'default_ttl': '100', 'unique_key_policy': {'unique_keys': [{'paths': ['/testPath']}]}, 'conflict_resolution_policy': {'mode': 'LastWriterWins', 'conflict_resolution_path': '/path'}}, 'options': {}}
        result = self.mgmt_client.sql_resources.begin_create_update_sql_container(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, create_update_sql_container_parameters=BODY)
        result = result.result()
        BODY = {'resource': {'id': USER_DEFINED_FUNCTION_NAME, 'body': 'body'}, 'options': {}}
        result = self.mgmt_client.sql_resources.begin_create_update_sql_user_defined_function(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, user_defined_function_name=USER_DEFINED_FUNCTION_NAME, create_update_sql_user_defined_function_parameters=BODY)
        result = result.result()
        result = self.mgmt_client.sql_resources.get_sql_user_defined_function(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, user_defined_function_name=USER_DEFINED_FUNCTION_NAME)
        result = self.mgmt_client.sql_resources.list_sql_user_defined_functions(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME)
        result = self.mgmt_client.sql_resources.begin_delete_sql_user_defined_function(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME, user_defined_function_name=USER_DEFINED_FUNCTION_NAME)
        result = result.result()
        result = self.mgmt_client.sql_resources.begin_delete_sql_container(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME, container_name=CONTAINER_NAME)
        result = result.result()
        result = self.mgmt_client.sql_resources.begin_delete_sql_database(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, database_name=DATABASE_NAME)
        result = result.result()
        result = self.mgmt_client.database_accounts.begin_delete(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = result.result()