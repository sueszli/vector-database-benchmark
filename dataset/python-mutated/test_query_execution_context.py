import unittest
import uuid
import pytest
import azure.cosmos.cosmos_client as cosmos_client
from azure.cosmos._execution_context import base_execution_context as base_execution_context
import azure.cosmos._base as base
import test_config
from azure.cosmos.partition_key import PartitionKey
pytestmark = pytest.mark.cosmosEmulator

@pytest.mark.usefixtures('teardown')
class QueryExecutionContextEndToEndTests(unittest.TestCase):
    """Routing Map Functionalities end to end Tests.
    """
    host = test_config._test_config.host
    masterKey = test_config._test_config.masterKey
    connectionPolicy = test_config._test_config.connectionPolicy

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        if cls.masterKey == '[YOUR_KEY_HERE]' or cls.host == '[YOUR_ENDPOINT_HERE]':
            raise Exception("You must specify your Azure Cosmos account values for 'masterKey' and 'host' at the top of this class to run the tests.")
        cls.client = cosmos_client.CosmosClient(QueryExecutionContextEndToEndTests.host, QueryExecutionContextEndToEndTests.masterKey, consistency_level='Session', connection_policy=QueryExecutionContextEndToEndTests.connectionPolicy)
        cls.created_db = cls.client.create_database_if_not_exists(test_config._test_config.TEST_DATABASE_ID)
        cls.created_collection = cls.created_db.create_container(id='query_execution_context_tests_' + str(uuid.uuid4()), partition_key=PartitionKey(path='/id', kind='Hash'))
        cls.document_definitions = []
        for i in range(20):
            d = {'id': str(i), 'name': 'sample document', 'spam': 'eggs' + str(i), 'key': 'value'}
            cls.document_definitions.append(d)
        cls.insert_doc(cls.document_definitions)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        cls.created_db.delete_container(container=cls.created_collection)

    def setUp(self):
        if False:
            return 10
        partition_key_ranges = list(self.client.client_connection._ReadPartitionKeyRanges(self.GetDocumentCollectionLink(self.created_db, self.created_collection)))
        self.assertGreaterEqual(len(partition_key_ranges), 1)
        queried_docs = list(self.created_collection.read_all_items())
        self.assertEqual(len(queried_docs), len(self.document_definitions), 'create should increase the number of documents')

    def test_no_query_default_execution_context(self):
        if False:
            for i in range(10):
                print('nop')
        options = {'maxItemCount': 2}
        self._test_default_execution_context(options, None, 20)

    def test_no_query_default_execution_context_with_small_last_page(self):
        if False:
            for i in range(10):
                print('nop')
        options = {'maxItemCount': 3}
        self._test_default_execution_context(options, None, 20)

    def test_simple_query_default_execution_context(self):
        if False:
            while True:
                i = 10
        query = {'query': 'SELECT * FROM root r WHERE r.id != @id', 'parameters': [{'name': '@id', 'value': '5'}]}
        options = {'enableCrossPartitionQuery': True, 'maxItemCount': 2}
        res = self.created_collection.query_items(query=query, enable_cross_partition_query=True, max_item_count=2)
        self.assertEqual(len(list(res)), 19)
        self._test_default_execution_context(options, query, 19)

    def test_simple_query_default_execution_context_with_small_last_page(self):
        if False:
            return 10
        query = {'query': 'SELECT * FROM root r WHERE r.id != @id', 'parameters': [{'name': '@id', 'value': '5'}]}
        options = {}
        options['enableCrossPartitionQuery'] = True
        options['maxItemCount'] = 3
        self._test_default_execution_context(options, query, 19)

    def _test_default_execution_context(self, options, query, expected_number_of_results):
        if False:
            for i in range(10):
                print('nop')
        page_size = options['maxItemCount']
        collection_link = self.GetDocumentCollectionLink(self.created_db, self.created_collection)
        path = base.GetPathFromLink(collection_link, 'docs')
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)

        def fetch_fn(options):
            if False:
                print('Hello World!')
            return self.client.client_connection.QueryFeed(path, collection_id, query, options)
        ex = base_execution_context._DefaultQueryExecutionContext(self.client.client_connection, options, fetch_fn)
        it = ex.__iter__()

        def invokeNext():
            if False:
                print('Hello World!')
            return next(it)
        results = {}
        for _ in range(expected_number_of_results):
            item = invokeNext()
            results[item['id']] = item
        self.assertEqual(len(results), expected_number_of_results)
        self.assertRaises(StopIteration, invokeNext)
        ex = base_execution_context._DefaultQueryExecutionContext(self.client.client_connection, options, fetch_fn)
        results = {}
        cnt = 0
        while True:
            fetched_res = ex.fetch_next_block()
            fetched_size = len(fetched_res)
            for item in fetched_res:
                results[item['id']] = item
            cnt += fetched_size
            if cnt < expected_number_of_results:
                self.assertEqual(fetched_size, page_size, 'page size')
            elif cnt == expected_number_of_results:
                self.assertTrue(fetched_size <= page_size, 'last page size')
                break
            else:
                self.fail('more results than expected')
        self.assertEqual(len(results), expected_number_of_results)
        self.assertEqual(ex.fetch_next_block(), [])

    @classmethod
    def insert_doc(cls, document_definitions):
        if False:
            while True:
                i = 10
        created_docs = []
        for d in document_definitions:
            created_doc = cls.created_collection.create_item(body=d)
            created_docs.append(created_doc)
        return created_docs

    def GetDatabaseLink(self, database):
        if False:
            print('Hello World!')
        return 'dbs/' + database.id

    def GetDocumentCollectionLink(self, database, document_collection):
        if False:
            for i in range(10):
                print('nop')
        return self.GetDatabaseLink(database) + '/colls/' + document_collection.id
if __name__ == '__main__':
    unittest.main()