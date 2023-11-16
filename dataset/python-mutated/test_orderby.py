import unittest
import uuid
import pytest
from azure.core.paging import ItemPaged
from azure.cosmos.partition_key import PartitionKey
import azure.cosmos.cosmos_client as cosmos_client
from azure.cosmos import _query_iterable as query_iterable
import azure.cosmos._base as base
import test_config
pytestmark = pytest.mark.cosmosEmulator

@pytest.mark.usefixtures('teardown')
class CrossPartitionTopOrderByTest(unittest.TestCase):
    """Orderby Tests.
    """
    host = test_config._test_config.host
    masterKey = test_config._test_config.masterKey
    connectionPolicy = test_config._test_config.connectionPolicy

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        if cls.masterKey == '[YOUR_KEY_HERE]' or cls.host == '[YOUR_ENDPOINT_HERE]':
            raise Exception("You must specify your Azure Cosmos account values for 'masterKey' and 'host' at the top of this class to run the tests.")
        cls.client = cosmos_client.CosmosClient(cls.host, cls.masterKey, 'Session', connection_policy=cls.connectionPolicy)
        cls.created_db = cls.client.create_database_if_not_exists(test_config._test_config.TEST_DATABASE_ID)
        cls.created_collection = cls.created_db.create_container(id='orderby_tests collection ' + str(uuid.uuid4()), indexing_policy={'includedPaths': [{'path': '/', 'indexes': [{'kind': 'Range', 'dataType': 'Number'}, {'kind': 'Range', 'dataType': 'String'}]}]}, partition_key=PartitionKey(path='/id'), offer_throughput=30000)
        cls.collection_link = cls.GetDocumentCollectionLink(cls.created_db, cls.created_collection)
        cls.document_definitions = []
        for i in range(20):
            d = {'id': str(i), 'name': 'sample document', 'spam': 'eggs' + str(i), 'cnt': i, 'key': 'value', 'spam2': 'eggs' + str(i) if i == 3 else i, 'boolVar': i % 2 == 0, 'number': 1.1 * i}
            cls.created_collection.create_item(d)
            cls.document_definitions.append(d)

    def test_orderby_query(self):
        if False:
            for i in range(10):
                print('nop')
        query = {'query': 'SELECT * FROM root r order by r.spam'}

        def get_order_by_key(r):
            if False:
                return 10
            return r['spam']
        expected_ordered_ids = [r['id'] for r in sorted(self.document_definitions, key=get_order_by_key)]
        self.execute_query_and_validate_results(query, expected_ordered_ids)

    def test_orderby_query_as_string(self):
        if False:
            for i in range(10):
                print('nop')
        query = 'SELECT * FROM root r order by r.spam'

        def get_order_by_key(r):
            if False:
                return 10
            return r['spam']
        expected_ordered_ids = [r['id'] for r in sorted(self.document_definitions, key=get_order_by_key)]
        self.execute_query_and_validate_results(query, expected_ordered_ids)

    def test_orderby_asc_query(self):
        if False:
            for i in range(10):
                print('nop')
        query = {'query': 'SELECT * FROM root r order by r.spam ASC'}

        def get_order_by_key(r):
            if False:
                for i in range(10):
                    print('nop')
            return r['spam']
        expected_ordered_ids = [r['id'] for r in sorted(self.document_definitions, key=get_order_by_key)]
        self.execute_query_and_validate_results(query, expected_ordered_ids)

    def test_orderby_desc_query(self):
        if False:
            for i in range(10):
                print('nop')
        query = {'query': 'SELECT * FROM root r order by r.spam DESC'}

        def get_order_by_key(r):
            if False:
                i = 10
                return i + 15
            return r['spam']
        expected_ordered_ids = [r['id'] for r in sorted(self.document_definitions, key=get_order_by_key, reverse=True)]
        self.execute_query_and_validate_results(query, expected_ordered_ids)

    def test_orderby_top_query(self):
        if False:
            for i in range(10):
                print('nop')
        top_count = 9
        self.assertLess(top_count, len(self.document_definitions))
        query = {'query': 'SELECT top %d * FROM root r order by r.spam' % top_count}

        def get_order_by_key(r):
            if False:
                i = 10
                return i + 15
            return r['spam']
        expected_ordered_ids = [r['id'] for r in sorted(self.document_definitions, key=get_order_by_key)[:top_count]]
        self.execute_query_and_validate_results(query, expected_ordered_ids)

    def test_orderby_top_query_less_results_than_top_counts(self):
        if False:
            while True:
                i = 10
        top_count = 30
        self.assertGreater(top_count, len(self.document_definitions))
        query = {'query': 'SELECT top %d * FROM root r order by r.spam' % top_count}

        def get_order_by_key(r):
            if False:
                return 10
            return r['spam']
        expected_ordered_ids = [r['id'] for r in sorted(self.document_definitions, key=get_order_by_key)]
        self.execute_query_and_validate_results(query, expected_ordered_ids)

    def test_top_query(self):
        if False:
            print('Hello World!')
        partition_key_ranges = list(self.client.client_connection._ReadPartitionKeyRanges(self.collection_link))
        docs_by_partition_key_range_id = self.find_docs_by_partition_key_range_id()
        cnt = 0
        first_two_ranges_results = []
        for r in partition_key_ranges:
            if cnt >= 2:
                break
            p_id = r['id']
            if len(docs_by_partition_key_range_id[p_id]) > 0:
                first_two_ranges_results.extend(docs_by_partition_key_range_id[p_id])
                cnt += 1
        self.assertEqual(cnt, 2)
        self.assertLess(2, len(partition_key_ranges))
        self.assertLess(len(first_two_ranges_results), len(self.document_definitions))
        self.assertGreater(len(first_two_ranges_results), 1)
        expected_ordered_ids = [d['id'] for d in first_two_ranges_results]
        query = {'query': 'SELECT top %d * FROM root r' % len(expected_ordered_ids)}
        self.execute_query_and_validate_results(query, expected_ordered_ids)

    def test_top_query_as_string(self):
        if False:
            print('Hello World!')
        partition_key_ranges = list(self.client.client_connection._ReadPartitionKeyRanges(self.collection_link))
        docs_by_partition_key_range_id = self.find_docs_by_partition_key_range_id()
        cnt = 0
        first_two_ranges_results = []
        for r in partition_key_ranges:
            if cnt >= 2:
                break
            p_id = r['id']
            if len(docs_by_partition_key_range_id[p_id]) > 0:
                first_two_ranges_results.extend(docs_by_partition_key_range_id[p_id])
                cnt += 1
        self.assertEqual(cnt, 2)
        self.assertLess(2, len(partition_key_ranges))
        self.assertLess(len(first_two_ranges_results), len(self.document_definitions))
        self.assertGreater(len(first_two_ranges_results), 1)
        expected_ordered_ids = [d['id'] for d in first_two_ranges_results]
        query = 'SELECT top %d * FROM root r' % len(expected_ordered_ids)
        self.execute_query_and_validate_results(query, expected_ordered_ids)

    def test_parametrized_top_query(self):
        if False:
            return 10
        partition_key_ranges = list(self.client.client_connection._ReadPartitionKeyRanges(self.collection_link))
        docs_by_partition_key_range_id = self.find_docs_by_partition_key_range_id()
        cnt = 0
        first_two_ranges_results = []
        for r in partition_key_ranges:
            if cnt >= 2:
                break
            p_id = r['id']
            if len(docs_by_partition_key_range_id[p_id]) > 0:
                first_two_ranges_results.extend(docs_by_partition_key_range_id[p_id])
                cnt += 1
        self.assertEqual(cnt, 2)
        self.assertLess(2, len(partition_key_ranges))
        self.assertLess(len(first_two_ranges_results), len(self.document_definitions))
        self.assertGreater(len(first_two_ranges_results), 1)
        expected_ordered_ids = [d['id'] for d in first_two_ranges_results]
        query = {'query': 'SELECT top @n * FROM root r', 'parameters': [{'name': '@n', 'value': len(expected_ordered_ids)}]}
        self.execute_query_and_validate_results(query, expected_ordered_ids)

    def test_orderby_query_with_parametrized_top(self):
        if False:
            for i in range(10):
                print('nop')
        top_count = 9
        self.assertLess(top_count, len(self.document_definitions))

        def get_order_by_key(r):
            if False:
                while True:
                    i = 10
            return r['spam']
        expected_ordered_ids = [r['id'] for r in sorted(self.document_definitions, key=get_order_by_key)[:top_count]]
        query = {'query': 'SELECT top @n * FROM root r order by r.spam', 'parameters': [{'name': '@n', 'value': top_count}]}
        self.execute_query_and_validate_results(query, expected_ordered_ids)

    def test_orderby_query_with_parametrized_predicate(self):
        if False:
            i = 10
            return i + 15
        query = {'query': 'SELECT * FROM root r where r.cnt > @cnt order by r.spam', 'parameters': [{'name': '@cnt', 'value': 5}]}

        def get_order_by_key(r):
            if False:
                for i in range(10):
                    print('nop')
            return r['spam']
        expected_ordered_ids = [r['id'] for r in sorted(self.document_definitions, key=get_order_by_key) if r['cnt'] > 5]
        self.execute_query_and_validate_results(query, expected_ordered_ids)

    def test_orderby_query_noncomparable_orderby_item(self):
        if False:
            return 10
        query = {'query': 'SELECT * FROM root r order by r.spam2 DESC'}

        def get_order_by_key(r):
            if False:
                while True:
                    i = 10
            return r['id']
        expected_ordered_ids = [r['id'] for r in sorted(self.document_definitions, key=get_order_by_key)]
        try:
            self.execute_query_and_validate_results(query, expected_ordered_ids)
            self.fail('non comparable order by items did not result in failure.')
        except ValueError as e:
            self.assertTrue(e.args[0] == 'Expected String, but got Number.' or e.message == 'Expected Number, but got String.')

    def test_orderby_integer_query(self):
        if False:
            print('Hello World!')
        query = {'query': 'SELECT * FROM root r order by r.cnt'}

        def get_order_by_key(r):
            if False:
                print('Hello World!')
            return r['cnt']
        expected_ordered_ids = [r['id'] for r in sorted(self.document_definitions, key=get_order_by_key)]
        self.execute_query_and_validate_results(query, expected_ordered_ids)

    def test_orderby_floating_point_number_query(self):
        if False:
            i = 10
            return i + 15
        query = {'query': 'SELECT * FROM root r order by r.number'}

        def get_order_by_key(r):
            if False:
                while True:
                    i = 10
            return r['number']
        expected_ordered_ids = [r['id'] for r in sorted(self.document_definitions, key=get_order_by_key)]
        self.execute_query_and_validate_results(query, expected_ordered_ids)

    def test_orderby_boolean_query(self):
        if False:
            return 10
        query = {'query': 'SELECT * FROM root r order by r.boolVar'}
        result_iterable = self.created_collection.query_items(query=query, enable_cross_partition_query=True, max_item_count=2)
        results = list(result_iterable)
        self.assertEqual(len(results), len(self.document_definitions))
        index = 0
        while index < len(results):
            if results[index]['boolVar']:
                break
            self.assertTrue(int(results[index]['id']) % 2 == 1)
            index = index + 1
        while index < len(results):
            self.assertTrue(results[index]['boolVar'])
            self.assertTrue(int(results[index]['id']) % 2 == 0)
            index = index + 1

    def find_docs_by_partition_key_range_id(self):
        if False:
            while True:
                i = 10
        query = {'query': 'SELECT * FROM root r'}
        partition_key_range = list(self.client.client_connection._ReadPartitionKeyRanges(self.collection_link))
        docs_by_partition_key_range_id = {}
        for r in partition_key_range:
            options = {}
            path = base.GetPathFromLink(self.collection_link, 'docs')
            collection_id = base.GetResourceIdOrFullNameFromLink(self.collection_link)

            def fetch_fn(options):
                if False:
                    i = 10
                    return i + 15
                return self.client.client_connection.QueryFeed(path, collection_id, query, options, r['id'])
            docResultsIterable = ItemPaged(self.client.client_connection, query, options, fetch_function=fetch_fn, collection_link=self.collection_link, page_iterator_class=query_iterable.QueryIterable)
            docs = list(docResultsIterable)
            self.assertFalse(r['id'] in docs_by_partition_key_range_id)
            docs_by_partition_key_range_id[r['id']] = docs
        return docs_by_partition_key_range_id

    def execute_query_and_validate_results(self, query, expected_ordered_ids):
        if False:
            print('Hello World!')
        page_size = 2
        result_iterable = self.created_collection.query_items(query=query, enable_cross_partition_query=True, max_item_count=page_size)
        self.assertTrue(isinstance(result_iterable, ItemPaged))
        self.assertEqual(result_iterable._page_iterator_class, query_iterable.QueryIterable)
        it = result_iterable.__iter__()

        def invokeNext():
            if False:
                for i in range(10):
                    print('nop')
            return next(it)
        for i in range(len(expected_ordered_ids)):
            item = invokeNext()
            self.assertEqual(item['id'], expected_ordered_ids[i])
        results = {}
        cnt = 0
        page_iter = result_iterable.by_page()
        for page in page_iter:
            fetched_res = list(page)
            fetched_size = len(fetched_res)
            for item in fetched_res:
                self.assertEqual(item['id'], expected_ordered_ids[cnt])
                results[cnt] = item
                cnt = cnt + 1
            if cnt < len(expected_ordered_ids):
                self.assertEqual(fetched_size, page_size, 'page size')
            elif cnt == len(expected_ordered_ids):
                self.assertTrue(fetched_size <= page_size, 'last page size')
                break
            else:
                self.fail('more results than expected')
        self.assertEqual(len(results), len(expected_ordered_ids))
        with self.assertRaises(StopIteration):
            next(page_iter)

    @classmethod
    def create_collection(self, client, created_db):
        if False:
            while True:
                i = 10
        created_collection = created_db.create_container(id='orderby_tests collection ' + str(uuid.uuid4()), indexing_policy={'includedPaths': [{'path': '/', 'indexes': [{'kind': 'Range', 'dataType': 'Number'}, {'kind': 'Range', 'dataType': 'String'}]}]}, partition_key=PartitionKey(path='/id', kind='Hash'), offer_throughput=30000)
        return created_collection

    @classmethod
    def insert_doc(cls):
        if False:
            print('Hello World!')
        created_docs = []
        for d in cls.document_definitions:
            created_doc = cls.created_collection.create_item(body=d)
            created_docs.append(created_doc)
        return created_docs

    @classmethod
    def GetDatabaseLink(cls, database, is_name_based=True):
        if False:
            for i in range(10):
                print('nop')
        if is_name_based:
            return 'dbs/' + database.id
        else:
            return database['_self']

    @classmethod
    def GetDocumentCollectionLink(cls, database, document_collection, is_name_based=True):
        if False:
            print('Hello World!')
        if is_name_based:
            return cls.GetDatabaseLink(database) + '/colls/' + document_collection.id
        else:
            return document_collection['_self']

    @classmethod
    def GetDocumentLink(cls, database, document_collection, document, is_name_based=True):
        if False:
            i = 10
            return i + 15
        if is_name_based:
            return cls.GetDocumentCollectionLink(database, document_collection) + '/docs/' + document['id']
        else:
            return document['_self']
if __name__ == '__main__':
    unittest.main()