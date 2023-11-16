"""Run the auth spec tests."""
from __future__ import annotations
import os
import sys
import time
import uuid
from typing import Any, Mapping
sys.path[0:0] = ['']
from test import IntegrationTest, unittest
from test.unified_format import generate_test_classes
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel
_TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'index_management')
_NAME = 'test-search-index'

class TestCreateSearchIndex(IntegrationTest):

    def test_inputs(self):
        if False:
            i = 10
            return i + 15
        if not os.environ.get('TEST_INDEX_MANAGEMENT'):
            raise unittest.SkipTest('Skipping index management tests')
        client = MongoClient()
        self.addCleanup(client.close)
        coll = client.test.test
        coll.drop()
        definition = dict(mappings=dict(dynamic=True))
        model_kwarg_list: list[Mapping[str, Any]] = [dict(definition=definition, name=None), dict(definition=definition, name='test')]
        for model_kwargs in model_kwarg_list:
            model = SearchIndexModel(**model_kwargs)
            with self.assertRaises(OperationFailure):
                coll.create_search_index(model)
            with self.assertRaises(OperationFailure):
                coll.create_search_index(model_kwargs)

class TestSearchIndexProse(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            while True:
                i = 10
        super().setUpClass()
        if not os.environ.get('TEST_INDEX_MANAGEMENT'):
            raise unittest.SkipTest('Skipping index management tests')
        url = os.environ.get('MONGODB_URI')
        username = os.environ['DB_USER']
        password = os.environ['DB_PASSWORD']
        cls.client = MongoClient(url, username=username, password=password)
        cls.client.drop_database(_NAME)
        cls.db = cls.client.test_search_index_prose

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        cls.client.drop_database(_NAME)
        cls.client.close()

    def wait_for_ready(self, coll, name=_NAME, predicate=None):
        if False:
            i = 10
            return i + 15
        'Wait for a search index to be ready.'
        indices: list[Mapping[str, Any]] = []
        if predicate is None:
            predicate = lambda index: index.get('queryable') is True
        while True:
            indices = list(coll.list_search_indexes(name))
            if len(indices) and predicate(indices[0]):
                return indices[0]
                break
            time.sleep(5)

    def test_case_1(self):
        if False:
            i = 10
            return i + 15
        'Driver can successfully create and list search indexes.'
        coll0 = self.db[f'col{uuid.uuid4()}']
        model = {'name': _NAME, 'definition': {'mappings': {'dynamic': False}}}
        coll0.insert_one({})
        resp = coll0.create_search_index(model)
        self.assertEqual(resp, _NAME)
        index = self.wait_for_ready(coll0)
        self.assertIn('latestDefinition', index)
        self.assertEqual(index['latestDefinition'], model['definition'])

    def test_case_2(self):
        if False:
            for i in range(10):
                print('nop')
        'Driver can successfully create multiple indexes in batch.'
        coll0 = self.db[f'col{uuid.uuid4()}']
        coll0.insert_one({})
        name1 = 'test-search-index-1'
        name2 = 'test-search-index-2'
        definition = {'mappings': {'dynamic': False}}
        index_definitions: list[dict[str, Any]] = [{'name': name1, 'definition': definition}, {'name': name2, 'definition': definition}]
        coll0.create_search_indexes([SearchIndexModel(i['definition'], i['name']) for i in index_definitions])
        indices = list(coll0.list_search_indexes())
        names = [i['name'] for i in indices]
        self.assertIn(name1, names)
        self.assertIn(name2, names)
        index1 = self.wait_for_ready(coll0, name1)
        index2 = self.wait_for_ready(coll0, name2)
        for index in [index1, index2]:
            self.assertIn('latestDefinition', index)
            self.assertEqual(index['latestDefinition'], definition)

    def test_case_3(self):
        if False:
            print('Hello World!')
        'Driver can successfully drop search indexes.'
        coll0 = self.db[f'col{uuid.uuid4()}']
        coll0.insert_one({})
        model = {'name': _NAME, 'definition': {'mappings': {'dynamic': False}}}
        resp = coll0.create_search_index(model)
        self.assertEqual(resp, 'test-search-index')
        self.wait_for_ready(coll0)
        coll0.drop_search_index(_NAME)
        t0 = time.time()
        while True:
            indices = list(coll0.list_search_indexes())
            if indices:
                break
            if (time.time() - t0) / 60 > 5:
                raise TimeoutError('Timed out waiting for index deletion')
            time.sleep(5)

    def test_case_4(self):
        if False:
            while True:
                i = 10
        'Driver can update a search index.'
        coll0 = self.db[f'col{uuid.uuid4()}']
        coll0.insert_one({})
        model = {'name': _NAME, 'definition': {'mappings': {'dynamic': False}}}
        resp = coll0.create_search_index(model)
        self.assertEqual(resp, _NAME)
        self.wait_for_ready(coll0)
        model2: dict[str, Any] = {'name': _NAME, 'definition': {'mappings': {'dynamic': True}}}
        coll0.update_search_index(_NAME, model2['definition'])
        predicate = lambda index: index.get('queryable') is True and index.get('status') == 'READY'
        self.wait_for_ready(coll0, predicate=predicate)
        index = list(coll0.list_search_indexes(_NAME))[0]
        self.assertIn('latestDefinition', index)
        self.assertEqual(index['latestDefinition'], model2['definition'])

    def test_case_5(self):
        if False:
            while True:
                i = 10
        '``dropSearchIndex`` suppresses namespace not found errors.'
        coll0 = self.db[f'col{uuid.uuid4()}']
        coll0.drop_search_index('foo')
if os.environ.get('TEST_INDEX_MANAGEMENT'):
    globals().update(generate_test_classes(_TEST_PATH, module=__name__))
else:

    class TestIndexManagementUnifiedTests(unittest.TestCase):

        @classmethod
        def setUpClass(cls) -> None:
            if False:
                i = 10
                return i + 15
            raise unittest.SkipTest('Skipping index management pending PYTHON-3951')

        def test_placeholder(self):
            if False:
                i = 10
                return i + 15
            pass
if __name__ == '__main__':
    unittest.main()