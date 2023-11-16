"""Test the collection module."""
from __future__ import annotations
import os
import sys
sys.path[0:0] = ['']
from test import IntegrationTest, unittest
from test.utils import SpecTestCreator, camel_to_snake, camel_to_snake_args, camel_to_upper_camel, drop_collections
from pymongo import WriteConcern, operations
from pymongo.command_cursor import CommandCursor
from pymongo.cursor import Cursor
from pymongo.errors import PyMongoError
from pymongo.operations import DeleteMany, DeleteOne, InsertOne, ReplaceOne, UpdateMany, UpdateOne
from pymongo.read_concern import ReadConcern
from pymongo.results import BulkWriteResult, _WriteResult
_TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crud', 'v1')

class TestAllScenarios(IntegrationTest):
    RUN_ON_SERVERLESS = True

def check_result(self, expected_result, result):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(result, _WriteResult):
        for res in expected_result:
            prop = camel_to_snake(res)
            msg = f'{prop} : {expected_result!r} != {result!r}'
            if prop == 'upserted_count' and (not isinstance(result, BulkWriteResult)):
                if result.upserted_id is not None:
                    upserted_count = 1
                else:
                    upserted_count = 0
                self.assertEqual(upserted_count, expected_result[res], msg)
            elif prop == 'inserted_ids':
                if isinstance(result, BulkWriteResult):
                    self.assertEqual(len(expected_result[res]), result.inserted_count)
                else:
                    ids = expected_result[res]
                    if isinstance(ids, dict):
                        ids = [ids[str(i)] for i in range(len(ids))]
                    self.assertEqual(ids, result.inserted_ids, msg)
            elif prop == 'upserted_ids':
                ids = expected_result[res]
                expected_ids = {}
                for str_index in ids:
                    expected_ids[int(str_index)] = ids[str_index]
                self.assertEqual(expected_ids, result.upserted_ids, msg)
            else:
                self.assertEqual(getattr(result, prop), expected_result[res], msg)
    else:
        self.assertEqual(result, expected_result)

def run_operation(collection, test):
    if False:
        i = 10
        return i + 15
    operation = camel_to_snake(test['operation']['name'])
    cmd = getattr(collection, operation)
    arguments = test['operation']['arguments']
    options = arguments.pop('options', {})
    for option_name in options:
        arguments[camel_to_snake(option_name)] = options[option_name]
    if operation == 'count':
        raise unittest.SkipTest('PyMongo does not support count')
    if operation == 'bulk_write':
        requests = []
        for request in arguments['requests']:
            bulk_model = camel_to_upper_camel(request['name'])
            bulk_class = getattr(operations, bulk_model)
            bulk_arguments = camel_to_snake_args(request['arguments'])
            requests.append(bulk_class(**bulk_arguments))
        arguments['requests'] = requests
    else:
        for arg_name in list(arguments):
            c2s = camel_to_snake(arg_name)
            if arg_name == 'sort':
                sort_dict = arguments[arg_name]
                arguments[arg_name] = list(sort_dict.items())
            if arg_name == 'fieldName':
                arguments['key'] = arguments.pop(arg_name)
            elif arg_name == 'batchSize' and operation == 'aggregate':
                continue
            elif arg_name == 'returnDocument':
                arguments[c2s] = arguments.pop(arg_name) == 'After'
            else:
                arguments[c2s] = arguments.pop(arg_name)
    result = cmd(**arguments)
    if isinstance(result, Cursor) or isinstance(result, CommandCursor):
        return list(result)
    return result

def create_test(scenario_def, test, name):
    if False:
        print('Hello World!')

    def run_scenario(self):
        if False:
            i = 10
            return i + 15
        drop_collections(self.db)
        data = scenario_def.get('data')
        if data:
            self.db.test.with_options(write_concern=WriteConcern(w='majority')).insert_many(scenario_def['data'])
        expected_result = test.get('outcome', {}).get('result')
        expected_error = test.get('outcome', {}).get('error')
        if expected_error is True:
            with self.assertRaises(PyMongoError):
                run_operation(self.db.test, test)
        else:
            result = run_operation(self.db.test, test)
            if expected_result is not None:
                check_result(self, expected_result, result)
        expected_c = test['outcome'].get('collection')
        if expected_c is not None:
            expected_name = expected_c.get('name')
            if expected_name is not None:
                db_coll = self.db[expected_name]
            else:
                db_coll = self.db.test
            db_coll = db_coll.with_options(read_concern=ReadConcern(level='local'))
            self.assertEqual(list(db_coll.find()), expected_c['data'])
    return run_scenario
test_creator = SpecTestCreator(create_test, TestAllScenarios, _TEST_PATH)
test_creator.create_tests()

class TestWriteOpsComparison(unittest.TestCase):

    def test_InsertOneEquals(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(InsertOne({'foo': 42}), InsertOne({'foo': 42}))

    def test_InsertOneNotEquals(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNotEqual(InsertOne({'foo': 42}), InsertOne({'foo': 23}))

    def test_DeleteOneEquals(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(DeleteOne({'foo': 42}), DeleteOne({'foo': 42}))
        self.assertEqual(DeleteOne({'foo': 42}, {'locale': 'en_US'}), DeleteOne({'foo': 42}, {'locale': 'en_US'}))
        self.assertEqual(DeleteOne({'foo': 42}, {'locale': 'en_US'}, {'hint': 1}), DeleteOne({'foo': 42}, {'locale': 'en_US'}, {'hint': 1}))

    def test_DeleteOneNotEquals(self):
        if False:
            return 10
        self.assertNotEqual(DeleteOne({'foo': 42}), DeleteOne({'foo': 23}))
        self.assertNotEqual(DeleteOne({'foo': 42}, {'locale': 'en_US'}), DeleteOne({'foo': 42}, {'locale': 'en_GB'}))
        self.assertNotEqual(DeleteOne({'foo': 42}, {'locale': 'en_US'}, {'hint': 1}), DeleteOne({'foo': 42}, {'locale': 'en_US'}, {'hint': 2}))

    def test_DeleteManyEquals(self):
        if False:
            while True:
                i = 10
        self.assertEqual(DeleteMany({'foo': 42}), DeleteMany({'foo': 42}))
        self.assertEqual(DeleteMany({'foo': 42}, {'locale': 'en_US'}), DeleteMany({'foo': 42}, {'locale': 'en_US'}))
        self.assertEqual(DeleteMany({'foo': 42}, {'locale': 'en_US'}, {'hint': 1}), DeleteMany({'foo': 42}, {'locale': 'en_US'}, {'hint': 1}))

    def test_DeleteManyNotEquals(self):
        if False:
            while True:
                i = 10
        self.assertNotEqual(DeleteMany({'foo': 42}), DeleteMany({'foo': 23}))
        self.assertNotEqual(DeleteMany({'foo': 42}, {'locale': 'en_US'}), DeleteMany({'foo': 42}, {'locale': 'en_GB'}))
        self.assertNotEqual(DeleteMany({'foo': 42}, {'locale': 'en_US'}, {'hint': 1}), DeleteMany({'foo': 42}, {'locale': 'en_US'}, {'hint': 2}))

    def test_DeleteOneNotEqualsDeleteMany(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNotEqual(DeleteOne({'foo': 42}), DeleteMany({'foo': 42}))

    def test_ReplaceOneEquals(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(ReplaceOne({'foo': 42}, {'bar': 42}, upsert=False), ReplaceOne({'foo': 42}, {'bar': 42}, upsert=False))

    def test_ReplaceOneNotEquals(self):
        if False:
            i = 10
            return i + 15
        self.assertNotEqual(ReplaceOne({'foo': 42}, {'bar': 42}, upsert=False), ReplaceOne({'foo': 42}, {'bar': 42}, upsert=True))

    def test_UpdateOneEquals(self):
        if False:
            return 10
        self.assertEqual(UpdateOne({'foo': 42}, {'$set': {'bar': 42}}), UpdateOne({'foo': 42}, {'$set': {'bar': 42}}))

    def test_UpdateOneNotEquals(self):
        if False:
            while True:
                i = 10
        self.assertNotEqual(UpdateOne({'foo': 42}, {'$set': {'bar': 42}}), UpdateOne({'foo': 42}, {'$set': {'bar': 23}}))

    def test_UpdateManyEquals(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(UpdateMany({'foo': 42}, {'$set': {'bar': 42}}), UpdateMany({'foo': 42}, {'$set': {'bar': 42}}))

    def test_UpdateManyNotEquals(self):
        if False:
            i = 10
            return i + 15
        self.assertNotEqual(UpdateMany({'foo': 42}, {'$set': {'bar': 42}}), UpdateMany({'foo': 42}, {'$set': {'bar': 23}}))

    def test_UpdateOneNotEqualsUpdateMany(self):
        if False:
            return 10
        self.assertNotEqual(UpdateOne({'foo': 42}, {'$set': {'bar': 42}}), UpdateMany({'foo': 42}, {'$set': {'bar': 42}}))
if __name__ == '__main__':
    unittest.main()