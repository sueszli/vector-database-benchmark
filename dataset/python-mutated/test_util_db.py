from __future__ import absolute_import
import mongoengine
import unittest2
from st2common.util import db as db_util

class DatabaseUtilTestCase(unittest2.TestCase):

    def test_noop_mongodb_to_python_types(self):
        if False:
            for i in range(10):
                print('nop')
        data = [123, 999.99, True, [10, 20, 30], {'a': 1, 'b': 2}, None]
        for item in data:
            self.assertEqual(db_util.mongodb_to_python_types(item), item)

    def test_mongodb_basedict_to_dict(self):
        if False:
            print('Hello World!')
        data = {'a': 1, 'b': 2}
        obj = mongoengine.base.datastructures.BaseDict(data, None, 'foobar')
        self.assertDictEqual(db_util.mongodb_to_python_types(obj), data)

    def test_mongodb_baselist_to_list(self):
        if False:
            print('Hello World!')
        data = [2, 4, 6]
        obj = mongoengine.base.datastructures.BaseList(data, None, 'foobar')
        self.assertListEqual(db_util.mongodb_to_python_types(obj), data)

    def test_nested_mongdb_to_python_types(self):
        if False:
            while True:
                i = 10
        data = {'a': mongoengine.base.datastructures.BaseList([1, 2, 3], None, 'a'), 'b': mongoengine.base.datastructures.BaseDict({'a': 1, 'b': 2}, None, 'b'), 'c': {'d': mongoengine.base.datastructures.BaseList([4, 5, 6], None, 'd'), 'e': mongoengine.base.datastructures.BaseDict({'c': 3, 'd': 4}, None, 'e')}, 'f': mongoengine.base.datastructures.BaseList([mongoengine.base.datastructures.BaseDict({'e': 5}, None, 'f1'), mongoengine.base.datastructures.BaseDict({'f': 6}, None, 'f2')], None, 'f'), 'g': mongoengine.base.datastructures.BaseDict({'h': mongoengine.base.datastructures.BaseList([mongoengine.base.datastructures.BaseDict({'g': 7}, None, 'h1'), mongoengine.base.datastructures.BaseDict({'h': 8}, None, 'h2')], None, 'h'), 'i': mongoengine.base.datastructures.BaseDict({'j': 9, 'k': 10}, None, 'i')}, None, 'g')}
        expected = {'a': [1, 2, 3], 'b': {'a': 1, 'b': 2}, 'c': {'d': [4, 5, 6], 'e': {'c': 3, 'd': 4}}, 'f': [{'e': 5}, {'f': 6}], 'g': {'h': [{'g': 7}, {'h': 8}], 'i': {'j': 9, 'k': 10}}}
        self.assertDictEqual(db_util.mongodb_to_python_types(data), expected)