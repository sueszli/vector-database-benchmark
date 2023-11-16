"""Tests for the dbref module."""
from __future__ import annotations
import pickle
import sys
from typing import Any
sys.path[0:0] = ['']
from copy import deepcopy
from test import unittest
from bson import decode, encode
from bson.dbref import DBRef
from bson.objectid import ObjectId

class TestDBRef(unittest.TestCase):

    def test_creation(self):
        if False:
            for i in range(10):
                print('nop')
        a = ObjectId()
        self.assertRaises(TypeError, DBRef)
        self.assertRaises(TypeError, DBRef, 'coll')
        self.assertRaises(TypeError, DBRef, 4, a)
        self.assertRaises(TypeError, DBRef, 1.5, a)
        self.assertRaises(TypeError, DBRef, a, a)
        self.assertRaises(TypeError, DBRef, None, a)
        self.assertRaises(TypeError, DBRef, 'coll', a, 5)
        self.assertTrue(DBRef('coll', a))
        self.assertTrue(DBRef('coll', 5))
        self.assertTrue(DBRef('coll', 5, 'database'))

    def test_read_only(self):
        if False:
            for i in range(10):
                print('nop')
        a = DBRef('coll', ObjectId())

        def foo():
            if False:
                print('Hello World!')
            a.collection = 'blah'

        def bar():
            if False:
                while True:
                    i = 10
            a.id = 'aoeu'
        self.assertEqual('coll', a.collection)
        a.id
        self.assertEqual(None, a.database)
        self.assertRaises(AttributeError, foo)
        self.assertRaises(AttributeError, bar)

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(repr(DBRef('coll', ObjectId('1234567890abcdef12345678'))), "DBRef('coll', ObjectId('1234567890abcdef12345678'))")
        self.assertEqual(repr(DBRef('coll', ObjectId('1234567890abcdef12345678'))), "DBRef({}, ObjectId('1234567890abcdef12345678'))".format(repr('coll')))
        self.assertEqual(repr(DBRef('coll', 5, foo='bar')), "DBRef('coll', 5, foo='bar')")
        self.assertEqual(repr(DBRef('coll', ObjectId('1234567890abcdef12345678'), 'foo')), "DBRef('coll', ObjectId('1234567890abcdef12345678'), 'foo')")

    def test_equality(self):
        if False:
            print('Hello World!')
        obj_id = ObjectId('1234567890abcdef12345678')
        self.assertEqual(DBRef('foo', 5), DBRef('foo', 5))
        self.assertEqual(DBRef('coll', obj_id), DBRef('coll', obj_id))
        self.assertNotEqual(DBRef('coll', obj_id), DBRef('coll', obj_id, 'foo'))
        self.assertNotEqual(DBRef('coll', obj_id), DBRef('col', obj_id))
        self.assertNotEqual(DBRef('coll', obj_id), DBRef('coll', ObjectId(b'123456789011')))
        self.assertNotEqual(DBRef('coll', obj_id), 4)
        self.assertNotEqual(DBRef('coll', obj_id, 'foo'), DBRef('coll', obj_id, 'bar'))
        self.assertFalse(DBRef('foo', 5) != DBRef('foo', 5))
        self.assertFalse(DBRef('coll', obj_id) != DBRef('coll', obj_id))
        self.assertFalse(DBRef('coll', obj_id, 'foo') != DBRef('coll', obj_id, 'foo'))

    def test_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(DBRef('coll', 5, foo='bar'), DBRef('coll', 5, foo='bar'))
        self.assertNotEqual(DBRef('coll', 5, foo='bar'), DBRef('coll', 5))
        self.assertNotEqual(DBRef('coll', 5, foo='bar'), DBRef('coll', 5, foo='baz'))
        self.assertEqual('bar', DBRef('coll', 5, foo='bar').foo)
        self.assertRaises(AttributeError, getattr, DBRef('coll', 5, foo='bar'), 'bar')

    def test_deepcopy(self):
        if False:
            print('Hello World!')
        a = DBRef('coll', 'asdf', 'db', x=[1])
        b = deepcopy(a)
        self.assertEqual(a, b)
        self.assertNotEqual(id(a), id(b.x))
        self.assertEqual(a.x, b.x)
        self.assertNotEqual(id(a.x), id(b.x))
        b.x[0] = 2
        self.assertEqual(a.x, [1])
        self.assertEqual(b.x, [2])

    def test_pickling(self):
        if False:
            while True:
                i = 10
        dbr = DBRef('coll', 5, foo='bar')
        for protocol in [0, 1, 2, -1]:
            pkl = pickle.dumps(dbr, protocol=protocol)
            dbr2 = pickle.loads(pkl)
            self.assertEqual(dbr, dbr2)

    def test_dbref_hash(self):
        if False:
            i = 10
            return i + 15
        dbref_1a = DBRef('collection', 'id', 'database')
        dbref_1b = DBRef('collection', 'id', 'database')
        self.assertEqual(hash(dbref_1a), hash(dbref_1b))
        dbref_2a = DBRef('collection', 'id', 'database', custom='custom')
        dbref_2b = DBRef('collection', 'id', 'database', custom='custom')
        self.assertEqual(hash(dbref_2a), hash(dbref_2b))
        self.assertNotEqual(hash(dbref_1a), hash(dbref_2a))

class TestDBRefSpec(unittest.TestCase):

    def test_decoding_1_2_3(self):
        if False:
            while True:
                i = 10
        doc: Any
        for doc in [{'$ref': 'coll0', '$id': ObjectId('60a6fe9a54f4180c86309efa')}, {'$ref': 'coll0', '$id': 1}, {'$ref': 'coll0', '$id': None}, {'$ref': 'coll0', '$id': 1, '$db': 'db0'}, {'$ref': 'coll0', '$id': 1, '$db': 'db0', 'foo': 'bar'}, {'$ref': 'coll0', '$id': 1, 'foo': True, 'bar': False}, {'$ref': 'coll0', '$id': 1, 'meta': {'foo': 1, 'bar': 2}}, {'$ref': 'coll0', '$id': 1, '$foo': 'bar'}, {'$ref': 'coll0', '$id': 1, 'foo.bar': 0}, {'$id': 1, '$ref': 'coll0'}, {'$db': 'db0', '$ref': 'coll0', '$id': 1}, {'foo': 1, '$id': 1, '$ref': 'coll0'}, {'foo': 1, '$ref': 'coll0', '$id': 1, '$db': 'db0'}, {'foo': 1, '$ref': 'coll0', '$id': 1, '$db': 'db0', 'bar': 1}]:
            with self.subTest(doc=doc):
                decoded = decode(encode({'dbref': doc}))
                dbref = decoded['dbref']
                self.assertIsInstance(dbref, DBRef)
                self.assertEqual(dbref.collection, doc['$ref'])
                self.assertEqual(dbref.id, doc['$id'])
                self.assertEqual(dbref.database, doc.get('$db'))
                for extra in set(doc.keys()) - {'$ref', '$id', '$db'}:
                    self.assertEqual(getattr(dbref, extra), doc[extra])

    def test_decoding_4_5(self):
        if False:
            i = 10
            return i + 15
        for doc in [{'$ref': 'coll0'}, {'$id': ObjectId('60a6fe9a54f4180c86309efa')}, {'$db': 'db0'}, {'$ref': True, '$id': 1}, {'$ref': 'coll0', '$id': 1, '$db': 1}]:
            with self.subTest(doc=doc):
                decoded = decode(encode({'dbref': doc}))
                dbref = decoded['dbref']
                self.assertIsInstance(dbref, dict)

    def test_encoding_1_2(self):
        if False:
            for i in range(10):
                print('nop')
        doc: Any
        for doc in [{'$ref': 'coll0', '$id': ObjectId('60a6fe9a54f4180c86309efa')}, {'$ref': 'coll0', '$id': 1}, {'$ref': 'coll0', '$id': None}, {'$ref': 'coll0', '$id': 1, '$db': 'db0'}, {'$ref': 'coll0', '$id': 1, '$db': 'db0', 'foo': 'bar'}, {'$ref': 'coll0', '$id': 1, 'foo': True, 'bar': False}, {'$ref': 'coll0', '$id': 1, 'meta': {'foo': 1, 'bar': 2}}, {'$ref': 'coll0', '$id': 1, '$foo': 'bar'}, {'$ref': 'coll0', '$id': 1, 'foo.bar': 0}]:
            with self.subTest(doc=doc):
                encoded_doc = encode({'dbref': doc})
                decoded = decode(encoded_doc)
                dbref = decoded['dbref']
                self.assertIsInstance(dbref, DBRef)
                encoded_dbref = encode(decoded)
                self.assertEqual(encoded_dbref, encoded_doc)
                for extra in set(doc.keys()) - {'$ref', '$id', '$db'}:
                    self.assertEqual(getattr(dbref, extra), doc[extra])

    def test_encoding_3(self):
        if False:
            i = 10
            return i + 15
        for doc in [{'$id': 1, '$ref': 'coll0'}, {'$db': 'db0', '$ref': 'coll0', '$id': 1}, {'foo': 1, '$id': 1, '$ref': 'coll0'}, {'foo': 1, '$ref': 'coll0', '$id': 1, '$db': 'db0'}, {'foo': 1, '$ref': 'coll0', '$id': 1, '$db': 'db0', 'bar': 1}]:
            with self.subTest(doc=doc):
                encoded_doc = encode({'dbref': doc})
                decoded = decode(encoded_doc)
                dbref = decoded['dbref']
                self.assertIsInstance(dbref, DBRef)
                encoded_dbref = encode(decoded)
                self.assertNotEqual(encoded_dbref, encoded_doc)
                self.assertEqual(decode(encoded_dbref), decode(encoded_doc))
                for extra in set(doc.keys()) - {'$ref', '$id', '$db'}:
                    self.assertEqual(getattr(dbref, extra), doc[extra])
if __name__ == '__main__':
    unittest.main()