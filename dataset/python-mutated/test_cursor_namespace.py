"""Test list_indexes with more than one batch."""
from __future__ import annotations
import unittest
from mockupdb import MockupDB, going
from pymongo import MongoClient

class TestCursorNamespace(unittest.TestCase):
    server: MockupDB
    client: MongoClient

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.server = MockupDB(auto_ismaster={'maxWireVersion': 6})
        cls.server.run()
        cls.client = MongoClient(cls.server.uri)

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.client.close()
        cls.server.stop()

    def _test_cursor_namespace(self, cursor_op, command):
        if False:
            return 10
        with going(cursor_op) as docs:
            request = self.server.receives(**{command: 'collection', 'namespace': 'test'})
            request.reply({'cursor': {'firstBatch': [{'doc': 1}], 'id': 123, 'ns': 'different_db.different.coll'}})
            request = self.server.receives(getMore=123, namespace='different_db', collection='different.coll')
            request.reply({'cursor': {'nextBatch': [{'doc': 2}], 'id': 0}})
        self.assertEqual([{'doc': 1}, {'doc': 2}], docs())

    def test_aggregate_cursor(self):
        if False:
            for i in range(10):
                print('nop')

        def op():
            if False:
                print('Hello World!')
            return list(self.client.test.collection.aggregate([]))
        self._test_cursor_namespace(op, 'aggregate')

    def test_find_cursor(self):
        if False:
            i = 10
            return i + 15

        def op():
            if False:
                return 10
            return list(self.client.test.collection.find())
        self._test_cursor_namespace(op, 'find')

    def test_list_indexes(self):
        if False:
            for i in range(10):
                print('nop')

        def op():
            if False:
                i = 10
                return i + 15
            return list(self.client.test.collection.list_indexes())
        self._test_cursor_namespace(op, 'listIndexes')

class TestKillCursorsNamespace(unittest.TestCase):
    server: MockupDB
    client: MongoClient

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.server = MockupDB(auto_ismaster={'maxWireVersion': 6})
        cls.server.run()
        cls.client = MongoClient(cls.server.uri)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        cls.client.close()
        cls.server.stop()

    def _test_killCursors_namespace(self, cursor_op, command):
        if False:
            for i in range(10):
                print('nop')
        with going(cursor_op):
            request = self.server.receives(**{command: 'collection', 'namespace': 'test'})
            request.reply({'cursor': {'firstBatch': [{'doc': 1}], 'id': 123, 'ns': 'different_db.different.coll'}})
            request = self.server.receives(**{'killCursors': 'different.coll', 'cursors': [123], '$db': 'different_db'})
            request.reply({'ok': 1, 'cursorsKilled': [123], 'cursorsNotFound': [], 'cursorsAlive': [], 'cursorsUnknown': []})

    def test_aggregate_killCursor(self):
        if False:
            for i in range(10):
                print('nop')

        def op():
            if False:
                return 10
            cursor = self.client.test.collection.aggregate([], batchSize=1)
            next(cursor)
            cursor.close()
        self._test_killCursors_namespace(op, 'aggregate')

    def test_find_killCursor(self):
        if False:
            print('Hello World!')

        def op():
            if False:
                return 10
            cursor = self.client.test.collection.find(batch_size=1)
            next(cursor)
            cursor.close()
        self._test_killCursors_namespace(op, 'find')
if __name__ == '__main__':
    unittest.main()