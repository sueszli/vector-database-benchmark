"""Test list_indexes with more than one batch."""
from __future__ import annotations
import unittest
from mockupdb import MockupDB, going
from bson import SON
from pymongo import MongoClient

class TestListIndexes(unittest.TestCase):

    def test_list_indexes_command(self):
        if False:
            while True:
                i = 10
        server = MockupDB(auto_ismaster={'maxWireVersion': 6})
        server.run()
        self.addCleanup(server.stop)
        client = MongoClient(server.uri)
        self.addCleanup(client.close)
        with going(client.test.collection.list_indexes) as cursor:
            request = server.receives(listIndexes='collection', namespace='test')
            request.reply({'cursor': {'firstBatch': [{'name': 'index_0'}], 'id': 123}})
        with going(list, cursor()) as indexes:
            request = server.receives(getMore=123, namespace='test', collection='collection')
            request.reply({'cursor': {'nextBatch': [{'name': 'index_1'}], 'id': 0}})
        self.assertEqual([{'name': 'index_0'}, {'name': 'index_1'}], indexes())
        for index_info in indexes():
            self.assertIsInstance(index_info, SON)
if __name__ == '__main__':
    unittest.main()