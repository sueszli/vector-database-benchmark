"""Test PyMongo cursor with a sharded cluster."""
from __future__ import annotations
import unittest
from queue import Queue
from mockupdb import MockupDB, going
from pymongo import MongoClient

class TestGetmoreSharded(unittest.TestCase):

    def test_getmore_sharded(self):
        if False:
            return 10
        servers = [MockupDB(), MockupDB()]
        q: Queue = Queue()
        for server in servers:
            server.subscribe(q.put)
            server.autoresponds('ismaster', ismaster=True, msg='isdbgrid', minWireVersion=2, maxWireVersion=6)
            server.run()
            self.addCleanup(server.stop)
        client = MongoClient('mongodb://%s:%d,%s:%d' % (servers[0].host, servers[0].port, servers[1].host, servers[1].port))
        self.addCleanup(client.close)
        collection = client.db.collection
        cursor = collection.find()
        with going(next, cursor):
            query = q.get(timeout=1)
            query.replies({'cursor': {'id': 123, 'firstBatch': [{}]}})
        for i in range(1, 10):
            with going(next, cursor):
                getmore = q.get(timeout=1)
                self.assertEqual(query.server, getmore.server)
                cursor_id = 123 if i < 9 else 0
                getmore.replies({'cursor': {'id': cursor_id, 'nextBatch': [{}]}})
if __name__ == '__main__':
    unittest.main()