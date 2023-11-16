from __future__ import annotations
import time
import unittest
from mockupdb import MockupDB, wait_until
from pymongo import MongoClient

class TestInitialIsMaster(unittest.TestCase):

    def test_initial_ismaster(self):
        if False:
            while True:
                i = 10
        server = MockupDB()
        server.run()
        self.addCleanup(server.stop)
        start = time.time()
        client = MongoClient(server.uri)
        self.addCleanup(client.close)
        self.assertFalse(client.nodes)
        server.receives('ismaster').ok(ismaster=True, minWireVersion=2, maxWireVersion=6)
        wait_until(lambda : client.nodes, 'update nodes', timeout=1)
        server.receives('ismaster').ok(ismaster=True, minWireVersion=2, maxWireVersion=6)
        self.assertGreaterEqual(time.time() - start, 10)
if __name__ == '__main__':
    unittest.main()