"""Test connections to RSGhost nodes."""
from __future__ import annotations
import datetime
import unittest
from mockupdb import MockupDB, going
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

class TestRSGhost(unittest.TestCase):

    def test_rsghost(self):
        if False:
            while True:
                i = 10
        rsother_response = {'ok': 1.0, 'ismaster': False, 'secondary': False, 'info': 'Does not have a valid replica set config', 'isreplicaset': True, 'maxBsonObjectSize': 16777216, 'maxMessageSizeBytes': 48000000, 'maxWriteBatchSize': 100000, 'localTime': datetime.datetime(2021, 11, 30, 0, 53, 4, 99000), 'logicalSessionTimeoutMinutes': 30, 'connectionId': 3, 'minWireVersion': 0, 'maxWireVersion': 15, 'readOnly': False}
        server = MockupDB(auto_ismaster=rsother_response)
        server.run()
        self.addCleanup(server.stop)
        with MongoClient(server.uri, serverSelectionTimeoutMS=250) as client:
            with self.assertRaises(ServerSelectionTimeoutError):
                client.test.command('ping')
        with MongoClient(server.uri, directConnection=True) as client:
            with going(client.test.command, 'ping'):
                request = server.receives(ping=1)
                request.reply()
if __name__ == '__main__':
    unittest.main()