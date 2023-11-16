from __future__ import annotations
import unittest
from mockupdb import MockupDB
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

class TestAuthRecoveringMember(unittest.TestCase):

    def test_auth_recovering_member(self):
        if False:
            i = 10
            return i + 15
        server = MockupDB()
        server.autoresponds('ismaster', {'minWireVersion': 2, 'maxWireVersion': 6, 'ismaster': False, 'secondary': False, 'setName': 'rs'})
        server.run()
        self.addCleanup(server.stop)
        client = MongoClient(server.uri, replicaSet='rs', serverSelectionTimeoutMS=100, socketTimeoutMS=100)
        self.addCleanup(client.close)
        with self.assertRaises(ServerSelectionTimeoutError):
            client.db.command('ping')
if __name__ == '__main__':
    unittest.main()