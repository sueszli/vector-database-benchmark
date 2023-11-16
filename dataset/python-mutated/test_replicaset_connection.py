import unittest
from pymongo import MongoClient, ReadPreference
import mongoengine
from mongoengine.connection import ConnectionFailure
CONN_CLASS = MongoClient
READ_PREF = ReadPreference.SECONDARY

class ConnectionTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        mongoengine.connection._connection_settings = {}
        mongoengine.connection._connections = {}
        mongoengine.connection._dbs = {}

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        mongoengine.connection._connection_settings = {}
        mongoengine.connection._connections = {}
        mongoengine.connection._dbs = {}

    def test_replicaset_uri_passes_read_preference(self):
        if False:
            print('Hello World!')
        'Requires a replica set called "rs" on port 27017'
        try:
            conn = mongoengine.connect(db='mongoenginetest', host='mongodb://localhost/mongoenginetest?replicaSet=rs', read_preference=READ_PREF)
        except ConnectionFailure:
            return
        if not isinstance(conn, CONN_CLASS):
            return
        assert conn.read_preference == READ_PREF
if __name__ == '__main__':
    unittest.main()