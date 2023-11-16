"""Test PyMongo query and read preference with a sharded cluster."""
from __future__ import annotations
import unittest
from mockupdb import MockupDB, OpMsg, going
from bson import SON
from pymongo import MongoClient
from pymongo.read_preferences import Nearest, Primary, PrimaryPreferred, Secondary, SecondaryPreferred

class TestQueryAndReadModeSharded(unittest.TestCase):

    def test_query_and_read_mode_sharded_op_msg(self):
        if False:
            print('Hello World!')
        'Test OP_MSG sends non-primary $readPreference and never $query.'
        server = MockupDB()
        server.autoresponds('ismaster', ismaster=True, msg='isdbgrid', minWireVersion=2, maxWireVersion=6)
        server.run()
        self.addCleanup(server.stop)
        client = MongoClient(server.uri)
        self.addCleanup(client.close)
        read_prefs = (Primary(), SecondaryPreferred(), PrimaryPreferred(), Secondary(), Nearest(), SecondaryPreferred([{'tag': 'value'}]))
        for query in ({'a': 1}, {'$query': {'a': 1}}):
            for pref in read_prefs:
                collection = client.db.get_collection('test', read_preference=pref)
                cursor = collection.find(query.copy())
                with going(next, cursor):
                    request = server.receives()
                    expected_cmd = SON([('find', 'test'), ('filter', {'a': 1})])
                    if pref.mode:
                        expected_cmd['$readPreference'] = pref.document
                    request.assert_matches(OpMsg(expected_cmd))
                    request.replies({'cursor': {'id': 0, 'firstBatch': [{}]}})
if __name__ == '__main__':
    unittest.main()