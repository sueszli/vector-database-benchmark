"""Test the read_concern module."""
from __future__ import annotations
import sys
import unittest
sys.path[0:0] = ['']
from test import IntegrationTest, client_context
from test.utils import OvertCommandListener, rs_or_single_client
from bson.son import SON
from pymongo.errors import OperationFailure
from pymongo.read_concern import ReadConcern

class TestReadConcern(IntegrationTest):
    listener: OvertCommandListener

    @classmethod
    @client_context.require_connection
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        cls.listener = OvertCommandListener()
        cls.client = rs_or_single_client(event_listeners=[cls.listener])
        cls.db = cls.client.pymongo_test
        client_context.client.pymongo_test.create_collection('coll')

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cls.client.close()
        client_context.client.pymongo_test.drop_collection('coll')
        super().tearDownClass()

    def tearDown(self):
        if False:
            return 10
        self.listener.reset()
        super().tearDown()

    def test_read_concern(self):
        if False:
            print('Hello World!')
        rc = ReadConcern()
        self.assertIsNone(rc.level)
        self.assertTrue(rc.ok_for_legacy)
        rc = ReadConcern('majority')
        self.assertEqual('majority', rc.level)
        self.assertFalse(rc.ok_for_legacy)
        rc = ReadConcern('local')
        self.assertEqual('local', rc.level)
        self.assertTrue(rc.ok_for_legacy)
        self.assertRaises(TypeError, ReadConcern, 42)

    def test_read_concern_uri(self):
        if False:
            print('Hello World!')
        uri = f'mongodb://{client_context.pair}/?readConcernLevel=majority'
        client = rs_or_single_client(uri, connect=False)
        self.assertEqual(ReadConcern('majority'), client.read_concern)

    def test_invalid_read_concern(self):
        if False:
            return 10
        coll = self.db.get_collection('coll', read_concern=ReadConcern('unknown'))
        with self.assertRaises(OperationFailure):
            coll.find_one()

    def test_find_command(self):
        if False:
            return 10
        coll = self.db.coll
        tuple(coll.find({'field': 'value'}))
        self.assertNotIn('readConcern', self.listener.started_events[0].command)
        self.listener.reset()
        coll = self.db.get_collection('coll', read_concern=ReadConcern('local'))
        tuple(coll.find({'field': 'value'}))
        self.assertEqualCommand(SON([('find', 'coll'), ('filter', {'field': 'value'}), ('readConcern', {'level': 'local'})]), self.listener.started_events[0].command)

    def test_command_cursor(self):
        if False:
            while True:
                i = 10
        coll = self.db.coll
        tuple(coll.aggregate([{'$match': {'field': 'value'}}]))
        self.assertNotIn('readConcern', self.listener.started_events[0].command)
        self.listener.reset()
        coll = self.db.get_collection('coll', read_concern=ReadConcern('local'))
        tuple(coll.aggregate([{'$match': {'field': 'value'}}]))
        self.assertEqual({'level': 'local'}, self.listener.started_events[0].command['readConcern'])

    def test_aggregate_out(self):
        if False:
            for i in range(10):
                print('nop')
        coll = self.db.get_collection('coll', read_concern=ReadConcern('local'))
        tuple(coll.aggregate([{'$match': {'field': 'value'}}, {'$out': 'output_collection'}]))
        if client_context.version >= (4, 1):
            self.assertIn('readConcern', self.listener.started_events[0].command)
        else:
            self.assertNotIn('readConcern', self.listener.started_events[0].command)
if __name__ == '__main__':
    unittest.main()