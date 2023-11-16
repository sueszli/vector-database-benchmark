"""Test PyMongo's SlaveOkay with:

- A direct connection to a standalone.
- A direct connection to a slave.
- A direct connection to a mongos.
"""
from __future__ import annotations
import itertools
import unittest
from queue import Queue
from mockupdb import MockupDB, going
from operations import operations
from pymongo import MongoClient
from pymongo.read_preferences import make_read_preference, read_pref_mode_from_name

class TestSlaveOkaySharded(unittest.TestCase):

    def setup_server(self):
        if False:
            print('Hello World!')
        (self.mongos1, self.mongos2) = (MockupDB(), MockupDB())
        self.q: Queue = Queue()
        for server in (self.mongos1, self.mongos2):
            server.subscribe(self.q.put)
            server.run()
            self.addCleanup(server.stop)
            server.autoresponds('ismaster', minWireVersion=2, maxWireVersion=6, ismaster=True, msg='isdbgrid')
        self.mongoses_uri = f'mongodb://{self.mongos1.address_string},{self.mongos2.address_string}'

def create_slave_ok_sharded_test(mode, operation):
    if False:
        for i in range(10):
            print('nop')

    def test(self):
        if False:
            i = 10
            return i + 15
        self.setup_server()
        if operation.op_type == 'always-use-secondary':
            slave_ok = True
        elif operation.op_type == 'may-use-secondary':
            slave_ok = mode != 'primary'
        elif operation.op_type == 'must-use-primary':
            slave_ok = False
        else:
            raise AssertionError('unrecognized op_type %r' % operation.op_type)
        pref = make_read_preference(read_pref_mode_from_name(mode), tag_sets=None)
        client = MongoClient(self.mongoses_uri, read_preference=pref)
        self.addCleanup(client.close)
        with going(operation.function, client):
            request = self.q.get(timeout=1)
            request.reply(operation.reply)
        if slave_ok:
            self.assertTrue(request.slave_ok, 'SlaveOkay not set')
        else:
            self.assertFalse(request.slave_ok, 'SlaveOkay set')
    return test

def generate_slave_ok_sharded_tests():
    if False:
        while True:
            i = 10
    modes = ('primary', 'secondary', 'nearest')
    matrix = itertools.product(modes, operations)
    for entry in matrix:
        (mode, operation) = entry
        test = create_slave_ok_sharded_test(mode, operation)
        test_name = 'test_{}_with_mode_{}'.format(operation.name.replace(' ', '_'), mode)
        test.__name__ = test_name
        setattr(TestSlaveOkaySharded, test_name, test)
generate_slave_ok_sharded_tests()
if __name__ == '__main__':
    unittest.main()