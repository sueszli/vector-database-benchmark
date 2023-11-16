"""Test PyMongo's SlaveOkay with a replica set connection.

Just make sure SlaveOkay is *not* set on primary reads.
"""
from __future__ import annotations
import unittest
from mockupdb import MockupDB, going
from operations import operations
from pymongo import MongoClient

class TestSlaveOkayRS(unittest.TestCase):

    def setup_server(self):
        if False:
            while True:
                i = 10
        (self.primary, self.secondary) = (MockupDB(), MockupDB())
        for server in (self.primary, self.secondary):
            server.run()
            self.addCleanup(server.stop)
        hosts = [server.address_string for server in (self.primary, self.secondary)]
        self.primary.autoresponds('ismaster', ismaster=True, setName='rs', hosts=hosts, minWireVersion=2, maxWireVersion=6)
        self.secondary.autoresponds('ismaster', ismaster=False, secondary=True, setName='rs', hosts=hosts, minWireVersion=2, maxWireVersion=6)

def create_slave_ok_rs_test(operation):
    if False:
        for i in range(10):
            print('nop')

    def test(self):
        if False:
            return 10
        self.setup_server()
        assert operation.op_type != 'always-use-secondary'
        client = MongoClient(self.primary.uri, replicaSet='rs')
        self.addCleanup(client.close)
        with going(operation.function, client):
            request = self.primary.receive()
            request.reply(operation.reply)
        self.assertFalse(request.slave_ok, 'SlaveOkay set read mode "primary"')
    return test

def generate_slave_ok_rs_tests():
    if False:
        print('Hello World!')
    for operation in operations:
        if operation.op_type == 'always-use-secondary':
            continue
        test = create_slave_ok_rs_test(operation)
        test_name = 'test_%s' % operation.name.replace(' ', '_')
        test.__name__ = test_name
        setattr(TestSlaveOkayRS, test_name, test)
generate_slave_ok_rs_tests()
if __name__ == '__main__':
    unittest.main()