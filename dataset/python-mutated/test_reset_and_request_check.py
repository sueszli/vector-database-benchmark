from __future__ import annotations
import itertools
import time
import unittest
from mockupdb import MockupDB, going, wait_until
from operations import operations
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pymongo.server_type import SERVER_TYPE

class TestResetAndRequestCheck(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.ismaster_time = 0.0
        self.client = None
        self.server = None

    def setup_server(self):
        if False:
            while True:
                i = 10
        self.server = MockupDB()

        def responder(request):
            if False:
                while True:
                    i = 10
            self.ismaster_time = time.time()
            return request.ok(ismaster=True, minWireVersion=2, maxWireVersion=6)
        self.server.autoresponds('ismaster', responder)
        self.server.run()
        self.addCleanup(self.server.stop)
        kwargs = {'socketTimeoutMS': 100}
        kwargs['retryReads'] = False
        self.client = MongoClient(self.server.uri, **kwargs)
        wait_until(lambda : self.client.nodes, 'connect to standalone')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'client') and self.client:
            self.client.close()

    def _test_disconnect(self, operation):
        if False:
            while True:
                i = 10
        self.setup_server()
        assert self.server is not None
        assert self.client is not None
        with self.assertRaises(ConnectionFailure):
            with going(operation.function, self.client):
                self.server.receives().hangup()
        topology = self.client._topology
        with self.assertRaises(ConnectionFailure):
            topology.select_server_by_address(self.server.address, 0)
        time.sleep(0.5)
        after = time.time()
        with going(self.client.db.command, 'buildinfo'):
            self.server.receives('buildinfo').ok()
        last = self.ismaster_time
        self.assertGreaterEqual(last, after, 'called ismaster before needed')

    def _test_timeout(self, operation):
        if False:
            return 10
        self.setup_server()
        assert self.server is not None
        assert self.client is not None
        with self.assertRaises(ConnectionFailure):
            with going(operation.function, self.client):
                self.server.receives()
                before = self.ismaster_time
                time.sleep(0.5)
        topology = self.client._topology
        server = topology.select_server_by_address(self.server.address, 0)
        assert server is not None
        self.assertEqual(SERVER_TYPE.Standalone, server.description.server_type)
        after = self.ismaster_time
        self.assertEqual(after, before, 'unneeded ismaster call')

    def _test_not_master(self, operation):
        if False:
            print('Hello World!')
        self.setup_server()
        assert self.server is not None
        assert self.client is not None
        with self.assertRaises(ConnectionFailure):
            with going(operation.function, self.client):
                request = self.server.receives()
                before = self.ismaster_time
                request.replies(operation.not_master)
                time.sleep(1)
        topology = self.client._topology
        server = topology.select_server_by_address(self.server.address, 0)
        assert server is not None
        self.assertEqual(SERVER_TYPE.Standalone, server.description.server_type)
        after = self.ismaster_time
        self.assertGreater(after, before, 'ismaster not called')

def create_reset_test(operation, test_method):
    if False:
        print('Hello World!')

    def test(self):
        if False:
            i = 10
            return i + 15
        test_method(self, operation)
    return test

def generate_reset_tests():
    if False:
        for i in range(10):
            print('nop')
    test_methods = [(TestResetAndRequestCheck._test_disconnect, 'test_disconnect'), (TestResetAndRequestCheck._test_timeout, 'test_timeout'), (TestResetAndRequestCheck._test_not_master, 'test_not_master')]
    matrix = itertools.product(operations, test_methods)
    for entry in matrix:
        (operation, (test_method, name)) = entry
        test = create_reset_test(operation, test_method)
        test_name = '{}_{}'.format(name, operation.name.replace(' ', '_'))
        test.__name__ = test_name
        setattr(TestResetAndRequestCheck, test_name, test)
generate_reset_tests()
if __name__ == '__main__':
    unittest.main()