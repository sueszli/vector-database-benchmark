"""Test PyMongo with a mixed-version cluster."""
from __future__ import annotations
import time
import unittest
from queue import Queue
from mockupdb import MockupDB, go
from operations import upgrades
from pymongo import MongoClient

class TestMixedVersionSharded(unittest.TestCase):

    def setup_server(self, upgrade):
        if False:
            return 10
        (self.mongos_old, self.mongos_new) = (MockupDB(), MockupDB())
        self.q: Queue = Queue()
        for server in (self.mongos_old, self.mongos_new):
            server.subscribe(self.q.put)
            server.autoresponds('getlasterror')
            server.run()
            self.addCleanup(server.stop)
        self.mongos_old.autoresponds('ismaster', ismaster=True, msg='isdbgrid', maxWireVersion=upgrade.wire_version - 1)
        self.mongos_new.autoresponds('ismaster', ismaster=True, msg='isdbgrid', maxWireVersion=upgrade.wire_version)
        self.mongoses_uri = 'mongodb://{},{}'.format(self.mongos_old.address_string, self.mongos_new.address_string)
        self.client = MongoClient(self.mongoses_uri)

    def tearDown(self):
        if False:
            while True:
                i = 10
        if hasattr(self, 'client') and self.client:
            self.client.close()

def create_mixed_version_sharded_test(upgrade):
    if False:
        return 10

    def test(self):
        if False:
            print('Hello World!')
        self.setup_server(upgrade)
        start = time.time()
        servers_used: set = set()
        while len(servers_used) < 2:
            go(upgrade.function, self.client)
            request = self.q.get(timeout=1)
            servers_used.add(request.server)
            request.assert_matches(upgrade.old if request.server is self.mongos_old else upgrade.new)
            if time.time() > start + 10:
                self.fail('never used both mongoses')
    return test

def generate_mixed_version_sharded_tests():
    if False:
        while True:
            i = 10
    for upgrade in upgrades:
        test = create_mixed_version_sharded_test(upgrade)
        test_name = 'test_%s' % upgrade.name.replace(' ', '_')
        test.__name__ = test_name
        setattr(TestMixedVersionSharded, test_name, test)
generate_mixed_version_sharded_tests()
if __name__ == '__main__':
    unittest.main()