"""Test clients and replica set configuration changes, using mocks."""
from __future__ import annotations
import sys
sys.path[0:0] = ['']
from test import MockClientTest, client_context, client_knobs, unittest
from test.pymongo_mocks import MockClient
from test.utils import wait_until
from pymongo import ReadPreference
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

@client_context.require_connection
@client_context.require_no_load_balancer
def setUpModule():
    if False:
        while True:
            i = 10
    pass

class TestSecondaryBecomesStandalone(MockClientTest):

    def test_client(self):
        if False:
            i = 10
            return i + 15
        c = MockClient(standalones=[], members=['a:1', 'b:2', 'c:3'], mongoses=[], host='a:1,b:2,c:3', replicaSet='rs', serverSelectionTimeoutMS=100, connect=False)
        self.addCleanup(c.close)
        c.mock_members.remove('c:3')
        c.mock_standalones.append('c:3')
        c.kill_host('a:1')
        c.kill_host('b:2')
        with self.assertRaises(ServerSelectionTimeoutError):
            c.db.command('ping')
        self.assertEqual(c.address, None)
        c.revive_host('a:1')
        wait_until(lambda : c.address is not None, 'connect to primary')
        self.assertEqual(c.address, ('a', 1))

    def test_replica_set_client(self):
        if False:
            return 10
        c = MockClient(standalones=[], members=['a:1', 'b:2', 'c:3'], mongoses=[], host='a:1,b:2,c:3', replicaSet='rs')
        self.addCleanup(c.close)
        wait_until(lambda : ('b', 2) in c.secondaries, 'discover host "b"')
        wait_until(lambda : ('c', 3) in c.secondaries, 'discover host "c"')
        c.mock_members.remove('c:3')
        c.mock_standalones.append('c:3')
        wait_until(lambda : {('b', 2)} == c.secondaries, 'update the list of secondaries')
        self.assertEqual(('a', 1), c.primary)

class TestSecondaryRemoved(MockClientTest):

    def test_replica_set_client(self):
        if False:
            i = 10
            return i + 15
        c = MockClient(standalones=[], members=['a:1', 'b:2', 'c:3'], mongoses=[], host='a:1,b:2,c:3', replicaSet='rs')
        self.addCleanup(c.close)
        wait_until(lambda : ('b', 2) in c.secondaries, 'discover host "b"')
        wait_until(lambda : ('c', 3) in c.secondaries, 'discover host "c"')
        c.mock_hello_hosts.remove('c:3')
        wait_until(lambda : {('b', 2)} == c.secondaries, 'update list of secondaries')
        self.assertEqual(('a', 1), c.primary)

class TestSocketError(MockClientTest):

    def test_socket_error_marks_member_down(self):
        if False:
            print('Hello World!')
        with client_knobs(heartbeat_frequency=999999):
            c = MockClient(standalones=[], members=['a:1', 'b:2'], mongoses=[], host='a:1', replicaSet='rs', serverSelectionTimeoutMS=100)
            self.addCleanup(c.close)
            wait_until(lambda : len(c.nodes) == 2, 'discover both nodes')
            c.mock_down_hosts.append('b:2')
            self.assertRaises(ConnectionFailure, c.db.collection.with_options(read_preference=ReadPreference.SECONDARY).find_one)
            self.assertEqual(1, len(c.nodes))

class TestSecondaryAdded(MockClientTest):

    def test_client(self):
        if False:
            return 10
        c = MockClient(standalones=[], members=['a:1', 'b:2'], mongoses=[], host='a:1', replicaSet='rs')
        self.addCleanup(c.close)
        wait_until(lambda : len(c.nodes) == 2, 'discover both nodes')
        self.assertEqual(c.address, ('a', 1))
        self.assertEqual({('a', 1), ('b', 2)}, c.nodes)
        c.mock_members.append('c:3')
        c.mock_hello_hosts.append('c:3')
        c.db.command('ping')
        self.assertEqual(c.address, ('a', 1))
        wait_until(lambda : {('a', 1), ('b', 2), ('c', 3)} == c.nodes, 'reconnect to both secondaries')

    def test_replica_set_client(self):
        if False:
            return 10
        c = MockClient(standalones=[], members=['a:1', 'b:2'], mongoses=[], host='a:1', replicaSet='rs')
        self.addCleanup(c.close)
        wait_until(lambda : c.primary == ('a', 1), 'discover the primary')
        wait_until(lambda : {('b', 2)} == c.secondaries, 'discover the secondary')
        c.mock_members.append('c:3')
        c.mock_hello_hosts.append('c:3')
        wait_until(lambda : {('b', 2), ('c', 3)} == c.secondaries, 'discover the new secondary')
        self.assertEqual(('a', 1), c.primary)
if __name__ == '__main__':
    unittest.main()