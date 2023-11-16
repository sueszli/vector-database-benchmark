"""Test MongoClient's mongos load balancing using a mock."""
from __future__ import annotations
import sys
import threading
sys.path[0:0] = ['']
from test import MockClientTest, client_context, unittest
from test.pymongo_mocks import MockClient
from test.utils import connected, wait_until
from pymongo.errors import AutoReconnect, InvalidOperation
from pymongo.server_selectors import writable_server_selector
from pymongo.topology_description import TOPOLOGY_TYPE

@client_context.require_connection
@client_context.require_no_load_balancer
def setUpModule():
    if False:
        for i in range(10):
            print('nop')
    pass

class SimpleOp(threading.Thread):

    def __init__(self, client):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.client = client
        self.passed = False

    def run(self):
        if False:
            i = 10
            return i + 15
        self.client.db.command('ping')
        self.passed = True

def do_simple_op(client, nthreads):
    if False:
        while True:
            i = 10
    threads = [SimpleOp(client) for _ in range(nthreads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for t in threads:
        assert t.passed

def writable_addresses(topology):
    if False:
        i = 10
        return i + 15
    return {server.description.address for server in topology.select_servers(writable_server_selector)}

class TestMongosLoadBalancing(MockClientTest):

    def mock_client(self, **kwargs):
        if False:
            i = 10
            return i + 15
        mock_client = MockClient(standalones=[], members=[], mongoses=['a:1', 'b:2', 'c:3'], host='a:1,b:2,c:3', connect=False, **kwargs)
        self.addCleanup(mock_client.close)
        mock_client.mock_rtts['a:1'] = 0.02
        mock_client.mock_rtts['b:2'] = 0.025
        mock_client.mock_rtts['c:3'] = 0.045
        return mock_client

    def test_lazy_connect(self):
        if False:
            return 10
        nthreads = 10
        client = self.mock_client()
        self.assertEqual(0, len(client.nodes))
        do_simple_op(client, nthreads)
        wait_until(lambda : len(client.nodes) == 3, 'connect to all mongoses')

    def test_failover(self):
        if False:
            return 10
        nthreads = 10
        client = connected(self.mock_client(localThresholdMS=0.001))
        wait_until(lambda : len(client.nodes) == 3, 'connect to all mongoses')
        client.kill_host('a:1')
        passed = []

        def f():
            if False:
                for i in range(10):
                    print('nop')
            try:
                client.db.command('ping')
            except AutoReconnect:
                client.db.command('ping')
            passed.append(True)
        threads = [threading.Thread(target=f) for _ in range(nthreads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(nthreads, len(passed))
        self.assertEqual(2, len(client.nodes))

    def test_local_threshold(self):
        if False:
            for i in range(10):
                print('nop')
        client = connected(self.mock_client(localThresholdMS=30))
        self.assertEqual(30, client.options.local_threshold_ms)
        wait_until(lambda : len(client.nodes) == 3, 'connect to all mongoses')
        topology = client._topology
        self.assertEqual({('a', 1), ('b', 2), ('c', 3)}, writable_addresses(topology))
        client.admin.command('ping')
        client = connected(self.mock_client(localThresholdMS=0))
        self.assertEqual(0, client.options.local_threshold_ms)
        client.db.command('ping')
        client.kill_host('{}:{}'.format(*next(iter(client.nodes))))
        try:
            client.db.command('ping')
        except:
            pass

        def connect_to_new_mongos():
            if False:
                i = 10
                return i + 15
            try:
                return client.db.command('ping')
            except AutoReconnect:
                pass
        wait_until(connect_to_new_mongos, 'connect to a new mongos')

    def test_load_balancing(self):
        if False:
            print('Hello World!')
        client = connected(self.mock_client())
        wait_until(lambda : len(client.nodes) == 3, 'connect to all mongoses')
        with self.assertRaises(InvalidOperation):
            client.address
        topology = client._topology
        self.assertEqual(TOPOLOGY_TYPE.Sharded, topology.description.topology_type)
        self.assertEqual({('a', 1), ('b', 2)}, writable_addresses(topology))
        client.mock_rtts['a:1'] = 0.045
        wait_until(lambda : {('b', 2)} == writable_addresses(topology), 'discover server "a" is too far')
if __name__ == '__main__':
    unittest.main()