"""Test the topology module's Server Selection Spec implementation."""
from __future__ import annotations
import os
import sys
from pymongo import MongoClient, ReadPreference
from pymongo.errors import ServerSelectionTimeoutError
from pymongo.hello import HelloCompat
from pymongo.server_selectors import writable_server_selector
from pymongo.settings import TopologySettings
from pymongo.topology import Topology
from pymongo.typings import strip_optional
sys.path[0:0] = ['']
from test import IntegrationTest, client_context, unittest
from test.utils import EventListener, FunctionCallRecorder, rs_or_single_client, wait_until
from test.utils_selection_tests import create_selection_tests, get_addresses, get_topology_settings_dict, make_server_description
_TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join('server_selection', 'server_selection'))

class SelectionStoreSelector:
    """No-op selector that keeps track of what was passed to it."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.selection = None

    def __call__(self, selection):
        if False:
            for i in range(10):
                print('nop')
        self.selection = selection
        return selection

class TestAllScenarios(create_selection_tests(_TEST_PATH)):
    pass

class TestCustomServerSelectorFunction(IntegrationTest):

    @client_context.require_replica_set
    def test_functional_select_max_port_number_host(self):
        if False:
            i = 10
            return i + 15

        def custom_selector(servers):
            if False:
                for i in range(10):
                    print('nop')
            ports = [s.address[1] for s in servers]
            idx = ports.index(max(ports))
            return [servers[idx]]
        listener = EventListener()
        client = rs_or_single_client(server_selector=custom_selector, event_listeners=[listener])
        self.addCleanup(client.close)
        coll = client.get_database('testdb', read_preference=ReadPreference.NEAREST).coll
        self.addCleanup(client.drop_database, 'testdb')

        def all_hosts_started():
            if False:
                while True:
                    i = 10
            return len(client.admin.command(HelloCompat.LEGACY_CMD)['hosts']) == len(client._topology._description.readable_servers)
        wait_until(all_hosts_started, 'receive heartbeat from all hosts')
        expected_port = max([strip_optional(n.address[1]) for n in client._topology._description.readable_servers])
        coll.insert_one({'name': 'John Doe'})
        for _ in range(10):
            coll.find_one({'name': 'John Doe'})
        for command in listener.started_events:
            if command.command_name == 'find':
                self.assertEqual(command.connection_id[1], expected_port)

    def test_invalid_server_selector(self):
        if False:
            while True:
                i = 10
        for selector_candidate in [[], 10, 'string', {}]:
            with self.assertRaisesRegex(ValueError, 'must be a callable'):
                MongoClient(connect=False, server_selector=selector_candidate)
        MongoClient(connect=False, server_selector=None)

    @client_context.require_replica_set
    def test_selector_called(self):
        if False:
            i = 10
            return i + 15
        selector = FunctionCallRecorder(lambda x: x)
        mongo_client = rs_or_single_client(server_selector=selector)
        test_collection = mongo_client.testdb.test_collection
        self.addCleanup(mongo_client.close)
        self.addCleanup(mongo_client.drop_database, 'testdb')
        test_collection.insert_one({'age': 20, 'name': 'John'})
        test_collection.insert_one({'age': 31, 'name': 'Jane'})
        test_collection.update_one({'name': 'Jane'}, {'$set': {'age': 21}})
        test_collection.find_one({'name': 'Roe'})
        self.assertGreaterEqual(selector.call_count, 4)

    @client_context.require_replica_set
    def test_latency_threshold_application(self):
        if False:
            for i in range(10):
                print('nop')
        selector = SelectionStoreSelector()
        scenario_def: dict = {'topology_description': {'type': 'ReplicaSetWithPrimary', 'servers': [{'address': 'b:27017', 'avg_rtt_ms': 10000, 'type': 'RSSecondary', 'tag': {}}, {'address': 'c:27017', 'avg_rtt_ms': 20000, 'type': 'RSSecondary', 'tag': {}}, {'address': 'a:27017', 'avg_rtt_ms': 30000, 'type': 'RSPrimary', 'tag': {}}]}}
        rtt_times = [srv['avg_rtt_ms'] for srv in scenario_def['topology_description']['servers']]
        min_rtt_idx = rtt_times.index(min(rtt_times))
        (seeds, hosts) = get_addresses(scenario_def['topology_description']['servers'])
        settings = get_topology_settings_dict(heartbeat_frequency=1, local_threshold_ms=1, seeds=seeds, server_selector=selector)
        topology = Topology(TopologySettings(**settings))
        topology.open()
        for server in scenario_def['topology_description']['servers']:
            server_description = make_server_description(server, hosts)
            topology.on_change(server_description)
        server = topology.select_server(ReadPreference.NEAREST)
        assert selector.selection is not None
        self.assertEqual(len(selector.selection), len(topology.description.server_descriptions()))
        self.assertEqual(server.description.address, seeds[min_rtt_idx])

    @client_context.require_replica_set
    def test_server_selector_bypassed(self):
        if False:
            i = 10
            return i + 15
        selector = FunctionCallRecorder(lambda x: x)
        scenario_def = {'topology_description': {'type': 'ReplicaSetNoPrimary', 'servers': [{'address': 'b:27017', 'avg_rtt_ms': 10000, 'type': 'RSSecondary', 'tag': {}}, {'address': 'c:27017', 'avg_rtt_ms': 20000, 'type': 'RSSecondary', 'tag': {}}, {'address': 'a:27017', 'avg_rtt_ms': 30000, 'type': 'RSSecondary', 'tag': {}}]}}
        (seeds, hosts) = get_addresses(scenario_def['topology_description']['servers'])
        settings = get_topology_settings_dict(heartbeat_frequency=1, local_threshold_ms=1, seeds=seeds, server_selector=selector)
        topology = Topology(TopologySettings(**settings))
        topology.open()
        for server in scenario_def['topology_description']['servers']:
            server_description = make_server_description(server, hosts)
            topology.on_change(server_description)
        with self.assertRaisesRegex(ServerSelectionTimeoutError, 'No primary available for writes'):
            topology.select_server(writable_server_selector, server_selection_timeout=0.1)
        self.assertEqual(selector.call_count, 0)
if __name__ == '__main__':
    unittest.main()