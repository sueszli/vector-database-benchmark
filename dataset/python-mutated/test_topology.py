"""Test the topology module."""
from __future__ import annotations
import sys
sys.path[0:0] = ['']
from test import client_knobs, unittest
from test.pymongo_mocks import DummyMonitor
from test.utils import MockPool, wait_until
from bson.objectid import ObjectId
from pymongo import common
from pymongo.errors import AutoReconnect, ConfigurationError, ConnectionFailure
from pymongo.hello import Hello, HelloCompat
from pymongo.monitor import Monitor
from pymongo.pool import PoolOptions
from pymongo.read_preferences import ReadPreference, Secondary
from pymongo.server import Server
from pymongo.server_description import ServerDescription
from pymongo.server_selectors import any_server_selector, writable_server_selector
from pymongo.server_type import SERVER_TYPE
from pymongo.settings import TopologySettings
from pymongo.topology import Topology, _ErrorContext, _filter_servers
from pymongo.topology_description import TOPOLOGY_TYPE

class SetNameDiscoverySettings(TopologySettings):

    def get_topology_type(self):
        if False:
            while True:
                i = 10
        return TOPOLOGY_TYPE.ReplicaSetNoPrimary
address = ('a', 27017)

def create_mock_topology(seeds=None, replica_set_name=None, monitor_class=DummyMonitor, direct_connection=False):
    if False:
        return 10
    partitioned_seeds = list(map(common.partition_node, seeds or ['a']))
    topology_settings = TopologySettings(partitioned_seeds, replica_set_name=replica_set_name, pool_class=MockPool, monitor_class=monitor_class, direct_connection=direct_connection)
    t = Topology(topology_settings)
    t.open()
    return t

def got_hello(topology, server_address, hello_response):
    if False:
        while True:
            i = 10
    server_description = ServerDescription(server_address, Hello(hello_response), 0)
    topology.on_change(server_description)

def disconnected(topology, server_address):
    if False:
        return 10
    topology.on_change(ServerDescription(server_address))

def get_server(topology, hostname):
    if False:
        print('Hello World!')
    return topology.get_server_by_address((hostname, 27017))

def get_type(topology, hostname):
    if False:
        while True:
            i = 10
    return get_server(topology, hostname).description.server_type

def get_monitor(topology, hostname):
    if False:
        i = 10
        return i + 15
    return get_server(topology, hostname)._monitor

class TopologyTest(unittest.TestCase):
    """Disables periodic monitoring, to make tests deterministic."""

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.client_knobs = client_knobs(heartbeat_frequency=999999)
        self.client_knobs.enable()
        self.addCleanup(self.client_knobs.disable)

class TestTopologyConfiguration(TopologyTest):

    def test_timeout_configuration(self):
        if False:
            for i in range(10):
                print('nop')
        pool_options = PoolOptions(connect_timeout=1, socket_timeout=2)
        topology_settings = TopologySettings(pool_options=pool_options)
        t = Topology(topology_settings=topology_settings)
        t.open()
        server = t.get_server_by_address(('localhost', 27017))
        self.assertEqual(1, server._pool.opts.connect_timeout)
        self.assertEqual(2, server._pool.opts.socket_timeout)
        monitor = server._monitor
        self.assertEqual(1, monitor._pool.opts.connect_timeout)
        self.assertEqual(1, monitor._pool.opts.socket_timeout)
        self.assertFalse(monitor._pool.handshake)

class TestSingleServerTopology(TopologyTest):

    def test_direct_connection(self):
        if False:
            i = 10
            return i + 15
        for (server_type, hello_response) in [(SERVER_TYPE.RSPrimary, {'ok': 1, HelloCompat.LEGACY_CMD: True, 'hosts': ['a'], 'setName': 'rs', 'maxWireVersion': 6}), (SERVER_TYPE.RSSecondary, {'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': True, 'hosts': ['a'], 'setName': 'rs', 'maxWireVersion': 6}), (SERVER_TYPE.Mongos, {'ok': 1, HelloCompat.LEGACY_CMD: True, 'msg': 'isdbgrid', 'maxWireVersion': 6}), (SERVER_TYPE.RSArbiter, {'ok': 1, HelloCompat.LEGACY_CMD: False, 'arbiterOnly': True, 'hosts': ['a'], 'setName': 'rs', 'maxWireVersion': 6}), (SERVER_TYPE.Standalone, {'ok': 1, HelloCompat.LEGACY_CMD: True, 'maxWireVersion': 6}), (SERVER_TYPE.Standalone, {'ok': 1, HelloCompat.LEGACY_CMD: False, 'maxWireVersion': 6})]:
            t = create_mock_topology(direct_connection=True)
            with self.assertRaisesRegex(ConnectionFailure, 'No servers found yet'):
                t.select_servers(any_server_selector, server_selection_timeout=0)
            got_hello(t, address, hello_response)
            self.assertEqual(TOPOLOGY_TYPE.Single, t.description.topology_type)
            s = t.select_server(writable_server_selector)
            self.assertEqual(server_type, s.description.server_type)
            self.assertEqual(t.description.topology_type_name, 'Single')
            self.assertTrue(t.description.has_writable_server())
            self.assertTrue(t.description.has_readable_server())
            self.assertTrue(t.description.has_readable_server(Secondary()))
            self.assertTrue(t.description.has_readable_server(Secondary(tag_sets=[{'tag': 'does-not-exist'}])))

    def test_reopen(self):
        if False:
            while True:
                i = 10
        t = create_mock_topology()
        t.open()
        t.open()

    def test_unavailable_seed(self):
        if False:
            for i in range(10):
                print('nop')
        t = create_mock_topology()
        disconnected(t, address)
        self.assertEqual(SERVER_TYPE.Unknown, get_type(t, 'a'))

    def test_round_trip_time(self):
        if False:
            while True:
                i = 10
        round_trip_time = 125
        available = True

        class TestMonitor(Monitor):

            def _check_with_socket(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                if available:
                    return (Hello({'ok': 1, 'maxWireVersion': 6}), round_trip_time)
                else:
                    raise AutoReconnect('mock monitor error')
        t = create_mock_topology(monitor_class=TestMonitor)
        self.addCleanup(t.close)
        s = t.select_server(writable_server_selector)
        self.assertEqual(125, s.description.round_trip_time)
        round_trip_time = 25
        t.request_check_all()
        self.assertAlmostEqual(105, s.description.round_trip_time)
        available = False
        t.request_check_all()

        def raises_err():
            if False:
                print('Hello World!')
            try:
                t.select_server(writable_server_selector, server_selection_timeout=0.1)
            except ConnectionFailure:
                return True
            else:
                return False
        wait_until(raises_err, 'discover server is down')
        self.assertIsNone(s.description.round_trip_time)
        available = True
        round_trip_time = 20

        def new_average():
            if False:
                print('Hello World!')
            description = s.description
            return description.round_trip_time is not None and round(abs(20 - description.round_trip_time), 7) == 0
        tries = 0
        while not new_average():
            t.request_check_all()
            tries += 1
            if tries > 10:
                self.fail("Didn't ever calculate correct new average")

class TestMultiServerTopology(TopologyTest):

    def test_readable_writable(self):
        if False:
            i = 10
            return i + 15
        t = create_mock_topology(replica_set_name='rs')
        got_hello(t, ('a', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a', 'b']})
        got_hello(t, ('b', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': True, 'setName': 'rs', 'hosts': ['a', 'b']})
        self.assertEqual(t.description.topology_type_name, 'ReplicaSetWithPrimary')
        self.assertTrue(t.description.has_writable_server())
        self.assertTrue(t.description.has_readable_server())
        self.assertTrue(t.description.has_readable_server(Secondary()))
        self.assertFalse(t.description.has_readable_server(Secondary(tag_sets=[{'tag': 'exists'}])))
        t = create_mock_topology(replica_set_name='rs')
        got_hello(t, ('a', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': False, 'setName': 'rs', 'hosts': ['a', 'b']})
        got_hello(t, ('b', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': True, 'setName': 'rs', 'hosts': ['a', 'b']})
        self.assertEqual(t.description.topology_type_name, 'ReplicaSetNoPrimary')
        self.assertFalse(t.description.has_writable_server())
        self.assertFalse(t.description.has_readable_server())
        self.assertTrue(t.description.has_readable_server(Secondary()))
        self.assertFalse(t.description.has_readable_server(Secondary(tag_sets=[{'tag': 'exists'}])))
        t = create_mock_topology(replica_set_name='rs')
        got_hello(t, ('a', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a', 'b']})
        got_hello(t, ('b', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': True, 'setName': 'rs', 'hosts': ['a', 'b'], 'tags': {'tag': 'exists'}})
        self.assertEqual(t.description.topology_type_name, 'ReplicaSetWithPrimary')
        self.assertTrue(t.description.has_writable_server())
        self.assertTrue(t.description.has_readable_server())
        self.assertTrue(t.description.has_readable_server(Secondary()))
        self.assertTrue(t.description.has_readable_server(Secondary(tag_sets=[{'tag': 'exists'}])))

    def test_close(self):
        if False:
            i = 10
            return i + 15
        t = create_mock_topology(replica_set_name='rs')
        got_hello(t, ('a', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a', 'b']})
        got_hello(t, ('b', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': True, 'setName': 'rs', 'hosts': ['a', 'b']})
        self.assertEqual(SERVER_TYPE.RSPrimary, get_type(t, 'a'))
        self.assertEqual(SERVER_TYPE.RSSecondary, get_type(t, 'b'))
        self.assertTrue(get_monitor(t, 'a').opened)
        self.assertTrue(get_monitor(t, 'b').opened)
        self.assertEqual(TOPOLOGY_TYPE.ReplicaSetWithPrimary, t.description.topology_type)
        t.close()
        self.assertEqual(2, len(t.description.server_descriptions()))
        self.assertEqual(SERVER_TYPE.Unknown, get_type(t, 'a'))
        self.assertEqual(SERVER_TYPE.Unknown, get_type(t, 'b'))
        self.assertFalse(get_monitor(t, 'a').opened)
        self.assertFalse(get_monitor(t, 'b').opened)
        self.assertEqual('rs', t.description.replica_set_name)
        self.assertEqual(TOPOLOGY_TYPE.ReplicaSetNoPrimary, t.description.topology_type)
        got_hello(t, ('a', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a', 'b', 'c']})
        self.assertEqual(2, len(t.description.server_descriptions()))
        self.assertEqual(SERVER_TYPE.Unknown, get_type(t, 'a'))
        self.assertEqual(SERVER_TYPE.Unknown, get_type(t, 'b'))
        self.assertFalse(get_monitor(t, 'a').opened)
        self.assertFalse(get_monitor(t, 'b').opened)
        self.assertEqual(None, get_server(t, 'c'))
        self.assertEqual('rs', t.description.replica_set_name)
        self.assertEqual(TOPOLOGY_TYPE.ReplicaSetNoPrimary, t.description.topology_type)

    def test_handle_error(self):
        if False:
            print('Hello World!')
        t = create_mock_topology(replica_set_name='rs')
        got_hello(t, ('a', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a', 'b']})
        got_hello(t, ('b', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': True, 'setName': 'rs', 'hosts': ['a', 'b']})
        errctx = _ErrorContext(AutoReconnect('mock'), 0, 0, True, None)
        t.handle_error(('a', 27017), errctx)
        self.assertEqual(SERVER_TYPE.Unknown, get_type(t, 'a'))
        self.assertEqual(SERVER_TYPE.RSSecondary, get_type(t, 'b'))
        self.assertEqual('rs', t.description.replica_set_name)
        self.assertEqual(TOPOLOGY_TYPE.ReplicaSetNoPrimary, t.description.topology_type)
        got_hello(t, ('a', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a', 'b']})
        self.assertEqual(SERVER_TYPE.RSPrimary, get_type(t, 'a'))
        self.assertEqual(TOPOLOGY_TYPE.ReplicaSetWithPrimary, t.description.topology_type)
        t.handle_error(('b', 27017), errctx)
        self.assertEqual(SERVER_TYPE.RSPrimary, get_type(t, 'a'))
        self.assertEqual(SERVER_TYPE.Unknown, get_type(t, 'b'))
        self.assertEqual('rs', t.description.replica_set_name)
        self.assertEqual(TOPOLOGY_TYPE.ReplicaSetWithPrimary, t.description.topology_type)

    def test_handle_error_removed_server(self):
        if False:
            return 10
        t = create_mock_topology(replica_set_name='rs')
        errctx = _ErrorContext(AutoReconnect('mock'), 0, 0, True, None)
        t.handle_error(('b', 27017), errctx)
        self.assertFalse(t.has_server(('b', 27017)))

    def test_discover_set_name_from_primary(self):
        if False:
            for i in range(10):
                print('nop')
        topology_settings = SetNameDiscoverySettings(seeds=[address], pool_class=MockPool, monitor_class=DummyMonitor)
        t = Topology(topology_settings)
        self.assertEqual(t.description.replica_set_name, None)
        self.assertEqual(t.description.topology_type, TOPOLOGY_TYPE.ReplicaSetNoPrimary)
        t.open()
        got_hello(t, address, {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a']})
        self.assertEqual(t.description.replica_set_name, 'rs')
        self.assertEqual(t.description.topology_type, TOPOLOGY_TYPE.ReplicaSetWithPrimary)
        got_hello(t, address, {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a']})
        self.assertEqual(t.description.replica_set_name, 'rs')
        self.assertEqual(t.description.topology_type, TOPOLOGY_TYPE.ReplicaSetWithPrimary)

    def test_discover_set_name_from_secondary(self):
        if False:
            print('Hello World!')
        topology_settings = SetNameDiscoverySettings(seeds=[address], pool_class=MockPool, monitor_class=DummyMonitor)
        t = Topology(topology_settings)
        self.assertEqual(t.description.replica_set_name, None)
        self.assertEqual(t.description.topology_type, TOPOLOGY_TYPE.ReplicaSetNoPrimary)
        t.open()
        got_hello(t, address, {'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': True, 'setName': 'rs', 'hosts': ['a']})
        self.assertEqual(t.description.replica_set_name, 'rs')
        self.assertEqual(t.description.topology_type, TOPOLOGY_TYPE.ReplicaSetNoPrimary)

    def test_wire_version(self):
        if False:
            while True:
                i = 10
        t = create_mock_topology(replica_set_name='rs')
        t.description.check_compatible()
        got_hello(t, address, {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a']})
        server = t.get_server_by_address(address)
        self.assertEqual(server.description.min_wire_version, 0)
        self.assertEqual(server.description.max_wire_version, 0)
        got_hello(t, address, {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a'], 'minWireVersion': 1, 'maxWireVersion': 6})
        self.assertEqual(server.description.min_wire_version, 1)
        self.assertEqual(server.description.max_wire_version, 6)
        t.select_servers(any_server_selector)
        got_hello(t, address, {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a'], 'minWireVersion': 22, 'maxWireVersion': 24})
        try:
            t.select_servers(any_server_selector)
        except ConfigurationError as e:
            self.assertEqual(str(e), 'Server at a:27017 requires wire version 22, but this version of PyMongo only supports up to %d.' % (common.MAX_SUPPORTED_WIRE_VERSION,))
        else:
            self.fail('No error with incompatible wire version')
        got_hello(t, address, {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a'], 'minWireVersion': 0, 'maxWireVersion': 0})
        try:
            t.select_servers(any_server_selector)
        except ConfigurationError as e:
            self.assertEqual(str(e), 'Server at a:27017 reports wire version 0, but this version of PyMongo requires at least %d (MongoDB %s).' % (common.MIN_SUPPORTED_WIRE_VERSION, common.MIN_SUPPORTED_SERVER_VERSION))
        else:
            self.fail('No error with incompatible wire version')

    def test_max_write_batch_size(self):
        if False:
            while True:
                i = 10
        t = create_mock_topology(seeds=['a', 'b'], replica_set_name='rs')

        def write_batch_size():
            if False:
                for i in range(10):
                    print('nop')
            s = t.select_server(writable_server_selector)
            return s.description.max_write_batch_size
        got_hello(t, ('a', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a', 'b'], 'maxWireVersion': 6, 'maxWriteBatchSize': 1})
        got_hello(t, ('b', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': True, 'setName': 'rs', 'hosts': ['a', 'b'], 'maxWireVersion': 6, 'maxWriteBatchSize': 2})
        self.assertEqual(1, write_batch_size())
        got_hello(t, ('b', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a', 'b'], 'maxWireVersion': 6, 'maxWriteBatchSize': 2})
        self.assertEqual(2, write_batch_size())

    def test_topology_repr(self):
        if False:
            while True:
                i = 10
        t = create_mock_topology(replica_set_name='rs')
        self.addCleanup(t.close)
        got_hello(t, ('a', 27017), {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a', 'c', 'b']})
        self.assertEqual(repr(t.description), f"<TopologyDescription id: {t._topology_id}, topology_type: ReplicaSetWithPrimary, servers: [<ServerDescription ('a', 27017) server_type: RSPrimary, rtt: 0>, <ServerDescription ('b', 27017) server_type: Unknown, rtt: None>, <ServerDescription ('c', 27017) server_type: Unknown, rtt: None>]>")

    def test_unexpected_load_balancer(self):
        if False:
            while True:
                i = 10
        t = create_mock_topology(seeds=['a'])
        mock_lb_response = {'ok': 1, 'msg': 'isdbgrid', 'serviceId': ObjectId(), 'maxWireVersion': 13}
        got_hello(t, ('a', 27017), mock_lb_response)
        sds = t.description.server_descriptions()
        self.assertIn(('a', 27017), sds)
        self.assertEqual(sds['a', 27017].server_type_name, 'LoadBalancer')
        self.assertEqual(t.description.topology_type_name, 'Single')
        self.assertTrue(t.description.has_writable_server())
        t = create_mock_topology(seeds=['a', 'b'])
        got_hello(t, ('a', 27017), mock_lb_response)
        self.assertNotIn(('a', 27017), t.description.server_descriptions())
        self.assertEqual(t.description.topology_type_name, 'Unknown')

    def test_filtered_server_selection(self):
        if False:
            for i in range(10):
                print('nop')
        s1 = Server(ServerDescription(('localhost', 27017)), pool=object(), monitor=object())
        s2 = Server(ServerDescription(('localhost2', 27017)), pool=object(), monitor=object())
        servers = [s1, s2]
        result = _filter_servers(servers, deprioritized_servers=[s2])
        self.assertEqual(result, [s1])
        result = _filter_servers(servers, deprioritized_servers=[s1, s2])
        self.assertEqual(result, servers)
        result = _filter_servers(servers, deprioritized_servers=[])
        self.assertEqual(result, servers)
        result = _filter_servers(servers)
        self.assertEqual(result, servers)

def wait_for_primary(topology):
    if False:
        i = 10
        return i + 15
    'Wait for a Topology to discover a writable server.\n\n    If the monitor is currently calling hello, a blocking call to\n    select_server from this thread can trigger a spurious wake of the monitor\n    thread. In applications this is harmless but it would break some tests,\n    so we pass server_selection_timeout=0 and poll instead.\n    '

    def get_primary():
        if False:
            return 10
        try:
            return topology.select_server(writable_server_selector, 0)
        except ConnectionFailure:
            return None
    return wait_until(get_primary, 'find primary')

class TestTopologyErrors(TopologyTest):

    def test_pool_reset(self):
        if False:
            print('Hello World!')
        hello_count = [0]

        class TestMonitor(Monitor):

            def _check_with_socket(self, *args, **kwargs):
                if False:
                    return 10
                hello_count[0] += 1
                if hello_count[0] == 1:
                    return (Hello({'ok': 1, 'maxWireVersion': 6}), 0)
                else:
                    raise AutoReconnect('mock monitor error')
        t = create_mock_topology(monitor_class=TestMonitor)
        self.addCleanup(t.close)
        server = wait_for_primary(t)
        self.assertEqual(1, hello_count[0])
        generation = server.pool.gen.get_overall()
        t.request_check_all()
        self.assertNotEqual(generation, server.pool.gen.get_overall())

    def test_hello_retry(self):
        if False:
            while True:
                i = 10
        hello_count = [0]

        class TestMonitor(Monitor):

            def _check_with_socket(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                hello_count[0] += 1
                if hello_count[0] in (1, 3):
                    return (Hello({'ok': 1, 'maxWireVersion': 6}), 0)
                else:
                    raise AutoReconnect(f'mock monitor error #{hello_count[0]}')
        t = create_mock_topology(monitor_class=TestMonitor)
        self.addCleanup(t.close)
        server = wait_for_primary(t)
        self.assertEqual(1, hello_count[0])
        self.assertEqual(SERVER_TYPE.Standalone, server.description.server_type)
        t.request_check_all()
        server = t.select_server(writable_server_selector, 0.25)
        self.assertEqual(SERVER_TYPE.Standalone, server.description.server_type)
        self.assertEqual(3, hello_count[0])

    def test_internal_monitor_error(self):
        if False:
            while True:
                i = 10
        exception = AssertionError('internal error')

        class TestMonitor(Monitor):

            def _check_with_socket(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                raise exception
        t = create_mock_topology(monitor_class=TestMonitor)
        self.addCleanup(t.close)
        with self.assertRaisesRegex(ConnectionFailure, 'internal error'):
            t.select_server(any_server_selector, server_selection_timeout=0.5)

class TestServerSelectionErrors(TopologyTest):

    def assertMessage(self, message, topology, selector=any_server_selector):
        if False:
            while True:
                i = 10
        with self.assertRaises(ConnectionFailure) as context:
            topology.select_server(selector, server_selection_timeout=0)
        self.assertIn(message, str(context.exception))

    def test_no_primary(self):
        if False:
            print('Hello World!')
        t = create_mock_topology(replica_set_name='rs')
        got_hello(t, address, {'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': True, 'setName': 'rs', 'hosts': ['a']})
        self.assertMessage('No replica set members match selector "Primary()"', t, ReadPreference.PRIMARY)
        self.assertMessage('No primary available for writes', t, writable_server_selector)

    def test_no_secondary(self):
        if False:
            while True:
                i = 10
        t = create_mock_topology(replica_set_name='rs')
        got_hello(t, address, {'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'hosts': ['a']})
        self.assertMessage('No replica set members match selector "Secondary(tag_sets=None, max_staleness=-1, hedge=None)"', t, ReadPreference.SECONDARY)
        self.assertMessage('No replica set members match selector "Secondary(tag_sets=[{\'dc\': \'ny\'}], max_staleness=-1, hedge=None)"', t, Secondary(tag_sets=[{'dc': 'ny'}]))

    def test_bad_replica_set_name(self):
        if False:
            while True:
                i = 10
        t = create_mock_topology(replica_set_name='rs')
        got_hello(t, address, {'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': True, 'setName': 'wrong', 'hosts': ['a']})
        self.assertMessage('No replica set members available for replica set name "rs"', t)

    def test_multiple_standalones(self):
        if False:
            for i in range(10):
                print('nop')
        t = create_mock_topology(seeds=['a', 'b'])
        got_hello(t, ('a', 27017), {'ok': 1})
        got_hello(t, ('b', 27017), {'ok': 1})
        self.assertMessage('No servers available', t)

    def test_no_mongoses(self):
        if False:
            return 10
        t = create_mock_topology(seeds=['a', 'b'])
        got_hello(t, ('a', 27017), {'ok': 1, 'msg': 'isdbgrid'})
        got_hello(t, ('a', 27017), {'ok': 1})
        got_hello(t, ('b', 27017), {'ok': 1})
        self.assertMessage('No mongoses available', t)
if __name__ == '__main__':
    unittest.main()