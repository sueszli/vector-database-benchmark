"""Test the replica_set_connection module."""
from __future__ import annotations
import contextlib
import copy
import pickle
import random
import sys
from typing import Any
sys.path[0:0] = ['']
from test import IntegrationTest, SkipTest, client_context, unittest
from test.utils import OvertCommandListener, connected, one, rs_client, single_client, wait_until
from test.version import Version
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
from pymongo.message import _maybe_add_read_preference
from pymongo.mongo_client import MongoClient
from pymongo.read_preferences import MovingAverage, Nearest, Primary, PrimaryPreferred, ReadPreference, Secondary, SecondaryPreferred
from pymongo.server_description import ServerDescription
from pymongo.server_selectors import Selection, readable_server_selector
from pymongo.server_type import SERVER_TYPE
from pymongo.write_concern import WriteConcern

class TestSelections(IntegrationTest):

    @client_context.require_connection
    def test_bool(self):
        if False:
            return 10
        client = single_client()
        wait_until(lambda : client.address, 'discover primary')
        selection = Selection.from_topology_description(client._topology.description)
        self.assertTrue(selection)
        self.assertFalse(selection.with_server_descriptions([]))

class TestReadPreferenceObjects(unittest.TestCase):
    prefs = [Primary(), PrimaryPreferred(), Secondary(), Nearest(tag_sets=[{'a': 1}, {'b': 2}]), SecondaryPreferred(max_staleness=30)]

    def test_pickle(self):
        if False:
            print('Hello World!')
        for pref in self.prefs:
            self.assertEqual(pref, pickle.loads(pickle.dumps(pref)))

    def test_copy(self):
        if False:
            for i in range(10):
                print('nop')
        for pref in self.prefs:
            self.assertEqual(pref, copy.copy(pref))

    def test_deepcopy(self):
        if False:
            print('Hello World!')
        for pref in self.prefs:
            self.assertEqual(pref, copy.deepcopy(pref))

class TestReadPreferencesBase(IntegrationTest):

    @classmethod
    @client_context.require_secondaries_count(1)
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super().setUpClass()

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.client.pymongo_test.test.drop()
        self.client.get_database('pymongo_test', write_concern=WriteConcern(w=client_context.w)).test.insert_many([{'_id': i} for i in range(10)])
        self.addCleanup(self.client.pymongo_test.test.drop)

    def read_from_which_host(self, client):
        if False:
            return 10
        'Do a find() on the client and return which host was used'
        cursor = client.pymongo_test.test.find()
        next(cursor)
        return cursor.address

    def read_from_which_kind(self, client):
        if False:
            i = 10
            return i + 15
        "Do a find() on the client and return 'primary' or 'secondary'\n        depending on which the client used.\n        "
        address = self.read_from_which_host(client)
        if address == client.primary:
            return 'primary'
        elif address in client.secondaries:
            return 'secondary'
        else:
            self.fail(f'Cursor used address {address}, expected either primary {client.primary} or secondaries {client.secondaries}')
            return None

    def assertReadsFrom(self, expected, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        c = rs_client(**kwargs)
        wait_until(lambda : len(c.nodes - c.arbiters) == client_context.w, 'discovered all nodes')
        used = self.read_from_which_kind(c)
        self.assertEqual(expected, used, f'Cursor used {used}, expected {expected}')

class TestSingleSecondaryOk(TestReadPreferencesBase):

    def test_reads_from_secondary(self):
        if False:
            i = 10
            return i + 15
        (host, port) = next(iter(self.client.secondaries))
        client = single_client(host, port)
        self.assertFalse(client.is_primary)
        self.assertEqual(client.read_preference, ReadPreference.PRIMARY)
        db = client.pymongo_test
        coll = db.test
        self.assertIsNotNone(coll.find_one())
        self.assertEqual(10, len(list(coll.find())))
        self.assertIsNotNone(db.list_collection_names())
        self.assertIsNotNone(db.validate_collection('test'))
        self.assertIsNotNone(db.command('ping'))
        self.assertEqual(10, coll.count_documents({}))
        self.assertEqual(10, len(coll.distinct('_id')))
        self.assertIsNotNone(coll.aggregate([]))
        self.assertIsNotNone(coll.index_information())

class TestReadPreferences(TestReadPreferencesBase):

    def test_mode_validation(self):
        if False:
            print('Hello World!')
        for mode in (ReadPreference.PRIMARY, ReadPreference.PRIMARY_PREFERRED, ReadPreference.SECONDARY, ReadPreference.SECONDARY_PREFERRED, ReadPreference.NEAREST):
            self.assertEqual(mode, rs_client(read_preference=mode).read_preference)
        self.assertRaises(TypeError, rs_client, read_preference='foo')

    def test_tag_sets_validation(self):
        if False:
            while True:
                i = 10
        S = Secondary(tag_sets=[{}])
        self.assertEqual([{}], rs_client(read_preference=S).read_preference.tag_sets)
        S = Secondary(tag_sets=[{'k': 'v'}])
        self.assertEqual([{'k': 'v'}], rs_client(read_preference=S).read_preference.tag_sets)
        S = Secondary(tag_sets=[{'k': 'v'}, {}])
        self.assertEqual([{'k': 'v'}, {}], rs_client(read_preference=S).read_preference.tag_sets)
        self.assertRaises(ValueError, Secondary, tag_sets=[])
        self.assertRaises(TypeError, Secondary, tag_sets={'k': 'v'})
        self.assertRaises(TypeError, Secondary, tag_sets='foo')
        self.assertRaises(TypeError, Secondary, tag_sets=['foo'])

    def test_threshold_validation(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(17, rs_client(localThresholdMS=17, connect=False).options.local_threshold_ms)
        self.assertEqual(42, rs_client(localThresholdMS=42, connect=False).options.local_threshold_ms)
        self.assertEqual(666, rs_client(localThresholdMS=666, connect=False).options.local_threshold_ms)
        self.assertEqual(0, rs_client(localThresholdMS=0, connect=False).options.local_threshold_ms)
        self.assertRaises(ValueError, rs_client, localthresholdms=-1)

    def test_zero_latency(self):
        if False:
            while True:
                i = 10
        ping_times: set = set()
        while len(ping_times) < len(self.client.nodes):
            ping_times.add(random.random())
        for (ping_time, host) in zip(ping_times, self.client.nodes):
            ServerDescription._host_to_round_trip_time[host] = ping_time
        try:
            client = connected(rs_client(readPreference='nearest', localThresholdMS=0))
            wait_until(lambda : client.nodes == self.client.nodes, 'discovered all nodes')
            host = self.read_from_which_host(client)
            for _ in range(5):
                self.assertEqual(host, self.read_from_which_host(client))
        finally:
            ServerDescription._host_to_round_trip_time.clear()

    def test_primary(self):
        if False:
            print('Hello World!')
        self.assertReadsFrom('primary', read_preference=ReadPreference.PRIMARY)

    def test_primary_with_tags(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ConfigurationError, rs_client, tag_sets=[{'dc': 'ny'}])

    def test_primary_preferred(self):
        if False:
            i = 10
            return i + 15
        self.assertReadsFrom('primary', read_preference=ReadPreference.PRIMARY_PREFERRED)

    def test_secondary(self):
        if False:
            print('Hello World!')
        self.assertReadsFrom('secondary', read_preference=ReadPreference.SECONDARY)

    def test_secondary_preferred(self):
        if False:
            while True:
                i = 10
        self.assertReadsFrom('secondary', read_preference=ReadPreference.SECONDARY_PREFERRED)

    def test_nearest(self):
        if False:
            for i in range(10):
                print('nop')
        c = rs_client(read_preference=ReadPreference.NEAREST, localThresholdMS=10000)
        data_members = {self.client.primary} | self.client.secondaries
        used: set = set()
        i = 0
        while data_members.difference(used) and i < 10000:
            address = self.read_from_which_host(c)
            used.add(address)
            i += 1
        not_used = data_members.difference(used)
        latencies = ', '.join(('%s: %sms' % (server.description.address, server.description.round_trip_time) for server in c._get_topology().select_servers(readable_server_selector)))
        self.assertFalse(not_used, f"Expected to use primary and all secondaries for mode NEAREST, but didn't use {not_used}\nlatencies: {latencies}")

class ReadPrefTester(MongoClient):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.has_read_from = set()
        client_options = client_context.client_options
        client_options.update(kwargs)
        super().__init__(*args, **client_options)

    @contextlib.contextmanager
    def _conn_for_reads(self, read_preference, session):
        if False:
            i = 10
            return i + 15
        context = super()._conn_for_reads(read_preference, session)
        with context as (conn, read_preference):
            self.record_a_read(conn.address)
            yield (conn, read_preference)

    @contextlib.contextmanager
    def _conn_from_server(self, read_preference, server, session):
        if False:
            for i in range(10):
                print('nop')
        context = super()._conn_from_server(read_preference, server, session)
        with context as (conn, read_preference):
            self.record_a_read(conn.address)
            yield (conn, read_preference)

    def record_a_read(self, address):
        if False:
            for i in range(10):
                print('nop')
        server = self._get_topology().select_server_by_address(address, 0)
        self.has_read_from.add(server)
_PREF_MAP = [(Primary, SERVER_TYPE.RSPrimary), (PrimaryPreferred, SERVER_TYPE.RSPrimary), (Secondary, SERVER_TYPE.RSSecondary), (SecondaryPreferred, SERVER_TYPE.RSSecondary), (Nearest, 'any')]

class TestCommandAndReadPreference(IntegrationTest):
    c: ReadPrefTester
    client_version: Version

    @classmethod
    @client_context.require_secondaries_count(1)
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super().setUpClass()
        cls.c = ReadPrefTester(client_context.pair, localThresholdMS=1000 * 1000)
        cls.client_version = Version.from_client(cls.c)
        coll = cls.c.pymongo_test.get_collection('test', write_concern=WriteConcern(w=client_context.w))
        coll.insert_one({})

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        cls.c.drop_database('pymongo_test')
        cls.c.close()

    def executed_on_which_server(self, client, fn, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Execute fn(*args, **kwargs) and return the Server instance used.'
        client.has_read_from.clear()
        fn(*args, **kwargs)
        self.assertEqual(1, len(client.has_read_from))
        return one(client.has_read_from)

    def assertExecutedOn(self, server_type, client, fn, *args, **kwargs):
        if False:
            print('Hello World!')
        server = self.executed_on_which_server(client, fn, *args, **kwargs)
        self.assertEqual(SERVER_TYPE._fields[server_type], SERVER_TYPE._fields[server.description.server_type])

    def _test_fn(self, server_type, fn):
        if False:
            while True:
                i = 10
        for _ in range(10):
            if server_type == 'any':
                used = set()
                for _ in range(1000):
                    server = self.executed_on_which_server(self.c, fn)
                    used.add(server.description.address)
                    if len(used) == len(self.c.secondaries) + 1:
                        break
                assert self.c.primary is not None
                unused = self.c.secondaries.union({self.c.primary}).difference(used)
                if unused:
                    self.fail('Some members not used for NEAREST: %s' % unused)
            else:
                self.assertExecutedOn(server_type, self.c, fn)

    def _test_primary_helper(self, func):
        if False:
            while True:
                i = 10
        self._test_fn(SERVER_TYPE.RSPrimary, func)

    def _test_coll_helper(self, secondary_ok, coll, meth, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        for (mode, server_type) in _PREF_MAP:
            new_coll = coll.with_options(read_preference=mode())

            def func():
                if False:
                    print('Hello World!')
                return getattr(new_coll, meth)(*args, **kwargs)
            if secondary_ok:
                self._test_fn(server_type, func)
            else:
                self._test_fn(SERVER_TYPE.RSPrimary, func)

    def test_command(self):
        if False:
            print('Hello World!')
        for (mode, server_type) in _PREF_MAP:

            def func():
                if False:
                    for i in range(10):
                        print('nop')
                return self.c.pymongo_test.command('dbStats', read_preference=mode())
            self._test_fn(server_type, func)

    def test_create_collection(self):
        if False:
            print('Hello World!')
        self._test_primary_helper(lambda : self.c.pymongo_test.create_collection('some_collection%s' % random.randint(0, sys.maxsize)))

    def test_count_documents(self):
        if False:
            while True:
                i = 10
        self._test_coll_helper(True, self.c.pymongo_test.test, 'count_documents', {})

    def test_estimated_document_count(self):
        if False:
            while True:
                i = 10
        self._test_coll_helper(True, self.c.pymongo_test.test, 'estimated_document_count')

    def test_distinct(self):
        if False:
            i = 10
            return i + 15
        self._test_coll_helper(True, self.c.pymongo_test.test, 'distinct', 'a')

    def test_aggregate(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_coll_helper(True, self.c.pymongo_test.test, 'aggregate', [{'$project': {'_id': 1}}])

    def test_aggregate_write(self):
        if False:
            i = 10
            return i + 15
        secondary_ok = client_context.version.at_least(5, 0)
        self._test_coll_helper(secondary_ok, self.c.pymongo_test.test, 'aggregate', [{'$project': {'_id': 1}}, {'$out': 'agg_write_test'}])

class TestMovingAverage(unittest.TestCase):

    def test_moving_average(self):
        if False:
            while True:
                i = 10
        avg = MovingAverage()
        self.assertIsNone(avg.get())
        avg.add_sample(10)
        self.assertAlmostEqual(10, avg.get())
        avg.add_sample(20)
        self.assertAlmostEqual(12, avg.get())
        avg.add_sample(30)
        self.assertAlmostEqual(15.6, avg.get())

class TestMongosAndReadPreference(IntegrationTest):

    def test_read_preference_document(self):
        if False:
            return 10
        pref = Primary()
        self.assertEqual(pref.document, {'mode': 'primary'})
        pref = PrimaryPreferred()
        self.assertEqual(pref.document, {'mode': 'primaryPreferred'})
        pref = PrimaryPreferred(tag_sets=[{'dc': 'sf'}])
        self.assertEqual(pref.document, {'mode': 'primaryPreferred', 'tags': [{'dc': 'sf'}]})
        pref = PrimaryPreferred(tag_sets=[{'dc': 'sf'}], max_staleness=30)
        self.assertEqual(pref.document, {'mode': 'primaryPreferred', 'tags': [{'dc': 'sf'}], 'maxStalenessSeconds': 30})
        pref = Secondary()
        self.assertEqual(pref.document, {'mode': 'secondary'})
        pref = Secondary(tag_sets=[{'dc': 'sf'}])
        self.assertEqual(pref.document, {'mode': 'secondary', 'tags': [{'dc': 'sf'}]})
        pref = Secondary(tag_sets=[{'dc': 'sf'}], max_staleness=30)
        self.assertEqual(pref.document, {'mode': 'secondary', 'tags': [{'dc': 'sf'}], 'maxStalenessSeconds': 30})
        pref = SecondaryPreferred()
        self.assertEqual(pref.document, {'mode': 'secondaryPreferred'})
        pref = SecondaryPreferred(tag_sets=[{'dc': 'sf'}])
        self.assertEqual(pref.document, {'mode': 'secondaryPreferred', 'tags': [{'dc': 'sf'}]})
        pref = SecondaryPreferred(tag_sets=[{'dc': 'sf'}], max_staleness=30)
        self.assertEqual(pref.document, {'mode': 'secondaryPreferred', 'tags': [{'dc': 'sf'}], 'maxStalenessSeconds': 30})
        pref = Nearest()
        self.assertEqual(pref.document, {'mode': 'nearest'})
        pref = Nearest(tag_sets=[{'dc': 'sf'}])
        self.assertEqual(pref.document, {'mode': 'nearest', 'tags': [{'dc': 'sf'}]})
        pref = Nearest(tag_sets=[{'dc': 'sf'}], max_staleness=30)
        self.assertEqual(pref.document, {'mode': 'nearest', 'tags': [{'dc': 'sf'}], 'maxStalenessSeconds': 30})
        with self.assertRaises(TypeError):
            Nearest(max_staleness=1.5)
        with self.assertRaises(ValueError):
            Nearest(max_staleness=0)
        with self.assertRaises(ValueError):
            Nearest(max_staleness=-2)

    def test_read_preference_document_hedge(self):
        if False:
            while True:
                i = 10
        cases = {'primaryPreferred': PrimaryPreferred, 'secondary': Secondary, 'secondaryPreferred': SecondaryPreferred, 'nearest': Nearest}
        for (mode, cls) in cases.items():
            with self.assertRaises(TypeError):
                cls(hedge=[])
            pref = cls(hedge={})
            self.assertEqual(pref.document, {'mode': mode})
            out = _maybe_add_read_preference({}, pref)
            if cls == SecondaryPreferred:
                self.assertEqual(out, {})
            else:
                self.assertEqual(out, SON([('$query', {}), ('$readPreference', pref.document)]))
            hedge: dict[str, Any] = {'enabled': True}
            pref = cls(hedge=hedge)
            self.assertEqual(pref.document, {'mode': mode, 'hedge': hedge})
            out = _maybe_add_read_preference({}, pref)
            self.assertEqual(out, SON([('$query', {}), ('$readPreference', pref.document)]))
            hedge = {'enabled': False}
            pref = cls(hedge=hedge)
            self.assertEqual(pref.document, {'mode': mode, 'hedge': hedge})
            out = _maybe_add_read_preference({}, pref)
            self.assertEqual(out, SON([('$query', {}), ('$readPreference', pref.document)]))
            hedge = {'enabled': False, 'extra': 'option'}
            pref = cls(hedge=hedge)
            self.assertEqual(pref.document, {'mode': mode, 'hedge': hedge})
            out = _maybe_add_read_preference({}, pref)
            self.assertEqual(out, SON([('$query', {}), ('$readPreference', pref.document)]))

    def test_send_hedge(self):
        if False:
            while True:
                i = 10
        cases = {'primaryPreferred': PrimaryPreferred, 'secondaryPreferred': SecondaryPreferred, 'nearest': Nearest}
        if client_context.supports_secondary_read_pref:
            cases['secondary'] = Secondary
        listener = OvertCommandListener()
        client = rs_client(event_listeners=[listener])
        self.addCleanup(client.close)
        client.admin.command('ping')
        for (_mode, cls) in cases.items():
            pref = cls(hedge={'enabled': True})
            coll = client.test.get_collection('test', read_preference=pref)
            listener.reset()
            coll.find_one()
            started = listener.started_events
            self.assertEqual(len(started), 1, started)
            cmd = started[0].command
            if client_context.is_rs or client_context.is_mongos:
                self.assertIn('$readPreference', cmd)
                self.assertEqual(cmd['$readPreference'], pref.document)
            else:
                self.assertNotIn('$readPreference', cmd)

    def test_maybe_add_read_preference(self):
        if False:
            return 10
        out = _maybe_add_read_preference({}, Primary())
        self.assertEqual(out, {})
        pref = PrimaryPreferred()
        out = _maybe_add_read_preference({}, pref)
        self.assertEqual(out, SON([('$query', {}), ('$readPreference', pref.document)]))
        pref = PrimaryPreferred(tag_sets=[{'dc': 'nyc'}])
        out = _maybe_add_read_preference({}, pref)
        self.assertEqual(out, SON([('$query', {}), ('$readPreference', pref.document)]))
        pref = Secondary()
        out = _maybe_add_read_preference({}, pref)
        self.assertEqual(out, SON([('$query', {}), ('$readPreference', pref.document)]))
        pref = Secondary(tag_sets=[{'dc': 'nyc'}])
        out = _maybe_add_read_preference({}, pref)
        self.assertEqual(out, SON([('$query', {}), ('$readPreference', pref.document)]))
        pref = SecondaryPreferred()
        out = _maybe_add_read_preference({}, pref)
        self.assertEqual(out, {})
        pref = SecondaryPreferred(tag_sets=[{'dc': 'nyc'}])
        out = _maybe_add_read_preference({}, pref)
        self.assertEqual(out, SON([('$query', {}), ('$readPreference', pref.document)]))
        pref = SecondaryPreferred(max_staleness=120)
        out = _maybe_add_read_preference({}, pref)
        self.assertEqual(out, SON([('$query', {}), ('$readPreference', pref.document)]))
        pref = Nearest()
        out = _maybe_add_read_preference({}, pref)
        self.assertEqual(out, SON([('$query', {}), ('$readPreference', pref.document)]))
        pref = Nearest(tag_sets=[{'dc': 'nyc'}])
        out = _maybe_add_read_preference({}, pref)
        self.assertEqual(out, SON([('$query', {}), ('$readPreference', pref.document)]))
        criteria = SON([('$query', {}), ('$orderby', SON([('_id', 1)]))])
        pref = Nearest()
        out = _maybe_add_read_preference(criteria, pref)
        self.assertEqual(out, SON([('$query', {}), ('$orderby', SON([('_id', 1)])), ('$readPreference', pref.document)]))
        pref = Nearest(tag_sets=[{'dc': 'nyc'}])
        out = _maybe_add_read_preference(criteria, pref)
        self.assertEqual(out, SON([('$query', {}), ('$orderby', SON([('_id', 1)])), ('$readPreference', pref.document)]))

    @client_context.require_mongos
    def test_mongos(self):
        if False:
            for i in range(10):
                print('nop')
        res = client_context.client.config.shards.find_one()
        assert res is not None
        shard = res['host']
        num_members = shard.count(',') + 1
        if num_members == 1:
            raise SkipTest('Need a replica set shard to test.')
        coll = client_context.client.pymongo_test.get_collection('test', write_concern=WriteConcern(w=num_members))
        coll.drop()
        res = coll.insert_many([{} for _ in range(5)])
        first_id = res.inserted_ids[0]
        last_id = res.inserted_ids[-1]
        for pref in (Primary(), PrimaryPreferred(), Secondary(), SecondaryPreferred(), Nearest()):
            qcoll = coll.with_options(read_preference=pref)
            results = list(qcoll.find().sort([('_id', 1)]))
            self.assertEqual(first_id, results[0]['_id'])
            self.assertEqual(last_id, results[-1]['_id'])
            results = list(qcoll.find().sort([('_id', -1)]))
            self.assertEqual(first_id, results[-1]['_id'])
            self.assertEqual(last_id, results[0]['_id'])

    @client_context.require_mongos
    def test_mongos_max_staleness(self):
        if False:
            i = 10
            return i + 15
        coll = client_context.client.pymongo_test.get_collection('test', read_preference=SecondaryPreferred(max_staleness=120))
        coll.find_one()
        coll = client_context.client.pymongo_test.get_collection('test', read_preference=SecondaryPreferred(max_staleness=10))
        try:
            coll.find_one()
        except OperationFailure as exc:
            self.assertEqual(160, exc.code)
        else:
            self.fail('mongos accepted invalid staleness')
        coll = single_client(readPreference='secondaryPreferred', maxStalenessSeconds=120).pymongo_test.test
        coll.find_one()
        coll = single_client(readPreference='secondaryPreferred', maxStalenessSeconds=10).pymongo_test.test
        try:
            coll.find_one()
        except OperationFailure as exc:
            self.assertEqual(160, exc.code)
        else:
            self.fail('mongos accepted invalid staleness')
if __name__ == '__main__':
    unittest.main()