"""Test retryable reads spec."""
from __future__ import annotations
import os
import pprint
import sys
import threading
from bson import SON
from pymongo.errors import AutoReconnect
sys.path[0:0] = ['']
from test import IntegrationTest, PyMongoTestCase, client_context, client_knobs, unittest
from test.utils import CMAPListener, EventListener, OvertCommandListener, SpecTestCreator, rs_client, rs_or_single_client, set_fail_point
from test.utils_spec_runner import SpecRunner
from pymongo.mongo_client import MongoClient
from pymongo.monitoring import ConnectionCheckedOutEvent, ConnectionCheckOutFailedEvent, ConnectionCheckOutFailedReason, PoolClearedEvent
from pymongo.write_concern import WriteConcern
_TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'retryable_reads', 'legacy')

class TestClientOptions(PyMongoTestCase):

    def test_default(self):
        if False:
            for i in range(10):
                print('nop')
        client = MongoClient(connect=False)
        self.assertEqual(client.options.retry_reads, True)

    def test_kwargs(self):
        if False:
            while True:
                i = 10
        client = MongoClient(retryReads=True, connect=False)
        self.assertEqual(client.options.retry_reads, True)
        client = MongoClient(retryReads=False, connect=False)
        self.assertEqual(client.options.retry_reads, False)

    def test_uri(self):
        if False:
            print('Hello World!')
        client = MongoClient('mongodb://h/?retryReads=true', connect=False)
        self.assertEqual(client.options.retry_reads, True)
        client = MongoClient('mongodb://h/?retryReads=false', connect=False)
        self.assertEqual(client.options.retry_reads, False)

class TestSpec(SpecRunner):
    RUN_ON_LOAD_BALANCER = True
    RUN_ON_SERVERLESS = True

    @classmethod
    @client_context.require_failCommand_fail_point
    @client_context.require_no_mmap
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()

    def maybe_skip_scenario(self, test):
        if False:
            i = 10
            return i + 15
        super().maybe_skip_scenario(test)
        skip_names = ['listCollectionObjects', 'listIndexNames', 'listDatabaseObjects']
        for name in skip_names:
            if name.lower() in test['description'].lower():
                self.skipTest(f'PyMongo does not support {name}')
        if client_context.serverless:
            for operation in test['operations']:
                if operation['name'] == 'aggregate':
                    for stage in operation['arguments']['pipeline']:
                        if '$out' in stage:
                            self.skipTest('MongoDB Serverless does not support $out')
                if 'collation' in operation['arguments']:
                    self.skipTest('MongoDB Serverless does not support collations')
        test_name = self.id().rsplit('.')[-1]
        if 'changestream' in test_name.lower():
            if client_context.storage_engine == 'mmapv1':
                self.skipTest('MMAPv1 does not support change streams.')
            if client_context.serverless:
                self.skipTest('Serverless does not support change streams.')

    def get_scenario_coll_name(self, scenario_def):
        if False:
            i = 10
            return i + 15
        "Override a test's collection name to support GridFS tests."
        if 'bucket_name' in scenario_def:
            return scenario_def['bucket_name']
        return super().get_scenario_coll_name(scenario_def)

    def setup_scenario(self, scenario_def):
        if False:
            return 10
        "Override a test's setup to support GridFS tests."
        if 'bucket_name' in scenario_def:
            data = scenario_def['data']
            db_name = self.get_scenario_db_name(scenario_def)
            db = client_context.client[db_name]
            wc = WriteConcern(w='majority')
            if data:
                db['fs.chunks'].drop()
                db['fs.files'].drop()
                db['fs.chunks'].insert_many(data['fs.chunks'])
                db.get_collection('fs.files', write_concern=wc).insert_many(data['fs.files'])
            else:
                db.get_collection('fs.chunks').drop()
                db.get_collection('fs.files', write_concern=wc).drop()
        else:
            super().setup_scenario(scenario_def)

def create_test(scenario_def, test, name):
    if False:
        while True:
            i = 10

    @client_context.require_test_commands
    def run_scenario(self):
        if False:
            while True:
                i = 10
        self.run_scenario(scenario_def, test)
    return run_scenario
test_creator = SpecTestCreator(create_test, TestSpec, _TEST_PATH)
test_creator.create_tests()

class FindThread(threading.Thread):

    def __init__(self, collection):
        if False:
            return 10
        super().__init__()
        self.daemon = True
        self.collection = collection
        self.passed = False

    def run(self):
        if False:
            return 10
        self.collection.find_one({})
        self.passed = True

class TestPoolPausedError(IntegrationTest):
    RUN_ON_LOAD_BALANCER = False
    RUN_ON_SERVERLESS = False

    @client_context.require_failCommand_blockConnection
    @client_knobs(heartbeat_frequency=0.05, min_heartbeat_interval=0.05)
    def test_pool_paused_error_is_retryable(self):
        if False:
            print('Hello World!')
        if 'PyPy' in sys.version:
            self.skipTest('Test is flakey on PyPy')
        cmap_listener = CMAPListener()
        cmd_listener = OvertCommandListener()
        client = rs_or_single_client(maxPoolSize=1, event_listeners=[cmap_listener, cmd_listener])
        self.addCleanup(client.close)
        for _ in range(10):
            cmap_listener.reset()
            cmd_listener.reset()
            threads = [FindThread(client.pymongo_test.test) for _ in range(2)]
            fail_command = {'mode': {'times': 1}, 'data': {'failCommands': ['find'], 'blockConnection': True, 'blockTimeMS': 1000, 'errorCode': 91}}
            with self.fail_point(fail_command):
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                for thread in threads:
                    self.assertTrue(thread.passed)
            if cmap_listener.event_count(ConnectionCheckOutFailedEvent):
                break
        cmap_events = cmap_listener.events_by_type((ConnectionCheckedOutEvent, ConnectionCheckOutFailedEvent, PoolClearedEvent))
        msg = pprint.pformat(cmap_listener.events)
        self.assertIsInstance(cmap_events[0], ConnectionCheckedOutEvent, msg)
        self.assertIsInstance(cmap_events[1], PoolClearedEvent, msg)
        self.assertIsInstance(cmap_events[2], ConnectionCheckOutFailedEvent, msg)
        self.assertEqual(cmap_events[2].reason, ConnectionCheckOutFailedReason.CONN_ERROR, msg)
        self.assertIsInstance(cmap_events[3], ConnectionCheckedOutEvent, msg)
        started = cmd_listener.started_events
        msg = pprint.pformat(cmd_listener.results)
        self.assertEqual(3, len(started), msg)
        succeeded = cmd_listener.succeeded_events
        self.assertEqual(2, len(succeeded), msg)
        failed = cmd_listener.failed_events
        self.assertEqual(1, len(failed), msg)

class TestRetryableReads(IntegrationTest):

    @client_context.require_multiple_mongoses
    @client_context.require_failCommand_fail_point
    def test_retryable_reads_in_sharded_cluster_multiple_available(self):
        if False:
            while True:
                i = 10
        fail_command = {'configureFailPoint': 'failCommand', 'mode': {'times': 1}, 'data': {'failCommands': ['find'], 'closeConnection': True, 'appName': 'retryableReadTest'}}
        mongos_clients = []
        for mongos in client_context.mongos_seeds().split(','):
            client = rs_or_single_client(mongos)
            set_fail_point(client, fail_command)
            self.addCleanup(client.close)
            mongos_clients.append(client)
        listener = OvertCommandListener()
        client = rs_or_single_client(client_context.mongos_seeds(), appName='retryableReadTest', event_listeners=[listener], retryReads=True)
        with self.fail_point(fail_command):
            with self.assertRaises(AutoReconnect):
                client.t.t.find_one({})
        for client in mongos_clients:
            fail_command['mode'] = 'off'
            set_fail_point(client, fail_command)
        self.assertEqual(len(listener.failed_events), 2)
        self.assertEqual(len(listener.succeeded_events), 0)
if __name__ == '__main__':
    unittest.main()