"""Test compliance with the connections survive primary step down spec."""
from __future__ import annotations
import sys
sys.path[0:0] = ['']
from test import IntegrationTest, client_context, unittest
from test.utils import CMAPListener, ensure_all_connected, repl_set_step_down, rs_or_single_client
from bson import SON
from pymongo import monitoring
from pymongo.collection import Collection
from pymongo.errors import NotPrimaryError
from pymongo.write_concern import WriteConcern

class TestConnectionsSurvivePrimaryStepDown(IntegrationTest):
    listener: CMAPListener
    coll: Collection

    @classmethod
    @client_context.require_replica_set
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        cls.listener = CMAPListener()
        cls.client = rs_or_single_client(event_listeners=[cls.listener], retryWrites=False, heartbeatFrequencyMS=500)
        ensure_all_connected(cls.client)
        cls.listener.reset()
        cls.db = cls.client.get_database('step-down', write_concern=WriteConcern('majority'))
        cls.coll = cls.db.get_collection('step-down', write_concern=WriteConcern('majority'))

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        cls.client.close()

    def setUp(self):
        if False:
            print('Hello World!')
        self.db.drop_collection('step-down')
        self.db.create_collection('step-down')
        self.listener.reset()

    def set_fail_point(self, command_args):
        if False:
            print('Hello World!')
        cmd = SON([('configureFailPoint', 'failCommand')])
        cmd.update(command_args)
        self.client.admin.command(cmd)

    def verify_pool_cleared(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.listener.event_count(monitoring.PoolClearedEvent), 1)

    def verify_pool_not_cleared(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.listener.event_count(monitoring.PoolClearedEvent), 0)

    @client_context.require_version_min(4, 2, -1)
    def test_get_more_iteration(self):
        if False:
            i = 10
            return i + 15
        self.coll.insert_many([{'data': k} for k in range(5)])
        batch_size = 2
        cursor = self.coll.find(batch_size=batch_size)
        for _ in range(batch_size):
            cursor.next()
        repl_set_step_down(self.client, replSetStepDown=5, force=True)
        for _ in range(batch_size):
            cursor.next()
        self.verify_pool_not_cleared()
        try:
            self.coll.insert_one({})
        except NotPrimaryError:
            pass
        self.coll.insert_one({})
        self.verify_pool_not_cleared()

    def run_scenario(self, error_code, retry, pool_status_checker):
        if False:
            while True:
                i = 10
        self.set_fail_point({'mode': {'times': 1}, 'data': {'failCommands': ['insert'], 'errorCode': error_code}})
        self.addCleanup(self.set_fail_point, {'mode': 'off'})
        with self.assertRaises(NotPrimaryError) as exc:
            self.coll.insert_one({'test': 1})
        self.assertEqual(exc.exception.details['code'], error_code)
        if retry:
            self.coll.insert_one({'test': 1})
        pool_status_checker()
        self.coll.insert_one({'test': 1})

    @client_context.require_version_min(4, 2, -1)
    @client_context.require_test_commands
    def test_not_primary_keep_connection_pool(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_scenario(10107, True, self.verify_pool_not_cleared)

    @client_context.require_version_min(4, 0, 0)
    @client_context.require_version_max(4, 1, 0, -1)
    @client_context.require_test_commands
    def test_not_primary_reset_connection_pool(self):
        if False:
            print('Hello World!')
        self.run_scenario(10107, False, self.verify_pool_cleared)

    @client_context.require_version_min(4, 0, 0)
    @client_context.require_test_commands
    def test_shutdown_in_progress(self):
        if False:
            i = 10
            return i + 15
        self.run_scenario(91, False, self.verify_pool_cleared)

    @client_context.require_version_min(4, 0, 0)
    @client_context.require_test_commands
    def test_interrupted_at_shutdown(self):
        if False:
            i = 10
            return i + 15
        self.run_scenario(11600, False, self.verify_pool_cleared)
if __name__ == '__main__':
    unittest.main()