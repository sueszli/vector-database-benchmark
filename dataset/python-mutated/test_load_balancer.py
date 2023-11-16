"""Test the Load Balancer unified spec tests."""
from __future__ import annotations
import gc
import os
import sys
import threading
sys.path[0:0] = ['']
from test import IntegrationTest, client_context, unittest
from test.unified_format import generate_test_classes
from test.utils import ExceptionCatchingThread, get_pool, rs_client, wait_until
TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'load_balancer')
globals().update(generate_test_classes(TEST_PATH, module=__name__))

class TestLB(IntegrationTest):
    RUN_ON_LOAD_BALANCER = True
    RUN_ON_SERVERLESS = True

    def test_connections_are_only_returned_once(self):
        if False:
            i = 10
            return i + 15
        if 'PyPy' in sys.version:
            self.skipTest('Test is flaky on PyPy')
        pool = get_pool(self.client)
        n_conns = len(pool.conns)
        self.db.test.find_one({})
        self.assertEqual(len(pool.conns), n_conns)
        list(self.db.test.aggregate([{'$limit': 1}]))
        self.assertEqual(len(pool.conns), n_conns)

    @client_context.require_load_balancer
    def test_unpin_committed_transaction(self):
        if False:
            for i in range(10):
                print('nop')
        client = rs_client()
        self.addCleanup(client.close)
        pool = get_pool(client)
        coll = client[self.db.name].test
        with client.start_session() as session:
            with session.start_transaction():
                self.assertEqual(pool.active_sockets, 0)
                coll.insert_one({}, session=session)
                self.assertEqual(pool.active_sockets, 1)
            self.assertEqual(pool.active_sockets, 1)
        self.assertEqual(pool.active_sockets, 0)

    @client_context.require_failCommand_fail_point
    def test_cursor_gc(self):
        if False:
            i = 10
            return i + 15

        def create_resource(coll):
            if False:
                return 10
            cursor = coll.find({}, batch_size=3)
            next(cursor)
            return cursor
        self._test_no_gc_deadlock(create_resource)

    @client_context.require_failCommand_fail_point
    def test_command_cursor_gc(self):
        if False:
            while True:
                i = 10

        def create_resource(coll):
            if False:
                for i in range(10):
                    print('nop')
            cursor = coll.aggregate([], batchSize=3)
            next(cursor)
            return cursor
        self._test_no_gc_deadlock(create_resource)

    def _test_no_gc_deadlock(self, create_resource):
        if False:
            return 10
        client = rs_client()
        self.addCleanup(client.close)
        pool = get_pool(client)
        coll = client[self.db.name].test
        coll.insert_many([{} for _ in range(10)])
        self.assertEqual(pool.active_sockets, 0)
        args = {'mode': {'times': 1}, 'data': {'failCommands': ['find', 'aggregate'], 'closeConnection': True}}
        with self.fail_point(args):
            resource = create_resource(coll)
            if client_context.load_balancer:
                self.assertEqual(pool.active_sockets, 1)
        thread = PoolLocker(pool)
        thread.start()
        self.assertTrue(thread.locked.wait(5), 'timed out')
        del resource
        for _ in range(3):
            gc.collect()
        thread.unlock.set()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(thread.exc)
        wait_until(lambda : pool.active_sockets == 0, 'return socket')
        coll.delete_many({})

    @client_context.require_transactions
    def test_session_gc(self):
        if False:
            print('Hello World!')
        client = rs_client()
        self.addCleanup(client.close)
        pool = get_pool(client)
        session = client.start_session()
        session.start_transaction()
        client.test_session_gc.test.find_one({}, session=session)
        if not client_context.serverless:
            self.addCleanup(self.client.admin.command, 'killSessions', [session.session_id])
        if client_context.load_balancer:
            self.assertEqual(pool.active_sockets, 1)
        thread = PoolLocker(pool)
        thread.start()
        self.assertTrue(thread.locked.wait(5), 'timed out')
        del session
        for _ in range(3):
            gc.collect()
        thread.unlock.set()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(thread.exc)
        wait_until(lambda : pool.active_sockets == 0, 'return socket')
        client[self.db.name].test.delete_many({})

class PoolLocker(ExceptionCatchingThread):

    def __init__(self, pool):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(target=self.lock_pool)
        self.pool = pool
        self.daemon = True
        self.locked = threading.Event()
        self.unlock = threading.Event()

    def lock_pool(self):
        if False:
            for i in range(10):
                print('nop')
        with self.pool.lock:
            self.locked.set()
            unlock_pool = self.unlock.wait(10)
            if not unlock_pool:
                raise Exception('timed out waiting for unlock signal: deadlock?')
if __name__ == '__main__':
    unittest.main()