from __future__ import annotations
import os
import sys
sys.path[0:0] = ['']
from test import IntegrationTest, client_context, unittest
from test.unified_format import generate_test_classes
from test.utils import OvertCommandListener, rs_or_single_client
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi, ServerApiVersion
TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'versioned-api')
globals().update(generate_test_classes(TEST_PATH, module=__name__))

class TestServerApi(IntegrationTest):
    RUN_ON_LOAD_BALANCER = True
    RUN_ON_SERVERLESS = True

    def test_server_api_defaults(self):
        if False:
            print('Hello World!')
        api = ServerApi(ServerApiVersion.V1)
        self.assertEqual(api.version, '1')
        self.assertIsNone(api.strict)
        self.assertIsNone(api.deprecation_errors)

    def test_server_api_explicit_false(self):
        if False:
            return 10
        api = ServerApi('1', strict=False, deprecation_errors=False)
        self.assertEqual(api.version, '1')
        self.assertFalse(api.strict)
        self.assertFalse(api.deprecation_errors)

    def test_server_api_strict(self):
        if False:
            for i in range(10):
                print('nop')
        api = ServerApi('1', strict=True, deprecation_errors=True)
        self.assertEqual(api.version, '1')
        self.assertTrue(api.strict)
        self.assertTrue(api.deprecation_errors)

    def test_server_api_validation(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            ServerApi('2')
        with self.assertRaises(TypeError):
            ServerApi('1', strict='not-a-bool')
        with self.assertRaises(TypeError):
            ServerApi('1', deprecation_errors='not-a-bool')
        with self.assertRaises(TypeError):
            MongoClient(server_api='not-a-ServerApi')

    def assertServerApi(self, event):
        if False:
            i = 10
            return i + 15
        self.assertIn('apiVersion', event.command)
        self.assertEqual(event.command['apiVersion'], '1')

    def assertNoServerApi(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.assertNotIn('apiVersion', event.command)

    def assertServerApiInAllCommands(self, events):
        if False:
            print('Hello World!')
        for event in events:
            self.assertServerApi(event)

    @client_context.require_version_min(4, 7)
    def test_command_options(self):
        if False:
            print('Hello World!')
        listener = OvertCommandListener()
        client = rs_or_single_client(server_api=ServerApi('1'), event_listeners=[listener])
        self.addCleanup(client.close)
        coll = client.test.test
        coll.insert_many([{} for _ in range(100)])
        self.addCleanup(coll.delete_many, {})
        list(coll.find(batch_size=25))
        client.admin.command('ping')
        self.assertServerApiInAllCommands(listener.started_events)

    @client_context.require_version_min(4, 7)
    @client_context.require_transactions
    def test_command_options_txn(self):
        if False:
            print('Hello World!')
        listener = OvertCommandListener()
        client = rs_or_single_client(server_api=ServerApi('1'), event_listeners=[listener])
        self.addCleanup(client.close)
        coll = client.test.test
        coll.insert_many([{} for _ in range(100)])
        self.addCleanup(coll.delete_many, {})
        listener.reset()
        with client.start_session() as s, s.start_transaction():
            coll.insert_many([{} for _ in range(100)], session=s)
            list(coll.find(batch_size=25, session=s))
            client.test.command('find', 'test', session=s)
            self.assertServerApiInAllCommands(listener.started_events)
if __name__ == '__main__':
    unittest.main()