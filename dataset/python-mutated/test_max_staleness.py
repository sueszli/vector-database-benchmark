"""Test maxStalenessSeconds support."""
from __future__ import annotations
import os
import sys
import time
import warnings
sys.path[0:0] = ['']
from test import client_context, unittest
from test.utils import rs_or_single_client
from test.utils_selection_tests import create_selection_tests
from pymongo import MongoClient
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import writable_server_selector
_TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'max_staleness')

class TestAllScenarios(create_selection_tests(_TEST_PATH)):
    pass

class TestMaxStaleness(unittest.TestCase):

    def test_max_staleness(self):
        if False:
            print('Hello World!')
        client = MongoClient()
        self.assertEqual(-1, client.read_preference.max_staleness)
        client = MongoClient('mongodb://a/?readPreference=secondary')
        self.assertEqual(-1, client.read_preference.max_staleness)
        with self.assertRaises(ConfigurationError):
            MongoClient('mongodb://a/?maxStalenessSeconds=120')
        with self.assertRaises(ConfigurationError):
            MongoClient('mongodb://a/?readPreference=primary&maxStalenessSeconds=120')
        client = MongoClient('mongodb://host/?maxStalenessSeconds=-1')
        self.assertEqual(-1, client.read_preference.max_staleness)
        client = MongoClient('mongodb://host/?readPreference=primary&maxStalenessSeconds=-1')
        self.assertEqual(-1, client.read_preference.max_staleness)
        client = MongoClient('mongodb://host/?readPreference=secondary&maxStalenessSeconds=120')
        self.assertEqual(120, client.read_preference.max_staleness)
        client = MongoClient('mongodb://a/?readPreference=secondary&maxStalenessSeconds=1')
        self.assertEqual(1, client.read_preference.max_staleness)
        client = MongoClient('mongodb://a/?readPreference=secondary&maxStalenessSeconds=-1')
        self.assertEqual(-1, client.read_preference.max_staleness)
        client = MongoClient(maxStalenessSeconds=-1, readPreference='nearest')
        self.assertEqual(-1, client.read_preference.max_staleness)
        with self.assertRaises(TypeError):
            MongoClient(maxStalenessSeconds=None, readPreference='nearest')

    def test_max_staleness_float(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError) as ctx:
            rs_or_single_client(maxStalenessSeconds=1.5, readPreference='nearest')
        self.assertIn('must be an integer', str(ctx.exception))
        with warnings.catch_warnings(record=True) as ctx:
            warnings.simplefilter('always')
            client = MongoClient('mongodb://host/?maxStalenessSeconds=1.5&readPreference=nearest')
            self.assertEqual(-1, client.read_preference.max_staleness)
            self.assertIn('must be an integer', str(ctx[0]))

    def test_max_staleness_zero(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError) as ctx:
            rs_or_single_client(maxStalenessSeconds=0, readPreference='nearest')
        self.assertIn('must be a positive integer', str(ctx.exception))
        with warnings.catch_warnings(record=True) as ctx:
            warnings.simplefilter('always')
            client = MongoClient('mongodb://host/?maxStalenessSeconds=0&readPreference=nearest')
            self.assertEqual(-1, client.read_preference.max_staleness)
            self.assertIn('must be a positive integer', str(ctx[0]))

    @client_context.require_replica_set
    def test_last_write_date(self):
        if False:
            i = 10
            return i + 15
        client = rs_or_single_client(heartbeatFrequencyMS=500)
        client.pymongo_test.test.insert_one({})
        time.sleep(1)
        server = client._topology.select_server(writable_server_selector)
        first = server.description.last_write_date
        self.assertTrue(first)
        time.sleep(1)
        client.pymongo_test.test.insert_one({})
        time.sleep(1)
        server = client._topology.select_server(writable_server_selector)
        second = server.description.last_write_date
        assert first is not None
        assert second is not None
        self.assertGreater(second, first)
        self.assertLess(second, first + 10)
if __name__ == '__main__':
    unittest.main()