"""Test MONGODB-AWS Authentication."""
from __future__ import annotations
import os
import sys
import unittest
from unittest.mock import patch
sys.path[0:0] = ['']
from pymongo_auth_aws import AwsCredential, auth
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from pymongo.uri_parser import parse_uri

class TestAuthAWS(unittest.TestCase):
    uri: str

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.uri = os.environ['MONGODB_URI']

    def test_should_fail_without_credentials(self):
        if False:
            for i in range(10):
                print('nop')
        if '@' not in self.uri:
            self.skipTest('MONGODB_URI already has no credentials')
        hosts = ['{}:{}'.format(*addr) for addr in parse_uri(self.uri)['nodelist']]
        self.assertTrue(hosts)
        with MongoClient(hosts) as client:
            with self.assertRaises(OperationFailure):
                client.aws.test.find_one()

    def test_should_fail_incorrect_credentials(self):
        if False:
            while True:
                i = 10
        with MongoClient(self.uri, username='fake', password='fake', authMechanism='MONGODB-AWS') as client:
            with self.assertRaises(OperationFailure):
                client.get_database().test.find_one()

    def test_connect_uri(self):
        if False:
            print('Hello World!')
        with MongoClient(self.uri) as client:
            client.get_database().test.find_one()

    def setup_cache(self):
        if False:
            while True:
                i = 10
        if os.environ.get('AWS_ACCESS_KEY_ID', None) or '@' in self.uri:
            self.skipTest('Not testing cached credentials')
        if not hasattr(auth, 'set_cached_credentials'):
            self.skipTest('Cached credentials not available')
        auth.set_cached_credentials(None)
        self.assertEqual(auth.get_cached_credentials(), None)
        client = MongoClient(self.uri)
        client.get_database().test.find_one()
        client.close()
        return auth.get_cached_credentials()

    def test_cache_credentials(self):
        if False:
            while True:
                i = 10
        creds = self.setup_cache()
        self.assertIsNotNone(creds)

    def test_cache_about_to_expire(self):
        if False:
            print('Hello World!')
        creds = self.setup_cache()
        client = MongoClient(self.uri)
        self.addCleanup(client.close)
        creds = auth.get_cached_credentials()
        assert creds is not None
        creds = AwsCredential(creds.username, creds.password, creds.token, lambda x: True)
        auth.set_cached_credentials(creds)
        client.get_database().test.find_one()
        new_creds = auth.get_cached_credentials()
        self.assertNotEqual(creds, new_creds)

    def test_poisoned_cache(self):
        if False:
            i = 10
            return i + 15
        creds = self.setup_cache()
        client = MongoClient(self.uri)
        self.addCleanup(client.close)
        assert creds is not None
        creds = AwsCredential('a' * 24, 'b' * 24, 'c' * 24)
        auth.set_cached_credentials(creds)
        with self.assertRaises(OperationFailure):
            client.get_database().test.find_one()
        self.assertEqual(auth.get_cached_credentials(), None)
        client.get_database().test.find_one()
        self.assertNotEqual(auth.get_cached_credentials(), None)

    def test_environment_variables_ignored(self):
        if False:
            i = 10
            return i + 15
        creds = self.setup_cache()
        self.assertIsNotNone(creds)
        os.environ.copy()
        client = MongoClient(self.uri)
        self.addCleanup(client.close)
        client.get_database().test.find_one()
        self.assertIsNotNone(auth.get_cached_credentials())
        mock_env = {'AWS_ACCESS_KEY_ID': 'foo', 'AWS_SECRET_ACCESS_KEY': 'bar', 'AWS_SESSION_TOKEN': 'baz'}
        with patch.dict('os.environ', mock_env):
            self.assertEqual(os.environ['AWS_ACCESS_KEY_ID'], 'foo')
            client.get_database().test.find_one()
        auth.set_cached_credentials(None)
        client2 = MongoClient(self.uri)
        self.addCleanup(client2.close)
        with patch.dict('os.environ', mock_env):
            self.assertEqual(os.environ['AWS_ACCESS_KEY_ID'], 'foo')
            with self.assertRaises(OperationFailure):
                client2.get_database().test.find_one()

    def test_no_cache_environment_variables(self):
        if False:
            print('Hello World!')
        creds = self.setup_cache()
        self.assertIsNotNone(creds)
        auth.set_cached_credentials(None)
        mock_env = {'AWS_ACCESS_KEY_ID': creds.username, 'AWS_SECRET_ACCESS_KEY': creds.password}
        if creds.token:
            mock_env['AWS_SESSION_TOKEN'] = creds.token
        client = MongoClient(self.uri)
        self.addCleanup(client.close)
        with patch.dict(os.environ, mock_env):
            self.assertEqual(os.environ['AWS_ACCESS_KEY_ID'], creds.username)
            client.get_database().test.find_one()
        self.assertIsNone(auth.get_cached_credentials())
        mock_env['AWS_ACCESS_KEY_ID'] = 'foo'
        client2 = MongoClient(self.uri)
        self.addCleanup(client2.close)
        with patch.dict('os.environ', mock_env), self.assertRaises(OperationFailure):
            self.assertEqual(os.environ['AWS_ACCESS_KEY_ID'], 'foo')
            client2.get_database().test.find_one()

class TestAWSLambdaExamples(unittest.TestCase):

    def test_shared_client(self):
        if False:
            return 10
        import os
        from pymongo import MongoClient
        client = MongoClient(host=os.environ['MONGODB_URI'])

        def lambda_handler(event, context):
            if False:
                print('Hello World!')
            return client.db.command('ping')

    def test_IAM_auth(self):
        if False:
            for i in range(10):
                print('nop')
        import os
        from pymongo import MongoClient
        client = MongoClient(host=os.environ['MONGODB_URI'], authSource='$external', authMechanism='MONGODB-AWS')

        def lambda_handler(event, context):
            if False:
                while True:
                    i = 10
            return client.db.command('ping')
if __name__ == '__main__':
    unittest.main()