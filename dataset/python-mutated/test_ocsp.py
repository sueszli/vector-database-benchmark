"""Test OCSP."""
from __future__ import annotations
import logging
import os
import sys
import unittest
sys.path[0:0] = ['']
import pymongo
from pymongo.errors import ServerSelectionTimeoutError
CA_FILE = os.environ.get('CA_FILE')
OCSP_TLS_SHOULD_SUCCEED = os.environ.get('OCSP_TLS_SHOULD_SUCCEED') == 'true'
FORMAT = '%(asctime)s %(levelname)s %(module)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
if sys.platform == 'win32':
    TIMEOUT_MS = 5000
else:
    TIMEOUT_MS = 500

def _connect(options):
    if False:
        for i in range(10):
            print('nop')
    uri = 'mongodb://localhost:27017/?serverSelectionTimeoutMS={}&tlsCAFile={}&{}'.format(TIMEOUT_MS, CA_FILE, options)
    print(uri)
    client = pymongo.MongoClient(uri)
    client.admin.command('ping')

class TestOCSP(unittest.TestCase):

    def test_tls_insecure(self):
        if False:
            for i in range(10):
                print('nop')
        options = 'tls=true&tlsInsecure=true'
        _connect(options)

    def test_allow_invalid_certificates(self):
        if False:
            print('Hello World!')
        options = 'tls=true&tlsAllowInvalidCertificates=true'
        _connect(options)

    def test_tls(self):
        if False:
            print('Hello World!')
        options = 'tls=true'
        if not OCSP_TLS_SHOULD_SUCCEED:
            self.assertRaisesRegex(ServerSelectionTimeoutError, 'invalid status response', _connect, options)
        else:
            _connect(options)
if __name__ == '__main__':
    unittest.main()