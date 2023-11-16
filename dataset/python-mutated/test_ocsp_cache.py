"""Test the pymongo ocsp_support module."""
from __future__ import annotations
import random
import sys
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from os import urandom
from time import sleep
from typing import Any
sys.path[0:0] = ['']
from test import unittest
from pymongo.ocsp_cache import _OCSPCache

class TestOcspCache(unittest.TestCase):
    MockHashAlgorithm: Any
    MockOcspRequest: Any
    MockOcspResponse: Any

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.MockHashAlgorithm = namedtuple('MockHashAlgorithm', ['name'])
        cls.MockOcspRequest = namedtuple('MockOcspRequest', ['hash_algorithm', 'issuer_name_hash', 'issuer_key_hash', 'serial_number'])
        cls.MockOcspResponse = namedtuple('MockOcspResponse', ['this_update', 'next_update'])

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.cache = _OCSPCache()

    def _create_mock_request(self):
        if False:
            print('Hello World!')
        hash_algorithm = self.MockHashAlgorithm(random.choice(['sha1', 'md5', 'sha256']))
        issuer_name_hash = urandom(8)
        issuer_key_hash = urandom(8)
        serial_number = random.randint(0, 10 ** 10)
        return self.MockOcspRequest(hash_algorithm=hash_algorithm, issuer_name_hash=issuer_name_hash, issuer_key_hash=issuer_key_hash, serial_number=serial_number)

    def _create_mock_response(self, this_update_delta_seconds, next_update_delta_seconds):
        if False:
            for i in range(10):
                print('nop')
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        this_update = now + timedelta(seconds=this_update_delta_seconds)
        if next_update_delta_seconds is not None:
            next_update = now + timedelta(seconds=next_update_delta_seconds)
        else:
            next_update = None
        return self.MockOcspResponse(this_update=this_update, next_update=next_update)

    def _add_mock_cache_entry(self, mock_request, mock_response):
        if False:
            print('Hello World!')
        key = self.cache._get_cache_key(mock_request)
        self.cache._data[key] = mock_response

    def test_simple(self):
        if False:
            return 10
        request = self._create_mock_request()
        response = self._create_mock_response(-10, +3600)
        self._add_mock_cache_entry(request, response)
        self.assertEqual(self.cache[request], response)
        response_1 = self._create_mock_response(-20, +1800)
        self.cache[request] = response_1
        self.assertEqual(self.cache[request], response)
        response_2 = self._create_mock_response(+20, +1800)
        self.cache[request] = response_2
        self.assertEqual(self.cache[request], response)
        response_3 = self._create_mock_response(-10, -5)
        self.cache[request] = response_3
        self.assertEqual(self.cache[request], response)
        response_new = self._create_mock_response(-5, +7200)
        self.cache[request] = response_new
        self.assertEqual(self.cache[request], response_new)
        response_notset = self._create_mock_response(-5, None)
        self.cache[request] = response_notset
        with self.assertRaises(KeyError):
            _ = self.cache[request]

    def test_invalidate(self):
        if False:
            i = 10
            return i + 15
        request = self._create_mock_request()
        response = self._create_mock_response(-10, +0.25)
        self._add_mock_cache_entry(request, response)
        self.assertEqual(self.cache[request], response)
        sleep(0.5)
        with self.assertRaises(KeyError):
            _ = self.cache[request]

    def test_non_existent(self):
        if False:
            i = 10
            return i + 15
        request = self._create_mock_request()
        response = self._create_mock_response(-10, +10)
        self._add_mock_cache_entry(request, response)
        with self.assertRaises(KeyError):
            _ = self.cache[self._create_mock_request()]
if __name__ == '__main__':
    unittest.main()