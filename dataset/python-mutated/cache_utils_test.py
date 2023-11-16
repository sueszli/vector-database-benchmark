import math
import unittest
from datetime import timedelta
from typing import Any
from parameterized import parameterized
from streamlit.runtime.caching.cache_errors import BadTTLStringError
from streamlit.runtime.caching.cache_utils import ttl_to_seconds
NORMAL_PARAMS = [('float', 3.5, 3.5), ('timedelta', timedelta(minutes=3), 60 * 3), ('str 1 arg', '1d', 24 * 60 * 60), ('str 2 args', '1d23h', 24 * 60 * 60 + 23 * 60 * 60), ('complex str 3 args', '1 day 23hr 45minutes', 24 * 60 * 60 + 23 * 60 * 60 + 45 * 60), ('str 2 args with float', '1.5d23.5h', 1.5 * 24 * 60 * 60 + 23.5 * 60 * 60)]

class CacheUtilsTest(unittest.TestCase):

    @parameterized.expand([*NORMAL_PARAMS, ('None', None, math.inf)])
    def test_ttl_to_seconds_coerced(self, _, input_value: Any, expected_seconds: float):
        if False:
            for i in range(10):
                print('nop')
        'Test the various types of input that ttl_to_seconds accepts.'
        self.assertEqual(expected_seconds, ttl_to_seconds(input_value))

    @parameterized.expand([*NORMAL_PARAMS, ('None', None, None)])
    def test_ttl_to_seconds_not_coerced(self, _, input_value: Any, expected_seconds: float):
        if False:
            print('Hello World!')
        'Test the various types of input that ttl_to_seconds accepts.'
        self.assertEqual(expected_seconds, ttl_to_seconds(input_value, coerce_none_to_inf=False))

    def test_ttl_str_exception(self):
        if False:
            while True:
                i = 10
        'Test that a badly-formatted TTL string raises an exception.'
        with self.assertRaises(BadTTLStringError):
            ttl_to_seconds('')
        with self.assertRaises(BadTTLStringError):
            ttl_to_seconds('1 flecond')