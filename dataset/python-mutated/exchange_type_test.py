"""
Tests for pika.exchange_type

"""
import unittest
from pika.exchange_type import ExchangeType

class ExchangeTypeTests(unittest.TestCase):

    def test_exchange_type_direct(self):
        if False:
            while True:
                i = 10
        self.assertEqual(ExchangeType.direct.value, 'direct')

    def test_exchange_type_fanout(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(ExchangeType.fanout.value, 'fanout')