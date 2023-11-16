"""Tests for Decimal128."""
from __future__ import annotations
import pickle
import sys
from decimal import Decimal
sys.path[0:0] = ['']
from test import client_context, unittest
from bson.decimal128 import Decimal128, create_decimal128_context

class TestDecimal128(unittest.TestCase):

    @client_context.require_connection
    def test_round_trip(self):
        if False:
            for i in range(10):
                print('nop')
        coll = client_context.client.pymongo_test.test
        coll.drop()
        dec128 = Decimal128.from_bid(b'\x00@cR\xbf\xc6\x01\x00\x00\x00\x00\x00\x00\x00\x1c0')
        coll.insert_one({'dec128': dec128})
        doc = coll.find_one({'dec128': dec128})
        assert doc is not None
        self.assertIsNotNone(doc)
        self.assertEqual(doc['dec128'], dec128)

    def test_pickle(self):
        if False:
            while True:
                i = 10
        dec128 = Decimal128.from_bid(b'\x00@cR\xbf\xc6\x01\x00\x00\x00\x00\x00\x00\x00\x1c0')
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            pkl = pickle.dumps(dec128, protocol=protocol)
            self.assertEqual(dec128, pickle.loads(pkl))

    def test_special(self):
        if False:
            i = 10
            return i + 15
        dnan = Decimal('NaN')
        dnnan = Decimal('-NaN')
        dsnan = Decimal('sNaN')
        dnsnan = Decimal('-sNaN')
        dnan128 = Decimal128(dnan)
        dnnan128 = Decimal128(dnnan)
        dsnan128 = Decimal128(dsnan)
        dnsnan128 = Decimal128(dnsnan)
        self.assertEqual(str(dnan), str(dnan128.to_decimal()))
        self.assertEqual(str(dnnan), str(dnnan128.to_decimal()))
        self.assertEqual(str(dsnan), str(dsnan128.to_decimal()))
        self.assertEqual(str(dnsnan), str(dnsnan128.to_decimal()))

    def test_decimal128_context(self):
        if False:
            while True:
                i = 10
        ctx = create_decimal128_context()
        self.assertEqual('NaN', str(ctx.copy().create_decimal('.13.1')))
        self.assertEqual('Infinity', str(ctx.copy().create_decimal('1E6145')))
        self.assertEqual('0E-6176', str(ctx.copy().create_decimal('1E-6177')))
if __name__ == '__main__':
    unittest.main()