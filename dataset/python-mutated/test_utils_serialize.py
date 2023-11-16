import dataclasses
import datetime
import json
import unittest
from decimal import Decimal
import attr
from twisted.internet import defer
from scrapy.http import Request, Response
from scrapy.utils.serialize import ScrapyJSONEncoder

class JsonEncoderTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.encoder = ScrapyJSONEncoder(sort_keys=True)

    def test_encode_decode(self):
        if False:
            i = 10
            return i + 15
        dt = datetime.datetime(2010, 1, 2, 10, 11, 12)
        dts = '2010-01-02 10:11:12'
        d = datetime.date(2010, 1, 2)
        ds = '2010-01-02'
        t = datetime.time(10, 11, 12)
        ts = '10:11:12'
        dec = Decimal('1000.12')
        decs = '1000.12'
        s = {'foo'}
        ss = ['foo']
        dt_set = {dt}
        dt_sets = [dts]
        for (input, output) in [('foo', 'foo'), (d, ds), (t, ts), (dt, dts), (dec, decs), (['foo', d], ['foo', ds]), (s, ss), (dt_set, dt_sets)]:
            self.assertEqual(self.encoder.encode(input), json.dumps(output, sort_keys=True))

    def test_encode_deferred(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIn('Deferred', self.encoder.encode(defer.Deferred()))

    def test_encode_request(self):
        if False:
            return 10
        r = Request('http://www.example.com/lala')
        rs = self.encoder.encode(r)
        self.assertIn(r.method, rs)
        self.assertIn(r.url, rs)

    def test_encode_response(self):
        if False:
            for i in range(10):
                print('nop')
        r = Response('http://www.example.com/lala')
        rs = self.encoder.encode(r)
        self.assertIn(r.url, rs)
        self.assertIn(str(r.status), rs)

    def test_encode_dataclass_item(self) -> None:
        if False:
            while True:
                i = 10

        @dataclasses.dataclass
        class TestDataClass:
            name: str
            url: str
            price: int
        item = TestDataClass(name='Product', url='http://product.org', price=1)
        encoded = self.encoder.encode(item)
        self.assertEqual(encoded, '{"name": "Product", "price": 1, "url": "http://product.org"}')

    def test_encode_attrs_item(self):
        if False:
            print('Hello World!')

        @attr.s
        class AttrsItem:
            name = attr.ib(type=str)
            url = attr.ib(type=str)
            price = attr.ib(type=int)
        item = AttrsItem(name='Product', url='http://product.org', price=1)
        encoded = self.encoder.encode(item)
        self.assertEqual(encoded, '{"name": "Product", "price": 1, "url": "http://product.org"}')