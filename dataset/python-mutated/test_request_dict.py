import unittest
from scrapy import Request, Spider
from scrapy.http import FormRequest, JsonRequest
from scrapy.utils.request import request_from_dict

class CustomRequest(Request):
    pass

class RequestSerializationTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.spider = TestSpider()

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        r = Request('http://www.example.com')
        self._assert_serializes_ok(r)

    def test_all_attributes(self):
        if False:
            while True:
                i = 10
        r = Request(url='http://www.example.com', callback=self.spider.parse_item, errback=self.spider.handle_error, method='POST', body=b'some body', headers={'content-encoding': 'text/html; charset=latin-1'}, cookies={'currency': 'руб'}, encoding='latin-1', priority=20, meta={'a': 'b'}, cb_kwargs={'k': 'v'}, flags=['testFlag'])
        self._assert_serializes_ok(r, spider=self.spider)

    def test_latin1_body(self):
        if False:
            return 10
        r = Request('http://www.example.com', body=b'\xa3')
        self._assert_serializes_ok(r)

    def test_utf8_body(self):
        if False:
            print('Hello World!')
        r = Request('http://www.example.com', body=b'\xc2\xa3')
        self._assert_serializes_ok(r)

    def _assert_serializes_ok(self, request, spider=None):
        if False:
            for i in range(10):
                print('nop')
        d = request.to_dict(spider=spider)
        request2 = request_from_dict(d, spider=spider)
        self._assert_same_request(request, request2)

    def _assert_same_request(self, r1, r2):
        if False:
            while True:
                i = 10
        self.assertEqual(r1.__class__, r2.__class__)
        self.assertEqual(r1.url, r2.url)
        self.assertEqual(r1.callback, r2.callback)
        self.assertEqual(r1.errback, r2.errback)
        self.assertEqual(r1.method, r2.method)
        self.assertEqual(r1.body, r2.body)
        self.assertEqual(r1.headers, r2.headers)
        self.assertEqual(r1.cookies, r2.cookies)
        self.assertEqual(r1.meta, r2.meta)
        self.assertEqual(r1.cb_kwargs, r2.cb_kwargs)
        self.assertEqual(r1.encoding, r2.encoding)
        self.assertEqual(r1._encoding, r2._encoding)
        self.assertEqual(r1.priority, r2.priority)
        self.assertEqual(r1.dont_filter, r2.dont_filter)
        self.assertEqual(r1.flags, r2.flags)
        if isinstance(r1, JsonRequest):
            self.assertEqual(r1.dumps_kwargs, r2.dumps_kwargs)

    def test_request_class(self):
        if False:
            i = 10
            return i + 15
        r1 = FormRequest('http://www.example.com')
        self._assert_serializes_ok(r1, spider=self.spider)
        r2 = CustomRequest('http://www.example.com')
        self._assert_serializes_ok(r2, spider=self.spider)
        r3 = JsonRequest('http://www.example.com', dumps_kwargs={'indent': 4})
        self._assert_serializes_ok(r3, spider=self.spider)

    def test_callback_serialization(self):
        if False:
            for i in range(10):
                print('nop')
        r = Request('http://www.example.com', callback=self.spider.parse_item, errback=self.spider.handle_error)
        self._assert_serializes_ok(r, spider=self.spider)

    def test_reference_callback_serialization(self):
        if False:
            i = 10
            return i + 15
        r = Request('http://www.example.com', callback=self.spider.parse_item_reference, errback=self.spider.handle_error_reference)
        self._assert_serializes_ok(r, spider=self.spider)
        request_dict = r.to_dict(spider=self.spider)
        self.assertEqual(request_dict['callback'], 'parse_item_reference')
        self.assertEqual(request_dict['errback'], 'handle_error_reference')

    def test_private_reference_callback_serialization(self):
        if False:
            print('Hello World!')
        r = Request('http://www.example.com', callback=self.spider._TestSpider__parse_item_reference, errback=self.spider._TestSpider__handle_error_reference)
        self._assert_serializes_ok(r, spider=self.spider)
        request_dict = r.to_dict(spider=self.spider)
        self.assertEqual(request_dict['callback'], '_TestSpider__parse_item_reference')
        self.assertEqual(request_dict['errback'], '_TestSpider__handle_error_reference')

    def test_private_callback_serialization(self):
        if False:
            for i in range(10):
                print('nop')
        r = Request('http://www.example.com', callback=self.spider._TestSpider__parse_item_private, errback=self.spider.handle_error)
        self._assert_serializes_ok(r, spider=self.spider)

    def test_mixin_private_callback_serialization(self):
        if False:
            return 10
        r = Request('http://www.example.com', callback=self.spider._TestSpiderMixin__mixin_callback, errback=self.spider.handle_error)
        self._assert_serializes_ok(r, spider=self.spider)

    def test_delegated_callback_serialization(self):
        if False:
            i = 10
            return i + 15
        r = Request('http://www.example.com', callback=self.spider.delegated_callback, errback=self.spider.handle_error)
        self._assert_serializes_ok(r, spider=self.spider)

    def test_unserializable_callback1(self):
        if False:
            while True:
                i = 10
        r = Request('http://www.example.com', callback=lambda x: x)
        self.assertRaises(ValueError, r.to_dict, spider=self.spider)

    def test_unserializable_callback2(self):
        if False:
            while True:
                i = 10
        r = Request('http://www.example.com', callback=self.spider.parse_item)
        self.assertRaises(ValueError, r.to_dict, spider=None)

    def test_unserializable_callback3(self):
        if False:
            return 10
        'Parser method is removed or replaced dynamically.'

        class MySpider(Spider):
            name = 'my_spider'

            def parse(self, response):
                if False:
                    while True:
                        i = 10
                pass
        spider = MySpider()
        r = Request('http://www.example.com', callback=spider.parse)
        setattr(spider, 'parse', None)
        self.assertRaises(ValueError, r.to_dict, spider=spider)

    def test_callback_not_available(self):
        if False:
            while True:
                i = 10
        'Callback method is not available in the spider passed to from_dict'
        spider = TestSpiderDelegation()
        r = Request('http://www.example.com', callback=spider.delegated_callback)
        d = r.to_dict(spider=spider)
        self.assertRaises(ValueError, request_from_dict, d, spider=Spider('foo'))

class TestSpiderMixin:

    def __mixin_callback(self, response):
        if False:
            i = 10
            return i + 15
        pass

class TestSpiderDelegation:

    def delegated_callback(self, response):
        if False:
            while True:
                i = 10
        pass

def parse_item(response):
    if False:
        print('Hello World!')
    pass

def handle_error(failure):
    if False:
        i = 10
        return i + 15
    pass

def private_parse_item(response):
    if False:
        for i in range(10):
            print('nop')
    pass

def private_handle_error(failure):
    if False:
        while True:
            i = 10
    pass

class TestSpider(Spider, TestSpiderMixin):
    name = 'test'
    parse_item_reference = parse_item
    handle_error_reference = handle_error
    __parse_item_reference = private_parse_item
    __handle_error_reference = private_handle_error

    def __init__(self):
        if False:
            while True:
                i = 10
        self.delegated_callback = TestSpiderDelegation().delegated_callback

    def parse_item(self, response):
        if False:
            return 10
        pass

    def handle_error(self, failure):
        if False:
            for i in range(10):
                print('nop')
        pass

    def __parse_item_private(self, response):
        if False:
            for i in range(10):
                print('nop')
        pass