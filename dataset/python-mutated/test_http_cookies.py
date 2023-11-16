from unittest import TestCase
from urllib.parse import urlparse
from scrapy.http import Request, Response
from scrapy.http.cookies import WrappedRequest, WrappedResponse

class WrappedRequestTest(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.request = Request('http://www.example.com/page.html', headers={'Content-Type': 'text/html'})
        self.wrapped = WrappedRequest(self.request)

    def test_get_full_url(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.wrapped.get_full_url(), self.request.url)
        self.assertEqual(self.wrapped.full_url, self.request.url)

    def test_get_host(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.wrapped.get_host(), urlparse(self.request.url).netloc)
        self.assertEqual(self.wrapped.host, urlparse(self.request.url).netloc)

    def test_get_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.wrapped.get_type(), urlparse(self.request.url).scheme)
        self.assertEqual(self.wrapped.type, urlparse(self.request.url).scheme)

    def test_is_unverifiable(self):
        if False:
            return 10
        self.assertFalse(self.wrapped.is_unverifiable())
        self.assertFalse(self.wrapped.unverifiable)

    def test_is_unverifiable2(self):
        if False:
            while True:
                i = 10
        self.request.meta['is_unverifiable'] = True
        self.assertTrue(self.wrapped.is_unverifiable())
        self.assertTrue(self.wrapped.unverifiable)

    def test_get_origin_req_host(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.wrapped.origin_req_host, 'www.example.com')

    def test_has_header(self):
        if False:
            return 10
        self.assertTrue(self.wrapped.has_header('content-type'))
        self.assertFalse(self.wrapped.has_header('xxxxx'))

    def test_get_header(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.wrapped.get_header('content-type'), 'text/html')
        self.assertEqual(self.wrapped.get_header('xxxxx', 'def'), 'def')

    def test_header_items(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.wrapped.header_items(), [('Content-Type', ['text/html'])])

    def test_add_unredirected_header(self):
        if False:
            while True:
                i = 10
        self.wrapped.add_unredirected_header('hello', 'world')
        self.assertEqual(self.request.headers['hello'], b'world')

class WrappedResponseTest(TestCase):

    def setUp(self):
        if False:
            return 10
        self.response = Response('http://www.example.com/page.html', headers={'Content-TYpe': 'text/html'})
        self.wrapped = WrappedResponse(self.response)

    def test_info(self):
        if False:
            print('Hello World!')
        self.assertIs(self.wrapped.info(), self.wrapped)

    def test_get_all(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.wrapped.get_all('content-type'), ['text/html'])