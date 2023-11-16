import asyncio
from unittest import mock
from pytest import mark
from twisted.internet import defer
from twisted.internet.defer import Deferred
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from scrapy.core.downloader.middleware import DownloaderMiddlewareManager
from scrapy.exceptions import _InvalidOutput
from scrapy.http import Request, Response
from scrapy.spiders import Spider
from scrapy.utils.python import to_bytes
from scrapy.utils.test import get_crawler, get_from_asyncio_queue

class ManagerTestCase(TestCase):
    settings_dict = None

    def setUp(self):
        if False:
            return 10
        self.crawler = get_crawler(Spider, self.settings_dict)
        self.spider = self.crawler._create_spider('foo')
        self.mwman = DownloaderMiddlewareManager.from_crawler(self.crawler)
        self.crawler.stats.open_spider(self.spider)
        return self.mwman.open_spider(self.spider)

    def tearDown(self):
        if False:
            return 10
        self.crawler.stats.close_spider(self.spider, '')
        return self.mwman.close_spider(self.spider)

    def _download(self, request, response=None):
        if False:
            i = 10
            return i + 15
        "Executes downloader mw manager's download method and returns\n        the result (Request or Response) or raise exception in case of\n        failure.\n        "
        if not response:
            response = Response(request.url)

        def download_func(**kwargs):
            if False:
                i = 10
                return i + 15
            return response
        dfd = self.mwman.download(download_func, request, self.spider)
        results = []
        dfd.addBoth(results.append)
        self._wait(dfd)
        ret = results[0]
        if isinstance(ret, Failure):
            ret.raiseException()
        return ret

class DefaultsTest(ManagerTestCase):
    """Tests default behavior with default settings"""

    def test_request_response(self):
        if False:
            while True:
                i = 10
        req = Request('http://example.com/index.html')
        resp = Response(req.url, status=200)
        ret = self._download(req, resp)
        self.assertTrue(isinstance(ret, Response), 'Non-response returned')

    def test_3xx_and_invalid_gzipped_body_must_redirect(self):
        if False:
            i = 10
            return i + 15
        "Regression test for a failure when redirecting a compressed\n        request.\n\n        This happens when httpcompression middleware is executed before redirect\n        middleware and attempts to decompress a non-compressed body.\n        In particular when some website returns a 30x response with header\n        'Content-Encoding: gzip' giving as result the error below:\n\n            BadGzipFile: Not a gzipped file (...)\n\n        "
        req = Request('http://example.com')
        body = b'<p>You are being redirected</p>'
        resp = Response(req.url, status=302, body=body, headers={'Content-Length': str(len(body)), 'Content-Type': 'text/html', 'Content-Encoding': 'gzip', 'Location': 'http://example.com/login'})
        ret = self._download(request=req, response=resp)
        self.assertTrue(isinstance(ret, Request), f'Not redirected: {ret!r}')
        self.assertEqual(to_bytes(ret.url), resp.headers['Location'], 'Not redirected to location header')

    def test_200_and_invalid_gzipped_body_must_fail(self):
        if False:
            print('Hello World!')
        req = Request('http://example.com')
        body = b'<p>You are being redirected</p>'
        resp = Response(req.url, status=200, body=body, headers={'Content-Length': str(len(body)), 'Content-Type': 'text/html', 'Content-Encoding': 'gzip', 'Location': 'http://example.com/login'})
        self.assertRaises(OSError, self._download, request=req, response=resp)

class ResponseFromProcessRequestTest(ManagerTestCase):
    """Tests middleware returning a response from process_request."""

    def test_download_func_not_called(self):
        if False:
            print('Hello World!')
        resp = Response('http://example.com/index.html')

        class ResponseMiddleware:

            def process_request(self, request, spider):
                if False:
                    print('Hello World!')
                return resp
        self.mwman._add_middleware(ResponseMiddleware())
        req = Request('http://example.com/index.html')
        download_func = mock.MagicMock()
        dfd = self.mwman.download(download_func, req, self.spider)
        results = []
        dfd.addBoth(results.append)
        self._wait(dfd)
        self.assertIs(results[0], resp)
        self.assertFalse(download_func.called)

class ProcessRequestInvalidOutput(ManagerTestCase):
    """Invalid return value for process_request method should raise an exception"""

    def test_invalid_process_request(self):
        if False:
            i = 10
            return i + 15
        req = Request('http://example.com/index.html')

        class InvalidProcessRequestMiddleware:

            def process_request(self, request, spider):
                if False:
                    print('Hello World!')
                return 1
        self.mwman._add_middleware(InvalidProcessRequestMiddleware())
        download_func = mock.MagicMock()
        dfd = self.mwman.download(download_func, req, self.spider)
        results = []
        dfd.addBoth(results.append)
        self.assertIsInstance(results[0], Failure)
        self.assertIsInstance(results[0].value, _InvalidOutput)

class ProcessResponseInvalidOutput(ManagerTestCase):
    """Invalid return value for process_response method should raise an exception"""

    def test_invalid_process_response(self):
        if False:
            while True:
                i = 10
        req = Request('http://example.com/index.html')

        class InvalidProcessResponseMiddleware:

            def process_response(self, request, response, spider):
                if False:
                    i = 10
                    return i + 15
                return 1
        self.mwman._add_middleware(InvalidProcessResponseMiddleware())
        download_func = mock.MagicMock()
        dfd = self.mwman.download(download_func, req, self.spider)
        results = []
        dfd.addBoth(results.append)
        self.assertIsInstance(results[0], Failure)
        self.assertIsInstance(results[0].value, _InvalidOutput)

class ProcessExceptionInvalidOutput(ManagerTestCase):
    """Invalid return value for process_exception method should raise an exception"""

    def test_invalid_process_exception(self):
        if False:
            print('Hello World!')
        req = Request('http://example.com/index.html')

        class InvalidProcessExceptionMiddleware:

            def process_request(self, request, spider):
                if False:
                    while True:
                        i = 10
                raise Exception()

            def process_exception(self, request, exception, spider):
                if False:
                    for i in range(10):
                        print('nop')
                return 1
        self.mwman._add_middleware(InvalidProcessExceptionMiddleware())
        download_func = mock.MagicMock()
        dfd = self.mwman.download(download_func, req, self.spider)
        results = []
        dfd.addBoth(results.append)
        self.assertIsInstance(results[0], Failure)
        self.assertIsInstance(results[0].value, _InvalidOutput)

class MiddlewareUsingDeferreds(ManagerTestCase):
    """Middlewares using Deferreds should work"""

    def test_deferred(self):
        if False:
            print('Hello World!')
        resp = Response('http://example.com/index.html')

        class DeferredMiddleware:

            def cb(self, result):
                if False:
                    i = 10
                    return i + 15
                return result

            def process_request(self, request, spider):
                if False:
                    return 10
                d = Deferred()
                d.addCallback(self.cb)
                d.callback(resp)
                return d
        self.mwman._add_middleware(DeferredMiddleware())
        req = Request('http://example.com/index.html')
        download_func = mock.MagicMock()
        dfd = self.mwman.download(download_func, req, self.spider)
        results = []
        dfd.addBoth(results.append)
        self._wait(dfd)
        self.assertIs(results[0], resp)
        self.assertFalse(download_func.called)

@mark.usefixtures('reactor_pytest')
class MiddlewareUsingCoro(ManagerTestCase):
    """Middlewares using asyncio coroutines should work"""

    def test_asyncdef(self):
        if False:
            for i in range(10):
                print('nop')
        resp = Response('http://example.com/index.html')

        class CoroMiddleware:

            async def process_request(self, request, spider):
                await defer.succeed(42)
                return resp
        self.mwman._add_middleware(CoroMiddleware())
        req = Request('http://example.com/index.html')
        download_func = mock.MagicMock()
        dfd = self.mwman.download(download_func, req, self.spider)
        results = []
        dfd.addBoth(results.append)
        self._wait(dfd)
        self.assertIs(results[0], resp)
        self.assertFalse(download_func.called)

    @mark.only_asyncio()
    def test_asyncdef_asyncio(self):
        if False:
            print('Hello World!')
        resp = Response('http://example.com/index.html')

        class CoroMiddleware:

            async def process_request(self, request, spider):
                await asyncio.sleep(0.1)
                result = await get_from_asyncio_queue(resp)
                return result
        self.mwman._add_middleware(CoroMiddleware())
        req = Request('http://example.com/index.html')
        download_func = mock.MagicMock()
        dfd = self.mwman.download(download_func, req, self.spider)
        results = []
        dfd.addBoth(results.append)
        self._wait(dfd)
        self.assertIs(results[0], resp)
        self.assertFalse(download_func.called)