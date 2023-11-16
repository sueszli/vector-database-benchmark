import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Type
from unittest import SkipTest, mock
from testfixtures import LogCapture
from twisted.cred import checkers, credentials, portal
from twisted.internet import defer, error, reactor
from twisted.protocols.policies import WrappingFactory
from twisted.trial import unittest
from twisted.web import resource, server, static, util
from twisted.web._newclient import ResponseFailed
from twisted.web.http import _DataLoss
from w3lib.url import path_to_file_uri
from scrapy.core.downloader.handlers import DownloadHandlers
from scrapy.core.downloader.handlers.datauri import DataURIDownloadHandler
from scrapy.core.downloader.handlers.file import FileDownloadHandler
from scrapy.core.downloader.handlers.http import HTTPDownloadHandler
from scrapy.core.downloader.handlers.http10 import HTTP10DownloadHandler
from scrapy.core.downloader.handlers.http11 import HTTP11DownloadHandler
from scrapy.core.downloader.handlers.s3 import S3DownloadHandler
from scrapy.exceptions import NotConfigured
from scrapy.http import Headers, HtmlResponse, Request
from scrapy.http.response.text import TextResponse
from scrapy.responsetypes import responsetypes
from scrapy.spiders import Spider
from scrapy.utils.misc import create_instance
from scrapy.utils.python import to_bytes
from scrapy.utils.test import get_crawler, skip_if_no_boto
from tests import NON_EXISTING_RESOLVABLE
from tests.mockserver import Echo, ForeverTakingResource, HostHeaderResource, MockServer, NoLengthResource, PayloadResource, ssl_context_factory
from tests.spiders import SingleRequestSpider

class DummyDH:
    lazy = False

class DummyLazyDH:
    pass

class OffDH:
    lazy = False

    def __init__(self, crawler):
        if False:
            i = 10
            return i + 15
        raise NotConfigured

    @classmethod
    def from_crawler(cls, crawler):
        if False:
            return 10
        return cls(crawler)

class LoadTestCase(unittest.TestCase):

    def test_enabled_handler(self):
        if False:
            print('Hello World!')
        handlers = {'scheme': DummyDH}
        crawler = get_crawler(settings_dict={'DOWNLOAD_HANDLERS': handlers})
        dh = DownloadHandlers(crawler)
        self.assertIn('scheme', dh._schemes)
        self.assertIn('scheme', dh._handlers)
        self.assertNotIn('scheme', dh._notconfigured)

    def test_not_configured_handler(self):
        if False:
            print('Hello World!')
        handlers = {'scheme': OffDH}
        crawler = get_crawler(settings_dict={'DOWNLOAD_HANDLERS': handlers})
        dh = DownloadHandlers(crawler)
        self.assertIn('scheme', dh._schemes)
        self.assertNotIn('scheme', dh._handlers)
        self.assertIn('scheme', dh._notconfigured)

    def test_disabled_handler(self):
        if False:
            for i in range(10):
                print('nop')
        handlers = {'scheme': None}
        crawler = get_crawler(settings_dict={'DOWNLOAD_HANDLERS': handlers})
        dh = DownloadHandlers(crawler)
        self.assertNotIn('scheme', dh._schemes)
        for scheme in handlers:
            dh._get_handler(scheme)
        self.assertNotIn('scheme', dh._handlers)
        self.assertIn('scheme', dh._notconfigured)

    def test_lazy_handlers(self):
        if False:
            i = 10
            return i + 15
        handlers = {'scheme': DummyLazyDH}
        crawler = get_crawler(settings_dict={'DOWNLOAD_HANDLERS': handlers})
        dh = DownloadHandlers(crawler)
        self.assertIn('scheme', dh._schemes)
        self.assertNotIn('scheme', dh._handlers)
        for scheme in handlers:
            dh._get_handler(scheme)
        self.assertIn('scheme', dh._handlers)
        self.assertNotIn('scheme', dh._notconfigured)

class FileTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tmpname = Path(self.mktemp() + '^')
        Path(self.tmpname).write_text('0123456789', encoding='utf-8')
        handler = create_instance(FileDownloadHandler, None, get_crawler())
        self.download_request = handler.download_request

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tmpname.unlink()

    def test_download(self):
        if False:
            print('Hello World!')

        def _test(response):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(response.url, request.url)
            self.assertEqual(response.status, 200)
            self.assertEqual(response.body, b'0123456789')
            self.assertEqual(response.protocol, None)
        request = Request(path_to_file_uri(str(self.tmpname)))
        assert request.url.upper().endswith('%5E')
        return self.download_request(request, Spider('foo')).addCallback(_test)

    def test_non_existent(self):
        if False:
            for i in range(10):
                print('nop')
        request = Request(path_to_file_uri(self.mktemp()))
        d = self.download_request(request, Spider('foo'))
        return self.assertFailure(d, OSError)

class ContentLengthHeaderResource(resource.Resource):
    """
    A testing resource which renders itself as the value of the Content-Length
    header from the request.
    """

    def render(self, request):
        if False:
            i = 10
            return i + 15
        return request.requestHeaders.getRawHeaders(b'content-length')[0]

class ChunkedResource(resource.Resource):

    def render(self, request):
        if False:
            i = 10
            return i + 15

        def response():
            if False:
                while True:
                    i = 10
            request.write(b'chunked ')
            request.write(b'content\n')
            request.finish()
        reactor.callLater(0, response)
        return server.NOT_DONE_YET

class BrokenChunkedResource(resource.Resource):

    def render(self, request):
        if False:
            for i in range(10):
                print('nop')

        def response():
            if False:
                return 10
            request.write(b'chunked ')
            request.write(b'content\n')
            request.chunked = False
            closeConnection(request)
        reactor.callLater(0, response)
        return server.NOT_DONE_YET

class BrokenDownloadResource(resource.Resource):

    def render(self, request):
        if False:
            print('Hello World!')

        def response():
            if False:
                for i in range(10):
                    print('nop')
            request.setHeader(b'Content-Length', b'20')
            request.write(b'partial')
            closeConnection(request)
        reactor.callLater(0, response)
        return server.NOT_DONE_YET

def closeConnection(request):
    if False:
        i = 10
        return i + 15
    if hasattr(request.channel, 'loseConnection'):
        request.channel.loseConnection()
    else:
        request.channel.transport.loseConnection()
    request.finish()

class EmptyContentTypeHeaderResource(resource.Resource):
    """
    A testing resource which renders itself as the value of request body
    without content-type header in response.
    """

    def render(self, request):
        if False:
            while True:
                i = 10
        request.setHeader('content-type', '')
        return request.content.read()

class LargeChunkedFileResource(resource.Resource):

    def render(self, request):
        if False:
            while True:
                i = 10

        def response():
            if False:
                return 10
            for i in range(1024):
                request.write(b'x' * 1024)
            request.finish()
        reactor.callLater(0, response)
        return server.NOT_DONE_YET

class DuplicateHeaderResource(resource.Resource):

    def render(self, request):
        if False:
            i = 10
            return i + 15
        request.responseHeaders.setRawHeaders(b'Set-Cookie', [b'a=b', b'c=d'])
        return b''

class HttpTestCase(unittest.TestCase):
    scheme = 'http'
    download_handler_cls: Type = HTTPDownloadHandler
    keyfile = 'keys/localhost.key'
    certfile = 'keys/localhost.crt'

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tmpname = Path(self.mktemp())
        self.tmpname.mkdir()
        (self.tmpname / 'file').write_bytes(b'0123456789')
        r = static.File(str(self.tmpname))
        r.putChild(b'redirect', util.Redirect(b'/file'))
        r.putChild(b'wait', ForeverTakingResource())
        r.putChild(b'hang-after-headers', ForeverTakingResource(write=True))
        r.putChild(b'nolength', NoLengthResource())
        r.putChild(b'host', HostHeaderResource())
        r.putChild(b'payload', PayloadResource())
        r.putChild(b'broken', BrokenDownloadResource())
        r.putChild(b'chunked', ChunkedResource())
        r.putChild(b'broken-chunked', BrokenChunkedResource())
        r.putChild(b'contentlength', ContentLengthHeaderResource())
        r.putChild(b'nocontenttype', EmptyContentTypeHeaderResource())
        r.putChild(b'largechunkedfile', LargeChunkedFileResource())
        r.putChild(b'duplicate-header', DuplicateHeaderResource())
        r.putChild(b'echo', Echo())
        self.site = server.Site(r, timeout=None)
        self.wrapper = WrappingFactory(self.site)
        self.host = 'localhost'
        if self.scheme == 'https':
            self.port = reactor.listenSSL(0, self.site, ssl_context_factory(self.keyfile, self.certfile), interface=self.host)
        else:
            self.port = reactor.listenTCP(0, self.wrapper, interface=self.host)
        self.portno = self.port.getHost().port
        self.download_handler = create_instance(self.download_handler_cls, None, get_crawler())
        self.download_request = self.download_handler.download_request

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            print('Hello World!')
        yield self.port.stopListening()
        if hasattr(self.download_handler, 'close'):
            yield self.download_handler.close()
        shutil.rmtree(self.tmpname)

    def getURL(self, path):
        if False:
            return 10
        return f'{self.scheme}://{self.host}:{self.portno}/{path}'

    def test_download(self):
        if False:
            for i in range(10):
                print('nop')
        request = Request(self.getURL('file'))
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b'0123456789')
        return d

    def test_download_head(self):
        if False:
            return 10
        request = Request(self.getURL('file'), method='HEAD')
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b'')
        return d

    def test_redirect_status(self):
        if False:
            return 10
        request = Request(self.getURL('redirect'))
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.status)
        d.addCallback(self.assertEqual, 302)
        return d

    def test_redirect_status_head(self):
        if False:
            for i in range(10):
                print('nop')
        request = Request(self.getURL('redirect'), method='HEAD')
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.status)
        d.addCallback(self.assertEqual, 302)
        return d

    @defer.inlineCallbacks
    def test_timeout_download_from_spider_nodata_rcvd(self):
        if False:
            print('Hello World!')
        if self.reactor_pytest == 'asyncio' and sys.platform == 'win32':
            raise unittest.SkipTest('This test produces DirtyReactorAggregateError on Windows with asyncio')
        spider = Spider('foo')
        meta = {'download_timeout': 0.5}
        request = Request(self.getURL('wait'), meta=meta)
        d = self.download_request(request, spider)
        yield self.assertFailure(d, defer.TimeoutError, error.TimeoutError)

    @defer.inlineCallbacks
    def test_timeout_download_from_spider_server_hangs(self):
        if False:
            print('Hello World!')
        if self.reactor_pytest == 'asyncio' and sys.platform == 'win32':
            raise unittest.SkipTest('This test produces DirtyReactorAggregateError on Windows with asyncio')
        spider = Spider('foo')
        meta = {'download_timeout': 0.5}
        request = Request(self.getURL('hang-after-headers'), meta=meta)
        d = self.download_request(request, spider)
        yield self.assertFailure(d, defer.TimeoutError, error.TimeoutError)

    def test_host_header_not_in_request_headers(self):
        if False:
            return 10

        def _test(response):
            if False:
                while True:
                    i = 10
            self.assertEqual(response.body, to_bytes(f'{self.host}:{self.portno}'))
            self.assertEqual(request.headers, {})
        request = Request(self.getURL('host'))
        return self.download_request(request, Spider('foo')).addCallback(_test)

    def test_host_header_seted_in_request_headers(self):
        if False:
            for i in range(10):
                print('nop')
        host = self.host + ':' + str(self.portno)

        def _test(response):
            if False:
                print('Hello World!')
            self.assertEqual(response.body, host.encode())
            self.assertEqual(request.headers.get('Host'), host.encode())
        request = Request(self.getURL('host'), headers={'Host': host})
        return self.download_request(request, Spider('foo')).addCallback(_test)
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b'localhost')
        return d

    def test_content_length_zero_bodyless_post_request_headers(self):
        if False:
            i = 10
            return i + 15
        'Tests if "Content-Length: 0" is sent for bodyless POST requests.\n\n        This is not strictly required by HTTP RFCs but can cause trouble\n        for some web servers.\n        See:\n        https://github.com/scrapy/scrapy/issues/823\n        https://issues.apache.org/jira/browse/TS-2902\n        https://github.com/kennethreitz/requests/issues/405\n        https://bugs.python.org/issue14721\n        '

        def _test(response):
            if False:
                print('Hello World!')
            self.assertEqual(response.body, b'0')
        request = Request(self.getURL('contentlength'), method='POST')
        return self.download_request(request, Spider('foo')).addCallback(_test)

    def test_content_length_zero_bodyless_post_only_one(self):
        if False:
            print('Hello World!')

        def _test(response):
            if False:
                return 10
            import json
            headers = Headers(json.loads(response.text)['headers'])
            contentlengths = headers.getlist('Content-Length')
            self.assertEqual(len(contentlengths), 1)
            self.assertEqual(contentlengths, [b'0'])
        request = Request(self.getURL('echo'), method='POST')
        return self.download_request(request, Spider('foo')).addCallback(_test)

    def test_payload(self):
        if False:
            i = 10
            return i + 15
        body = b'1' * 100
        request = Request(self.getURL('payload'), method='POST', body=body)
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, body)
        return d

    def test_response_header_content_length(self):
        if False:
            return 10
        request = Request(self.getURL('file'), method=b'GET')
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.headers[b'content-length'])
        d.addCallback(self.assertEqual, b'159')
        return d

    def _test_response_class(self, filename, body, response_class):
        if False:
            while True:
                i = 10

        def _test(response):
            if False:
                print('Hello World!')
            self.assertEqual(type(response), response_class)
        request = Request(self.getURL(filename), body=body)
        return self.download_request(request, Spider('foo')).addCallback(_test)

    def test_response_class_from_url(self):
        if False:
            for i in range(10):
                print('nop')
        return self._test_response_class('foo.html', b'', HtmlResponse)

    def test_response_class_from_body(self):
        if False:
            print('Hello World!')
        return self._test_response_class('foo', b'<!DOCTYPE html>\n<title>.</title>', HtmlResponse)

    def test_get_duplicate_header(self):
        if False:
            while True:
                i = 10

        def _test(response):
            if False:
                while True:
                    i = 10
            self.assertEqual(response.headers.getlist(b'Set-Cookie'), [b'a=b', b'c=d'])
        request = Request(self.getURL('duplicate-header'))
        return self.download_request(request, Spider('foo')).addCallback(_test)

class Http10TestCase(HttpTestCase):
    """HTTP 1.0 test case"""
    download_handler_cls: Type = HTTP10DownloadHandler

    def test_protocol(self):
        if False:
            print('Hello World!')
        request = Request(self.getURL('host'), method='GET')
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.protocol)
        d.addCallback(self.assertEqual, 'HTTP/1.0')
        return d

class Https10TestCase(Http10TestCase):
    scheme = 'https'

class Http11TestCase(HttpTestCase):
    """HTTP 1.1 test case"""
    download_handler_cls: Type = HTTP11DownloadHandler

    def test_download_without_maxsize_limit(self):
        if False:
            for i in range(10):
                print('nop')
        request = Request(self.getURL('file'))
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b'0123456789')
        return d

    def test_response_class_choosing_request(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests choosing of correct response type\n        in case of Content-Type is empty but body contains text.\n        '
        body = b'Some plain text\ndata with tabs\t and null bytes\x00'

        def _test_type(response):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(type(response), TextResponse)
        request = Request(self.getURL('nocontenttype'), body=body)
        d = self.download_request(request, Spider('foo'))
        d.addCallback(_test_type)
        return d

    @defer.inlineCallbacks
    def test_download_with_maxsize(self):
        if False:
            while True:
                i = 10
        request = Request(self.getURL('file'))
        d = self.download_request(request, Spider('foo', download_maxsize=10))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b'0123456789')
        yield d
        d = self.download_request(request, Spider('foo', download_maxsize=9))
        yield self.assertFailure(d, defer.CancelledError, error.ConnectionAborted)

    @defer.inlineCallbacks
    def test_download_with_maxsize_very_large_file(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch('scrapy.core.downloader.handlers.http11.logger') as logger:
            request = Request(self.getURL('largechunkedfile'))

            def check(logger):
                if False:
                    for i in range(10):
                        print('nop')
                logger.warning.assert_called_once_with(mock.ANY, mock.ANY)
            d = self.download_request(request, Spider('foo', download_maxsize=1500))
            yield self.assertFailure(d, defer.CancelledError, error.ConnectionAborted)
            d = defer.Deferred()
            d.addCallback(check)
            reactor.callLater(0.1, d.callback, logger)
            yield d

    @defer.inlineCallbacks
    def test_download_with_maxsize_per_req(self):
        if False:
            while True:
                i = 10
        meta = {'download_maxsize': 2}
        request = Request(self.getURL('file'), meta=meta)
        d = self.download_request(request, Spider('foo'))
        yield self.assertFailure(d, defer.CancelledError, error.ConnectionAborted)

    @defer.inlineCallbacks
    def test_download_with_small_maxsize_per_spider(self):
        if False:
            i = 10
            return i + 15
        request = Request(self.getURL('file'))
        d = self.download_request(request, Spider('foo', download_maxsize=2))
        yield self.assertFailure(d, defer.CancelledError, error.ConnectionAborted)

    def test_download_with_large_maxsize_per_spider(self):
        if False:
            i = 10
            return i + 15
        request = Request(self.getURL('file'))
        d = self.download_request(request, Spider('foo', download_maxsize=100))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b'0123456789')
        return d

    def test_download_chunked_content(self):
        if False:
            i = 10
            return i + 15
        request = Request(self.getURL('chunked'))
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b'chunked content\n')
        return d

    def test_download_broken_content_cause_data_loss(self, url='broken'):
        if False:
            while True:
                i = 10
        request = Request(self.getURL(url))
        d = self.download_request(request, Spider('foo'))

        def checkDataLoss(failure):
            if False:
                i = 10
                return i + 15
            if failure.check(ResponseFailed):
                if any((r.check(_DataLoss) for r in failure.value.reasons)):
                    return None
            return failure
        d.addCallback(lambda _: self.fail('No DataLoss exception'))
        d.addErrback(checkDataLoss)
        return d

    def test_download_broken_chunked_content_cause_data_loss(self):
        if False:
            return 10
        return self.test_download_broken_content_cause_data_loss('broken-chunked')

    def test_download_broken_content_allow_data_loss(self, url='broken'):
        if False:
            return 10
        request = Request(self.getURL(url), meta={'download_fail_on_dataloss': False})
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.flags)
        d.addCallback(self.assertEqual, ['dataloss'])
        return d

    def test_download_broken_chunked_content_allow_data_loss(self):
        if False:
            while True:
                i = 10
        return self.test_download_broken_content_allow_data_loss('broken-chunked')

    def test_download_broken_content_allow_data_loss_via_setting(self, url='broken'):
        if False:
            i = 10
            return i + 15
        crawler = get_crawler(settings_dict={'DOWNLOAD_FAIL_ON_DATALOSS': False})
        download_handler = create_instance(self.download_handler_cls, None, crawler)
        request = Request(self.getURL(url))
        d = download_handler.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.flags)
        d.addCallback(self.assertEqual, ['dataloss'])
        return d

    def test_download_broken_chunked_content_allow_data_loss_via_setting(self):
        if False:
            for i in range(10):
                print('nop')
        return self.test_download_broken_content_allow_data_loss_via_setting('broken-chunked')

    def test_protocol(self):
        if False:
            i = 10
            return i + 15
        request = Request(self.getURL('host'), method='GET')
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.protocol)
        d.addCallback(self.assertEqual, 'HTTP/1.1')
        return d

class Https11TestCase(Http11TestCase):
    scheme = 'https'
    tls_log_message = 'SSL connection certificate: issuer "/C=IE/O=Scrapy/CN=localhost", subject "/C=IE/O=Scrapy/CN=localhost"'

    @defer.inlineCallbacks
    def test_tls_logging(self):
        if False:
            print('Hello World!')
        crawler = get_crawler(settings_dict={'DOWNLOADER_CLIENT_TLS_VERBOSE_LOGGING': True})
        download_handler = create_instance(self.download_handler_cls, None, crawler)
        try:
            with LogCapture() as log_capture:
                request = Request(self.getURL('file'))
                d = download_handler.download_request(request, Spider('foo'))
                d.addCallback(lambda r: r.body)
                d.addCallback(self.assertEqual, b'0123456789')
                yield d
                log_capture.check_present(('scrapy.core.downloader.tls', 'DEBUG', self.tls_log_message))
        finally:
            yield download_handler.close()

class Https11WrongHostnameTestCase(Http11TestCase):
    scheme = 'https'
    keyfile = 'keys/example-com.key.pem'
    certfile = 'keys/example-com.cert.pem'

class Https11InvalidDNSId(Https11TestCase):
    """Connect to HTTPS hosts with IP while certificate uses domain names IDs."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.host = '127.0.0.1'

class Https11InvalidDNSPattern(Https11TestCase):
    """Connect to HTTPS hosts where the certificate are issued to an ip instead of a domain."""
    keyfile = 'keys/localhost.ip.key'
    certfile = 'keys/localhost.ip.crt'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            from service_identity.exceptions import CertificateError
        except ImportError:
            raise unittest.SkipTest('cryptography lib is too old')
        self.tls_log_message = 'SSL connection certificate: issuer "/C=IE/O=Scrapy/CN=127.0.0.1", subject "/C=IE/O=Scrapy/CN=127.0.0.1"'
        super().setUp()

class Https11CustomCiphers(unittest.TestCase):
    scheme = 'https'
    download_handler_cls: Type = HTTP11DownloadHandler
    keyfile = 'keys/localhost.key'
    certfile = 'keys/localhost.crt'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tmpname = Path(self.mktemp())
        self.tmpname.mkdir()
        (self.tmpname / 'file').write_bytes(b'0123456789')
        r = static.File(str(self.tmpname))
        self.site = server.Site(r, timeout=None)
        self.host = 'localhost'
        self.port = reactor.listenSSL(0, self.site, ssl_context_factory(self.keyfile, self.certfile, cipher_string='CAMELLIA256-SHA'), interface=self.host)
        self.portno = self.port.getHost().port
        crawler = get_crawler(settings_dict={'DOWNLOADER_CLIENT_TLS_CIPHERS': 'CAMELLIA256-SHA'})
        self.download_handler = create_instance(self.download_handler_cls, None, crawler)
        self.download_request = self.download_handler.download_request

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            while True:
                i = 10
        yield self.port.stopListening()
        if hasattr(self.download_handler, 'close'):
            yield self.download_handler.close()
        shutil.rmtree(self.tmpname)

    def getURL(self, path):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.scheme}://{self.host}:{self.portno}/{path}'

    def test_download(self):
        if False:
            for i in range(10):
                print('nop')
        request = Request(self.getURL('file'))
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b'0123456789')
        return d

class Http11MockServerTestCase(unittest.TestCase):
    """HTTP 1.1 test case with MockServer"""
    settings_dict: Optional[dict] = None

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.mockserver = MockServer()
        self.mockserver.__enter__()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.mockserver.__exit__(None, None, None)

    @defer.inlineCallbacks
    def test_download_with_content_length(self):
        if False:
            for i in range(10):
                print('nop')
        crawler = get_crawler(SingleRequestSpider, self.settings_dict)
        yield crawler.crawl(seed=Request(url=self.mockserver.url('/partial'), meta={'download_maxsize': 1000}))
        failure = crawler.spider.meta['failure']
        self.assertIsInstance(failure.value, defer.CancelledError)

    @defer.inlineCallbacks
    def test_download(self):
        if False:
            i = 10
            return i + 15
        crawler = get_crawler(SingleRequestSpider, self.settings_dict)
        yield crawler.crawl(seed=Request(url=self.mockserver.url('')))
        failure = crawler.spider.meta.get('failure')
        self.assertTrue(failure is None)
        reason = crawler.spider.meta['close_reason']
        self.assertTrue(reason, 'finished')

    @defer.inlineCallbacks
    def test_download_gzip_response(self):
        if False:
            while True:
                i = 10
        crawler = get_crawler(SingleRequestSpider, self.settings_dict)
        body = b'1' * 100
        request = Request(self.mockserver.url('/payload'), method='POST', body=body, meta={'download_maxsize': 50})
        yield crawler.crawl(seed=request)
        failure = crawler.spider.meta['failure']
        self.assertIsInstance(failure.value, defer.CancelledError)
        raise unittest.SkipTest('xpayload fails on PY3')
        crawler = get_crawler(SingleRequestSpider, self.settings_dict)
        request.headers.setdefault(b'Accept-Encoding', b'gzip,deflate')
        request = request.replace(url=self.mockserver.url('/xpayload'))
        yield crawler.crawl(seed=request)
        failure = crawler.spider.meta.get('failure')
        self.assertIsNone(failure)
        reason = crawler.spider.meta['close_reason']
        self.assertTrue(reason, 'finished')

class UriResource(resource.Resource):
    """Return the full uri that was requested"""

    def getChild(self, path, request):
        if False:
            while True:
                i = 10
        return self

    def render(self, request):
        if False:
            return 10
        if request.method != b'CONNECT':
            return request.uri
        return b''

class HttpProxyTestCase(unittest.TestCase):
    download_handler_cls: Type = HTTPDownloadHandler
    expected_http_proxy_request_body = b'http://example.com'

    def setUp(self):
        if False:
            return 10
        site = server.Site(UriResource(), timeout=None)
        wrapper = WrappingFactory(site)
        self.port = reactor.listenTCP(0, wrapper, interface='127.0.0.1')
        self.portno = self.port.getHost().port
        self.download_handler = create_instance(self.download_handler_cls, None, get_crawler())
        self.download_request = self.download_handler.download_request

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            i = 10
            return i + 15
        yield self.port.stopListening()
        if hasattr(self.download_handler, 'close'):
            yield self.download_handler.close()

    def getURL(self, path):
        if False:
            i = 10
            return i + 15
        return f'http://127.0.0.1:{self.portno}/{path}'

    def test_download_with_proxy(self):
        if False:
            print('Hello World!')

        def _test(response):
            if False:
                while True:
                    i = 10
            self.assertEqual(response.status, 200)
            self.assertEqual(response.url, request.url)
            self.assertEqual(response.body, self.expected_http_proxy_request_body)
        http_proxy = self.getURL('')
        request = Request('http://example.com', meta={'proxy': http_proxy})
        return self.download_request(request, Spider('foo')).addCallback(_test)

    def test_download_without_proxy(self):
        if False:
            print('Hello World!')

        def _test(response):
            if False:
                i = 10
                return i + 15
            self.assertEqual(response.status, 200)
            self.assertEqual(response.url, request.url)
            self.assertEqual(response.body, b'/path/to/resource')
        request = Request(self.getURL('path/to/resource'))
        return self.download_request(request, Spider('foo')).addCallback(_test)

class Http10ProxyTestCase(HttpProxyTestCase):
    download_handler_cls: Type = HTTP10DownloadHandler

    def test_download_with_proxy_https_noconnect(self):
        if False:
            i = 10
            return i + 15
        raise unittest.SkipTest('noconnect is not supported in HTTP10DownloadHandler')

class Http11ProxyTestCase(HttpProxyTestCase):
    download_handler_cls: Type = HTTP11DownloadHandler

    @defer.inlineCallbacks
    def test_download_with_proxy_https_timeout(self):
        if False:
            print('Hello World!')
        'Test TunnelingTCP4ClientEndpoint'
        if NON_EXISTING_RESOLVABLE:
            raise SkipTest('Non-existing hosts are resolvable')
        http_proxy = self.getURL('')
        domain = 'https://no-such-domain.nosuch'
        request = Request(domain, meta={'proxy': http_proxy, 'download_timeout': 0.2})
        d = self.download_request(request, Spider('foo'))
        timeout = (yield self.assertFailure(d, error.TimeoutError))
        self.assertIn(domain, timeout.osError)

    def test_download_with_proxy_without_http_scheme(self):
        if False:
            print('Hello World!')

        def _test(response):
            if False:
                while True:
                    i = 10
            self.assertEqual(response.status, 200)
            self.assertEqual(response.url, request.url)
            self.assertEqual(response.body, self.expected_http_proxy_request_body)
        http_proxy = self.getURL('').replace('http://', '')
        request = Request('http://example.com', meta={'proxy': http_proxy})
        return self.download_request(request, Spider('foo')).addCallback(_test)

class HttpDownloadHandlerMock:

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        pass

    def download_request(self, request, spider):
        if False:
            while True:
                i = 10
        return request

class S3AnonTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        skip_if_no_boto()
        crawler = get_crawler()
        self.s3reqh = create_instance(objcls=S3DownloadHandler, settings=None, crawler=crawler, httpdownloadhandler=HttpDownloadHandlerMock)
        self.download_request = self.s3reqh.download_request
        self.spider = Spider('foo')

    def test_anon_request(self):
        if False:
            print('Hello World!')
        req = Request('s3://aws-publicdatasets/')
        httpreq = self.download_request(req, self.spider)
        self.assertEqual(hasattr(self.s3reqh, 'anon'), True)
        self.assertEqual(self.s3reqh.anon, True)
        self.assertEqual(httpreq.url, 'http://aws-publicdatasets.s3.amazonaws.com/')

class S3TestCase(unittest.TestCase):
    download_handler_cls: Type = S3DownloadHandler
    AWS_ACCESS_KEY_ID = '0PN5J17HBGZHT7JJ3X82'
    AWS_SECRET_ACCESS_KEY = 'uV3F3YluFJax1cknvbcGwgjvx4QpvB+leU8dUj2o'

    def setUp(self):
        if False:
            while True:
                i = 10
        skip_if_no_boto()
        crawler = get_crawler()
        s3reqh = create_instance(objcls=S3DownloadHandler, settings=None, crawler=crawler, aws_access_key_id=self.AWS_ACCESS_KEY_ID, aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY, httpdownloadhandler=HttpDownloadHandlerMock)
        self.download_request = s3reqh.download_request
        self.spider = Spider('foo')

    @contextlib.contextmanager
    def _mocked_date(self, date):
        if False:
            i = 10
            return i + 15
        try:
            import botocore.auth
        except ImportError:
            yield
        else:
            with mock.patch('botocore.auth.formatdate') as mock_formatdate:
                mock_formatdate.return_value = date
                yield

    def test_extra_kw(self):
        if False:
            return 10
        try:
            crawler = get_crawler()
            create_instance(objcls=S3DownloadHandler, settings=None, crawler=crawler, extra_kw=True)
        except Exception as e:
            self.assertIsInstance(e, (TypeError, NotConfigured))
        else:
            assert False

    def test_request_signing1(self):
        if False:
            return 10
        date = 'Tue, 27 Mar 2007 19:36:42 +0000'
        req = Request('s3://johnsmith/photos/puppy.jpg', headers={'Date': date})
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], b'AWS 0PN5J17HBGZHT7JJ3X82:xXjDGYUmKxnwqr5KXNPGldn5LbA=')

    def test_request_signing2(self):
        if False:
            while True:
                i = 10
        date = 'Tue, 27 Mar 2007 21:15:45 +0000'
        req = Request('s3://johnsmith/photos/puppy.jpg', method='PUT', headers={'Content-Type': 'image/jpeg', 'Date': date, 'Content-Length': '94328'})
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], b'AWS 0PN5J17HBGZHT7JJ3X82:hcicpDDvL9SsO6AkvxqmIWkmOuQ=')

    def test_request_signing3(self):
        if False:
            print('Hello World!')
        date = 'Tue, 27 Mar 2007 19:42:41 +0000'
        req = Request('s3://johnsmith/?prefix=photos&max-keys=50&marker=puppy', method='GET', headers={'User-Agent': 'Mozilla/5.0', 'Date': date})
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], b'AWS 0PN5J17HBGZHT7JJ3X82:jsRt/rhG+Vtp88HrYL706QhE4w4=')

    def test_request_signing4(self):
        if False:
            print('Hello World!')
        date = 'Tue, 27 Mar 2007 19:44:46 +0000'
        req = Request('s3://johnsmith/?acl', method='GET', headers={'Date': date})
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], b'AWS 0PN5J17HBGZHT7JJ3X82:thdUi9VAkzhkniLj96JIrOPGi0g=')

    def test_request_signing6(self):
        if False:
            return 10
        date = 'Tue, 27 Mar 2007 21:06:08 +0000'
        req = Request('s3://static.johnsmith.net:8080/db-backup.dat.gz', method='PUT', headers={'User-Agent': 'curl/7.15.5', 'Host': 'static.johnsmith.net:8080', 'Date': date, 'x-amz-acl': 'public-read', 'content-type': 'application/x-download', 'Content-MD5': '4gJE4saaMU4BqNR0kLY+lw==', 'X-Amz-Meta-ReviewedBy': 'joe@johnsmith.net,jane@johnsmith.net', 'X-Amz-Meta-FileChecksum': '0x02661779', 'X-Amz-Meta-ChecksumAlgorithm': 'crc32', 'Content-Disposition': 'attachment; filename=database.dat', 'Content-Encoding': 'gzip', 'Content-Length': '5913339'})
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], b'AWS 0PN5J17HBGZHT7JJ3X82:C0FlOtU8Ylb9KDTpZqYkZPX91iI=')

    def test_request_signing7(self):
        if False:
            while True:
                i = 10
        date = 'Tue, 27 Mar 2007 19:42:41 +0000'
        req = Request('s3://johnsmith/photos/my puppy.jpg?response-content-disposition=my puppy.jpg', method='GET', headers={'Date': date})
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], b'AWS 0PN5J17HBGZHT7JJ3X82:+CfvG8EZ3YccOrRVMXNaK2eKZmM=')

class BaseFTPTestCase(unittest.TestCase):
    username = 'scrapy'
    password = 'passwd'
    req_meta = {'ftp_user': username, 'ftp_password': password}
    test_files = (('file.txt', b'I have the power!'), ('file with spaces.txt', b'Moooooooooo power!'), ('html-file-without-extension', b'<!DOCTYPE html>\n<title>.</title>'))

    def setUp(self):
        if False:
            print('Hello World!')
        from twisted.protocols.ftp import FTPFactory, FTPRealm
        from scrapy.core.downloader.handlers.ftp import FTPDownloadHandler
        self.directory = Path(self.mktemp())
        self.directory.mkdir()
        userdir = self.directory / self.username
        userdir.mkdir()
        for (filename, content) in self.test_files:
            (userdir / filename).write_bytes(content)
        realm = FTPRealm(anonymousRoot=str(self.directory), userHome=str(self.directory))
        p = portal.Portal(realm)
        users_checker = checkers.InMemoryUsernamePasswordDatabaseDontUse()
        users_checker.addUser(self.username, self.password)
        p.registerChecker(users_checker, credentials.IUsernamePassword)
        self.factory = FTPFactory(portal=p)
        self.port = reactor.listenTCP(0, self.factory, interface='127.0.0.1')
        self.portNum = self.port.getHost().port
        crawler = get_crawler()
        self.download_handler = create_instance(FTPDownloadHandler, crawler.settings, crawler)
        self.addCleanup(self.port.stopListening)

    def tearDown(self):
        if False:
            return 10
        shutil.rmtree(self.directory)

    def _add_test_callbacks(self, deferred, callback=None, errback=None):
        if False:
            return 10

        def _clean(data):
            if False:
                while True:
                    i = 10
            self.download_handler.client.transport.loseConnection()
            return data
        deferred.addCallback(_clean)
        if callback:
            deferred.addCallback(callback)
        if errback:
            deferred.addErrback(errback)
        return deferred

    def test_ftp_download_success(self):
        if False:
            i = 10
            return i + 15
        request = Request(url=f'ftp://127.0.0.1:{self.portNum}/file.txt', meta=self.req_meta)
        d = self.download_handler.download_request(request, None)

        def _test(r):
            if False:
                i = 10
                return i + 15
            self.assertEqual(r.status, 200)
            self.assertEqual(r.body, b'I have the power!')
            self.assertEqual(r.headers, {b'Local Filename': [b''], b'Size': [b'17']})
            self.assertIsNone(r.protocol)
        return self._add_test_callbacks(d, _test)

    def test_ftp_download_path_with_spaces(self):
        if False:
            while True:
                i = 10
        request = Request(url=f'ftp://127.0.0.1:{self.portNum}/file with spaces.txt', meta=self.req_meta)
        d = self.download_handler.download_request(request, None)

        def _test(r):
            if False:
                i = 10
                return i + 15
            self.assertEqual(r.status, 200)
            self.assertEqual(r.body, b'Moooooooooo power!')
            self.assertEqual(r.headers, {b'Local Filename': [b''], b'Size': [b'18']})
        return self._add_test_callbacks(d, _test)

    def test_ftp_download_nonexistent(self):
        if False:
            i = 10
            return i + 15
        request = Request(url=f'ftp://127.0.0.1:{self.portNum}/nonexistent.txt', meta=self.req_meta)
        d = self.download_handler.download_request(request, None)

        def _test(r):
            if False:
                i = 10
                return i + 15
            self.assertEqual(r.status, 404)
        return self._add_test_callbacks(d, _test)

    def test_ftp_local_filename(self):
        if False:
            return 10
        (f, local_fname) = tempfile.mkstemp()
        fname_bytes = to_bytes(local_fname)
        local_fname = Path(local_fname)
        os.close(f)
        meta = {'ftp_local_filename': fname_bytes}
        meta.update(self.req_meta)
        request = Request(url=f'ftp://127.0.0.1:{self.portNum}/file.txt', meta=meta)
        d = self.download_handler.download_request(request, None)

        def _test(r):
            if False:
                print('Hello World!')
            self.assertEqual(r.body, fname_bytes)
            self.assertEqual(r.headers, {b'Local Filename': [fname_bytes], b'Size': [b'17']})
            self.assertTrue(local_fname.exists())
            self.assertEqual(local_fname.read_bytes(), b'I have the power!')
            local_fname.unlink()
        return self._add_test_callbacks(d, _test)

    def _test_response_class(self, filename, response_class):
        if False:
            i = 10
            return i + 15
        (f, local_fname) = tempfile.mkstemp()
        local_fname = Path(local_fname)
        os.close(f)
        meta = {}
        meta.update(self.req_meta)
        request = Request(url=f'ftp://127.0.0.1:{self.portNum}/{filename}', meta=meta)
        d = self.download_handler.download_request(request, None)

        def _test(r):
            if False:
                while True:
                    i = 10
            self.assertEqual(type(r), response_class)
            local_fname.unlink()
        return self._add_test_callbacks(d, _test)

    def test_response_class_from_url(self):
        if False:
            i = 10
            return i + 15
        return self._test_response_class('file.txt', TextResponse)

    def test_response_class_from_body(self):
        if False:
            return 10
        return self._test_response_class('html-file-without-extension', HtmlResponse)

class FTPTestCase(BaseFTPTestCase):

    def test_invalid_credentials(self):
        if False:
            while True:
                i = 10
        if self.reactor_pytest == 'asyncio' and sys.platform == 'win32':
            raise unittest.SkipTest('This test produces DirtyReactorAggregateError on Windows with asyncio')
        from twisted.protocols.ftp import ConnectionLost
        meta = dict(self.req_meta)
        meta.update({'ftp_password': 'invalid'})
        request = Request(url=f'ftp://127.0.0.1:{self.portNum}/file.txt', meta=meta)
        d = self.download_handler.download_request(request, None)

        def _test(r):
            if False:
                print('Hello World!')
            self.assertEqual(r.type, ConnectionLost)
        return self._add_test_callbacks(d, errback=_test)

class AnonymousFTPTestCase(BaseFTPTestCase):
    username = 'anonymous'
    req_meta = {}

    def setUp(self):
        if False:
            while True:
                i = 10
        from twisted.protocols.ftp import FTPFactory, FTPRealm
        from scrapy.core.downloader.handlers.ftp import FTPDownloadHandler
        self.directory = Path(self.mktemp())
        self.directory.mkdir()
        for (filename, content) in self.test_files:
            (self.directory / filename).write_bytes(content)
        realm = FTPRealm(anonymousRoot=str(self.directory))
        p = portal.Portal(realm)
        p.registerChecker(checkers.AllowAnonymousAccess(), credentials.IAnonymous)
        self.factory = FTPFactory(portal=p, userAnonymous=self.username)
        self.port = reactor.listenTCP(0, self.factory, interface='127.0.0.1')
        self.portNum = self.port.getHost().port
        crawler = get_crawler()
        self.download_handler = create_instance(FTPDownloadHandler, crawler.settings, crawler)
        self.addCleanup(self.port.stopListening)

    def tearDown(self):
        if False:
            while True:
                i = 10
        shutil.rmtree(self.directory)

class DataURITestCase(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        crawler = get_crawler()
        self.download_handler = create_instance(DataURIDownloadHandler, crawler.settings, crawler)
        self.download_request = self.download_handler.download_request
        self.spider = Spider('foo')

    def test_response_attrs(self):
        if False:
            while True:
                i = 10
        uri = 'data:,A%20brief%20note'

        def _test(response):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(response.url, uri)
            self.assertFalse(response.headers)
        request = Request(uri)
        return self.download_request(request, self.spider).addCallback(_test)

    def test_default_mediatype_encoding(self):
        if False:
            print('Hello World!')

        def _test(response):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(response.text, 'A brief note')
            self.assertEqual(type(response), responsetypes.from_mimetype('text/plain'))
            self.assertEqual(response.encoding, 'US-ASCII')
        request = Request('data:,A%20brief%20note')
        return self.download_request(request, self.spider).addCallback(_test)

    def test_default_mediatype(self):
        if False:
            i = 10
            return i + 15

        def _test(response):
            if False:
                while True:
                    i = 10
            self.assertEqual(response.text, 'ΎΣΎ')
            self.assertEqual(type(response), responsetypes.from_mimetype('text/plain'))
            self.assertEqual(response.encoding, 'iso-8859-7')
        request = Request('data:;charset=iso-8859-7,%be%d3%be')
        return self.download_request(request, self.spider).addCallback(_test)

    def test_text_charset(self):
        if False:
            return 10

        def _test(response):
            if False:
                print('Hello World!')
            self.assertEqual(response.text, 'ΎΣΎ')
            self.assertEqual(response.body, b'\xbe\xd3\xbe')
            self.assertEqual(response.encoding, 'iso-8859-7')
        request = Request('data:text/plain;charset=iso-8859-7,%be%d3%be')
        return self.download_request(request, self.spider).addCallback(_test)

    def test_mediatype_parameters(self):
        if False:
            for i in range(10):
                print('nop')

        def _test(response):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(response.text, 'ΎΣΎ')
            self.assertEqual(type(response), responsetypes.from_mimetype('text/plain'))
            self.assertEqual(response.encoding, 'utf-8')
        request = Request('data:text/plain;foo=%22foo;bar%5C%22%22;charset=utf-8;bar=%22foo;%5C%22 foo ;/,%22,%CE%8E%CE%A3%CE%8E')
        return self.download_request(request, self.spider).addCallback(_test)

    def test_base64(self):
        if False:
            return 10

        def _test(response):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(response.text, 'Hello, world.')
        request = Request('data:text/plain;base64,SGVsbG8sIHdvcmxkLg%3D%3D')
        return self.download_request(request, self.spider).addCallback(_test)

    def test_protocol(self):
        if False:
            return 10

        def _test(response):
            if False:
                while True:
                    i = 10
            self.assertIsNone(response.protocol)
        request = Request('data:,')
        return self.download_request(request, self.spider).addCallback(_test)