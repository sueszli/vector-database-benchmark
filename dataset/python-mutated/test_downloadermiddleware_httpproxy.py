import os
import pytest
from twisted.trial.unittest import TestCase
from scrapy.downloadermiddlewares.httpproxy import HttpProxyMiddleware
from scrapy.exceptions import NotConfigured
from scrapy.http import Request
from scrapy.spiders import Spider
from scrapy.utils.test import get_crawler
spider = Spider('foo')

class TestHttpProxyMiddleware(TestCase):
    failureException = AssertionError

    def setUp(self):
        if False:
            while True:
                i = 10
        self._oldenv = os.environ.copy()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        os.environ = self._oldenv

    def test_not_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        crawler = get_crawler(Spider, {'HTTPPROXY_ENABLED': False})
        with pytest.raises(NotConfigured):
            HttpProxyMiddleware.from_crawler(crawler)

    def test_no_environment_proxies(self):
        if False:
            for i in range(10):
                print('nop')
        os.environ = {'dummy_proxy': 'reset_env_and_do_not_raise'}
        mw = HttpProxyMiddleware()
        for url in ('http://e.com', 'https://e.com', 'file:///tmp/a'):
            req = Request(url)
            assert mw.process_request(req, spider) is None
            self.assertEqual(req.url, url)
            self.assertEqual(req.meta, {})

    def test_environment_proxies(self):
        if False:
            print('Hello World!')
        os.environ['http_proxy'] = http_proxy = 'https://proxy.for.http:3128'
        os.environ['https_proxy'] = https_proxy = 'http://proxy.for.https:8080'
        os.environ.pop('file_proxy', None)
        mw = HttpProxyMiddleware()
        for (url, proxy) in [('http://e.com', http_proxy), ('https://e.com', https_proxy), ('file://tmp/a', None)]:
            req = Request(url)
            assert mw.process_request(req, spider) is None
            self.assertEqual(req.url, url)
            self.assertEqual(req.meta.get('proxy'), proxy)

    def test_proxy_precedence_meta(self):
        if False:
            print('Hello World!')
        os.environ['http_proxy'] = 'https://proxy.com'
        mw = HttpProxyMiddleware()
        req = Request('http://scrapytest.org', meta={'proxy': 'https://new.proxy:3128'})
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta, {'proxy': 'https://new.proxy:3128'})

    def test_proxy_auth(self):
        if False:
            for i in range(10):
                print('nop')
        os.environ['http_proxy'] = 'https://user:pass@proxy:3128'
        mw = HttpProxyMiddleware()
        req = Request('http://scrapytest.org')
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta['proxy'], 'https://proxy:3128')
        self.assertEqual(req.headers.get('Proxy-Authorization'), b'Basic dXNlcjpwYXNz')
        req = Request('http://scrapytest.org', meta={'proxy': 'https://username:password@proxy:3128'})
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta['proxy'], 'https://proxy:3128')
        self.assertEqual(req.headers.get('Proxy-Authorization'), b'Basic dXNlcm5hbWU6cGFzc3dvcmQ=')

    def test_proxy_auth_empty_passwd(self):
        if False:
            i = 10
            return i + 15
        os.environ['http_proxy'] = 'https://user:@proxy:3128'
        mw = HttpProxyMiddleware()
        req = Request('http://scrapytest.org')
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta['proxy'], 'https://proxy:3128')
        self.assertEqual(req.headers.get('Proxy-Authorization'), b'Basic dXNlcjo=')
        req = Request('http://scrapytest.org', meta={'proxy': 'https://username:@proxy:3128'})
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta['proxy'], 'https://proxy:3128')
        self.assertEqual(req.headers.get('Proxy-Authorization'), b'Basic dXNlcm5hbWU6')

    def test_proxy_auth_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        os.environ['http_proxy'] = 'https://mán:pass@proxy:3128'
        mw = HttpProxyMiddleware(auth_encoding='utf-8')
        req = Request('http://scrapytest.org')
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta['proxy'], 'https://proxy:3128')
        self.assertEqual(req.headers.get('Proxy-Authorization'), b'Basic bcOhbjpwYXNz')
        req = Request('http://scrapytest.org', meta={'proxy': 'https://üser:pass@proxy:3128'})
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta['proxy'], 'https://proxy:3128')
        self.assertEqual(req.headers.get('Proxy-Authorization'), b'Basic w7xzZXI6cGFzcw==')
        mw = HttpProxyMiddleware(auth_encoding='latin-1')
        req = Request('http://scrapytest.org')
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta['proxy'], 'https://proxy:3128')
        self.assertEqual(req.headers.get('Proxy-Authorization'), b'Basic beFuOnBhc3M=')
        req = Request('http://scrapytest.org', meta={'proxy': 'https://üser:pass@proxy:3128'})
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta['proxy'], 'https://proxy:3128')
        self.assertEqual(req.headers.get('Proxy-Authorization'), b'Basic /HNlcjpwYXNz')

    def test_proxy_already_seted(self):
        if False:
            print('Hello World!')
        os.environ['http_proxy'] = 'https://proxy.for.http:3128'
        mw = HttpProxyMiddleware()
        req = Request('http://noproxy.com', meta={'proxy': None})
        assert mw.process_request(req, spider) is None
        assert 'proxy' in req.meta and req.meta['proxy'] is None

    def test_no_proxy(self):
        if False:
            while True:
                i = 10
        os.environ['http_proxy'] = 'https://proxy.for.http:3128'
        mw = HttpProxyMiddleware()
        os.environ['no_proxy'] = '*'
        req = Request('http://noproxy.com')
        assert mw.process_request(req, spider) is None
        assert 'proxy' not in req.meta
        os.environ['no_proxy'] = 'other.com'
        req = Request('http://noproxy.com')
        assert mw.process_request(req, spider) is None
        assert 'proxy' in req.meta
        os.environ['no_proxy'] = 'other.com,noproxy.com'
        req = Request('http://noproxy.com')
        assert mw.process_request(req, spider) is None
        assert 'proxy' not in req.meta
        os.environ['no_proxy'] = '*'
        req = Request('http://noproxy.com', meta={'proxy': 'http://proxy.com'})
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta, {'proxy': 'http://proxy.com'})

    def test_no_proxy_invalid_values(self):
        if False:
            i = 10
            return i + 15
        os.environ['no_proxy'] = '/var/run/docker.sock'
        mw = HttpProxyMiddleware()
        assert 'no' not in mw.proxies

    def test_add_proxy_without_credentials(self):
        if False:
            return 10
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com')
        assert middleware.process_request(request, spider) is None
        request.meta['proxy'] = 'https://example.com'
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.com')
        self.assertNotIn(b'Proxy-Authorization', request.headers)

    def test_add_proxy_with_credentials(self):
        if False:
            return 10
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com')
        assert middleware.process_request(request, spider) is None
        request.meta['proxy'] = 'https://user1:password1@example.com'
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.com')
        encoded_credentials = middleware._basic_auth_header('user1', 'password1')
        self.assertEqual(request.headers['Proxy-Authorization'], b'Basic ' + encoded_credentials)

    def test_remove_proxy_without_credentials(self):
        if False:
            print('Hello World!')
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com', meta={'proxy': 'https://example.com'})
        assert middleware.process_request(request, spider) is None
        request.meta['proxy'] = None
        assert middleware.process_request(request, spider) is None
        self.assertIsNone(request.meta['proxy'])
        self.assertNotIn(b'Proxy-Authorization', request.headers)

    def test_remove_proxy_with_credentials(self):
        if False:
            return 10
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com', meta={'proxy': 'https://user1:password1@example.com'})
        assert middleware.process_request(request, spider) is None
        request.meta['proxy'] = None
        assert middleware.process_request(request, spider) is None
        self.assertIsNone(request.meta['proxy'])
        self.assertNotIn(b'Proxy-Authorization', request.headers)

    def test_add_credentials(self):
        if False:
            for i in range(10):
                print('nop')
        'If the proxy request meta switches to a proxy URL with the same\n        proxy and adds credentials (there were no credentials before), the new\n        credentials must be used.'
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com', meta={'proxy': 'https://example.com'})
        assert middleware.process_request(request, spider) is None
        request.meta['proxy'] = 'https://user1:password1@example.com'
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.com')
        encoded_credentials = middleware._basic_auth_header('user1', 'password1')
        self.assertEqual(request.headers['Proxy-Authorization'], b'Basic ' + encoded_credentials)

    def test_change_credentials(self):
        if False:
            print('Hello World!')
        'If the proxy request meta switches to a proxy URL with different\n        credentials, those new credentials must be used.'
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com', meta={'proxy': 'https://user1:password1@example.com'})
        assert middleware.process_request(request, spider) is None
        request.meta['proxy'] = 'https://user2:password2@example.com'
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.com')
        encoded_credentials = middleware._basic_auth_header('user2', 'password2')
        self.assertEqual(request.headers['Proxy-Authorization'], b'Basic ' + encoded_credentials)

    def test_remove_credentials(self):
        if False:
            print('Hello World!')
        'If the proxy request meta switches to a proxy URL with the same\n        proxy but no credentials, the original credentials must be still\n        used.\n\n        To remove credentials while keeping the same proxy URL, users must\n        delete the Proxy-Authorization header.\n        '
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com', meta={'proxy': 'https://user1:password1@example.com'})
        assert middleware.process_request(request, spider) is None
        request.meta['proxy'] = 'https://example.com'
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.com')
        encoded_credentials = middleware._basic_auth_header('user1', 'password1')
        self.assertEqual(request.headers['Proxy-Authorization'], b'Basic ' + encoded_credentials)
        request.meta['proxy'] = 'https://example.com'
        del request.headers[b'Proxy-Authorization']
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.com')
        self.assertNotIn(b'Proxy-Authorization', request.headers)

    def test_change_proxy_add_credentials(self):
        if False:
            print('Hello World!')
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com', meta={'proxy': 'https://example.com'})
        assert middleware.process_request(request, spider) is None
        request.meta['proxy'] = 'https://user1:password1@example.org'
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.org')
        encoded_credentials = middleware._basic_auth_header('user1', 'password1')
        self.assertEqual(request.headers['Proxy-Authorization'], b'Basic ' + encoded_credentials)

    def test_change_proxy_keep_credentials(self):
        if False:
            print('Hello World!')
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com', meta={'proxy': 'https://user1:password1@example.com'})
        assert middleware.process_request(request, spider) is None
        request.meta['proxy'] = 'https://user1:password1@example.org'
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.org')
        encoded_credentials = middleware._basic_auth_header('user1', 'password1')
        self.assertEqual(request.headers['Proxy-Authorization'], b'Basic ' + encoded_credentials)
        request.meta['proxy'] = 'https://example.com'
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.com')
        self.assertNotIn(b'Proxy-Authorization', request.headers)

    def test_change_proxy_change_credentials(self):
        if False:
            for i in range(10):
                print('nop')
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com', meta={'proxy': 'https://user1:password1@example.com'})
        assert middleware.process_request(request, spider) is None
        request.meta['proxy'] = 'https://user2:password2@example.org'
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.org')
        encoded_credentials = middleware._basic_auth_header('user2', 'password2')
        self.assertEqual(request.headers['Proxy-Authorization'], b'Basic ' + encoded_credentials)

    def test_change_proxy_remove_credentials(self):
        if False:
            return 10
        'If the proxy request meta switches to a proxy URL with a different\n        proxy and no credentials, no credentials must be used.'
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com', meta={'proxy': 'https://user1:password1@example.com'})
        assert middleware.process_request(request, spider) is None
        request.meta['proxy'] = 'https://example.org'
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta, {'proxy': 'https://example.org'})
        self.assertNotIn(b'Proxy-Authorization', request.headers)

    def test_change_proxy_remove_credentials_preremoved_header(self):
        if False:
            print('Hello World!')
        'Corner case of proxy switch with credentials removal where the\n        credentials have been removed beforehand.\n\n        It ensures that our implementation does not assume that the credentials\n        header exists when trying to remove it.\n        '
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com', meta={'proxy': 'https://user1:password1@example.com'})
        assert middleware.process_request(request, spider) is None
        request.meta['proxy'] = 'https://example.org'
        del request.headers[b'Proxy-Authorization']
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta, {'proxy': 'https://example.org'})
        self.assertNotIn(b'Proxy-Authorization', request.headers)

    def test_proxy_authentication_header_undefined_proxy(self):
        if False:
            for i in range(10):
                print('nop')
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com', headers={'Proxy-Authorization': 'Basic foo'})
        assert middleware.process_request(request, spider) is None
        self.assertNotIn('proxy', request.meta)
        self.assertNotIn(b'Proxy-Authorization', request.headers)

    def test_proxy_authentication_header_disabled_proxy(self):
        if False:
            while True:
                i = 10
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com', headers={'Proxy-Authorization': 'Basic foo'}, meta={'proxy': None})
        assert middleware.process_request(request, spider) is None
        self.assertIsNone(request.meta['proxy'])
        self.assertNotIn(b'Proxy-Authorization', request.headers)

    def test_proxy_authentication_header_proxy_without_credentials(self):
        if False:
            while True:
                i = 10
        'As long as the proxy URL in request metadata remains the same, the\n        Proxy-Authorization header is used and kept, and may even be\n        changed.'
        middleware = HttpProxyMiddleware()
        request = Request('https://example.com', headers={'Proxy-Authorization': 'Basic foo'}, meta={'proxy': 'https://example.com'})
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.com')
        self.assertEqual(request.headers['Proxy-Authorization'], b'Basic foo')
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.com')
        self.assertEqual(request.headers['Proxy-Authorization'], b'Basic foo')
        request.headers['Proxy-Authorization'] = b'Basic bar'
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.com')
        self.assertEqual(request.headers['Proxy-Authorization'], b'Basic bar')

    def test_proxy_authentication_header_proxy_with_same_credentials(self):
        if False:
            while True:
                i = 10
        middleware = HttpProxyMiddleware()
        encoded_credentials = middleware._basic_auth_header('user1', 'password1')
        request = Request('https://example.com', headers={'Proxy-Authorization': b'Basic ' + encoded_credentials}, meta={'proxy': 'https://user1:password1@example.com'})
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.com')
        self.assertEqual(request.headers['Proxy-Authorization'], b'Basic ' + encoded_credentials)

    def test_proxy_authentication_header_proxy_with_different_credentials(self):
        if False:
            return 10
        middleware = HttpProxyMiddleware()
        encoded_credentials1 = middleware._basic_auth_header('user1', 'password1')
        request = Request('https://example.com', headers={'Proxy-Authorization': b'Basic ' + encoded_credentials1}, meta={'proxy': 'https://user2:password2@example.com'})
        assert middleware.process_request(request, spider) is None
        self.assertEqual(request.meta['proxy'], 'https://example.com')
        encoded_credentials2 = middleware._basic_auth_header('user2', 'password2')
        self.assertEqual(request.headers['Proxy-Authorization'], b'Basic ' + encoded_credentials2)