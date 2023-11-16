import unittest
from scrapy.downloadermiddlewares.redirect import MetaRefreshMiddleware, RedirectMiddleware
from scrapy.exceptions import IgnoreRequest
from scrapy.http import HtmlResponse, Request, Response
from scrapy.spiders import Spider
from scrapy.utils.test import get_crawler

class RedirectMiddlewareTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.crawler = get_crawler(Spider)
        self.spider = self.crawler._create_spider('foo')
        self.mw = RedirectMiddleware.from_crawler(self.crawler)

    def test_priority_adjust(self):
        if False:
            return 10
        req = Request('http://a.com')
        rsp = Response('http://a.com', headers={'Location': 'http://a.com/redirected'}, status=301)
        req2 = self.mw.process_response(req, rsp, self.spider)
        assert req2.priority > req.priority

    def test_redirect_3xx_permanent(self):
        if False:
            i = 10
            return i + 15

        def _test(method, status=301):
            if False:
                print('Hello World!')
            url = f'http://www.example.com/{status}'
            url2 = 'http://www.example.com/redirected'
            req = Request(url, method=method)
            rsp = Response(url, headers={'Location': url2}, status=status)
            req2 = self.mw.process_response(req, rsp, self.spider)
            assert isinstance(req2, Request)
            self.assertEqual(req2.url, url2)
            self.assertEqual(req2.method, method)
            del rsp.headers['Location']
            assert self.mw.process_response(req, rsp, self.spider) is rsp
        _test('GET')
        _test('POST')
        _test('HEAD')
        _test('GET', status=307)
        _test('POST', status=307)
        _test('HEAD', status=307)
        _test('GET', status=308)
        _test('POST', status=308)
        _test('HEAD', status=308)

    def test_dont_redirect(self):
        if False:
            return 10
        url = 'http://www.example.com/301'
        url2 = 'http://www.example.com/redirected'
        req = Request(url, meta={'dont_redirect': True})
        rsp = Response(url, headers={'Location': url2}, status=301)
        r = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(r, Response)
        assert r is rsp
        req = Request(url, meta={'dont_redirect': False})
        rsp = Response(url2, status=200)
        r = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(r, Response)
        assert r is rsp

    def test_redirect_302(self):
        if False:
            i = 10
            return i + 15
        url = 'http://www.example.com/302'
        url2 = 'http://www.example.com/redirected2'
        req = Request(url, method='POST', body='test', headers={'Content-Type': 'text/plain', 'Content-length': '4'})
        rsp = Response(url, headers={'Location': url2}, status=302)
        req2 = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req2, Request)
        self.assertEqual(req2.url, url2)
        self.assertEqual(req2.method, 'GET')
        assert 'Content-Type' not in req2.headers, 'Content-Type header must not be present in redirected request'
        assert 'Content-Length' not in req2.headers, 'Content-Length header must not be present in redirected request'
        assert not req2.body, f"Redirected body must be empty, not '{req2.body}'"
        del rsp.headers['Location']
        assert self.mw.process_response(req, rsp, self.spider) is rsp

    def test_redirect_302_head(self):
        if False:
            return 10
        url = 'http://www.example.com/302'
        url2 = 'http://www.example.com/redirected2'
        req = Request(url, method='HEAD')
        rsp = Response(url, headers={'Location': url2}, status=302)
        req2 = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req2, Request)
        self.assertEqual(req2.url, url2)
        self.assertEqual(req2.method, 'HEAD')
        del rsp.headers['Location']
        assert self.mw.process_response(req, rsp, self.spider) is rsp

    def test_redirect_302_relative(self):
        if False:
            return 10
        url = 'http://www.example.com/302'
        url2 = '///i8n.example2.com/302'
        url3 = 'http://i8n.example2.com/302'
        req = Request(url, method='HEAD')
        rsp = Response(url, headers={'Location': url2}, status=302)
        req2 = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req2, Request)
        self.assertEqual(req2.url, url3)
        self.assertEqual(req2.method, 'HEAD')
        del rsp.headers['Location']
        assert self.mw.process_response(req, rsp, self.spider) is rsp

    def test_max_redirect_times(self):
        if False:
            while True:
                i = 10
        self.mw.max_redirect_times = 1
        req = Request('http://scrapytest.org/302')
        rsp = Response('http://scrapytest.org/302', headers={'Location': '/redirected'}, status=302)
        req = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req, Request)
        assert 'redirect_times' in req.meta
        self.assertEqual(req.meta['redirect_times'], 1)
        self.assertRaises(IgnoreRequest, self.mw.process_response, req, rsp, self.spider)

    def test_ttl(self):
        if False:
            i = 10
            return i + 15
        self.mw.max_redirect_times = 100
        req = Request('http://scrapytest.org/302', meta={'redirect_ttl': 1})
        rsp = Response('http://www.scrapytest.org/302', headers={'Location': '/redirected'}, status=302)
        req = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req, Request)
        self.assertRaises(IgnoreRequest, self.mw.process_response, req, rsp, self.spider)

    def test_redirect_urls(self):
        if False:
            return 10
        req1 = Request('http://scrapytest.org/first')
        rsp1 = Response('http://scrapytest.org/first', headers={'Location': '/redirected'}, status=302)
        req2 = self.mw.process_response(req1, rsp1, self.spider)
        rsp2 = Response('http://scrapytest.org/redirected', headers={'Location': '/redirected2'}, status=302)
        req3 = self.mw.process_response(req2, rsp2, self.spider)
        self.assertEqual(req2.url, 'http://scrapytest.org/redirected')
        self.assertEqual(req2.meta['redirect_urls'], ['http://scrapytest.org/first'])
        self.assertEqual(req3.url, 'http://scrapytest.org/redirected2')
        self.assertEqual(req3.meta['redirect_urls'], ['http://scrapytest.org/first', 'http://scrapytest.org/redirected'])

    def test_redirect_reasons(self):
        if False:
            return 10
        req1 = Request('http://scrapytest.org/first')
        rsp1 = Response('http://scrapytest.org/first', headers={'Location': '/redirected1'}, status=301)
        req2 = self.mw.process_response(req1, rsp1, self.spider)
        rsp2 = Response('http://scrapytest.org/redirected1', headers={'Location': '/redirected2'}, status=301)
        req3 = self.mw.process_response(req2, rsp2, self.spider)
        self.assertEqual(req2.meta['redirect_reasons'], [301])
        self.assertEqual(req3.meta['redirect_reasons'], [301, 301])

    def test_spider_handling(self):
        if False:
            return 10
        smartspider = self.crawler._create_spider('smarty')
        smartspider.handle_httpstatus_list = [404, 301, 302]
        url = 'http://www.example.com/301'
        url2 = 'http://www.example.com/redirected'
        req = Request(url)
        rsp = Response(url, headers={'Location': url2}, status=301)
        r = self.mw.process_response(req, rsp, smartspider)
        self.assertIs(r, rsp)

    def test_request_meta_handling(self):
        if False:
            i = 10
            return i + 15
        url = 'http://www.example.com/301'
        url2 = 'http://www.example.com/redirected'

        def _test_passthrough(req):
            if False:
                i = 10
                return i + 15
            rsp = Response(url, headers={'Location': url2}, status=301, request=req)
            r = self.mw.process_response(req, rsp, self.spider)
            self.assertIs(r, rsp)
        _test_passthrough(Request(url, meta={'handle_httpstatus_list': [404, 301, 302]}))
        _test_passthrough(Request(url, meta={'handle_httpstatus_all': True}))

    def test_latin1_location(self):
        if False:
            while True:
                i = 10
        req = Request('http://scrapytest.org/first')
        latin1_location = '/ação'.encode('latin1')
        resp = Response('http://scrapytest.org/first', headers={'Location': latin1_location}, status=302)
        req_result = self.mw.process_response(req, resp, self.spider)
        perc_encoded_utf8_url = 'http://scrapytest.org/a%E7%E3o'
        self.assertEqual(perc_encoded_utf8_url, req_result.url)

    def test_utf8_location(self):
        if False:
            while True:
                i = 10
        req = Request('http://scrapytest.org/first')
        utf8_location = '/ação'.encode('utf-8')
        resp = Response('http://scrapytest.org/first', headers={'Location': utf8_location}, status=302)
        req_result = self.mw.process_response(req, resp, self.spider)
        perc_encoded_utf8_url = 'http://scrapytest.org/a%C3%A7%C3%A3o'
        self.assertEqual(perc_encoded_utf8_url, req_result.url)

class MetaRefreshMiddlewareTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        crawler = get_crawler(Spider)
        self.spider = crawler._create_spider('foo')
        self.mw = MetaRefreshMiddleware.from_crawler(crawler)

    def _body(self, interval=5, url='http://example.org/newpage'):
        if False:
            for i in range(10):
                print('nop')
        html = f'<html><head><meta http-equiv="refresh" content="{interval};url={url}"/></head></html>'
        return html.encode('utf-8')

    def test_priority_adjust(self):
        if False:
            return 10
        req = Request('http://a.com')
        rsp = HtmlResponse(req.url, body=self._body())
        req2 = self.mw.process_response(req, rsp, self.spider)
        assert req2.priority > req.priority

    def test_meta_refresh(self):
        if False:
            while True:
                i = 10
        req = Request(url='http://example.org')
        rsp = HtmlResponse(req.url, body=self._body())
        req2 = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req2, Request)
        self.assertEqual(req2.url, 'http://example.org/newpage')

    def test_meta_refresh_with_high_interval(self):
        if False:
            return 10
        req = Request(url='http://example.org')
        rsp = HtmlResponse(url='http://example.org', body=self._body(interval=1000), encoding='utf-8')
        rsp2 = self.mw.process_response(req, rsp, self.spider)
        assert rsp is rsp2

    def test_meta_refresh_trough_posted_request(self):
        if False:
            i = 10
            return i + 15
        req = Request(url='http://example.org', method='POST', body='test', headers={'Content-Type': 'text/plain', 'Content-length': '4'})
        rsp = HtmlResponse(req.url, body=self._body())
        req2 = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req2, Request)
        self.assertEqual(req2.url, 'http://example.org/newpage')
        self.assertEqual(req2.method, 'GET')
        assert 'Content-Type' not in req2.headers, 'Content-Type header must not be present in redirected request'
        assert 'Content-Length' not in req2.headers, 'Content-Length header must not be present in redirected request'
        assert not req2.body, f"Redirected body must be empty, not '{req2.body}'"

    def test_max_redirect_times(self):
        if False:
            for i in range(10):
                print('nop')
        self.mw.max_redirect_times = 1
        req = Request('http://scrapytest.org/max')
        rsp = HtmlResponse(req.url, body=self._body())
        req = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req, Request)
        assert 'redirect_times' in req.meta
        self.assertEqual(req.meta['redirect_times'], 1)
        self.assertRaises(IgnoreRequest, self.mw.process_response, req, rsp, self.spider)

    def test_ttl(self):
        if False:
            for i in range(10):
                print('nop')
        self.mw.max_redirect_times = 100
        req = Request('http://scrapytest.org/302', meta={'redirect_ttl': 1})
        rsp = HtmlResponse(req.url, body=self._body())
        req = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req, Request)
        self.assertRaises(IgnoreRequest, self.mw.process_response, req, rsp, self.spider)

    def test_redirect_urls(self):
        if False:
            for i in range(10):
                print('nop')
        req1 = Request('http://scrapytest.org/first')
        rsp1 = HtmlResponse(req1.url, body=self._body(url='/redirected'))
        req2 = self.mw.process_response(req1, rsp1, self.spider)
        assert isinstance(req2, Request), req2
        rsp2 = HtmlResponse(req2.url, body=self._body(url='/redirected2'))
        req3 = self.mw.process_response(req2, rsp2, self.spider)
        assert isinstance(req3, Request), req3
        self.assertEqual(req2.url, 'http://scrapytest.org/redirected')
        self.assertEqual(req2.meta['redirect_urls'], ['http://scrapytest.org/first'])
        self.assertEqual(req3.url, 'http://scrapytest.org/redirected2')
        self.assertEqual(req3.meta['redirect_urls'], ['http://scrapytest.org/first', 'http://scrapytest.org/redirected'])

    def test_redirect_reasons(self):
        if False:
            i = 10
            return i + 15
        req1 = Request('http://scrapytest.org/first')
        rsp1 = HtmlResponse('http://scrapytest.org/first', body=self._body(url='/redirected'))
        req2 = self.mw.process_response(req1, rsp1, self.spider)
        rsp2 = HtmlResponse('http://scrapytest.org/redirected', body=self._body(url='/redirected1'))
        req3 = self.mw.process_response(req2, rsp2, self.spider)
        self.assertEqual(req2.meta['redirect_reasons'], ['meta refresh'])
        self.assertEqual(req3.meta['redirect_reasons'], ['meta refresh', 'meta refresh'])

    def test_ignore_tags_default(self):
        if False:
            for i in range(10):
                print('nop')
        req = Request(url='http://example.org')
        body = '<noscript><meta http-equiv="refresh" content="0;URL=\'http://example.org/newpage\'"></noscript>'
        rsp = HtmlResponse(req.url, body=body.encode())
        req2 = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req2, Request)
        self.assertEqual(req2.url, 'http://example.org/newpage')

    def test_ignore_tags_1_x_list(self):
        if False:
            print('Hello World!')
        'Test that Scrapy 1.x behavior remains possible'
        settings = {'METAREFRESH_IGNORE_TAGS': ['script', 'noscript']}
        crawler = get_crawler(Spider, settings)
        mw = MetaRefreshMiddleware.from_crawler(crawler)
        req = Request(url='http://example.org')
        body = '<noscript><meta http-equiv="refresh" content="0;URL=\'http://example.org/newpage\'"></noscript>'
        rsp = HtmlResponse(req.url, body=body.encode())
        response = mw.process_response(req, rsp, self.spider)
        assert isinstance(response, Response)
if __name__ == '__main__':
    unittest.main()