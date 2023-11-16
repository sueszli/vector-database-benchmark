from gzip import GzipFile
from io import BytesIO
from pathlib import Path
from unittest import SkipTest, TestCase
from w3lib.encoding import resolve_encoding
from scrapy.downloadermiddlewares.httpcompression import ACCEPTED_ENCODINGS, HttpCompressionMiddleware
from scrapy.exceptions import NotConfigured
from scrapy.http import HtmlResponse, Request, Response
from scrapy.responsetypes import responsetypes
from scrapy.spiders import Spider
from scrapy.utils.gz import gunzip
from scrapy.utils.test import get_crawler
from tests import tests_datadir
SAMPLEDIR = Path(tests_datadir, 'compressed')
FORMAT = {'gzip': ('html-gzip.bin', 'gzip'), 'x-gzip': ('html-gzip.bin', 'gzip'), 'rawdeflate': ('html-rawdeflate.bin', 'deflate'), 'zlibdeflate': ('html-zlibdeflate.bin', 'deflate'), 'br': ('html-br.bin', 'br'), 'zstd-static-content-size': ('html-zstd-static-content-size.bin', 'zstd'), 'zstd-static-no-content-size': ('html-zstd-static-no-content-size.bin', 'zstd'), 'zstd-streaming-no-content-size': ('html-zstd-streaming-no-content-size.bin', 'zstd')}

class HttpCompressionTest(TestCase):

    def setUp(self):
        if False:
            return 10
        self.crawler = get_crawler(Spider)
        self.spider = self.crawler._create_spider('scrapytest.org')
        self.mw = HttpCompressionMiddleware.from_crawler(self.crawler)
        self.crawler.stats.open_spider(self.spider)

    def _getresponse(self, coding):
        if False:
            return 10
        if coding not in FORMAT:
            raise ValueError()
        (samplefile, contentencoding) = FORMAT[coding]
        body = (SAMPLEDIR / samplefile).read_bytes()
        headers = {'Server': 'Yaws/1.49 Yet Another Web Server', 'Date': 'Sun, 08 Mar 2009 00:41:03 GMT', 'Content-Length': len(body), 'Content-Type': 'text/html', 'Content-Encoding': contentencoding}
        response = Response('http://scrapytest.org/', body=body, headers=headers)
        response.request = Request('http://scrapytest.org', headers={'Accept-Encoding': 'gzip, deflate'})
        return response

    def assertStatsEqual(self, key, value):
        if False:
            print('Hello World!')
        self.assertEqual(self.crawler.stats.get_value(key, spider=self.spider), value, str(self.crawler.stats.get_stats(self.spider)))

    def test_setting_false_compression_enabled(self):
        if False:
            print('Hello World!')
        self.assertRaises(NotConfigured, HttpCompressionMiddleware.from_crawler, get_crawler(settings_dict={'COMPRESSION_ENABLED': False}))

    def test_setting_default_compression_enabled(self):
        if False:
            i = 10
            return i + 15
        self.assertIsInstance(HttpCompressionMiddleware.from_crawler(get_crawler()), HttpCompressionMiddleware)

    def test_setting_true_compression_enabled(self):
        if False:
            while True:
                i = 10
        self.assertIsInstance(HttpCompressionMiddleware.from_crawler(get_crawler(settings_dict={'COMPRESSION_ENABLED': True})), HttpCompressionMiddleware)

    def test_process_request(self):
        if False:
            for i in range(10):
                print('nop')
        request = Request('http://scrapytest.org')
        assert 'Accept-Encoding' not in request.headers
        self.mw.process_request(request, self.spider)
        self.assertEqual(request.headers.get('Accept-Encoding'), b', '.join(ACCEPTED_ENCODINGS))

    def test_process_response_gzip(self):
        if False:
            for i in range(10):
                print('nop')
        response = self._getresponse('gzip')
        request = response.request
        self.assertEqual(response.headers['Content-Encoding'], b'gzip')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        assert newresponse.body.startswith(b'<!DOCTYPE')
        assert 'Content-Encoding' not in newresponse.headers
        self.assertStatsEqual('httpcompression/response_count', 1)
        self.assertStatsEqual('httpcompression/response_bytes', 74837)

    def test_process_response_gzip_no_stats(self):
        if False:
            print('Hello World!')
        mw = HttpCompressionMiddleware()
        response = self._getresponse('gzip')
        request = response.request
        self.assertEqual(response.headers['Content-Encoding'], b'gzip')
        newresponse = mw.process_response(request, response, self.spider)
        self.assertEqual(mw.stats, None)
        assert newresponse is not response
        assert newresponse.body.startswith(b'<!DOCTYPE')
        assert 'Content-Encoding' not in newresponse.headers

    def test_process_response_br(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            import brotli
        except ImportError:
            raise SkipTest('no brotli')
        response = self._getresponse('br')
        request = response.request
        self.assertEqual(response.headers['Content-Encoding'], b'br')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        assert newresponse.body.startswith(b'<!DOCTYPE')
        assert 'Content-Encoding' not in newresponse.headers
        self.assertStatsEqual('httpcompression/response_count', 1)
        self.assertStatsEqual('httpcompression/response_bytes', 74837)

    def test_process_response_zstd(self):
        if False:
            i = 10
            return i + 15
        try:
            import zstandard
        except ImportError:
            raise SkipTest('no zstd support (zstandard)')
        raw_content = None
        for check_key in FORMAT:
            if not check_key.startswith('zstd-'):
                continue
            response = self._getresponse(check_key)
            request = response.request
            self.assertEqual(response.headers['Content-Encoding'], b'zstd')
            newresponse = self.mw.process_response(request, response, self.spider)
            if raw_content is None:
                raw_content = newresponse.body
            else:
                assert raw_content == newresponse.body
            assert newresponse is not response
            assert newresponse.body.startswith(b'<!DOCTYPE')
            assert 'Content-Encoding' not in newresponse.headers

    def test_process_response_rawdeflate(self):
        if False:
            for i in range(10):
                print('nop')
        response = self._getresponse('rawdeflate')
        request = response.request
        self.assertEqual(response.headers['Content-Encoding'], b'deflate')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        assert newresponse.body.startswith(b'<!DOCTYPE')
        assert 'Content-Encoding' not in newresponse.headers
        self.assertStatsEqual('httpcompression/response_count', 1)
        self.assertStatsEqual('httpcompression/response_bytes', 74840)

    def test_process_response_zlibdelate(self):
        if False:
            while True:
                i = 10
        response = self._getresponse('zlibdeflate')
        request = response.request
        self.assertEqual(response.headers['Content-Encoding'], b'deflate')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        assert newresponse.body.startswith(b'<!DOCTYPE')
        assert 'Content-Encoding' not in newresponse.headers
        self.assertStatsEqual('httpcompression/response_count', 1)
        self.assertStatsEqual('httpcompression/response_bytes', 74840)

    def test_process_response_plain(self):
        if False:
            print('Hello World!')
        response = Response('http://scrapytest.org', body=b'<!DOCTYPE...')
        request = Request('http://scrapytest.org')
        assert not response.headers.get('Content-Encoding')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is response
        assert newresponse.body.startswith(b'<!DOCTYPE')
        self.assertStatsEqual('httpcompression/response_count', None)
        self.assertStatsEqual('httpcompression/response_bytes', None)

    def test_multipleencodings(self):
        if False:
            print('Hello World!')
        response = self._getresponse('gzip')
        response.headers['Content-Encoding'] = ['uuencode', 'gzip']
        request = response.request
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        self.assertEqual(newresponse.headers.getlist('Content-Encoding'), [b'uuencode'])

    def test_process_response_encoding_inside_body(self):
        if False:
            while True:
                i = 10
        headers = {'Content-Type': 'text/html', 'Content-Encoding': 'gzip'}
        f = BytesIO()
        plainbody = b'<html><head><title>Some page</title><meta http-equiv="Content-Type" content="text/html; charset=gb2312">'
        zf = GzipFile(fileobj=f, mode='wb')
        zf.write(plainbody)
        zf.close()
        response = Response('http;//www.example.com/', headers=headers, body=f.getvalue())
        request = Request('http://www.example.com/')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert isinstance(newresponse, HtmlResponse)
        self.assertEqual(newresponse.body, plainbody)
        self.assertEqual(newresponse.encoding, resolve_encoding('gb2312'))
        self.assertStatsEqual('httpcompression/response_count', 1)
        self.assertStatsEqual('httpcompression/response_bytes', len(plainbody))

    def test_process_response_force_recalculate_encoding(self):
        if False:
            print('Hello World!')
        headers = {'Content-Type': 'text/html', 'Content-Encoding': 'gzip'}
        f = BytesIO()
        plainbody = b'<html><head><title>Some page</title><meta http-equiv="Content-Type" content="text/html; charset=gb2312">'
        zf = GzipFile(fileobj=f, mode='wb')
        zf.write(plainbody)
        zf.close()
        response = HtmlResponse('http;//www.example.com/page.html', headers=headers, body=f.getvalue())
        request = Request('http://www.example.com/')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert isinstance(newresponse, HtmlResponse)
        self.assertEqual(newresponse.body, plainbody)
        self.assertEqual(newresponse.encoding, resolve_encoding('gb2312'))
        self.assertStatsEqual('httpcompression/response_count', 1)
        self.assertStatsEqual('httpcompression/response_bytes', len(plainbody))

    def test_process_response_no_content_type_header(self):
        if False:
            for i in range(10):
                print('nop')
        headers = {'Content-Encoding': 'identity'}
        plainbody = b'<html><head><title>Some page</title><meta http-equiv="Content-Type" content="text/html; charset=gb2312">'
        respcls = responsetypes.from_args(url='http://www.example.com/index', headers=headers, body=plainbody)
        response = respcls('http://www.example.com/index', headers=headers, body=plainbody)
        request = Request('http://www.example.com/index')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert isinstance(newresponse, respcls)
        self.assertEqual(newresponse.body, plainbody)
        self.assertEqual(newresponse.encoding, resolve_encoding('gb2312'))
        self.assertStatsEqual('httpcompression/response_count', 1)
        self.assertStatsEqual('httpcompression/response_bytes', len(plainbody))

    def test_process_response_gzipped_contenttype(self):
        if False:
            return 10
        response = self._getresponse('gzip')
        response.headers['Content-Type'] = 'application/gzip'
        request = response.request
        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertIsNot(newresponse, response)
        self.assertTrue(newresponse.body.startswith(b'<!DOCTYPE'))
        self.assertNotIn('Content-Encoding', newresponse.headers)
        self.assertStatsEqual('httpcompression/response_count', 1)
        self.assertStatsEqual('httpcompression/response_bytes', 74837)

    def test_process_response_gzip_app_octetstream_contenttype(self):
        if False:
            print('Hello World!')
        response = self._getresponse('gzip')
        response.headers['Content-Type'] = 'application/octet-stream'
        request = response.request
        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertIsNot(newresponse, response)
        self.assertTrue(newresponse.body.startswith(b'<!DOCTYPE'))
        self.assertNotIn('Content-Encoding', newresponse.headers)
        self.assertStatsEqual('httpcompression/response_count', 1)
        self.assertStatsEqual('httpcompression/response_bytes', 74837)

    def test_process_response_gzip_binary_octetstream_contenttype(self):
        if False:
            return 10
        response = self._getresponse('x-gzip')
        response.headers['Content-Type'] = 'binary/octet-stream'
        request = response.request
        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertIsNot(newresponse, response)
        self.assertTrue(newresponse.body.startswith(b'<!DOCTYPE'))
        self.assertNotIn('Content-Encoding', newresponse.headers)
        self.assertStatsEqual('httpcompression/response_count', 1)
        self.assertStatsEqual('httpcompression/response_bytes', 74837)

    def test_process_response_gzipped_gzip_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that a gzip Content-Encoded .gz file is gunzipped\n        only once by the middleware, leaving gunzipping of the file\n        to upper layers.\n        '
        headers = {'Content-Type': 'application/gzip', 'Content-Encoding': 'gzip'}
        f = BytesIO()
        plainbody = b'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.google.com/schemas/sitemap/0.84">\n  <url>\n    <loc>http://www.example.com/</loc>\n    <lastmod>2009-08-16</lastmod>\n    <changefreq>daily</changefreq>\n    <priority>1</priority>\n  </url>\n  <url>\n    <loc>http://www.example.com/Special-Offers.html</loc>\n    <lastmod>2009-08-16</lastmod>\n    <changefreq>weekly</changefreq>\n    <priority>0.8</priority>\n  </url>\n</urlset>'
        gz_file = GzipFile(fileobj=f, mode='wb')
        gz_file.write(plainbody)
        gz_file.close()
        r = BytesIO()
        gz_resp = GzipFile(fileobj=r, mode='wb')
        gz_resp.write(f.getvalue())
        gz_resp.close()
        response = Response('http;//www.example.com/', headers=headers, body=r.getvalue())
        request = Request('http://www.example.com/')
        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertEqual(gunzip(newresponse.body), plainbody)
        self.assertStatsEqual('httpcompression/response_count', 1)
        self.assertStatsEqual('httpcompression/response_bytes', 230)

    def test_process_response_head_request_no_decode_required(self):
        if False:
            i = 10
            return i + 15
        response = self._getresponse('gzip')
        response.headers['Content-Type'] = 'application/gzip'
        request = response.request
        request.method = 'HEAD'
        response = response.replace(body=None)
        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertIs(newresponse, response)
        self.assertEqual(response.body, b'')
        self.assertStatsEqual('httpcompression/response_count', None)
        self.assertStatsEqual('httpcompression/response_bytes', None)