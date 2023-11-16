import gzip
import inspect
import warnings
from io import BytesIO
from typing import Any
from unittest import mock
from testfixtures import LogCapture
from twisted.internet.defer import inlineCallbacks
from twisted.trial import unittest
from w3lib.url import safe_url_string
from scrapy import signals
from scrapy.crawler import Crawler
from scrapy.http import HtmlResponse, Request, Response, TextResponse, XmlResponse
from scrapy.linkextractors import LinkExtractor
from scrapy.settings import Settings
from scrapy.spiders import CrawlSpider, CSVFeedSpider, Rule, SitemapSpider, Spider, XMLFeedSpider
from scrapy.spiders.init import InitSpider
from scrapy.utils.test import get_crawler
from tests import get_testdata

class SpiderTest(unittest.TestCase):
    spider_class = Spider

    def setUp(self):
        if False:
            while True:
                i = 10
        warnings.simplefilter('always')

    def tearDown(self):
        if False:
            while True:
                i = 10
        warnings.resetwarnings()

    def test_base_spider(self):
        if False:
            print('Hello World!')
        spider = self.spider_class('example.com')
        self.assertEqual(spider.name, 'example.com')
        self.assertEqual(spider.start_urls, [])

    def test_start_requests(self):
        if False:
            i = 10
            return i + 15
        spider = self.spider_class('example.com')
        start_requests = spider.start_requests()
        self.assertTrue(inspect.isgenerator(start_requests))
        self.assertEqual(list(start_requests), [])

    def test_spider_args(self):
        if False:
            i = 10
            return i + 15
        '``__init__`` method arguments are assigned to spider attributes'
        spider = self.spider_class('example.com', foo='bar')
        self.assertEqual(spider.foo, 'bar')

    def test_spider_without_name(self):
        if False:
            return 10
        '``__init__`` method arguments are assigned to spider attributes'
        self.assertRaises(ValueError, self.spider_class)
        self.assertRaises(ValueError, self.spider_class, somearg='foo')

    def test_from_crawler_crawler_and_settings_population(self):
        if False:
            print('Hello World!')
        crawler = get_crawler()
        spider = self.spider_class.from_crawler(crawler, 'example.com')
        self.assertTrue(hasattr(spider, 'crawler'))
        self.assertIs(spider.crawler, crawler)
        self.assertTrue(hasattr(spider, 'settings'))
        self.assertIs(spider.settings, crawler.settings)

    def test_from_crawler_init_call(self):
        if False:
            print('Hello World!')
        with mock.patch.object(self.spider_class, '__init__', return_value=None) as mock_init:
            self.spider_class.from_crawler(get_crawler(), 'example.com', foo='bar')
            mock_init.assert_called_once_with('example.com', foo='bar')

    def test_closed_signal_call(self):
        if False:
            while True:
                i = 10

        class TestSpider(self.spider_class):
            closed_called = False

            def closed(self, reason):
                if False:
                    return 10
                self.closed_called = True
        crawler = get_crawler()
        spider = TestSpider.from_crawler(crawler, 'example.com')
        crawler.signals.send_catch_log(signal=signals.spider_opened, spider=spider)
        crawler.signals.send_catch_log(signal=signals.spider_closed, spider=spider, reason=None)
        self.assertTrue(spider.closed_called)

    def test_update_settings(self):
        if False:
            for i in range(10):
                print('nop')
        spider_settings = {'TEST1': 'spider', 'TEST2': 'spider'}
        project_settings = {'TEST1': 'project', 'TEST3': 'project'}
        self.spider_class.custom_settings = spider_settings
        settings = Settings(project_settings, priority='project')
        self.spider_class.update_settings(settings)
        self.assertEqual(settings.get('TEST1'), 'spider')
        self.assertEqual(settings.get('TEST2'), 'spider')
        self.assertEqual(settings.get('TEST3'), 'project')

    @inlineCallbacks
    def test_settings_in_from_crawler(self):
        if False:
            i = 10
            return i + 15
        spider_settings = {'TEST1': 'spider', 'TEST2': 'spider'}
        project_settings = {'TEST1': 'project', 'TEST3': 'project'}

        class TestSpider(self.spider_class):
            name = 'test'
            custom_settings = spider_settings

            @classmethod
            def from_crawler(cls, crawler: Crawler, *args: Any, **kwargs: Any):
                if False:
                    for i in range(10):
                        print('nop')
                spider = super().from_crawler(crawler, *args, **kwargs)
                spider.settings.set('TEST1', 'spider_instance', priority='spider')
                return spider
        crawler = Crawler(TestSpider, project_settings)
        self.assertEqual(crawler.settings.get('TEST1'), 'spider')
        self.assertEqual(crawler.settings.get('TEST2'), 'spider')
        self.assertEqual(crawler.settings.get('TEST3'), 'project')
        yield crawler.crawl()
        self.assertEqual(crawler.settings.get('TEST1'), 'spider_instance')

    def test_logger(self):
        if False:
            return 10
        spider = self.spider_class('example.com')
        with LogCapture() as lc:
            spider.logger.info('test log msg')
        lc.check(('example.com', 'INFO', 'test log msg'))
        record = lc.records[0]
        self.assertIn('spider', record.__dict__)
        self.assertIs(record.spider, spider)

    def test_log(self):
        if False:
            return 10
        spider = self.spider_class('example.com')
        with mock.patch('scrapy.spiders.Spider.logger') as mock_logger:
            spider.log('test log msg', 'INFO')
        mock_logger.log.assert_called_once_with('INFO', 'test log msg')

class InitSpiderTest(SpiderTest):
    spider_class = InitSpider

class XMLFeedSpiderTest(SpiderTest):
    spider_class = XMLFeedSpider

    def test_register_namespace(self):
        if False:
            print('Hello World!')
        body = b'<?xml version="1.0" encoding="UTF-8"?>\n        <urlset xmlns:x="http://www.google.com/schemas/sitemap/0.84"\n                xmlns:y="http://www.example.com/schemas/extras/1.0">\n        <url><x:loc>http://www.example.com/Special-Offers.html</loc><y:updated>2009-08-16</updated>\n            <other value="bar" y:custom="fuu"/>\n        </url>\n        <url><loc>http://www.example.com/</loc><y:updated>2009-08-16</updated><other value="foo"/></url>\n        </urlset>'
        response = XmlResponse(url='http://example.com/sitemap.xml', body=body)

        class _XMLSpider(self.spider_class):
            itertag = 'url'
            namespaces = (('a', 'http://www.google.com/schemas/sitemap/0.84'), ('b', 'http://www.example.com/schemas/extras/1.0'))

            def parse_node(self, response, selector):
                if False:
                    i = 10
                    return i + 15
                yield {'loc': selector.xpath('a:loc/text()').getall(), 'updated': selector.xpath('b:updated/text()').getall(), 'other': selector.xpath('other/@value').getall(), 'custom': selector.xpath('other/@b:custom').getall()}
        for iterator in ('iternodes', 'xml'):
            spider = _XMLSpider('example', iterator=iterator)
            output = list(spider._parse(response))
            self.assertEqual(len(output), 2, iterator)
            self.assertEqual(output, [{'loc': ['http://www.example.com/Special-Offers.html'], 'updated': ['2009-08-16'], 'custom': ['fuu'], 'other': ['bar']}, {'loc': [], 'updated': ['2009-08-16'], 'other': ['foo'], 'custom': []}], iterator)

class CSVFeedSpiderTest(SpiderTest):
    spider_class = CSVFeedSpider

    def test_parse_rows(self):
        if False:
            print('Hello World!')
        body = get_testdata('feeds', 'feed-sample6.csv')
        response = Response('http://example.org/dummy.csv', body=body)

        class _CrawlSpider(self.spider_class):
            name = 'test'
            delimiter = ','
            quotechar = "'"

            def parse_row(self, response, row):
                if False:
                    print('Hello World!')
                return row
        spider = _CrawlSpider()
        rows = list(spider.parse_rows(response))
        assert rows[0] == {'id': '1', 'name': 'alpha', 'value': 'foobar'}
        assert len(rows) == 4

class CrawlSpiderTest(SpiderTest):
    test_body = b'<html><head><title>Page title<title>\n    <body>\n    <p><a href="item/12.html">Item 12</a></p>\n    <div class=\'links\'>\n    <p><a href="/about.html">About us</a></p>\n    </div>\n    <div>\n    <p><a href="/nofollow.html">This shouldn\'t be followed</a></p>\n    </div>\n    </body></html>'
    spider_class = CrawlSpider

    def test_rule_without_link_extractor(self):
        if False:
            i = 10
            return i + 15
        response = HtmlResponse('http://example.org/somepage/index.html', body=self.test_body)

        class _CrawlSpider(self.spider_class):
            name = 'test'
            allowed_domains = ['example.org']
            rules = (Rule(),)
        spider = _CrawlSpider()
        output = list(spider._requests_to_follow(response))
        self.assertEqual(len(output), 3)
        self.assertTrue(all(map(lambda r: isinstance(r, Request), output)))
        self.assertEqual([r.url for r in output], ['http://example.org/somepage/item/12.html', 'http://example.org/about.html', 'http://example.org/nofollow.html'])

    def test_process_links(self):
        if False:
            while True:
                i = 10
        response = HtmlResponse('http://example.org/somepage/index.html', body=self.test_body)

        class _CrawlSpider(self.spider_class):
            name = 'test'
            allowed_domains = ['example.org']
            rules = (Rule(LinkExtractor(), process_links='dummy_process_links'),)

            def dummy_process_links(self, links):
                if False:
                    for i in range(10):
                        print('nop')
                return links
        spider = _CrawlSpider()
        output = list(spider._requests_to_follow(response))
        self.assertEqual(len(output), 3)
        self.assertTrue(all(map(lambda r: isinstance(r, Request), output)))
        self.assertEqual([r.url for r in output], ['http://example.org/somepage/item/12.html', 'http://example.org/about.html', 'http://example.org/nofollow.html'])

    def test_process_links_filter(self):
        if False:
            print('Hello World!')
        response = HtmlResponse('http://example.org/somepage/index.html', body=self.test_body)

        class _CrawlSpider(self.spider_class):
            import re
            name = 'test'
            allowed_domains = ['example.org']
            rules = (Rule(LinkExtractor(), process_links='filter_process_links'),)
            _test_regex = re.compile('nofollow')

            def filter_process_links(self, links):
                if False:
                    for i in range(10):
                        print('nop')
                return [link for link in links if not self._test_regex.search(link.url)]
        spider = _CrawlSpider()
        output = list(spider._requests_to_follow(response))
        self.assertEqual(len(output), 2)
        self.assertTrue(all(map(lambda r: isinstance(r, Request), output)))
        self.assertEqual([r.url for r in output], ['http://example.org/somepage/item/12.html', 'http://example.org/about.html'])

    def test_process_links_generator(self):
        if False:
            i = 10
            return i + 15
        response = HtmlResponse('http://example.org/somepage/index.html', body=self.test_body)

        class _CrawlSpider(self.spider_class):
            name = 'test'
            allowed_domains = ['example.org']
            rules = (Rule(LinkExtractor(), process_links='dummy_process_links'),)

            def dummy_process_links(self, links):
                if False:
                    print('Hello World!')
                for link in links:
                    yield link
        spider = _CrawlSpider()
        output = list(spider._requests_to_follow(response))
        self.assertEqual(len(output), 3)
        self.assertTrue(all(map(lambda r: isinstance(r, Request), output)))
        self.assertEqual([r.url for r in output], ['http://example.org/somepage/item/12.html', 'http://example.org/about.html', 'http://example.org/nofollow.html'])

    def test_process_request(self):
        if False:
            return 10
        response = HtmlResponse('http://example.org/somepage/index.html', body=self.test_body)

        def process_request_change_domain(request, response):
            if False:
                while True:
                    i = 10
            return request.replace(url=request.url.replace('.org', '.com'))

        class _CrawlSpider(self.spider_class):
            name = 'test'
            allowed_domains = ['example.org']
            rules = (Rule(LinkExtractor(), process_request=process_request_change_domain),)
        spider = _CrawlSpider()
        output = list(spider._requests_to_follow(response))
        self.assertEqual(len(output), 3)
        self.assertTrue(all(map(lambda r: isinstance(r, Request), output)))
        self.assertEqual([r.url for r in output], ['http://example.com/somepage/item/12.html', 'http://example.com/about.html', 'http://example.com/nofollow.html'])

    def test_process_request_with_response(self):
        if False:
            print('Hello World!')
        response = HtmlResponse('http://example.org/somepage/index.html', body=self.test_body)

        def process_request_meta_response_class(request, response):
            if False:
                i = 10
                return i + 15
            request.meta['response_class'] = response.__class__.__name__
            return request

        class _CrawlSpider(self.spider_class):
            name = 'test'
            allowed_domains = ['example.org']
            rules = (Rule(LinkExtractor(), process_request=process_request_meta_response_class),)
        spider = _CrawlSpider()
        output = list(spider._requests_to_follow(response))
        self.assertEqual(len(output), 3)
        self.assertTrue(all(map(lambda r: isinstance(r, Request), output)))
        self.assertEqual([r.url for r in output], ['http://example.org/somepage/item/12.html', 'http://example.org/about.html', 'http://example.org/nofollow.html'])
        self.assertEqual([r.meta['response_class'] for r in output], ['HtmlResponse', 'HtmlResponse', 'HtmlResponse'])

    def test_process_request_instance_method(self):
        if False:
            return 10
        response = HtmlResponse('http://example.org/somepage/index.html', body=self.test_body)

        class _CrawlSpider(self.spider_class):
            name = 'test'
            allowed_domains = ['example.org']
            rules = (Rule(LinkExtractor(), process_request='process_request_upper'),)

            def process_request_upper(self, request, response):
                if False:
                    i = 10
                    return i + 15
                return request.replace(url=request.url.upper())
        spider = _CrawlSpider()
        output = list(spider._requests_to_follow(response))
        self.assertEqual(len(output), 3)
        self.assertTrue(all(map(lambda r: isinstance(r, Request), output)))
        self.assertEqual([r.url for r in output], [safe_url_string('http://EXAMPLE.ORG/SOMEPAGE/ITEM/12.HTML'), safe_url_string('http://EXAMPLE.ORG/ABOUT.HTML'), safe_url_string('http://EXAMPLE.ORG/NOFOLLOW.HTML')])

    def test_process_request_instance_method_with_response(self):
        if False:
            while True:
                i = 10
        response = HtmlResponse('http://example.org/somepage/index.html', body=self.test_body)

        class _CrawlSpider(self.spider_class):
            name = 'test'
            allowed_domains = ['example.org']
            rules = (Rule(LinkExtractor(), process_request='process_request_meta_response_class'),)

            def process_request_meta_response_class(self, request, response):
                if False:
                    return 10
                request.meta['response_class'] = response.__class__.__name__
                return request
        spider = _CrawlSpider()
        output = list(spider._requests_to_follow(response))
        self.assertEqual(len(output), 3)
        self.assertTrue(all(map(lambda r: isinstance(r, Request), output)))
        self.assertEqual([r.url for r in output], ['http://example.org/somepage/item/12.html', 'http://example.org/about.html', 'http://example.org/nofollow.html'])
        self.assertEqual([r.meta['response_class'] for r in output], ['HtmlResponse', 'HtmlResponse', 'HtmlResponse'])

    def test_follow_links_attribute_population(self):
        if False:
            i = 10
            return i + 15
        crawler = get_crawler()
        spider = self.spider_class.from_crawler(crawler, 'example.com')
        self.assertTrue(hasattr(spider, '_follow_links'))
        self.assertTrue(spider._follow_links)
        settings_dict = {'CRAWLSPIDER_FOLLOW_LINKS': False}
        crawler = get_crawler(settings_dict=settings_dict)
        spider = self.spider_class.from_crawler(crawler, 'example.com')
        self.assertTrue(hasattr(spider, '_follow_links'))
        self.assertFalse(spider._follow_links)

    def test_start_url(self):
        if False:
            return 10
        spider = self.spider_class('example.com')
        spider.start_url = 'https://www.example.com'
        with self.assertRaisesRegex(AttributeError, '^Crawling could not start.*$'):
            list(spider.start_requests())

class SitemapSpiderTest(SpiderTest):
    spider_class = SitemapSpider
    BODY = b'SITEMAP'
    f = BytesIO()
    g = gzip.GzipFile(fileobj=f, mode='w+b')
    g.write(BODY)
    g.close()
    GZBODY = f.getvalue()

    def assertSitemapBody(self, response, body):
        if False:
            return 10
        spider = self.spider_class('example.com')
        self.assertEqual(spider._get_sitemap_body(response), body)

    def test_get_sitemap_body(self):
        if False:
            while True:
                i = 10
        r = XmlResponse(url='http://www.example.com/', body=self.BODY)
        self.assertSitemapBody(r, self.BODY)
        r = HtmlResponse(url='http://www.example.com/', body=self.BODY)
        self.assertSitemapBody(r, None)
        r = Response(url='http://www.example.com/favicon.ico', body=self.BODY)
        self.assertSitemapBody(r, None)

    def test_get_sitemap_body_gzip_headers(self):
        if False:
            return 10
        r = Response(url='http://www.example.com/sitemap', body=self.GZBODY, headers={'content-type': 'application/gzip'})
        self.assertSitemapBody(r, self.BODY)

    def test_get_sitemap_body_xml_url(self):
        if False:
            return 10
        r = TextResponse(url='http://www.example.com/sitemap.xml', body=self.BODY)
        self.assertSitemapBody(r, self.BODY)

    def test_get_sitemap_body_xml_url_compressed(self):
        if False:
            return 10
        r = Response(url='http://www.example.com/sitemap.xml.gz', body=self.GZBODY)
        self.assertSitemapBody(r, self.BODY)
        r = Response(url='http://www.example.com/sitemap.xml.gz', body=self.BODY)
        self.assertSitemapBody(r, self.BODY)

    def test_get_sitemap_urls_from_robotstxt(self):
        if False:
            print('Hello World!')
        robots = b'# Sitemap files\nSitemap: http://example.com/sitemap.xml\nSitemap: http://example.com/sitemap-product-index.xml\nSitemap: HTTP://example.com/sitemap-uppercase.xml\nSitemap: /sitemap-relative-url.xml\n'
        r = TextResponse(url='http://www.example.com/robots.txt', body=robots)
        spider = self.spider_class('example.com')
        self.assertEqual([req.url for req in spider._parse_sitemap(r)], ['http://example.com/sitemap.xml', 'http://example.com/sitemap-product-index.xml', 'http://example.com/sitemap-uppercase.xml', 'http://www.example.com/sitemap-relative-url.xml'])

    def test_alternate_url_locs(self):
        if False:
            i = 10
            return i + 15
        sitemap = b'<?xml version="1.0" encoding="UTF-8"?>\n    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"\n        xmlns:xhtml="http://www.w3.org/1999/xhtml">\n        <url>\n            <loc>http://www.example.com/english/</loc>\n            <xhtml:link rel="alternate" hreflang="de"\n                href="http://www.example.com/deutsch/"/>\n            <xhtml:link rel="alternate" hreflang="de-ch"\n                href="http://www.example.com/schweiz-deutsch/"/>\n            <xhtml:link rel="alternate" hreflang="it"\n                href="http://www.example.com/italiano/"/>\n            <xhtml:link rel="alternate" hreflang="it"/><!-- wrong tag without href -->\n        </url>\n    </urlset>'
        r = TextResponse(url='http://www.example.com/sitemap.xml', body=sitemap)
        spider = self.spider_class('example.com')
        self.assertEqual([req.url for req in spider._parse_sitemap(r)], ['http://www.example.com/english/'])
        spider.sitemap_alternate_links = True
        self.assertEqual([req.url for req in spider._parse_sitemap(r)], ['http://www.example.com/english/', 'http://www.example.com/deutsch/', 'http://www.example.com/schweiz-deutsch/', 'http://www.example.com/italiano/'])

    def test_sitemap_filter(self):
        if False:
            for i in range(10):
                print('nop')
        sitemap = b'<?xml version="1.0" encoding="UTF-8"?>\n    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"\n        xmlns:xhtml="http://www.w3.org/1999/xhtml">\n        <url>\n            <loc>http://www.example.com/english/</loc>\n            <lastmod>2010-01-01</lastmod>\n        </url>\n        <url>\n            <loc>http://www.example.com/portuguese/</loc>\n            <lastmod>2005-01-01</lastmod>\n        </url>\n    </urlset>'

        class FilteredSitemapSpider(self.spider_class):

            def sitemap_filter(self, entries):
                if False:
                    print('Hello World!')
                from datetime import datetime
                for entry in entries:
                    date_time = datetime.strptime(entry['lastmod'], '%Y-%m-%d')
                    if date_time.year > 2008:
                        yield entry
        r = TextResponse(url='http://www.example.com/sitemap.xml', body=sitemap)
        spider = self.spider_class('example.com')
        self.assertEqual([req.url for req in spider._parse_sitemap(r)], ['http://www.example.com/english/', 'http://www.example.com/portuguese/'])
        spider = FilteredSitemapSpider('example.com')
        self.assertEqual([req.url for req in spider._parse_sitemap(r)], ['http://www.example.com/english/'])

    def test_sitemap_filter_with_alternate_links(self):
        if False:
            for i in range(10):
                print('nop')
        sitemap = b'<?xml version="1.0" encoding="UTF-8"?>\n    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"\n        xmlns:xhtml="http://www.w3.org/1999/xhtml">\n        <url>\n            <loc>http://www.example.com/english/article_1/</loc>\n            <lastmod>2010-01-01</lastmod>\n            <xhtml:link rel="alternate" hreflang="de"\n                href="http://www.example.com/deutsch/article_1/"/>\n        </url>\n        <url>\n            <loc>http://www.example.com/english/article_2/</loc>\n            <lastmod>2015-01-01</lastmod>\n        </url>\n    </urlset>'

        class FilteredSitemapSpider(self.spider_class):

            def sitemap_filter(self, entries):
                if False:
                    i = 10
                    return i + 15
                for entry in entries:
                    alternate_links = entry.get('alternate', tuple())
                    for link in alternate_links:
                        if '/deutsch/' in link:
                            entry['loc'] = link
                            yield entry
        r = TextResponse(url='http://www.example.com/sitemap.xml', body=sitemap)
        spider = self.spider_class('example.com')
        self.assertEqual([req.url for req in spider._parse_sitemap(r)], ['http://www.example.com/english/article_1/', 'http://www.example.com/english/article_2/'])
        spider = FilteredSitemapSpider('example.com')
        self.assertEqual([req.url for req in spider._parse_sitemap(r)], ['http://www.example.com/deutsch/article_1/'])

    def test_sitemapindex_filter(self):
        if False:
            while True:
                i = 10
        sitemap = b'<?xml version="1.0" encoding="UTF-8"?>\n    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n        <sitemap>\n            <loc>http://www.example.com/sitemap1.xml</loc>\n            <lastmod>2004-01-01T20:00:00+00:00</lastmod>\n        </sitemap>\n        <sitemap>\n            <loc>http://www.example.com/sitemap2.xml</loc>\n            <lastmod>2005-01-01</lastmod>\n        </sitemap>\n    </sitemapindex>'

        class FilteredSitemapSpider(self.spider_class):

            def sitemap_filter(self, entries):
                if False:
                    for i in range(10):
                        print('nop')
                from datetime import datetime
                for entry in entries:
                    date_time = datetime.strptime(entry['lastmod'].split('T')[0], '%Y-%m-%d')
                    if date_time.year > 2004:
                        yield entry
        r = TextResponse(url='http://www.example.com/sitemap.xml', body=sitemap)
        spider = self.spider_class('example.com')
        self.assertEqual([req.url for req in spider._parse_sitemap(r)], ['http://www.example.com/sitemap1.xml', 'http://www.example.com/sitemap2.xml'])
        spider = FilteredSitemapSpider('example.com')
        self.assertEqual([req.url for req in spider._parse_sitemap(r)], ['http://www.example.com/sitemap2.xml'])

class DeprecationTest(unittest.TestCase):

    def test_crawl_spider(self):
        if False:
            return 10
        assert issubclass(CrawlSpider, Spider)
        assert isinstance(CrawlSpider(name='foo'), Spider)

class NoParseMethodSpiderTest(unittest.TestCase):
    spider_class = Spider

    def test_undefined_parse_method(self):
        if False:
            print('Hello World!')
        spider = self.spider_class('example.com')
        text = b'Random text'
        resp = TextResponse(url='http://www.example.com/random_url', body=text)
        exc_msg = 'Spider.parse callback is not defined'
        with self.assertRaisesRegex(NotImplementedError, exc_msg):
            spider.parse(resp)