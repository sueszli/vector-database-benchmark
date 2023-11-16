from unittest import TestCase
from testfixtures import LogCapture
from scrapy.http import Request, Response
from scrapy.settings import Settings
from scrapy.spidermiddlewares.urllength import UrlLengthMiddleware
from scrapy.spiders import Spider
from scrapy.utils.test import get_crawler

class TestUrlLengthMiddleware(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.maxlength = 25
        settings = Settings({'URLLENGTH_LIMIT': self.maxlength})
        crawler = get_crawler(Spider)
        self.spider = crawler._create_spider('foo')
        self.stats = crawler.stats
        self.mw = UrlLengthMiddleware.from_settings(settings)
        self.response = Response('http://scrapytest.org')
        self.short_url_req = Request('http://scrapytest.org/')
        self.long_url_req = Request('http://scrapytest.org/this_is_a_long_url')
        self.reqs = [self.short_url_req, self.long_url_req]

    def process_spider_output(self):
        if False:
            i = 10
            return i + 15
        return list(self.mw.process_spider_output(self.response, self.reqs, self.spider))

    def test_middleware_works(self):
        if False:
            return 10
        self.assertEqual(self.process_spider_output(), [self.short_url_req])

    def test_logging(self):
        if False:
            for i in range(10):
                print('nop')
        with LogCapture() as log:
            self.process_spider_output()
        ric = self.stats.get_value('urllength/request_ignored_count', spider=self.spider)
        self.assertEqual(ric, 1)
        self.assertIn(f'Ignoring link (url length > {self.maxlength})', str(log))