import shutil
from pathlib import Path
from typing import Optional, Set
from testfixtures import LogCapture
from twisted.internet import defer
from twisted.trial.unittest import TestCase
from w3lib.url import add_or_replace_parameter
from scrapy import signals
from scrapy.crawler import CrawlerRunner
from tests.mockserver import MockServer
from tests.spiders import SimpleSpider

class MediaDownloadSpider(SimpleSpider):
    name = 'mediadownload'

    def _process_url(self, url):
        if False:
            for i in range(10):
                print('nop')
        return url

    def parse(self, response):
        if False:
            for i in range(10):
                print('nop')
        self.logger.info(response.headers)
        self.logger.info(response.text)
        item = {self.media_key: [], self.media_urls_key: [self._process_url(response.urljoin(href)) for href in response.xpath('//table[thead/tr/th="Filename"]/tbody//a/@href').getall()]}
        yield item

class BrokenLinksMediaDownloadSpider(MediaDownloadSpider):
    name = 'brokenmedia'

    def _process_url(self, url):
        if False:
            while True:
                i = 10
        return url + '.foo'

class RedirectedMediaDownloadSpider(MediaDownloadSpider):
    name = 'redirectedmedia'

    def _process_url(self, url):
        if False:
            i = 10
            return i + 15
        return add_or_replace_parameter(self.mockserver.url('/redirect-to'), 'goto', url)

class FileDownloadCrawlTestCase(TestCase):
    pipeline_class = 'scrapy.pipelines.files.FilesPipeline'
    store_setting_key = 'FILES_STORE'
    media_key = 'files'
    media_urls_key = 'file_urls'
    expected_checksums: Optional[Set[str]] = {'5547178b89448faf0015a13f904c936e', 'c2281c83670e31d8aaab7cb642b824db', 'ed3f6538dc15d4d9179dae57319edc5f'}

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.mockserver = MockServer()
        self.mockserver.__enter__()
        self.tmpmediastore = Path(self.mktemp())
        self.tmpmediastore.mkdir()
        self.settings = {'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7', 'ITEM_PIPELINES': {self.pipeline_class: 1}, self.store_setting_key: str(self.tmpmediastore)}
        self.runner = CrawlerRunner(self.settings)
        self.items = []

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.tmpmediastore)
        self.items = []
        self.mockserver.__exit__(None, None, None)

    def _on_item_scraped(self, item):
        if False:
            print('Hello World!')
        self.items.append(item)

    def _create_crawler(self, spider_class, runner=None, **kwargs):
        if False:
            while True:
                i = 10
        if runner is None:
            runner = self.runner
        crawler = runner.create_crawler(spider_class, **kwargs)
        crawler.signals.connect(self._on_item_scraped, signals.item_scraped)
        return crawler

    def _assert_files_downloaded(self, items, logs):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(items), 1)
        self.assertIn(self.media_key, items[0])
        file_dl_success = 'File (downloaded): Downloaded file from'
        self.assertEqual(logs.count(file_dl_success), 3)
        for item in items:
            for i in item[self.media_key]:
                self.assertEqual(i['status'], 'downloaded')
        if self.expected_checksums is not None:
            checksums = set((i['checksum'] for item in items for i in item[self.media_key]))
            self.assertEqual(checksums, self.expected_checksums)
        for item in items:
            for i in item[self.media_key]:
                self.assertTrue((self.tmpmediastore / i['path']).exists())

    def _assert_files_download_failure(self, crawler, items, code, logs):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(items), 1)
        self.assertIn(self.media_key, items[0])
        self.assertFalse(items[0][self.media_key])
        self.assertEqual(crawler.stats.get_value('downloader/request_method_count/GET'), 4)
        self.assertEqual(crawler.stats.get_value('downloader/response_count'), 4)
        self.assertEqual(crawler.stats.get_value('downloader/response_status_count/200'), 1)
        self.assertEqual(crawler.stats.get_value(f'downloader/response_status_count/{code}'), 3)
        file_dl_failure = f'File (code: {code}): Error downloading file from'
        self.assertEqual(logs.count(file_dl_failure), 3)
        self.assertEqual([x for x in self.tmpmediastore.iterdir()], [])

    @defer.inlineCallbacks
    def test_download_media(self):
        if False:
            while True:
                i = 10
        crawler = self._create_crawler(MediaDownloadSpider)
        with LogCapture() as log:
            yield crawler.crawl(self.mockserver.url('/files/images/'), media_key=self.media_key, media_urls_key=self.media_urls_key)
        self._assert_files_downloaded(self.items, str(log))

    @defer.inlineCallbacks
    def test_download_media_wrong_urls(self):
        if False:
            for i in range(10):
                print('nop')
        crawler = self._create_crawler(BrokenLinksMediaDownloadSpider)
        with LogCapture() as log:
            yield crawler.crawl(self.mockserver.url('/files/images/'), media_key=self.media_key, media_urls_key=self.media_urls_key)
        self._assert_files_download_failure(crawler, self.items, 404, str(log))

    @defer.inlineCallbacks
    def test_download_media_redirected_default_failure(self):
        if False:
            return 10
        crawler = self._create_crawler(RedirectedMediaDownloadSpider)
        with LogCapture() as log:
            yield crawler.crawl(self.mockserver.url('/files/images/'), media_key=self.media_key, media_urls_key=self.media_urls_key, mockserver=self.mockserver)
        self._assert_files_download_failure(crawler, self.items, 302, str(log))

    @defer.inlineCallbacks
    def test_download_media_redirected_allowed(self):
        if False:
            i = 10
            return i + 15
        settings = dict(self.settings)
        settings.update({'MEDIA_ALLOW_REDIRECTS': True})
        runner = CrawlerRunner(settings)
        crawler = self._create_crawler(RedirectedMediaDownloadSpider, runner=runner)
        with LogCapture() as log:
            yield crawler.crawl(self.mockserver.url('/files/images/'), media_key=self.media_key, media_urls_key=self.media_urls_key, mockserver=self.mockserver)
        self._assert_files_downloaded(self.items, str(log))
        self.assertEqual(crawler.stats.get_value('downloader/response_status_count/302'), 3)
skip_pillow: Optional[str]
try:
    from PIL import Image
except ImportError:
    skip_pillow = 'Missing Python Imaging Library, install https://pypi.python.org/pypi/Pillow'
else:
    skip_pillow = None

class ImageDownloadCrawlTestCase(FileDownloadCrawlTestCase):
    skip = skip_pillow
    pipeline_class = 'scrapy.pipelines.images.ImagesPipeline'
    store_setting_key = 'IMAGES_STORE'
    media_key = 'images'
    media_urls_key = 'image_urls'
    expected_checksums = None