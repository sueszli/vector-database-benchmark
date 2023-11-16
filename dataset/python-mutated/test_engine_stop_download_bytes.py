from testfixtures import LogCapture
from twisted.internet import defer
from scrapy.exceptions import StopDownload
from tests.test_engine import AttrsItemsSpider, CrawlerRun, DataClassItemsSpider, DictItemsSpider, EngineTest, TestSpider

class BytesReceivedCrawlerRun(CrawlerRun):

    def bytes_received(self, data, request, spider):
        if False:
            i = 10
            return i + 15
        super().bytes_received(data, request, spider)
        raise StopDownload(fail=False)

class BytesReceivedEngineTest(EngineTest):

    @defer.inlineCallbacks
    def test_crawler(self):
        if False:
            while True:
                i = 10
        for spider in (TestSpider, DictItemsSpider, AttrsItemsSpider, DataClassItemsSpider):
            run = BytesReceivedCrawlerRun(spider)
            with LogCapture() as log:
                yield run.run()
                log.check_present(('scrapy.core.downloader.handlers.http11', 'DEBUG', f'Download stopped for <GET http://localhost:{run.portno}/redirected> from signal handler BytesReceivedCrawlerRun.bytes_received'))
                log.check_present(('scrapy.core.downloader.handlers.http11', 'DEBUG', f'Download stopped for <GET http://localhost:{run.portno}/> from signal handler BytesReceivedCrawlerRun.bytes_received'))
                log.check_present(('scrapy.core.downloader.handlers.http11', 'DEBUG', f'Download stopped for <GET http://localhost:{run.portno}/numbers> from signal handler BytesReceivedCrawlerRun.bytes_received'))
            self._assert_visited_urls(run)
            self._assert_scheduled_requests(run, count=9)
            self._assert_downloaded_responses(run, count=9)
            self._assert_signals_caught(run)
            self._assert_headers_received(run)
            self._assert_bytes_received(run)

    def _assert_bytes_received(self, run: CrawlerRun):
        if False:
            print('Hello World!')
        self.assertEqual(9, len(run.bytes))
        for (request, data) in run.bytes.items():
            joined_data = b''.join(data)
            self.assertTrue(len(data) == 1)
            if run.getpath(request.url) == '/numbers':
                numbers = [str(x).encode('utf8') for x in range(2 ** 18)]
                self.assertTrue(len(joined_data) < len(b''.join(numbers)))