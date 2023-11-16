import asyncio
import sys
from typing import Optional
from scrapy import Spider
from scrapy.crawler import CrawlerProcess
from scrapy.utils.defer import deferred_from_coro

class UppercasePipeline:

    async def _open_spider(self, spider):
        spider.logger.info('async pipeline opened!')
        await asyncio.sleep(0.1)

    def open_spider(self, spider):
        if False:
            for i in range(10):
                print('nop')
        return deferred_from_coro(self._open_spider(spider))

    def process_item(self, item, spider):
        if False:
            while True:
                i = 10
        return {'url': item['url'].upper()}

class UrlSpider(Spider):
    name = 'url_spider'
    start_urls = ['data:,']
    custom_settings = {'ITEM_PIPELINES': {UppercasePipeline: 100}}

    def parse(self, response):
        if False:
            return 10
        yield {'url': response.url}
if __name__ == '__main__':
    ASYNCIO_EVENT_LOOP: Optional[str]
    try:
        ASYNCIO_EVENT_LOOP = sys.argv[1]
    except IndexError:
        ASYNCIO_EVENT_LOOP = None
    process = CrawlerProcess(settings={'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor', 'ASYNCIO_EVENT_LOOP': ASYNCIO_EVENT_LOOP})
    process.crawl(UrlSpider)
    process.start()