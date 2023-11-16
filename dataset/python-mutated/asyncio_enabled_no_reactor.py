import scrapy
from scrapy.crawler import CrawlerProcess

class NoRequestsSpider(scrapy.Spider):
    name = 'no_request'

    def start_requests(self):
        if False:
            while True:
                i = 10
        return []
process = CrawlerProcess(settings={'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'})
process.crawl(NoRequestsSpider)
process.start()