from twisted.internet import reactor
from twisted.python import log
import scrapy
from scrapy.crawler import CrawlerProcess

class NoRequestsSpider(scrapy.Spider):
    name = 'no_request'

    def start_requests(self):
        if False:
            while True:
                i = 10
        return []
process = CrawlerProcess(settings={'TWISTED_REACTOR': 'twisted.internet.selectreactor.SelectReactor'})
d = process.crawl(NoRequestsSpider)
d.addErrback(log.err)
process.start()