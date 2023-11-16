from twisted.internet import reactor
import scrapy
from scrapy.crawler import CrawlerProcess

class NoRequestsSpider(scrapy.Spider):
    name = 'no_request'

    def start_requests(self):
        if False:
            return 10
        return []
process = CrawlerProcess(settings={})
process.crawl(NoRequestsSpider)
process.start()