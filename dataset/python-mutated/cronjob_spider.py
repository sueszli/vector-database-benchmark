from multiprocessing import Process
from apscheduler.schedulers.blocking import BlockingScheduler
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from twisted.internet import reactor
from useragentscraper.spiders.useragent import UserAgentSpider

class CrawlerRunnerProcess(Process):

    def __init__(self, spider):
        if False:
            for i in range(10):
                print('nop')
        Process.__init__(self)
        self.runner = CrawlerRunner(get_project_settings())
        self.spider = spider

    def run(self):
        if False:
            i = 10
            return i + 15
        deferred = self.runner.crawl(self.spider)
        deferred.addBoth(lambda _: reactor.stop())
        reactor.run(installSignalHandlers=False)

def run_spider(spider):
    if False:
        return 10
    crawler = CrawlerRunnerProcess(spider)
    crawler.start()
    crawler.join()
configure_logging()
scheduler = BlockingScheduler(timezone='Europe/Amsterdam')
scheduler.add_job(run_spider, 'cron', args=[UserAgentSpider], day_of_week='sun', hour=6, minute=10)
scheduler.start()