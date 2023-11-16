import optparse
from typing import List, Union
from scrapy.commands import crawl
from scrapy.crawler import Crawler

class Command(crawl.Command):

    def run(self, args: List[str], opts: optparse.Values) -> None:
        if False:
            while True:
                i = 10
        crawlers = []
        real_create_crawler = self.crawler_process.create_crawler

        def create_crawler(crawler_or_spidercls: Union[Crawler, str]) -> Crawler:
            if False:
                for i in range(10):
                    print('nop')
            crawler = real_create_crawler(crawler_or_spidercls)
            crawlers.append(crawler)
            return crawler
        self.crawler_process.create_crawler = create_crawler
        super().run(args, opts)
        if any((crawler.stats.get_value('log_count/ERROR') for crawler in crawlers)):
            self.exitcode = 1