import cdx_toolkit
from common.crawl import Crawl
from config.log import logger

class ArchiveCrawl(Crawl):

    def __init__(self, domain):
        if False:
            return 10
        Crawl.__init__(self)
        self.domain = domain
        self.module = 'Crawl'
        self.source = 'ArchiveCrawl'

    def crawl(self, domain, limit):
        if False:
            i = 10
            return i + 15
        '\n\n        :param domain:\n        :param limit:\n        '
        self.header = self.get_header()
        self.proxy = self.get_proxy(self.source)
        cdx = cdx_toolkit.CDXFetcher(source='ia')
        url = f'*.{domain}/*'
        size = cdx.get_size_estimate(url)
        logger.log('DEBUG', f'{url} ArchiveCrawl size estimate {size}')
        for resp in cdx.iter(url, limit=limit):
            if resp.data.get('status') not in ['301', '302']:
                url = resp.data.get('url')
                subdomains = self.match_subdomains(domain, url + resp.text)
                self.subdomains.update(subdomains)

    def run(self):
        if False:
            while True:
                i = 10
        '\n        类执行入口\n        '
        self.begin()
        self.crawl(self.domain, 50)
        for subdomain in self.subdomains:
            if subdomain != self.domain:
                self.crawl(subdomain, 10)
        self.finish()
        self.save_json()
        self.gen_result()
        self.save_db()

def run(domain):
    if False:
        while True:
            i = 10
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    crawl = ArchiveCrawl(domain)
    crawl.run()
if __name__ == '__main__':
    run('example.com')