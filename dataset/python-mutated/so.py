import time
from common.search import Search

class So(Search):

    def __init__(self, domain):
        if False:
            for i in range(10):
                print('nop')
        Search.__init__(self)
        self.domain = domain
        self.module = 'Search'
        self.source = 'SoSearch'
        self.addr = 'https://www.so.com/s'
        self.limit_num = 640
        self.per_page_num = 10

    def search(self, domain, filtered_subdomain=''):
        if False:
            while True:
                i = 10
        '\n        发送搜索请求并做子域匹配\n\n        :param str domain: 域名\n        :param str filtered_subdomain: 过滤的子域\n        '
        page_num = 1
        while True:
            time.sleep(self.delay)
            self.header = self.get_header()
            self.proxy = self.get_proxy(self.source)
            word = 'site:.' + domain + filtered_subdomain
            payload = {'q': word, 'pn': page_num}
            resp = self.get(url=self.addr, params=payload)
            subdomains = self.match_subdomains(resp, fuzzy=False)
            if not self.check_subdomains(subdomains):
                break
            self.subdomains.update(subdomains)
            page_num += 1
            if '<a id="snext"' not in resp.text:
                break
            if self.page_num * self.per_page_num >= self.limit_num:
                break

    def run(self):
        if False:
            return 10
        '\n        类执行入口\n        '
        self.begin()
        self.search(self.domain)
        for statement in self.filter(self.domain, self.subdomains):
            self.search(self.domain, filtered_subdomain=statement)
        if self.recursive_search:
            for subdomain in self.recursive_subdomain():
                self.search(subdomain)
        self.finish()
        self.save_json()
        self.gen_result()
        self.save_db()

def run(domain):
    if False:
        i = 10
        return i + 15
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    search = So(domain)
    search.run()
if __name__ == '__main__':
    run('mi.com')