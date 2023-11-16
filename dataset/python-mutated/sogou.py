from common.search import Search

class Sogou(Search):

    def __init__(self, domain):
        if False:
            for i in range(10):
                print('nop')
        Search.__init__(self)
        self.domain = domain
        self.module = 'Search'
        self.source = 'SogouSearch'
        self.addr = 'https://www.sogou.com/web'
        self.limit_num = 1000

    def search(self, domain, filtered_subdomain=''):
        if False:
            i = 10
            return i + 15
        '\n        发送搜索请求并做子域匹配\n\n        :param str domain: 域名\n        :param str filtered_subdomain: 过滤的子域\n        '
        self.page_num = 1
        while True:
            self.header = self.get_header()
            self.proxy = self.get_proxy(self.source)
            word = 'site:.' + domain + filtered_subdomain
            payload = {'query': word, 'page': self.page_num, 'num': self.per_page_num}
            resp = self.get(self.addr, payload)
            subdomains = self.match_subdomains(resp, fuzzy=False)
            if not self.check_subdomains(subdomains):
                break
            self.subdomains.update(subdomains)
            self.page_num += 1
            if '<a id="sogou_next"' not in resp.text:
                break
            if self.page_num * self.per_page_num >= self.limit_num:
                break

    def run(self):
        if False:
            print('Hello World!')
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
    search = Sogou(domain)
    search.run()
if __name__ == '__main__':
    run('example.com')