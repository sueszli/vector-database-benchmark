import time
from config import settings
from common.search import Search

class ZoomEyeAPI(Search):

    def __init__(self, domain):
        if False:
            print('Hello World!')
        Search.__init__(self)
        self.domain = domain
        self.module = 'Search'
        self.source = 'ZoomEyeAPISearch'
        self.addr = 'https://api.zoomeye.org/domain/search'
        self.delay = 2
        self.key = settings.zoomeye_api_key

    def search(self):
        if False:
            while True:
                i = 10
        '\n        发送搜索请求并做子域匹配\n        '
        self.per_page_num = 30
        self.page_num = 1
        while True:
            time.sleep(self.delay)
            self.header = self.get_header()
            self.header.update({'API-KEY': self.key})
            self.proxy = self.get_proxy(self.source)
            params = {'q': self.domain, 'page': self.page_num, 'type': 1}
            resp = self.get(self.addr, params)
            if not resp:
                return
            if resp.status_code == 403:
                break
            resp_json = resp.json()
            subdomains = self.match_subdomains(resp)
            if not subdomains:
                break
            self.subdomains.update(subdomains)
            total = resp_json.get('total')
            self.page_num += 1
            if self.page_num * self.per_page_num >= int(total):
                break
            if self.page_num > 400:
                break

    def run(self):
        if False:
            while True:
                i = 10
        '\n        类执行入口\n        '
        if not self.have_api(self.key):
            return
        self.begin()
        self.search()
        self.finish()
        self.save_json()
        self.gen_result()
        self.save_db()

def run(domain):
    if False:
        print('Hello World!')
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    search = ZoomEyeAPI(domain)
    search.run()
if __name__ == '__main__':
    run('zhipin.com')