import base64
import time
from config import settings
from common.search import Search

class Hunter(Search):

    def __init__(self, domain):
        if False:
            for i in range(10):
                print('nop')
        Search.__init__(self)
        self.domain = domain
        self.module = 'Search'
        self.source = 'HunterAPISearch'
        self.addr = 'https://hunter.qianxin.com/openApi/search'
        self.delay = 1
        self.key = settings.hunter_api_key

    def search(self):
        if False:
            i = 10
            return i + 15
        '\n        发送搜索请求并做子域匹配\n        '
        self.page_num = 1
        subdomain_encode = f'domain_suffix="{self.domain}"'.encode('utf-8')
        query_data = base64.b64encode(subdomain_encode)
        while True:
            time.sleep(self.delay)
            self.header = self.get_header()
            self.proxy = self.get_proxy(self.source)
            query = {'api-key': self.key, 'search': query_data, 'page': self.page_num, 'page_size': 100, 'is_web': 1}
            resp = self.get(self.addr, query)
            if not resp:
                return
            resp_json = resp.json()
            subdomains = self.match_subdomains(resp)
            if not subdomains:
                break
            self.subdomains.update(subdomains)
            total = resp_json.get('data').get('total')
            if self.page_num * 100 >= int(total):
                break
            self.page_num += 1

    def run(self):
        if False:
            return 10
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
        while True:
            i = 10
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    search = Hunter(domain)
    search.run()
if __name__ == '__main__':
    run('freebuf.com')