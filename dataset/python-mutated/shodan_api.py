from config import settings
from common.search import Search

class ShodanAPI(Search):

    def __init__(self, domain):
        if False:
            i = 10
            return i + 15
        Search.__init__(self)
        self.domain = domain
        self.module = 'Search'
        self.source = 'ShodanAPISearch'
        self.key = settings.shodan_api_key

    def search(self):
        if False:
            return 10
        '\n        发送搜索请求并做子域匹配\n        '
        self.header = self.get_header()
        self.proxy = self.get_proxy(self.source)
        url = f'https://api.shodan.io/dns/domain/{self.domain}?key={self.key}'
        resp = self.get(url)
        if not resp:
            return
        data = resp.json()
        names = data.get('subdomains')
        subdomain_str = str(set(map(lambda name: f'{name}.{self.domain}', names)))
        self.subdomains = self.collect_subdomains(subdomain_str)

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
        i = 10
        return i + 15
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    search = ShodanAPI(domain)
    search.run()
if __name__ == '__main__':
    run('freebuf.com')