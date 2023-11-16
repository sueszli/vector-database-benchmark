from config import settings
from common.query import Query

class SecurityTrailsAPI(Query):

    def __init__(self, domain):
        if False:
            print('Hello World!')
        Query.__init__(self)
        self.domain = domain
        self.module = 'Dataset'
        self.source = 'SecurityTrailsAPIQuery'
        self.addr = 'https://api.securitytrails.com/v1/domain/'
        self.api = settings.securitytrails_api
        self.delay = 2

    def query(self):
        if False:
            return 10
        '\n        向接口查询子域并做子域匹配\n        '
        self.header = self.get_header()
        self.proxy = self.get_proxy(self.source)
        params = {'apikey': self.api}
        url = f'{self.addr}{self.domain}/subdomains'
        resp = self.get(url, params)
        if not resp:
            return
        prefixs = resp.json()['subdomains']
        subdomains = [f'{prefix}.{self.domain}' for prefix in prefixs]
        if subdomains:
            self.subdomains.update(subdomains)

    def run(self):
        if False:
            return 10
        '\n        类执行入口\n        '
        if not self.have_api(self.api):
            return
        self.begin()
        self.query()
        self.finish()
        self.save_json()
        self.gen_result()
        self.save_db()

def run(domain):
    if False:
        return 10
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    query = SecurityTrailsAPI(domain)
    query.run()
if __name__ == '__main__':
    run('example.com')