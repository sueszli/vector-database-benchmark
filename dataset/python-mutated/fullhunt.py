from config import settings
from common.query import Query

class FullHuntAPI(Query):

    def __init__(self, domain):
        if False:
            for i in range(10):
                print('nop')
        Query.__init__(self)
        self.domain = domain
        self.module = 'Dataset'
        self.source = 'FullHuntAPIQuery'
        self.api = settings.fullhunt_api_key

    def query(self):
        if False:
            return 10
        '\n        向接口查询子域并做子域匹配\n        '
        self.header = self.get_header()
        self.header.update({'X-API-KEY': self.api})
        self.proxy = self.get_proxy(self.source)
        url = f'https://fullhunt.io/api/v1/domain/{self.domain}/subdomains'
        resp = self.get(url)
        self.subdomains = self.collect_subdomains(resp)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        类执行入口\n        '
        self.begin()
        self.query()
        self.finish()
        self.save_json()
        self.gen_result()
        self.save_db()

def run(domain):
    if False:
        i = 10
        return i + 15
    '\n    类统一调用入口\n\n    :param str domain: 域名\n\n    '
    query = FullHuntAPI(domain)
    query.run()
if __name__ == '__main__':
    run('qq.com')