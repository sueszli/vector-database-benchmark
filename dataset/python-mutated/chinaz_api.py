from config import settings
from common.query import Query

class ChinazAPI(Query):

    def __init__(self, domain):
        if False:
            while True:
                i = 10
        Query.__init__(self)
        self.domain = domain
        self.module = 'Dataset'
        self.source = 'ChinazAPIQuery'
        self.addr = 'https://apidata.chinaz.com/CallAPI/Alexa'
        self.api = settings.chinaz_api

    def query(self):
        if False:
            i = 10
            return i + 15
        '\n        向接口查询子域并做子域匹配\n        '
        self.header = self.get_header()
        self.proxy = self.get_proxy(self.source)
        params = {'key': self.api, 'domainName': self.domain}
        resp = self.get(self.addr, params)
        self.subdomains = self.collect_subdomains(resp)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
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
        for i in range(10):
            print('nop')
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    query = ChinazAPI(domain)
    query.run()
if __name__ == '__main__':
    run('example.com')