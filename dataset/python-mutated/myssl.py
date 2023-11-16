from common.query import Query

class MySSL(Query):

    def __init__(self, domain):
        if False:
            return 10
        Query.__init__(self)
        self.domain = domain
        self.module = 'Certificate'
        self.source = 'MySSLQuery'
        self.addr = 'https://myssl.com/api/v1/discover_sub_domain'

    def query(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        向接口查询子域并做子域匹配\n        '
        self.header = self.get_header()
        self.proxy = self.get_proxy(self.source)
        params = {'domain': self.domain}
        resp = self.get(self.addr, params)
        self.subdomains = self.collect_subdomains(resp)

    def run(self):
        if False:
            i = 10
            return i + 15
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
    query = MySSL(domain)
    query.run()
if __name__ == '__main__':
    run('freebuf.com')