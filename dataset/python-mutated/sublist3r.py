from common.query import Query

class Sublist3r(Query):

    def __init__(self, domain):
        if False:
            return 10
        Query.__init__(self)
        self.domain = domain
        self.module = 'Dataset'
        self.source = 'Sublist3rQuery'

    def query(self):
        if False:
            while True:
                i = 10
        '\n        向接口查询子域并做子域匹配\n        '
        self.header = self.get_header()
        self.proxy = self.get_proxy(self.source)
        addr = 'https://api.sublist3r.com/search.php'
        param = {'domain': self.domain}
        resp = self.get(addr, param)
        self.subdomains = self.collect_subdomains(resp)

    def run(self):
        if False:
            return 10
        '\n        类执行入口\n        '
        self.begin()
        self.query()
        self.finish()
        self.save_json()
        self.gen_result()
        self.save_db()

def run(domain):
    if False:
        print('Hello World!')
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    query = Sublist3r(domain)
    query.run()
if __name__ == '__main__':
    run('example.com')