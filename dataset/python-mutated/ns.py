from common.lookup import Lookup

class QueryNS(Lookup):

    def __init__(self, domain):
        if False:
            print('Hello World!')
        Lookup.__init__(self)
        self.domain = domain
        self.module = 'dnsquery'
        self.source = 'QueryNS'
        self.qtype = 'NS'

    def run(self):
        if False:
            while True:
                i = 10
        '\n        类执行入口\n        '
        self.begin()
        self.query()
        self.finish()
        self.save_json()
        self.gen_result()
        self.save_db()

def run(domain):
    if False:
        while True:
            i = 10
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    query = QueryNS(domain)
    query.run()
if __name__ == '__main__':
    run('cuit.edu.cn')