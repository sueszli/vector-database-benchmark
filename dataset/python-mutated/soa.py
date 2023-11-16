from common.lookup import Lookup

class QuerySOA(Lookup):

    def __init__(self, domain):
        if False:
            return 10
        Lookup.__init__(self)
        self.domain = domain
        self.module = 'dnsquery'
        self.source = 'QuerySOA'
        self.qtype = 'SOA'

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
        print('Hello World!')
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    query = QuerySOA(domain)
    query.run()
if __name__ == '__main__':
    run('cuit.edu.cn')