from common.lookup import Lookup

class QueryMX(Lookup):

    def __init__(self, domain):
        if False:
            for i in range(10):
                print('nop')
        Lookup.__init__(self)
        self.domain = domain
        self.module = 'dnsquery'
        self.source = 'QueryMX'
        self.type = 'MX'

    def run(self):
        if False:
            print('Hello World!')
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
    query = QueryMX(domain)
    query.run()
if __name__ == '__main__':
    run('cuit.edu.cn')