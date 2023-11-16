from common.lookup import Lookup

class QuerySPF(Lookup):

    def __init__(self, domain):
        if False:
            while True:
                i = 10
        Lookup.__init__(self)
        self.domain = domain
        self.module = 'dnsquery'
        self.source = 'QuerySPF'
        self.qtype = 'SPF'

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
        return 10
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    brute = QuerySPF(domain)
    brute.run()
if __name__ == '__main__':
    run('qq.com')