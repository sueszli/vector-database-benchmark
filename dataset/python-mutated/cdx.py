"""
检查crossdomain.xml文件收集子域名
"""
from common.check import Check

class CrossDomain(Check):

    def __init__(self, domain):
        if False:
            return 10
        Check.__init__(self)
        self.domain = domain
        self.module = 'check'
        self.source = 'CrossDomainCheck'

    def check(self):
        if False:
            i = 10
            return i + 15
        '\n        检查crossdomain.xml收集子域名\n        '
        filenames = {'crossdomain.xml'}
        self.to_check(filenames)

    def run(self):
        if False:
            print('Hello World!')
        '\n        类执行入口\n        '
        self.begin()
        self.check()
        self.finish()
        self.save_json()
        self.gen_result()
        self.save_db()

def run(domain):
    if False:
        print('Hello World!')
    '\n    类统一调用入口\n\n    :param domain: 域名\n    '
    check = CrossDomain(domain)
    check.run()
if __name__ == '__main__':
    run('example.com')