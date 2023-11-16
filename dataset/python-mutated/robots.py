"""
检查内容安全策略收集子域名收集子域名
"""
from common.check import Check

class Robots(Check):

    def __init__(self, domain):
        if False:
            for i in range(10):
                print('nop')
        Check.__init__(self)
        self.domain = domain
        self.module = 'check'
        self.source = 'RobotsCheck'

    def check(self):
        if False:
            print('Hello World!')
        '\n        正则匹配域名的robots.txt文件中的子域\n        '
        filenames = {'robots.txt'}
        self.to_check(filenames)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
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
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    check = Robots(domain)
    check.run()
if __name__ == '__main__':
    run('qq.com')