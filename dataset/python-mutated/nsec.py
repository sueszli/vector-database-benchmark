from common import utils
from common.check import Check

class NSEC(Check):

    def __init__(self, domain):
        if False:
            return 10
        Check.__init__(self)
        self.domain = domain
        self.module = 'check'
        self.source = 'NSECCheck'

    def walk(self):
        if False:
            for i in range(10):
                print('nop')
        domain = self.domain
        while True:
            answer = utils.dns_query(domain, 'NSEC')
            if answer is None:
                break
            subdomain = str()
            for item in answer:
                record = item.to_text()
                subdomains = self.match_subdomains(record)
                subdomain = ''.join(subdomains)
                self.subdomains.update(subdomains)
            if subdomain == self.domain:
                break
            if domain != self.domain:
                if domain.split('.')[0] == subdomain.split('.')[0]:
                    break
            domain = subdomain
        return self.subdomains

    def run(self):
        if False:
            i = 10
            return i + 15
        '\n        类执行入口\n        '
        self.begin()
        self.walk()
        self.finish()
        self.save_json()
        self.gen_result()
        self.save_db()

def run(domain):
    if False:
        return 10
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    brute = NSEC(domain)
    brute.run()
if __name__ == '__main__':
    run('iana.org')