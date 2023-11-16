"""
查询域名的NS记录(域名服务器记录，记录该域名由哪台域名服务器解析)，检查查出的域名服务器是
否开启DNS域传送，如果开启且没做访问控制和身份验证便加以利用获取域名的所有记录。

DNS域传送(DNS zone transfer)指的是一台备用域名服务器使用来自主域名服务器的数据刷新自己
的域数据库，目的是为了做冗余备份，防止主域名服务器出现故障时 dns 解析不可用。
当主服务器开启DNS域传送同时又对来请求的备用服务器未作访问控制和身份验证便可以利用此漏洞获
取某个域的所有记录。
"""
import dns.resolver
import dns.zone
from common import utils
from common.check import Check
from config.log import logger

class AXFR(Check):

    def __init__(self, domain):
        if False:
            i = 10
            return i + 15
        Check.__init__(self)
        self.domain = domain
        self.module = 'check'
        self.source = 'AXFRCheck'
        self.results = []

    def axfr(self, server):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform domain transfer\n\n        :param server: domain server\n        '
        logger.log('DEBUG', f'Trying to perform domain transfer in {server} of {self.domain}')
        try:
            xfr = dns.query.xfr(where=server, zone=self.domain, timeout=5.0, lifetime=10.0)
            zone = dns.zone.from_xfr(xfr)
        except Exception as e:
            logger.log('DEBUG', e.args)
            logger.log('DEBUG', f'Domain transfer to server {server} of {self.domain} failed')
            return
        names = zone.nodes.keys()
        for name in names:
            full_domain = str(name) + '.' + self.domain
            subdomain = self.match_subdomains(full_domain)
            self.subdomains.update(subdomain)
            record = zone[name].to_text(name)
            self.results.append(record)
        if self.results:
            logger.log('DEBUG', f'Found the domain transfer record of {self.domain} on {server}')
            logger.log('DEBUG', '\n'.join(self.results))
            self.results = []

    def check(self):
        if False:
            while True:
                i = 10
        '\n        check\n        '
        resolver = utils.dns_resolver()
        try:
            answers = resolver.query(self.domain, 'NS')
        except Exception as e:
            logger.log('ERROR', e.args)
            return
        nsservers = [str(answer) for answer in answers]
        if not len(nsservers):
            logger.log('ALERT', f'No name server record found for {self.domain}')
            return
        for nsserver in nsservers:
            self.axfr(nsserver)

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
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    check = AXFR(domain)
    check.run()
if __name__ == '__main__':
    run('ZoneTransfer.me')