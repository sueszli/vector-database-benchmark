from common.query import Query

class QianXun(Query):

    def __init__(self, domain):
        if False:
            for i in range(10):
                print('nop')
        Query.__init__(self)
        self.domain = domain
        self.module = 'Query'
        self.source = 'QianXunQuery'

    def query(self):
        if False:
            return 10
        '\n        向接口查询子域并做子域匹配\n        '
        self.header = self.get_header()
        self.proxy = self.get_proxy(self.source)
        num = 1
        while True:
            data = {'ecmsfrom': '', 'show': '', 'num': '', 'classid': '0', 'keywords': self.domain}
            url = f'https://www.dnsscan.cn/dns.html?keywords={self.domain}&page={num}'
            resp = self.post(url, data)
            subdomains = self.match_subdomains(resp)
            if not subdomains:
                break
            self.subdomains.update(subdomains)
            if '<div id="page" class="pagelist">' not in resp.text:
                break
            if '<li class="disabled"><span>&raquo;</span></li>' in resp.text:
                break
            num += 1

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
        for i in range(10):
            print('nop')
    '\n    类统一调用入口\n\n    :param str domain: 域名\n    '
    query = QianXun(domain)
    query.run()
if __name__ == '__main__':
    run('example.com')