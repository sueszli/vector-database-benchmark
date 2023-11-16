from config import settings
from common.query import Query
from config.log import logger

class IPv4InfoAPI(Query):

    def __init__(self, domain):
        if False:
            for i in range(10):
                print('nop')
        Query.__init__(self)
        self.domain = domain
        self.module = 'Dataset'
        self.source = 'IPv4InfoAPIQuery'
        self.addr = ' http://ipv4info.com/api_v1/'
        self.api = settings.ipv4info_api_key

    def query(self):
        if False:
            print('Hello World!')
        '\n        向接口查询子域并做子域匹配\n        '
        page = 0
        while True:
            self.header = self.get_header()
            self.proxy = self.get_proxy(self.source)
            params = {'type': 'SUBDOMAINS', 'key': self.api, 'value': self.domain, 'page': page}
            resp = self.get(self.addr, params)
            if not resp:
                return
            if resp.status_code != 200:
                break
            try:
                json = resp.json()
            except Exception as e:
                logger.log('DEBUG', e.args)
                break
            subdomains = self.match_subdomains(str(json))
            if not subdomains:
                break
            self.subdomains.update(subdomains)
            subdomains = json.get('Subdomains')
            if subdomains and len(subdomains) < 300:
                break
            page += 1
            if page >= 50:
                break

    def run(self):
        if False:
            print('Hello World!')
        '\n        类执行入口\n        '
        if not self.have_api(self.api):
            return
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
    query = IPv4InfoAPI(domain)
    query.run()
if __name__ == '__main__':
    run('example.com')