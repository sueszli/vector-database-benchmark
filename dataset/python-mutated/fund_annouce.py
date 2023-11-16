import datetime
import math
import sys
sys.path.append('..')
from common.BaseService import BaseService

class FundAnnouce(BaseService):

    def __init__(self):
        if False:
            return 10
        super(FundAnnouce, self).__init__('../log/fund_annouce.log')
        self.PAGE_SIZE = 30
        self.base_url = 'http://fund.szse.cn/api/disc/info/find/tannInfo?type=2&pageSize={}&pageNum={}'

    @property
    def headers(self):
        if False:
            while True:
                i = 10
        _header = {'Accept': 'application/json, text/javascript, */*; q=0.01', 'Accept-Encoding': 'gzip, deflate', 'Accept-Language': 'zh,en;q=0.9,en-US;q=0.8,zh-CN;q=0.7', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'Content-Type': 'application/json', 'Host': 'fund.szse.cn', 'Pragma': 'no-cache', 'Referer': 'http://fund.szse.cn/disclosurelist/index.html', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36', 'X-Request-Type': 'ajax', 'X-Requested-With': 'XMLHttpRequest'}
        return _header

    def get_page(self):
        if False:
            return 10
        content = self.get(self.base_url.format(self.PAGE_SIZE, 1), _json=True)
        announceCount = content['announceCount']
        total_page = math.ceil(announceCount / self.PAGE_SIZE)
        return total_page

    def run(self):
        if False:
            while True:
                i = 10
        total_page = self.get_page()
        if total_page < 1:
            self.logger.info('empty content')
            return
        for page in range(1, total_page):
            content = self.get(self.base_url.format(self.PAGE_SIZE, 1), _json=True)
            self.parse(content)

    def parse(self, content):
        if False:
            for i in range(10):
                print('nop')
        for item in content.get('data'):
            item['crawltime'] = datetime.datetime.now()

def main():
    if False:
        i = 10
        return i + 15
    app = FundAnnouce()
    app.run()
if __name__ == '__main__':
    main()