__author__ = 'Rocky'
'\nhttp://30daydo.com\nContact: weigesysu@qq.com\n'
import sys
sys.path.append('..')
from common.BaseService import BaseService
import tushare as ts
import pandas as pd
pd.set_option('display.max_rows', None)

class Monitor_Stock(BaseService):

    def __init__(self):
        if False:
            return 10
        super(Monitor_Stock, self).__init__('../log/bigdeal.log')

    def getBigDeal(self, code, vol):
        if False:
            while True:
                i = 10
        df = ts.get_today_ticks(code)
        print('df ', df)
        t = df[df['vol'] > vol]
        if len(t) > 0:
            self.logger.info('Big volume {}'.format(code))

    def init_market(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        获取全市场\n        '
        from configure.settings import get_tushare_pro
        pro = get_tushare_pro()
        data = pro.stock_basic(exchange='SSE', list_status='L')
        data = data[~data['ts_code'].str.startswith('A')]
        return data['symbol'].tolist()

    def run(self):
        if False:
            print('Hello World!')
        code_list = self.init_market()
        for i in code_list:
            try:
                self.getBigDeal(i, 1000)
            except Exception as e:
                print(e)

def main():
    if False:
        for i in range(10):
            print('nop')
    app = Monitor_Stock()
    app.run()
if __name__ == '__main__':
    main()