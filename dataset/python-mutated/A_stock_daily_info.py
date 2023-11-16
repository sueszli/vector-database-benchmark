import sys
sys.path.append('..')
from configure.settings import get_tushare_pro, DBSelector
from configure.util import calendar
import time

class AStockDailyInfo:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.pro = get_tushare_pro()
        self.conn = DBSelector().get_engine('db_stock_daily', 't')

    def run(self):
        if False:
            return 10
        date = calendar('2022-01-01', '2022-12-28')
        for d in date:
            print(d)
            df = self.pro.daily(trade_date=d)
            df.to_sql('tb_{}'.format(d), con=self.conn)
            time.sleep(1)

def main():
    if False:
        i = 10
        return i + 15
    app = AStockDailyInfo()
    app.run()
if __name__ == '__main__':
    main()