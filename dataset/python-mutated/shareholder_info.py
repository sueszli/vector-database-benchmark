import re
import sys
import pandas as pd
import time
import traceback
from configure.settings import DBSelector
from common.Base import pro
import json

class ShareHolderInfo:
    """
    十大股东与十大流通股东
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_mongo()
        self.tushare_init()

    def db_init(self):
        if False:
            return 10
        self.conn = DBSelector().get_mysql_conn('db_stock')
        self.cursor = self.conn.cursor()

    def init_mongo(self):
        if False:
            print('Hello World!')
        self.client = DBSelector().mongo('qq')
        self.doc_holder = self.client['db_stock']['shareHolder']
        self.doc_holder_float = self.client['db_stock']['shareHolder_float']

    def tushare_init(self):
        if False:
            for i in range(10):
                print('nop')
        self.pro = pro

    def exists(self, code):
        if False:
            print('Hello World!')
        result = self.doc_holder.find_one({'ts_code': code})
        return False if result is None else True

    def get_stock_list(self, exchange):
        if False:
            return 10
        df = self.pro.stock_basic(exchange=exchange, list_status='L')
        return dict(zip(list(df['ts_code'].values), list(df['name'].values)))

    @staticmethod
    def create_date():
        if False:
            i = 10
            return i + 15
        start_date = '20{}0101'
        end_date = '20{}1231'
        date_list = []
        for i in range(18, 0, -1):
            print(start_date.format(str(i).zfill(2)))
            print(end_date.format(str(i).zfill(2)))
            date_list.append(i)
        return date_list

    def get_stockholder(self, code, start, end):
        if False:
            while True:
                i = 10
        '\n        stockholder 十大\n        stockfloat 十大流通\n        '
        try:
            stockholder = self.pro.top10_holders(ts_code=code, start_date=start, end_date=end)
            stockfloat = self.pro.top10_floatholders(ts_code=code, start_date=start, end_date=end)
        except Exception as e:
            print(e)
            time.sleep(10)
            self.pro = pro
            stockholder = self.pro.top10_holders(ts_code=code, start_date=start, end_date=end)
            stockfloat = self.pro.top10_floatholders(ts_code=code, start_date=start, end_date=end)
        else:
            if stockholder.empty or stockfloat.empty:
                print('有空数据----> ', code)
                return (pd.DataFrame(), pd.DataFrame())
            else:
                return (stockholder, stockfloat)

    def dumpMongo(self, doc, df):
        if False:
            for i in range(10):
                print('nop')
        record_list = df.to_json(orient='records', force_ascii=False)
        record_list = json.loads(record_list)
        if len(record_list) == 0:
            return
        try:
            doc.insert_many(record_list)
        except Exception as e:
            (exc_type, exc_value, exc_obj) = sys.exc_info()
            traceback.print_exc()

    def valid_code(self, code):
        if False:
            return 10
        return True if re.search('^\\d{6}\\.\\S{2}', code) else False

    def run(self):
        if False:
            print('Hello World!')
        start_date = '20{}0101'
        end_date = '20{}1231'
        exchange_list = ['SSE', 'SZSE']
        for ex in exchange_list:
            code_dict = self.get_stock_list(ex)
            for (code, name) in code_dict.items():
                i = 21
                if not self.valid_code(code):
                    print('invalid code ', code)
                    continue
                if self.exists(code):
                    continue
                print('crawling -->', code)
                start = start_date.format(str(i).zfill(2))
                end = end_date.format(str(i).zfill(2))
                (df_holding, df_float) = self.get_stockholder(code, start, end)
                self.dumpMongo(self.doc_holder, df_holding)
                self.dumpMongo(self.doc_holder_float, df_float)
                time.sleep(0.1)

def main():
    if False:
        return 10
    app = ShareHolderInfo()
    app.run()
if __name__ == '__main__':
    main()