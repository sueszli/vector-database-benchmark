__author__ = 'Rocky'
'\nhttp://30daydo.com\nContact: weigesysu@qq.com\n'
import tushare as ts
import pandas as pd
import time
import os
import sys
sys.path.append('..')
from common.BaseService import BaseService
from configure.settings import DBSelector, config_dict

class FetchDaily(BaseService):

    def __init__(self):
        if False:
            print('Hello World!')
        super(FetchDaily, self).__init__(f'../log/{self.__class__.__name__}.log')
        self.path = config_dict('data_path')
        self.check_path(self.path)
        self.df_today_all = pd.DataFrame()
        self.TIMEOUT = 10
        self.DB = DBSelector()
        self.engine = self.DB.get_engine('db_daily', 'qq')

    def get_today_market(self, re_try=10):
        if False:
            for i in range(10):
                print('nop')
        while re_try > 0:
            try:
                df = ts.get_today_all()
                if df is None:
                    continue
                if len(df) == 0:
                    continue
            except Exception as e:
                self.logger.error(e)
                re_try = re_try - 1
                time.sleep(self.TIMEOUT)
            else:
                return df
        return None

    def run(self):
        if False:
            while True:
                i = 10
        self.df_today_all = self.get_today_market()
        filename = self.today + '_all_.xls'
        full_filename = os.path.join(self.path, filename)
        if self.df_today_all is not None:
            self.df_today_all['turnoverratio'] = self.df_today_all['turnoverratio'].map(lambda x: round(x, 2))
            self.df_today_all['per'] = self.df_today_all['per'].map(lambda x: round(x, 2))
            self.df_today_all['pb'] = self.df_today_all['pb'].map(lambda x: round(x, 2))
            try:
                self.df_today_all.to_excel(full_filename)
            except Exception as e:
                self.notify(title=f'{self.__class__}写excel出错')
                self.logger.error(e)
            try:
                self.df_today_all.to_sql(self.today, self.engine, if_exists='fail')
            except Exception as e:
                self.notify(title=f'{self.__class__}mysql出错')
                self.logger.error(e)

def main():
    if False:
        for i in range(10):
            print('nop')
    obj = FetchDaily()
    obj.run()
if __name__ == '__main__':
    main()