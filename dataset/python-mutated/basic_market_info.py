__author__ = 'Rocky'
'\nhttp://30daydo.com\nContact: weigesysu@qq.com\n'
import datetime
import time
import tushare as ts
import sys
sys.path.append('..')
from configure.settings import DBSelector, config_dict
from common.BaseService import BaseService

class BasicMarket(BaseService):

    def __init__(self):
        if False:
            return 10
        super(BasicMarket, self).__init__(f'../log/{self.__class__.__name__}.log')
        work_space = config_dict('data_path')
        ts_token = config_dict('ts_token')
        self.check_path(work_space)
        ts.set_token(ts_token)
        self.pro = ts.pro_api()

    def get_basic_info(self, retry=5):
        if False:
            print('Hello World!')
        '\n        保存全市场数据\n        :param retry:\n        :return:\n        '
        count = 0
        df = None
        while count < retry:
            try:
                df = self.pro.stock_basic(exchange='', list_status='', fields='')
            except Exception as e:
                self.logger.info(e)
                time.sleep(10)
                count += 1
                continue
            else:
                break
        if count == retry:
            self.notify(title=f'{self.__class__.__name__}获取股市市场全景数据失败')
            exit(0)
        if df is not None:
            df = df.reset_index(drop=True)
            df.rename(columns={'symbol': 'code'}, inplace=True)
            df['更新日期'] = datetime.datetime.now()
            engine = DBSelector().get_engine('db_stock', 'qq')
            try:
                df.to_sql('tb_basic_info', engine, if_exists='replace')
            except Exception as e:
                self.logger.error(e)
                self.notify(title=f'{self.__class__}mysql入库出错')
        return df

def main():
    if False:
        while True:
            i = 10
    obj = BasicMarket()
    obj.get_basic_info()
if __name__ == '__main__':
    main()