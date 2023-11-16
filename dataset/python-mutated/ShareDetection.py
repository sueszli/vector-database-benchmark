import sys
sys.path.append('..')
from sqlalchemy.orm import sessionmaker
from common.TushareUtil import TushareBaseUtil
from sqlalchemy import or_
import pandas as pd
from configure.util import send_from_aliyun
from common.BaseService import BaseService
from configure.settings import DBSelector
from LOF_Model import FundBaseInfoModel, ShareModel, Base
LOF_PERCENT = 10
LOF_DIFF_MAX = 200
ETF_PERCENT = 10
ETF_DIFF_MAX = 100000

class FundDetection(BaseService):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(FundDetection, self).__init__('../log/FundDetection.log')
        self.engine = self.get_engine()
        self.db_session = self.get_session()
        self.sess = self.db_session()
        self.ts_util = TushareBaseUtil()

    @staticmethod
    def get_engine():
        if False:
            while True:
                i = 10
        return DBSelector().get_engine('db_stock')

    def create_table(self):
        if False:
            return 10
        Base.metadata.create_all(self.engine)

    def get_session(self):
        if False:
            while True:
                i = 10
        return sessionmaker(bind=self.engine)

    def lof_start(self):
        if False:
            return 10
        category = 'LOF'
        (query_result_str, has_data) = self.query_big_volatility_share(category)
        if has_data:
            title = f'{self.today} LOF 申购波动数据'
            print(title)
            print(query_result_str)
            send_from_aliyun(title=title, content=query_result_str)
        else:
            self.logger.info(f'今天{self.today}没有数据')

    def etf_start(self):
        if False:
            while True:
                i = 10
        category = 'ETF'
        (query_result_str, has_data) = self.query_big_volatility_share(category)
        if has_data:
            title = f'{self.today} ETF 申购波动数据'
            send_from_aliyun(title, content=query_result_str)
        else:
            self.logger.info(f'今天{self.today}没有数据')

    def query_big_volatility_share(self, category):
        if False:
            while True:
                i = 10
        day = 0
        yesterday = self.ts_util.get_last_trade_date()
        yesterday = self.ts_util.date_convertor(yesterday)
        lastday_of_yesterday = self.ts_util.get_trade_date()[-3 - day]
        lastday_of_yesterday = self.ts_util.date_convertor(lastday_of_yesterday)
        string_arg = ''
        has_data = False
        string_arg += f'\n############ {category} ###############\n\n'
        if category == 'LOF':
            lastest_lofs = self.sess.query(FundBaseInfoModel.name, FundBaseInfoModel.code, ShareModel.share, ShareModel.date).join(ShareModel).filter(FundBaseInfoModel.category == category).filter(or_(ShareModel.date == yesterday, ShareModel.date == lastday_of_yesterday)).all()
            PERCENT = LOF_PERCENT
            DIFF_MAX = LOF_DIFF_MAX
        else:
            last_week = self.ts_util.get_last_week_trade_date()
            last_week = self.ts_util.date_convertor(last_week)
            lastest_lofs = self.sess.query(FundBaseInfoModel.name, FundBaseInfoModel.code, ShareModel.share, ShareModel.date).join(ShareModel).filter(FundBaseInfoModel.category == category).filter(ShareModel.date.between(last_week, yesterday)).all()
            PERCENT = ETF_PERCENT
            DIFF_MAX = ETF_DIFF_MAX
        current_df = pd.DataFrame(lastest_lofs, columns=['name', 'code', 'share', 'date'])
        current_df['date'] = current_df['date'].astype(str)
        current_df['share'] = current_df['share'].astype(float)
        current_df['code'] = current_df['code'].astype(str)
        current_df['name'] = current_df['name'].astype(str)
        for (code, sub_df) in current_df.groupby('code'):
            yesterday_share = sub_df[sub_df['date'] == yesterday]
            lastday_of_yesterday_share = sub_df[sub_df['date'] == lastday_of_yesterday]
            if len(yesterday_share) > 0 and len(lastday_of_yesterday_share) > 0:
                yesterday_share_num = yesterday_share['share'].to_list()[0]
                lastday_of_yesterday_num = lastday_of_yesterday_share['share'].to_list()[0]
                diff_part = yesterday_share_num - lastday_of_yesterday_num
                diff = diff_part * 1.0 / lastday_of_yesterday_num * 100.0
                diff = round(diff, 2)
                if abs(diff) >= PERCENT or abs(diff_part) > DIFF_MAX:
                    has_data = True
                    print(yesterday_share['name'].to_list()[0], yesterday_share['code'].to_list()[0], yesterday_share_num, lastday_of_yesterday_num, lastday_of_yesterday, diff, round(diff_part, 0))
                    string = self.formator(category, yesterday_share['name'].to_list()[0], yesterday_share['code'].to_list()[0], round(yesterday_share_num / 10000, 2), round(lastday_of_yesterday_num / 10000, 2), yesterday, diff, round(diff_part, 0))
                    string_arg += string + '\n'
        string_arg += '\n'
        return (string_arg, has_data)

    @staticmethod
    def formator(*args):
        if False:
            for i in range(10):
                print('nop')
        string = '{} {} {} {}亿份 {}亿份 {} {}% 多出了{}万份\n'.format(*args)
        return string