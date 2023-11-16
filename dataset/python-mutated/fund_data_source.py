import json
import sys
import akshare as ak
import pandas as pd
from loguru import logger
sys.path.append('../../')

class DataSource:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def get_fund_code(self):
        if False:
            while True:
                i = 10
        codes = []
        for code in self.doc.find({}, {'fund_id': 1, '_id': 0}):
            codes.append(code['fund_id'])
        return codes

    def get_fund_code_local(self):
        if False:
            print('Hello World!')
        codes = []
        with open('fund_code.json', 'r') as fp:
            js_data = json.load(fp)
        for code in js_data.keys():
            codes.append(code)
        return codes

    def get_data(self, source):
        if False:
            i = 10
            return i + 15
        if source == 'mongo':
            return self.get_data_from_mongo()
        else:
            logger.info('正在获取数据，请耐心等待。。。。。。【预测需要1分钟】')
            return self.get_data_from_ak()

    def df_into_csv(self, df):
        if False:
            while True:
                i = 10
        try:
            df.to_csv('fund_netvalue.csv', mode='a')
        except Exception as e:
            print(e)

    def all_market_data(self, source='local'):
        if False:
            return 10
        '\n        获取数据\n        '
        if source == 'local':
            for code in self.get_fund_code_local():
                df = self.get_closed_fund_netvalue(code)
                df['code'] = code
                df['净值日期'] = df['净值日期'].astype(str)
                self.df_into_csv(df)
        else:
            for code in self.get_fund_code():
                df = self.get_closed_fund_netvalue(code)
                df['code'] = code
                df['净值日期'] = df['净值日期'].astype(str)
                self.df_into_mongo(df)

    def df_into_mongo(self, df):
        if False:
            i = 10
            return i + 15
        data_list = df.to_json(orient='records', force_ascii=False)
        data_list = json.loads(data_list)
        try:
            self.bt_doc.insert_many(data_list)
        except Exception as e:
            print(e)

    def get_net_value(self, code):
        if False:
            while True:
                i = 10
        '\n        获取基金的净值\n        '
        fund_open_fund_info_em_df = ak.fund_open_fund_info_em(fund=code, indicator='单位净值走势')
        return fund_open_fund_info_em_df

    def get_closed_fund_netvalue(self, code):
        if False:
            print('Hello World!')
        fund_etf_fund_info_em_df = ak.fund_etf_fund_info_em(fund=code)
        return fund_etf_fund_info_em_df

    def get_data_from_mongo(self):
        if False:
            while True:
                i = 10
        from configure.settings import DBSelector
        self.mongo_client = DBSelector().mongo('qq')
        self.doc = self.mongo_client['closed_end_fund']['2022-02-28']
        self.bt_doc = self.mongo_client['db_stock']['closed_end_fund']
        result = []
        for item in self.bt_doc.find({'日增长率': {'$ne': None}}, {'_id': 0}):
            result.append(item)
        df = pd.DataFrame(result)
        return df

    def get_data_from_ak(self):
        if False:
            i = 10
            return i + 15
        result = []
        for code in self.get_fund_code_local():
            df = self.get_closed_fund_netvalue(code)
            df['code'] = code
            df['净值日期'] = df['净值日期'].astype(str)
            result.append(df)
        ret_df = pd.concat(result)
        return ret_df[~ret_df['日增长率'].isnull()]