import datetime
import sys
from collections import defaultdict
sys.path.append('..')
from configure.settings import DBSelector
from common.BaseService import BaseService
import pandas as pd
WEEK_DAY = -7

class DanjuanAnalyser(BaseService):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(DanjuanAnalyser, self).__init__('../log/Danjuan_analysis.log')

    def select_collection(self, current_date):
        if False:
            return 10
        '\n        根据日期选择数据库\n        '
        self.db = DBSelector().mongo(location_type='qq')
        doc = self.db['db_danjuan'][f'danjuan_fund_{current_date}']
        return doc

    def get_top_plan(self, collection, top=10):
        if False:
            return 10
        fund_dict = {}
        for item in collection.find({}, {'holding': 1}):
            plan_holding = item.get('holding', [])
            for hold in plan_holding:
                name = hold['fd_name']
                if hold['percent'] > 0:
                    fund_dict.setdefault(name, 0)
                    fund_dict[name] += 1
        fund_dict = list(sorted(fund_dict.items(), key=lambda x: x[1], reverse=True))[:top]
        return fund_dict

    def get_top_plan_percent(self, collection, top=10):
        if False:
            while True:
                i = 10
        fund_dict = {}
        for item in collection.find({}, {'holding': 1}):
            plan_holding = item.get('holding', [])
            for hold in plan_holding:
                name = hold['fd_name']
                percent = hold['percent']
                fund_dict.setdefault(name, 0)
                fund_dict[name] += percent
        fund_dict = list(sorted(fund_dict.items(), key=lambda x: x[1], reverse=True))[:top]
        return fund_dict

    def start(self):
        if False:
            print('Hello World!')
        today = datetime.datetime.now()
        last_week = today + datetime.timedelta(days=WEEK_DAY)
        last_week_str = last_week.strftime('%Y-%m-%d')
        last_week_str = '2021-04-20'
        today_doc = self.select_collection(self.today)
        last_week_doc = self.select_collection(last_week_str)
        fund_dict = self.get_top_plan(today_doc, 20)
        self.pretty(fund_dict, self.today, 'count')
        old_fund_dict = self.get_top_plan(last_week_doc, 20)
        self.pretty(old_fund_dict, last_week_str, 'count')
        diff_set = self.new_fund(fund_dict, old_fund_dict)
        print('新增的基金入围')
        print(diff_set)
        new_fund_percent = self.get_top_plan_percent(today_doc, 20)
        old_fund_percent = self.get_top_plan_percent(last_week_doc, 20)
        self.pretty(new_fund_percent, self.today, 'percent')
        self.pretty(old_fund_percent, last_week_str, 'percnet')
        clean_fund = self.clear_warehouse_fund(today_doc, 200)
        self.simple_display(clean_fund, self.today)

    def simple_display(self, data, date):
        if False:
            print('Hello World!')
        for i in data:
            print(i)
        df = pd.DataFrame(data, columns=['fund', 'clear_num'])
        print(df.head(100))
        df.to_excel(f'clear_{date}.xlsx')

    def pretty(self, fund_dict, date, kind):
        if False:
            print('Hello World!')
        df = pd.DataFrame(fund_dict, columns=['fund', 'holding_num'])
        print(df.head(100))
        df.to_excel(f'{date}-{kind}.xlsx')

    def new_fund(self, new_fund_dict, old_fund_dict):
        if False:
            i = 10
            return i + 15
        new_fund_list = list(map(lambda x: x[0], new_fund_dict))
        old_fund_list = list(map(lambda x: x[0], old_fund_dict))
        diff_set = set(old_fund_list) - set(new_fund_list)
        return diff_set

    def clear_warehouse_fund(self, collection, top):
        if False:
            for i in range(10):
                print('nop')
        '\n        清仓的基金\n        '
        fund_dict = {}
        for item in collection.find({}, {'holding': 1}):
            plan_holding = item.get('holding', [])
            for hold in plan_holding:
                name = hold['fd_name']
                percent = hold['percent']
                if percent > 0:
                    continue
                fund_dict.setdefault(name, 0)
                fund_dict[name] += 1
        fund_dict = list(sorted(fund_dict.items(), key=lambda x: x[1], reverse=True))[:top]
        return fund_dict

def main():
    if False:
        i = 10
        return i + 15
    app = DanjuanAnalyser()
    app.start()
if __name__ == '__main__':
    main()