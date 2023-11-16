__author__ = 'Rocky'
'\nhttp://30daydo.com\nContact: weigesysu@qq.com\n'
__doc__ = '\n复盘数据与流程\n'
from configure.settings import DBSelector
import pandas as pd
import pymongo
pd.set_option('expand_frame_repr', False)
client = pymongo.MongoClient('raspberrypi')
db = client['stock']
doc = db['industry']
today = '2018-05-08'
daily_df = pd.read_sql(today, daily_engine, index_col='index')

class IndustryFupan:
    """
    每天板块分析
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.engine = DBSelector()

    def save_industry(self):
        if False:
            i = 10
            return i + 15
        try:
            doc.drop()
        except Exception as e:
            print(e)
        engine = get_engine('db_stock')
        basic_df = pd.read_sql('tb_basic_info', engine, index_col='index')
        for (name, group) in basic_df.groupby('industry'):
            d = dict()
            d['板块名称'] = name
            d['代码'] = group['code'].values.tolist()
            d['更新日期'] = today
            try:
                doc.insert(d)
            except Exception as e:
                print(e)

def hot_industry():
    if False:
        return 10
    engine = get_engine('db_stock')
    basic_df = pd.read_sql('tb_basic_info', engine, index_col='index')
    industry_dict = {}
    for (name, group) in basic_df.groupby('industry'):
        industry_dict[name] = group['code'].values.tolist()
    result = {}
    for (k, v) in industry_dict.items():
        mean = 0.0
        for i in v:
            try:
                percent = daily_df[daily_df['code'] == i]['changepercent'].values[0]
                name = daily_df[daily_df['code'] == i]['name'].values[0]
            except:
                percent = 0
                name = ''
            mean = mean + float(percent)
        m = round(mean / len(v), 2)
        result[k] = m
    all_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    kind = '元器件'
    select_detail = {}
    for code in industry_dict.get(kind):
        try:
            percent = daily_df[daily_df['code'] == code]['changepercent'].values[0]
        except:
            percent = 0
        try:
            name = daily_df[daily_df['code'] == code]['name'].values[0]
        except:
            name = ''
        select_detail[name] = float(percent)
    print('\n\n{} detail\n'.format(kind))
    select_detail = sorted(select_detail.items(), key=lambda x: x[1], reverse=True)
    for (n, p) in select_detail:
        print(n, p)

def get_industry():
    if False:
        for i in range(10):
            print('nop')
    industry = {}
    for i in doc.find({}, {'_id': 0}):
        print(i.get('板块名称'))
        industry[i.get('板块名称')] = i.get('代码')
    return industry

def daily_hot_industry():
    if False:
        i = 10
        return i + 15
    industry = get_industry()
    result = {}
    for (item, code_list) in industry.items():
        for code in code_list:
            mean = 0.0
            try:
                percent = daily_df[daily_df['code'] == code]['changepercent'].values[0]
                name = daily_df[daily_df['code'] == code]['name'].values[0]
            except:
                percent = 0
                name = ''
            mean = mean + float(percent)
        m = round(mean / len(code_list), 2)
        result[item] = m
    all_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    return all_result

def industry_hot_mongo():
    if False:
        print('Hello World!')
    result = daily_hot_industry()
    collection = db['industry_rank']
    collection.drop()
    for item in result:
        d = {}
        d['板块'] = item[0]
        d['涨跌幅'] = item[1]
        d['日期'] = today
        try:
            collection.insert(d)
        except Exception as e:
            print(e)

def industry_detail(kind):
    if False:
        i = 10
        return i + 15
    select_detail = {}
    industry_list = get_industry()
    for code in industry_list.get(kind):
        try:
            percent = daily_df[daily_df['code'] == code]['changepercent'].values[0]
        except:
            percent = 0
        try:
            name = daily_df[daily_df['code'] == code]['name'].values[0]
        except:
            name = ''
        select_detail[name] = float(percent)
    print('\n\n{} detail\n'.format(kind))
    select_detail = sorted(select_detail.items(), key=lambda x: x[1], reverse=True)
    for (n, p) in select_detail:
        print(n, p)
if __name__ == '__main__':
    industry_detail('电器连锁')