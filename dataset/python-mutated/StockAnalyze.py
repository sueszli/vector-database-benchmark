"""
@author:rocky
@email:weigesysu@qq.com
@feature: 收盘事后分析
"""
from configure.settings import DBSelector
import pandas as pd
from scipy import stats
import tushare as ts
import datetime
import os
import numpy as np
pd.set_option('display.max_rows', None)

def volume_calculation(code, start, end):
    if False:
        i = 10
        return i + 15
    '\n    计算某个股票的某个时间段的累计成交量\n\n    :param start: 开始日期\n    :param end: 结束日期\n    :return: 成交量，占每天比例\n    '
    df = ts.get_today_ticks(code)
    df['time'] = df['time'].map(lambda x: datetime.datetime.strptime(str(x), '%H:%M:%S'))
    total = df['volume'].sum()
    start = datetime.datetime.strptime(start, '%H:%M:%S')
    end = datetime.datetime.strptime(end, '%H:%M:%S')
    new_df = df[(df['time'] >= start) & (df['time'] < end)]
    volume = new_df['volume'].sum()
    rate = round(volume * 1.0 / total * 100, 2)
    return (volume, rate)

def today_statistics(today):
    if False:
        for i in range(10):
            print('nop')
    '\n    :help: 今天涨跌幅的统计分析： 中位数，均值等数据\n    :param today: 日期 2019-01-01\n    :return:None\n    '
    engine = DBSelector().get_engine('db_daily')
    df = pd.read_sql(today, engine, index_col='index')
    df = df[df['volume'] != 0]
    median = round(df['changepercent'].median(), 2)
    mean = round(df['changepercent'].mean(), 2)
    std = round(df['changepercent'].std(), 2)
    p_25 = round(stats.scoreatpercentile(df['changepercent'], 25), 2)
    p_50 = round(stats.scoreatpercentile(df['changepercent'], 50), 2)
    p_75 = round(stats.scoreatpercentile(df['changepercent'], 75), 2)
    print('中位数: {}'.format(median))
    print('平均数: {}'.format(mean))
    print('方差: {}'.format(std))
    print('25%: {}'.format(p_25))
    print('50%: {}'.format(p_50))
    print('75%: {}'.format(p_75))

def zt_location(date):
    if False:
        return 10
    '\n    :help: 分析涨停的区域分布\n    :param date:日期格式 20180404\n    :return:\n    '
    engine_zdt = DBSelector().get_engine('db_zdt')
    engine_basic = DBSelector().get_engine('db_stock')
    df = pd.read_sql(date + 'zdt', engine_zdt, index_col='index')
    df_basic = pd.read_sql('tb_basic_info', engine_basic, index_col='index')
    result = {}
    for code in df['代码'].values:
        try:
            area = df_basic[df_basic['code'] == code]['area'].values[0]
            result.setdefault(area, 0)
            result[area] += 1
        except Exception as e:
            print(e)
    new_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    for (k, v) in new_result:
        print(k, v)

def show_percentage(price):
    if False:
        while True:
            i = 10
    '\n    :help: 根据收盘价计算每个百分比的价格\n    :param open_price: 开盘价\n    :return:\n    '
    for i in range(1, 11):
        print('{}\t+{}% -> {}'.format(price, i, round(price * (1 + 0.01 * i), 2)))
    for i in range(1, 11):
        print('{}\t-{}% -> {}'.format(price, i, round(price * (1 - 0.01 * i), 2)))

def stock_profit(code, start, end):
    if False:
        i = 10
        return i + 15
    '\n    :help: 计算某个时间段的收益率\n    :param code: 股票代码\n    :param start: 开始时间\n    :param end: 结束时间\n    :return: 收益率\n    '
    k_data = ts.get_k_data(start=start, end=end, code=code)
    if len(k_data) == 0:
        return np.nan
    start_price = k_data['close'].values[0]
    print('Start price: ', start_price)
    end_price = k_data['close'].values[-1]
    print('End price: ', end_price)
    earn_profit = (end_price - start_price) / start_price * 100
    print('Profit: ', round(earn_profit, 2))
    return round(earn_profit, 2)

def exclude_kcb(df):
    if False:
        return 10
    '\n    :help: 去除科创板\n    :param df:\n    :return:\n    '
    non_kcb = df[~df['code'].map(lambda x: True if x.startswith('688') else False)]
    return non_kcb

def plot_percent_distribution(date):
    if False:
        return 10
    '\n    :help:图形显示某一天的涨跌幅分布\n    :param date:\n    :return:\n    '
    import matplotlib.pyplot as plt
    total = []
    engine = DBSelector().get_engine('db_daily')
    df = pd.read_sql(date, con=engine)
    df = exclude_kcb(df)
    count = len(df[(df['changepercent'] >= -11) & (df['changepercent'] <= -9.5)])
    total.append(count)
    for i in range(-9, 9, 1):
        count = len(df[(df['changepercent'] >= i * 1.0) & (df['changepercent'] < (i + 1) * 1.0)])
        total.append(count)
    count = len(df[df['changepercent'] >= 9])
    total.append(count)
    df_figure = pd.Series(total)
    plt.figure(figsize=(16, 10))
    X = range(-10, 10)
    plt.bar(X, height=total, color='y')
    for (x, y) in zip(X, total):
        plt.text(x, y + 0.05, y, ha='center', va='bottom')
    plt.grid()
    plt.xticks(range(-10, 11))
    plt.show()

def year_price_change(year, ignore_new_stock=False):
    if False:
        print('Hello World!')
    '\n    :year: 年份\n    :ignore_new_stock: 排除当年上市的新股\n    计算某年个股的涨幅排名\n    :return: None 生成excel\n    '
    year = int(year)
    basic = ts.get_stock_basics()
    pro = []
    name = ''
    if ignore_new_stock:
        basic = basic[basic['timeToMarket'] < int('{}0101'.format(year))]
        name = '_ignore_new_stock'
    filename = '{}_all_price_change{}.xls'.format(year, name)
    for code in basic.index.values:
        p = stock_profit(code, '{}-01-01'.format(year), '{}-01-01'.format(year + 1))
        pro.append(p)
    basic['p_change_year'] = pro
    basic = basic.sort_values(by='p_change_year', ascending=False)
    basic.to_excel(filename, encoding='gbk')

def stock_analysis(filename):
    if False:
        while True:
            i = 10
    '\n    # 分析年度的数据\n    :return:\n    '
    df = pd.read_excel(filename, encoding='gbk')
    print('mean:\n', df['p_change_year'].mean())
    print('max:\n', df['p_change_year'].max())
    print('min:\n', df['p_change_year'].min())
    print('middle\n', df['p_change_year'].median())

def cb_stock_year():
    if False:
        return 10
    '\n    上一年可转债正股的涨跌幅排名\n    :return:\n    '
    engine = get_engine('db_stock')
    df_cb = pd.read_sql('tb_bond_jisilu', engine)
    filename = '2019_all_price_change_ignore_new_stock.xls'
    df_all = pd.read_excel(filename, encoding='gbk')
    zg_codes = list(df_cb['正股代码'].values)
    df = df_all[df_all['code'].isin(zg_codes)]
    df.to_excel('2019_cb_zg.xls', encoding='gbk')

def main():
    if False:
        print('Hello World!')
    cb_stock_year()
if __name__ == '__main__':
    main()