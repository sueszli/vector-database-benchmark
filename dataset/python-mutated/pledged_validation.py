import json
import tushare as ts
import pandas as pd
from configure.settings import get_engine
import matplotlib.pyplot as plt
with open('codes.txt', 'r') as f:
    codes = json.load(f)
stocks = codes.get('example1')
engine = get_engine('db_stock')

def pledge_info():
    if False:
        for i in range(10):
            print('nop')
    df = ts.stock_pledged()
    df.to_sql('tb_pledged_base', engine, if_exists='replace')
    df_list = []
    for stock in stocks:
        df_list.append(df[df['code'] == stock])
    df = pd.concat(df_list)
    df = df.reset_index(drop=True)
    df = df.sort_values('p_ratio', ascending=False)
    df['code'] = df['code'].astype('str')
    df['rest_ratio'] = df['rest_pledged'] / df['totals'] * 100
    df['rest_ratio'] = map(lambda x: round(x, 2), df['rest_ratio'])
    df['unrest_ratio'] = df['unrest_pledged'] / df['totals'] * 100
    df['unrest_ratio'] = map(lambda x: round(x, 2), df['unrest_ratio'])

def pledged_detail():
    if False:
        while True:
            i = 10
    df = ts.pledged_detail()
    print(df.tail(10))
    df.to_sql('tb_pledged_detail', engine)

def do_calculation():
    if False:
        return 10
    df = pd.read_sql('tb_pledged_base', engine, index_col='index')
    print('median ', df['p_ratio'].median())
    print('mean ', df['p_ratio'].mean())
    print('std ', df['p_ratio'].std())
    print('var ', df['p_ratio'].var())
    plt.figure()
    plt.hist(df['p_ratio'], 20)
    plt.show()
do_calculation()