"""
Created on 2015/08/24
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
import pandas as pd
import os
from tushare.stock import cons as ct
BK = 'bk'

def set_token(token):
    if False:
        print('Hello World!')
    df = pd.DataFrame([token], columns=['token'])
    user_home = os.path.expanduser('~')
    fp = os.path.join(user_home, ct.TOKEN_F_P)
    df.to_csv(fp, index=False)

def get_token():
    if False:
        while True:
            i = 10
    user_home = os.path.expanduser('~')
    fp = os.path.join(user_home, ct.TOKEN_F_P)
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        return str(df.ix[0]['token'])
    else:
        print(ct.TOKEN_ERR_MSG)
        return None

def set_broker(broker='', user='', passwd=''):
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame([[broker, user, passwd]], columns=['broker', 'user', 'passwd'], dtype=object)
    if os.path.exists(BK):
        all = pd.read_csv(BK, dtype=object)
        if all[all.broker == broker]['user'].any():
            all = all[all.broker != broker]
        all = all.append(df, ignore_index=True)
        all.to_csv(BK, index=False)
    else:
        df.to_csv(BK, index=False)

def get_broker(broker=''):
    if False:
        print('Hello World!')
    if os.path.exists(BK):
        df = pd.read_csv(BK, dtype=object)
        if broker == '':
            return df
        else:
            return df[df.broker == broker]
    else:
        return None

def remove_broker():
    if False:
        print('Hello World!')
    os.remove(BK)