"""
上海银行间同业拆放利率（Shibor）数据接口
Created on 2014/07/31
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
import pandas as pd
import numpy as np
from tushare.stock import cons as ct
from tushare.util import dateu as du
from tushare.util.netbase import Client
from pandas.compat import StringIO

def shibor_data(year=None):
    if False:
        return 10
    '\n    获取上海银行间同业拆放利率（Shibor）\n    Parameters\n    ------\n      year:年份(int)\n      \n    Return\n    ------\n    date:日期\n    ON:隔夜拆放利率\n    1W:1周拆放利率\n    2W:2周拆放利率\n    1M:1个月拆放利率\n    3M:3个月拆放利率\n    6M:6个月拆放利率\n    9M:9个月拆放利率\n    1Y:1年拆放利率\n    '
    year = du.get_year() if year is None else year
    lab = ct.SHIBOR_TYPE['Shibor']
    lab = lab.encode('utf-8') if ct.PY3 else lab
    try:
        clt = Client(url=ct.SHIBOR_DATA_URL % (ct.P_TYPE['http'], ct.DOMAINS['shibor'], ct.PAGES['dw'], 'Shibor', year, lab, year))
        content = clt.gvalue()
        df = pd.read_excel(StringIO(content))
        df.columns = ct.SHIBOR_COLS
        df['date'] = df['date'].map(lambda x: x.date())
        if pd.__version__ < '0.21':
            df['date'] = df['date'].astype(np.datetime64)
        else:
            df['date'] = df['date'].astype('datetime64[D]')
        return df
    except:
        return None

def shibor_quote_data(year=None):
    if False:
        i = 10
        return i + 15
    '\n    获取Shibor银行报价数据\n    Parameters\n    ------\n      year:年份(int)\n      \n    Return\n    ------\n    date:日期\n    bank:报价银行名称\n    ON:隔夜拆放利率\n    ON_B:隔夜拆放买入价\n    ON_A:隔夜拆放卖出价\n    1W_B:1周买入\n    1W_A:1周卖出\n    2W_B:买入\n    2W_A:卖出\n    1M_B:买入\n    1M_A:卖出\n    3M_B:买入\n    3M_A:卖出\n    6M_B:买入\n    6M_A:卖出\n    9M_B:买入\n    9M_A:卖出\n    1Y_B:买入\n    1Y_A:卖出\n    '
    year = du.get_year() if year is None else year
    lab = ct.SHIBOR_TYPE['Quote']
    lab = lab.encode('utf-8') if ct.PY3 else lab
    try:
        clt = Client(url=ct.SHIBOR_DATA_URL % (ct.P_TYPE['http'], ct.DOMAINS['shibor'], ct.PAGES['dw'], 'Quote', year, lab, year))
        content = clt.gvalue()
        df = pd.read_excel(StringIO(content), skiprows=[0])
        df.columns = ct.SHIBOR_Q_COLS
        df['date'] = df['date'].map(lambda x: x.date())
        if pd.__version__ < '0.21':
            df['date'] = df['date'].astype(np.datetime64)
        else:
            df['date'] = df['date'].astype('datetime64[D]')
        return df
    except:
        return None

def shibor_ma_data(year=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    获取Shibor均值数据\n    Parameters\n    ------\n      year:年份(int)\n      \n    Return\n    ------\n    date:日期\n       其它分别为各周期5、10、20均价\n    '
    year = du.get_year() if year is None else year
    lab = ct.SHIBOR_TYPE['Tendency']
    lab = lab.encode('utf-8') if ct.PY3 else lab
    try:
        clt = Client(url=ct.SHIBOR_DATA_URL % (ct.P_TYPE['http'], ct.DOMAINS['shibor'], ct.PAGES['dw'], 'Shibor_Tendency', year, lab, year))
        content = clt.gvalue()
        df = pd.read_excel(StringIO(content), skiprows=[0])
        df.columns = ct.SHIBOR_MA_COLS
        df['date'] = df['date'].map(lambda x: x.date())
        if pd.__version__ < '0.21':
            df['date'] = df['date'].astype(np.datetime64)
        else:
            df['date'] = df['date'].astype('datetime64[D]')
        return df
    except:
        return None

def lpr_data(year=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    获取贷款基础利率（LPR）\n    Parameters\n    ------\n      year:年份(int)\n      \n    Return\n    ------\n    date:日期\n    1Y:1年贷款基础利率\n    '
    year = du.get_year() if year is None else year
    lab = ct.SHIBOR_TYPE['LPR']
    lab = lab.encode('utf-8') if ct.PY3 else lab
    try:
        clt = Client(url=ct.SHIBOR_DATA_URL % (ct.P_TYPE['http'], ct.DOMAINS['shibor'], ct.PAGES['dw'], 'LPR', year, lab, year))
        content = clt.gvalue()
        df = pd.read_excel(StringIO(content), skiprows=[0])
        df.columns = ct.LPR_COLS
        df['date'] = df['date'].map(lambda x: x.date())
        if pd.__version__ < '0.21':
            df['date'] = df['date'].astype(np.datetime64)
        else:
            df['date'] = df['date'].astype('datetime64[D]')
        return df
    except:
        return None

def lpr_ma_data(year=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    获取贷款基础利率均值数据\n    Parameters\n    ------\n      year:年份(int)\n      \n    Return\n    ------\n    date:日期\n    1Y_5:5日均值\n    1Y_10:10日均值\n    1Y_20:20日均值\n    '
    year = du.get_year() if year is None else year
    lab = ct.SHIBOR_TYPE['LPR_Tendency']
    lab = lab.encode('utf-8') if ct.PY3 else lab
    try:
        clt = Client(url=ct.SHIBOR_DATA_URL % (ct.P_TYPE['http'], ct.DOMAINS['shibor'], ct.PAGES['dw'], 'LPR_Tendency', year, lab, year))
        content = clt.gvalue()
        df = pd.read_excel(StringIO(content), skiprows=[0])
        df.columns = ct.LPR_MA_COLS
        df['date'] = df['date'].map(lambda x: x.date())
        if pd.__version__ < '0.21':
            df['date'] = df['date'].astype(np.datetime64)
        else:
            df['date'] = df['date'].astype('datetime64[D]')
        return df
    except:
        return None