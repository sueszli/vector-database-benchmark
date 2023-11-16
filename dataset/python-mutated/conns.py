"""
connection for api 
Created on 2017/09/23
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API
from tushare.stock import cons as ct

def api(retry_count=3):
    if False:
        while True:
            i = 10
    for _ in range(retry_count):
        try:
            api = TdxHq_API(heartbeat=True)
            api.connect(ct._get_server(), ct.T_PORT)
        except Exception as e:
            print(e)
        else:
            return api
    raise IOError(ct.NETWORK_URL_ERROR_MSG)

def xapi(retry_count=3):
    if False:
        i = 10
        return i + 15
    for _ in range(retry_count):
        try:
            api = TdxExHq_API(heartbeat=True)
            api.connect(ct._get_xserver(), ct.X_PORT)
        except Exception as e:
            print(e)
        else:
            return api
    raise IOError(ct.NETWORK_URL_ERROR_MSG)

def xapi_x(retry_count=3):
    if False:
        print('Hello World!')
    for _ in range(retry_count):
        try:
            api = TdxExHq_API(heartbeat=True)
            api.connect(ct._get_xxserver(), ct.X_PORT)
        except Exception as e:
            print(e)
        else:
            return api
    raise IOError(ct.NETWORK_URL_ERROR_MSG)

def get_apis():
    if False:
        print('Hello World!')
    return (api(), xapi())

def close_apis(conn):
    if False:
        for i in range(10):
            print('nop')
    (api, xapi) = conn
    try:
        api.disconnect()
        xapi.disconnect()
    except Exception as e:
        print(e)