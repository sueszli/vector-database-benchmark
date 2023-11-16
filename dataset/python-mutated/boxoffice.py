"""
电影票房 
Created on 2015/12/24
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
import pandas as pd
from tushare.stock import cons as ct
from tushare.util import dateu as du
try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request
import time
import json

def realtime_boxoffice(retry_count=3, pause=0.001):
    if False:
        print('Hello World!')
    '\n    获取实时电影票房数据\n    数据来源：EBOT艺恩票房智库\n    Parameters\n    ------\n        retry_count : int, 默认 3\n                  如遇网络等问题重复执行的次数\n        pause : int, 默认 0\n                 重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题\n     return\n     -------\n        DataFrame \n              BoxOffice     实时票房（万） \n              Irank         排名\n              MovieName     影片名 \n              boxPer        票房占比 （%）\n              movieDay      上映天数\n              sumBoxOffice  累计票房（万） \n              time          数据获取时间\n    '
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            request = Request(ct.MOVIE_BOX % (ct.P_TYPE['http'], ct.DOMAINS['mbox'], ct.BOX, _random()))
            lines = urlopen(request, timeout=10).read()
            if len(lines) < 15:
                return None
        except Exception as e:
            print(e)
        else:
            js = json.loads(lines.decode('utf-8') if ct.PY3 else lines)
            df = pd.DataFrame(js['data2'])
            df = df.drop(['MovieImg', 'mId'], axis=1)
            df['time'] = du.get_now()
            return df

def day_boxoffice(date=None, retry_count=3, pause=0.001):
    if False:
        while True:
            i = 10
    '\n    获取单日电影票房数据\n    数据来源：EBOT艺恩票房智库\n    Parameters\n    ------\n        date:日期，默认为上一日\n        retry_count : int, 默认 3\n                  如遇网络等问题重复执行的次数\n        pause : int, 默认 0\n                 重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题\n     return\n     -------\n        DataFrame \n              AvgPrice      平均票价\n              AvpPeoPle     场均人次\n              BoxOffice     单日票房（万）\n              BoxOffice_Up  环比变化 （%）\n              IRank         排名\n              MovieDay      上映天数\n              MovieName     影片名 \n              SumBoxOffice  累计票房（万） \n              WomIndex      口碑指数 \n    '
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            if date is None:
                date = 0
            else:
                date = int(du.diff_day(du.today(), date)) + 1
            request = Request(ct.BOXOFFICE_DAY % (ct.P_TYPE['http'], ct.DOMAINS['mbox'], ct.BOX, date, _random()))
            lines = urlopen(request, timeout=10).read()
            if len(lines) < 15:
                return None
        except Exception as e:
            print(e)
        else:
            js = json.loads(lines.decode('utf-8') if ct.PY3 else lines)
            df = pd.DataFrame(js['data1'])
            df = df.drop(['MovieImg', 'BoxOffice1', 'MovieID', 'Director', 'IRank_pro'], axis=1)
            return df

def month_boxoffice(date=None, retry_count=3, pause=0.001):
    if False:
        while True:
            i = 10
    '\n    获取单月电影票房数据\n    数据来源：EBOT艺恩票房智库\n    Parameters\n    ------\n        date:日期，默认为上一月，格式YYYY-MM\n        retry_count : int, 默认 3\n                  如遇网络等问题重复执行的次数\n        pause : int, 默认 0\n                 重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题\n     return\n     -------\n        DataFrame \n              Irank         排名\n              MovieName     电影名称\n              WomIndex      口碑指数\n              avgboxoffice  平均票价\n              avgshowcount  场均人次\n              box_pro       月度占比\n              boxoffice     单月票房(万)     \n              days          月内天数\n              releaseTime   上映日期\n    '
    if date is None:
        date = du.day_last_week(-30)[0:7]
    elif len(date) > 8:
        print(ct.BOX_INPUT_ERR_MSG)
        return
    date += '-01'
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            request = Request(ct.BOXOFFICE_MONTH % (ct.P_TYPE['http'], ct.DOMAINS['mbox'], ct.BOX, date))
            lines = urlopen(request, timeout=10).read()
            if len(lines) < 15:
                return None
        except Exception as e:
            print(e)
        else:
            js = json.loads(lines.decode('utf-8') if ct.PY3 else lines)
            df = pd.DataFrame(js['data1'])
            df = df.drop(['defaultImage', 'EnMovieID'], axis=1)
            return df

def day_cinema(date=None, retry_count=3, pause=0.001):
    if False:
        return 10
    '\n        获取影院单日票房排行数据\n        数据来源：EBOT艺恩票房智库\n        Parameters\n        ------\n            date:日期，默认为上一日\n            retry_count : int, 默认 3\n                      如遇网络等问题重复执行的次数\n            pause : int, 默认 0\n                     重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题\n         return\n         -------\n            DataFrame \n                  Attendance         上座率\n                  AvgPeople          场均人次\n                  CinemaName         影院名称  \n                  RowNum             排名\n                  TodayAudienceCount 当日观众人数\n                  TodayBox           当日票房\n                  TodayShowCount     当日场次\n                  price              场均票价（元）\n    '
    if date is None:
        date = du.day_last_week(-1)
    data = pd.DataFrame()
    ct._write_head()
    for x in range(1, 11):
        df = _day_cinema(date, x, retry_count, pause)
        if df is not None:
            data = pd.concat([data, df])
    data = data.drop_duplicates()
    return data.reset_index(drop=True)

def _day_cinema(date=None, pNo=1, retry_count=3, pause=0.001):
    if False:
        print('Hello World!')
    ct._write_console()
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            request = Request(ct.BOXOFFICE_CBD % (ct.P_TYPE['http'], ct.DOMAINS['mbox'], ct.BOX, pNo, date))
            lines = urlopen(request, timeout=10).read()
            if len(lines) < 15:
                return None
        except Exception as e:
            print(e)
        else:
            js = json.loads(lines.decode('utf-8') if ct.PY3 else lines)
            df = pd.DataFrame(js['data1'])
            df = df.drop(['CinemaID'], axis=1)
            return df

def _random(n=13):
    if False:
        return 10
    from random import randint
    start = 10 ** (n - 1)
    end = 10 ** n - 1
    return str(randint(start, end))