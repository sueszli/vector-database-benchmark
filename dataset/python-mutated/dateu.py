import datetime
import time
import pandas as pd
from tushare.stock import cons as ct

def year_qua(date):
    if False:
        i = 10
        return i + 15
    mon = date[5:7]
    mon = int(mon)
    return [date[0:4], _quar(mon)]

def _quar(mon):
    if False:
        while True:
            i = 10
    if mon in [1, 2, 3]:
        return '1'
    elif mon in [4, 5, 6]:
        return '2'
    elif mon in [7, 8, 9]:
        return '3'
    elif mon in [10, 11, 12]:
        return '4'
    else:
        return None

def today():
    if False:
        while True:
            i = 10
    day = datetime.datetime.today().date()
    return str(day)

def get_year():
    if False:
        for i in range(10):
            print('nop')
    year = datetime.datetime.today().year
    return year

def get_month():
    if False:
        i = 10
        return i + 15
    month = datetime.datetime.today().month
    return month

def get_hour():
    if False:
        for i in range(10):
            print('nop')
    return datetime.datetime.today().hour

def today_last_year():
    if False:
        while True:
            i = 10
    lasty = datetime.datetime.today().date() + datetime.timedelta(-365)
    return str(lasty)

def day_last_week(days=-7):
    if False:
        return 10
    lasty = datetime.datetime.today().date() + datetime.timedelta(days)
    return str(lasty)

def get_now():
    if False:
        i = 10
        return i + 15
    return time.strftime('%Y-%m-%d %H:%M:%S')

def int2time(timestamp):
    if False:
        for i in range(10):
            print('nop')
    datearr = datetime.datetime.utcfromtimestamp(timestamp)
    timestr = datearr.strftime('%Y-%m-%d %H:%M:%S')
    return timestr

def diff_day(start=None, end=None):
    if False:
        while True:
            i = 10
    d1 = datetime.datetime.strptime(end, '%Y-%m-%d')
    d2 = datetime.datetime.strptime(start, '%Y-%m-%d')
    delta = d1 - d2
    return delta.days

def get_quarts(start, end):
    if False:
        return 10
    idx = pd.period_range('Q'.join(year_qua(start)), 'Q'.join(year_qua(end)), freq='Q-JAN')
    return [str(d).split('Q') for d in idx][::-1]

def trade_cal():
    if False:
        for i in range(10):
            print('nop')
    '\n            交易日历\n    isOpen=1是交易日，isOpen=0为休市\n    '
    df = pd.read_csv(ct.ALL_CAL_FILE)
    return df

def is_holiday(date):
    if False:
        print('Hello World!')
    '\n            判断是否为交易日，返回True or False\n    '
    df = trade_cal()
    holiday = df[df.isOpen == 0]['calendarDate'].values
    if isinstance(date, str):
        today = datetime.datetime.strptime(date, '%Y-%m-%d')
    if today.isoweekday() in [6, 7] or str(date) in holiday:
        return True
    else:
        return False

def last_tddate():
    if False:
        return 10
    today = datetime.datetime.today().date()
    today = int(today.strftime('%w'))
    if today == 0:
        return day_last_week(-2)
    else:
        return day_last_week(-1)

def tt_dates(start='', end=''):
    if False:
        for i in range(10):
            print('nop')
    startyear = int(start[0:4])
    endyear = int(end[0:4])
    dates = [d for d in range(startyear, endyear + 1, 2)]
    return dates

def _random(n=13):
    if False:
        for i in range(10):
            print('nop')
    from random import randint
    start = 10 ** (n - 1)
    end = 10 ** n - 1
    return str(randint(start, end))

def get_q_date(year=None, quarter=None):
    if False:
        while True:
            i = 10
    dt = {'1': '-03-31', '2': '-06-30', '3': '-09-30', '4': '-12-31'}
    return '%s%s' % (str(year), dt[str(quarter)])