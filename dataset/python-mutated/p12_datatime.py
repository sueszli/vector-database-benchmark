"""
Topic: 日期时间转换
Desc : 
"""
from datetime import timedelta
from datetime import datetime
from dateutil.relativedelta import relativedelta

def date_time():
    if False:
        return 10
    a = timedelta(days=2, hours=6)
    b = timedelta(hours=4.5)
    c = a + b
    print(c.days)
    print(c.seconds)
    print(c.seconds / 3600)
    print(c.total_seconds() / 3600)
    a = datetime(2012, 9, 23)
    print(a + timedelta(days=10))
    b = datetime(2012, 12, 21)
    d = b - a
    print(d.days)
    now = datetime.today()
    print(now)
    print(now + timedelta(minutes=10))
    a = datetime(2012, 9, 23)
    print(a + relativedelta(months=+1))
    print(a + relativedelta(months=+4))
    b = datetime(2012, 12, 21)
    d = b - a
    print(d)
    d = relativedelta(b, a)
    print(d)
    print(d.months, d.days)
if __name__ == '__main__':
    date_time()