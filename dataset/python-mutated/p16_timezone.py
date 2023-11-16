"""
Topic: 日期时区和本地化
Desc : 
"""
from datetime import datetime, timedelta
from pytz import timezone
import pytz

def tz_local():
    if False:
        for i in range(10):
            print('nop')
    d = datetime(2012, 12, 21, 9, 30, 0)
    print(d)
    central = timezone('US/Central')
    loc_d = central.localize(d)
    print(loc_d)
    bang_d = loc_d.astimezone(timezone('Asia/Kolkata'))
    print(bang_d)
    d = datetime(2013, 3, 10, 1, 45)
    loc_d = central.localize(d)
    print(loc_d)
    later = loc_d + timedelta(minutes=30)
    print(later)
    later = central.normalize(loc_d + timedelta(minutes=30))
    print(later)
    print(loc_d)
    utc_d = loc_d.astimezone(pytz.utc)
    print(utc_d)
    later_utc = utc_d + timedelta(minutes=30)
    print(later_utc.astimezone(central))
    print(pytz.country_timezones['IN'])
if __name__ == '__main__':
    tz_local()