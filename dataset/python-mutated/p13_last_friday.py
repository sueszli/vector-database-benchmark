"""
Topic: 最后的周五
Desc : 
"""
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil.rrule import *
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def get_previous_byday(dayname, start_date=None):
    if False:
        print('Hello World!')
    if start_date is None:
        start_date = datetime.today()
    day_num = start_date.weekday()
    day_num_target = weekdays.index(dayname)
    days_ago = (7 + day_num - day_num_target) % 7
    if days_ago == 0:
        days_ago = 7
    target_date = start_date - timedelta(days=days_ago)
    return target_date

def last_friday():
    if False:
        for i in range(10):
            print('nop')
    print(datetime.today())
    print(get_previous_byday('Monday'))
    print(get_previous_byday('Tuesday'))
    print(get_previous_byday('Friday'))
    print(get_previous_byday('Saturday'))
    print(get_previous_byday('Sunday', datetime(2012, 12, 21)))
    d = datetime.now()
    print(d + relativedelta(weekday=FR))
    print(d + relativedelta(weekday=FR(-1)))
    print(d + relativedelta(weekday=SA))
    print(d + relativedelta(weekday=SA(-1)))
if __name__ == '__main__':
    last_friday()