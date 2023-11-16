import pandas as pd
import pytz
from datetime import datetime
from dateutil import rrule
from functools import partial
start = pd.Timestamp('1990-01-01', tz='UTC')
end_base = pd.Timestamp('today', tz='UTC')
end = end_base + pd.Timedelta(days=365)

def canonicalize_datetime(dt):
    if False:
        for i in range(10):
            print('nop')
    return datetime(dt.year, dt.month, dt.day, tzinfo=pytz.utc)

def get_non_trading_days(start, end):
    if False:
        print('Hello World!')
    non_trading_rules = []
    start = canonicalize_datetime(start)
    end = canonicalize_datetime(end)
    weekends = rrule.rrule(rrule.YEARLY, byweekday=(rrule.SA, rrule.SU), cache=True, dtstart=start, until=end)
    non_trading_rules.append(weekends)
    new_years = rrule.rrule(rrule.MONTHLY, byyearday=1, cache=True, dtstart=start, until=end)
    non_trading_rules.append(new_years)
    new_years_sunday = rrule.rrule(rrule.MONTHLY, byyearday=2, byweekday=rrule.MO, cache=True, dtstart=start, until=end)
    non_trading_rules.append(new_years_sunday)
    mlk_day = rrule.rrule(rrule.MONTHLY, bymonth=1, byweekday=rrule.MO(+3), cache=True, dtstart=datetime(1998, 1, 1, tzinfo=pytz.utc), until=end)
    non_trading_rules.append(mlk_day)
    presidents_day = rrule.rrule(rrule.MONTHLY, bymonth=2, byweekday=rrule.MO(3), cache=True, dtstart=start, until=end)
    non_trading_rules.append(presidents_day)
    good_friday = rrule.rrule(rrule.DAILY, byeaster=-2, cache=True, dtstart=start, until=end)
    non_trading_rules.append(good_friday)
    memorial_day = rrule.rrule(rrule.MONTHLY, bymonth=5, byweekday=rrule.MO(-1), cache=True, dtstart=start, until=end)
    non_trading_rules.append(memorial_day)
    july_4th = rrule.rrule(rrule.MONTHLY, bymonth=7, bymonthday=4, cache=True, dtstart=start, until=end)
    non_trading_rules.append(july_4th)
    july_4th_sunday = rrule.rrule(rrule.MONTHLY, bymonth=7, bymonthday=5, byweekday=rrule.MO, cache=True, dtstart=start, until=end)
    non_trading_rules.append(july_4th_sunday)
    july_4th_saturday = rrule.rrule(rrule.MONTHLY, bymonth=7, bymonthday=3, byweekday=rrule.FR, cache=True, dtstart=start, until=end)
    non_trading_rules.append(july_4th_saturday)
    labor_day = rrule.rrule(rrule.MONTHLY, bymonth=9, byweekday=rrule.MO(1), cache=True, dtstart=start, until=end)
    non_trading_rules.append(labor_day)
    thanksgiving = rrule.rrule(rrule.MONTHLY, bymonth=11, byweekday=rrule.TH(4), cache=True, dtstart=start, until=end)
    non_trading_rules.append(thanksgiving)
    christmas = rrule.rrule(rrule.MONTHLY, bymonth=12, bymonthday=25, cache=True, dtstart=start, until=end)
    non_trading_rules.append(christmas)
    christmas_sunday = rrule.rrule(rrule.MONTHLY, bymonth=12, bymonthday=26, byweekday=rrule.MO, cache=True, dtstart=start, until=end)
    non_trading_rules.append(christmas_sunday)
    christmas_saturday = rrule.rrule(rrule.MONTHLY, bymonth=12, bymonthday=24, byweekday=rrule.FR, cache=True, dtstart=start, until=end)
    non_trading_rules.append(christmas_saturday)
    non_trading_ruleset = rrule.rruleset()
    for rule in non_trading_rules:
        non_trading_ruleset.rrule(rule)
    non_trading_days = non_trading_ruleset.between(start, end, inc=True)
    for day_num in range(11, 17):
        non_trading_days.append(datetime(2001, 9, day_num, tzinfo=pytz.utc))
    for day_num in range(29, 31):
        non_trading_days.append(datetime(2012, 10, day_num, tzinfo=pytz.utc))
    non_trading_days.append(datetime(1994, 4, 27, tzinfo=pytz.utc))
    non_trading_days.append(datetime(2004, 6, 11, tzinfo=pytz.utc))
    non_trading_days.append(datetime(2007, 1, 2, tzinfo=pytz.utc))
    non_trading_days.sort()
    return pd.DatetimeIndex(non_trading_days)
non_trading_days = get_non_trading_days(start, end)
trading_day = pd.tseries.offsets.CDay(holidays=non_trading_days)

def get_trading_days(start, end, trading_day=trading_day):
    if False:
        for i in range(10):
            print('nop')
    return pd.date_range(start=start.date(), end=end.date(), freq=trading_day).tz_localize('UTC')
trading_days = get_trading_days(start, end)

def get_early_closes(start, end):
    if False:
        for i in range(10):
            print('nop')
    start = canonicalize_datetime(start)
    end = canonicalize_datetime(end)
    start = max(start, datetime(1993, 1, 1, tzinfo=pytz.utc))
    end = max(end, datetime(1993, 1, 1, tzinfo=pytz.utc))
    early_close_rules = []
    day_after_thanksgiving = rrule.rrule(rrule.MONTHLY, bymonth=11, byweekday=rrule.FR, bymonthday=range(23, 30), cache=True, dtstart=start, until=end)
    early_close_rules.append(day_after_thanksgiving)
    christmas_eve = rrule.rrule(rrule.MONTHLY, bymonth=12, bymonthday=24, byweekday=(rrule.MO, rrule.TU, rrule.WE, rrule.TH), cache=True, dtstart=start, until=end)
    early_close_rules.append(christmas_eve)
    friday_after_christmas = rrule.rrule(rrule.MONTHLY, bymonth=12, bymonthday=26, byweekday=rrule.FR, cache=True, dtstart=start, until=min(end, datetime(2007, 12, 31, tzinfo=pytz.utc)))
    early_close_rules.append(friday_after_christmas)
    day_before_independence_day = rrule.rrule(rrule.MONTHLY, bymonth=7, bymonthday=3, byweekday=(rrule.MO, rrule.TU, rrule.TH), cache=True, dtstart=start, until=end)
    early_close_rules.append(day_before_independence_day)
    day_after_independence_day = rrule.rrule(rrule.MONTHLY, bymonth=7, bymonthday=5, byweekday=rrule.FR, cache=True, dtstart=start, until=min(end, datetime(2012, 12, 31, tzinfo=pytz.utc)))
    early_close_rules.append(day_after_independence_day)
    wednesday_before_independence_day = rrule.rrule(rrule.MONTHLY, bymonth=7, bymonthday=3, byweekday=rrule.WE, cache=True, dtstart=max(start, datetime(2013, 1, 1, tzinfo=pytz.utc)), until=max(end, datetime(2013, 1, 1, tzinfo=pytz.utc)))
    early_close_rules.append(wednesday_before_independence_day)
    early_close_ruleset = rrule.rruleset()
    for rule in early_close_rules:
        early_close_ruleset.rrule(rule)
    early_closes = early_close_ruleset.between(start, end, inc=True)
    nye_1999 = datetime(1999, 12, 31, tzinfo=pytz.utc)
    if start <= nye_1999 and nye_1999 <= end:
        early_closes.append(nye_1999)
    early_closes.sort()
    return pd.DatetimeIndex(early_closes)
early_closes = get_early_closes(start, end)

def get_open_and_close(day, early_closes):
    if False:
        print('Hello World!')
    market_open = pd.Timestamp(datetime(year=day.year, month=day.month, day=day.day, hour=9, minute=31), tz='US/Eastern').tz_convert('UTC')
    close_hour = 13 if day in early_closes else 16
    market_close = pd.Timestamp(datetime(year=day.year, month=day.month, day=day.day, hour=close_hour), tz='US/Eastern').tz_convert('UTC')
    return (market_open, market_close)

def get_open_and_closes(trading_days, early_closes, get_open_and_close):
    if False:
        for i in range(10):
            print('nop')
    open_and_closes = pd.DataFrame(index=trading_days, columns=('market_open', 'market_close'))
    get_o_and_c = partial(get_open_and_close, early_closes=early_closes)
    (open_and_closes['market_open'], open_and_closes['market_close']) = zip(*open_and_closes.index.map(get_o_and_c))
    return open_and_closes
open_and_closes = get_open_and_closes(trading_days, early_closes, get_open_and_close)