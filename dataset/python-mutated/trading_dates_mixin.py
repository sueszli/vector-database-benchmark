import datetime
from typing import Dict, Optional, Union
import pandas as pd
from rqalpha.utils.functools import lru_cache
from rqalpha.const import TRADING_CALENDAR_TYPE

def _to_timestamp(d: Union[datetime.date, str, int, float]):
    if False:
        for i in range(10):
            print('nop')
    return pd.Timestamp(d).replace(hour=0, minute=0, second=0, microsecond=0)

class TradingDatesMixin(object):

    def __init__(self, trading_calendars):
        if False:
            return 10
        self.trading_calendars = trading_calendars
        self.merged_trading_calendars = pd.DatetimeIndex(sorted(set.union(*(set(calendar) for calendar in trading_calendars.values()))))

    def get_trading_calendar(self, trading_calendar_type=None):
        if False:
            i = 10
            return i + 15
        if trading_calendar_type is None:
            return self.merged_trading_calendars
        try:
            return self.trading_calendars[trading_calendar_type]
        except KeyError:
            raise NotImplementedError('unsupported trading_calendar_type {}'.format(trading_calendar_type))

    def get_trading_dates(self, start_date, end_date, trading_calendar_type=None):
        if False:
            i = 10
            return i + 15
        trading_dates = self.get_trading_calendar(trading_calendar_type)
        start_date = _to_timestamp(start_date)
        end_date = _to_timestamp(end_date)
        left = trading_dates.searchsorted(start_date)
        right = trading_dates.searchsorted(end_date, side='right')
        return trading_dates[left:right]

    @lru_cache(64)
    def get_previous_trading_date(self, date, n=1, trading_calendar_type=None) -> pd.Timestamp:
        if False:
            return 10
        trading_dates = self.get_trading_calendar(trading_calendar_type)
        pos = trading_dates.searchsorted(_to_timestamp(date))
        if pos >= n:
            return trading_dates[pos - n]
        else:
            return trading_dates[0]

    @lru_cache(64)
    def get_next_trading_date(self, date, n=1, trading_calendar_type=None):
        if False:
            for i in range(10):
                print('nop')
        trading_dates = self.get_trading_calendar(trading_calendar_type)
        pos = trading_dates.searchsorted(_to_timestamp(date), side='right')
        if pos + n > len(trading_dates):
            return trading_dates[-1]
        else:
            return trading_dates[pos + n - 1]

    def is_trading_date(self, date, trading_calendar_type=None):
        if False:
            for i in range(10):
                print('nop')
        trading_dates = self.get_trading_calendar(trading_calendar_type)
        pos = trading_dates.searchsorted(_to_timestamp(date))
        return pos < len(trading_dates) and trading_dates[pos].date() == date

    def get_trading_dt(self, calendar_dt):
        if False:
            for i in range(10):
                print('nop')
        trading_date = self.get_future_trading_date(calendar_dt)
        return datetime.datetime.combine(trading_date, calendar_dt.time())

    def get_future_trading_date(self, dt):
        if False:
            while True:
                i = 10
        return self._get_future_trading_date(dt.replace(minute=0, second=0, microsecond=0))

    def get_n_trading_dates_until(self, dt, n, trading_calendar_type=None):
        if False:
            i = 10
            return i + 15
        trading_dates = self.get_trading_calendar(trading_calendar_type)
        pos = trading_dates.searchsorted(_to_timestamp(dt), side='right')
        if pos >= n:
            return trading_dates[pos - n:pos]
        return trading_dates[:pos]

    def count_trading_dates(self, start_date, end_date, trading_calendar_type=None):
        if False:
            i = 10
            return i + 15
        start_date = _to_timestamp(start_date)
        end_date = _to_timestamp(end_date)
        trading_dates = self.get_trading_calendar(trading_calendar_type)
        return trading_dates.searchsorted(end_date, side='right') - trading_dates.searchsorted(start_date)

    @lru_cache(512)
    def _get_future_trading_date(self, dt):
        if False:
            i = 10
            return i + 15
        dt1 = dt - datetime.timedelta(hours=4)
        td = pd.Timestamp(dt1.date())
        trading_dates = self.get_trading_calendar(TRADING_CALENDAR_TYPE.EXCHANGE)
        pos = trading_dates.searchsorted(td)
        if trading_dates[pos] != td:
            raise RuntimeError('invalid future calendar datetime: {}'.format(dt))
        if dt1.hour >= 16:
            return trading_dates[pos + 1]
        return td