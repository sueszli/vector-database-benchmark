from datetime import datetime
import numpy as np
import pytz
from pandas import Timestamp
from .tslib import _tzs

class TimestampConstruction:

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.npdatetime64 = np.datetime64('2020-01-01 00:00:00')
        self.dttime_unaware = datetime(2020, 1, 1, 0, 0, 0)
        self.dttime_aware = datetime(2020, 1, 1, 0, 0, 0, 0, pytz.UTC)
        self.ts = Timestamp('2020-01-01 00:00:00')

    def time_parse_iso8601_no_tz(self):
        if False:
            while True:
                i = 10
        Timestamp('2017-08-25 08:16:14')

    def time_parse_iso8601_tz(self):
        if False:
            for i in range(10):
                print('nop')
        Timestamp('2017-08-25 08:16:14-0500')

    def time_parse_dateutil(self):
        if False:
            return 10
        Timestamp('2017/08/25 08:16:14 AM')

    def time_parse_today(self):
        if False:
            while True:
                i = 10
        Timestamp('today')

    def time_parse_now(self):
        if False:
            while True:
                i = 10
        Timestamp('now')

    def time_fromordinal(self):
        if False:
            for i in range(10):
                print('nop')
        Timestamp.fromordinal(730120)

    def time_fromtimestamp(self):
        if False:
            i = 10
            return i + 15
        Timestamp.fromtimestamp(1515448538)

    def time_from_npdatetime64(self):
        if False:
            i = 10
            return i + 15
        Timestamp(self.npdatetime64)

    def time_from_datetime_unaware(self):
        if False:
            while True:
                i = 10
        Timestamp(self.dttime_unaware)

    def time_from_datetime_aware(self):
        if False:
            while True:
                i = 10
        Timestamp(self.dttime_aware)

    def time_from_pd_timestamp(self):
        if False:
            while True:
                i = 10
        Timestamp(self.ts)

class TimestampProperties:
    params = [_tzs]
    param_names = ['tz']

    def setup(self, tz):
        if False:
            return 10
        self.ts = Timestamp('2017-08-25 08:16:14', tzinfo=tz)

    def time_tz(self, tz):
        if False:
            print('Hello World!')
        self.ts.tz

    def time_dayofweek(self, tz):
        if False:
            for i in range(10):
                print('nop')
        self.ts.dayofweek

    def time_dayofyear(self, tz):
        if False:
            return 10
        self.ts.dayofyear

    def time_week(self, tz):
        if False:
            for i in range(10):
                print('nop')
        self.ts.week

    def time_quarter(self, tz):
        if False:
            for i in range(10):
                print('nop')
        self.ts.quarter

    def time_days_in_month(self, tz):
        if False:
            for i in range(10):
                print('nop')
        self.ts.days_in_month

    def time_is_month_start(self, tz):
        if False:
            for i in range(10):
                print('nop')
        self.ts.is_month_start

    def time_is_month_end(self, tz):
        if False:
            for i in range(10):
                print('nop')
        self.ts.is_month_end

    def time_is_quarter_start(self, tz):
        if False:
            i = 10
            return i + 15
        self.ts.is_quarter_start

    def time_is_quarter_end(self, tz):
        if False:
            i = 10
            return i + 15
        self.ts.is_quarter_end

    def time_is_year_start(self, tz):
        if False:
            while True:
                i = 10
        self.ts.is_year_start

    def time_is_year_end(self, tz):
        if False:
            return 10
        self.ts.is_year_end

    def time_is_leap_year(self, tz):
        if False:
            for i in range(10):
                print('nop')
        self.ts.is_leap_year

    def time_microsecond(self, tz):
        if False:
            return 10
        self.ts.microsecond

    def time_month_name(self, tz):
        if False:
            return 10
        self.ts.month_name()

    def time_weekday_name(self, tz):
        if False:
            i = 10
            return i + 15
        self.ts.day_name()

class TimestampOps:
    params = _tzs
    param_names = ['tz']

    def setup(self, tz):
        if False:
            while True:
                i = 10
        self.ts = Timestamp('2017-08-25 08:16:14', tz=tz)

    def time_replace_tz(self, tz):
        if False:
            return 10
        self.ts.replace(tzinfo=pytz.timezone('US/Eastern'))

    def time_replace_None(self, tz):
        if False:
            return 10
        self.ts.replace(tzinfo=None)

    def time_to_pydatetime(self, tz):
        if False:
            while True:
                i = 10
        self.ts.to_pydatetime()

    def time_normalize(self, tz):
        if False:
            while True:
                i = 10
        self.ts.normalize()

    def time_tz_convert(self, tz):
        if False:
            return 10
        if self.ts.tz is not None:
            self.ts.tz_convert(tz)

    def time_tz_localize(self, tz):
        if False:
            i = 10
            return i + 15
        if self.ts.tz is None:
            self.ts.tz_localize(tz)

    def time_to_julian_date(self, tz):
        if False:
            return 10
        self.ts.to_julian_date()

    def time_floor(self, tz):
        if False:
            print('Hello World!')
        self.ts.floor('5min')

    def time_ceil(self, tz):
        if False:
            return 10
        self.ts.ceil('5min')

class TimestampAcrossDst:

    def setup(self):
        if False:
            return 10
        dt = datetime(2016, 3, 27, 1)
        self.tzinfo = pytz.timezone('CET').localize(dt, is_dst=False).tzinfo
        self.ts2 = Timestamp(dt)

    def time_replace_across_dst(self):
        if False:
            for i in range(10):
                print('nop')
        self.ts2.replace(tzinfo=self.tzinfo)