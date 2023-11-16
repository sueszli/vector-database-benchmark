"""
offsets benchmarks that rely only on tslibs.  See benchmarks.offset for
offsets benchmarks that rely on other parts of pandas.
"""
from datetime import datetime
import numpy as np
from pandas import offsets
try:
    import pandas.tseries.holiday
except ImportError:
    pass
hcal = pandas.tseries.holiday.USFederalHolidayCalendar()
non_apply = [offsets.Day(), offsets.BYearEnd(), offsets.BYearBegin(), offsets.BQuarterEnd(), offsets.BQuarterBegin(), offsets.BMonthEnd(), offsets.BMonthBegin(), offsets.CustomBusinessDay(), offsets.CustomBusinessDay(calendar=hcal), offsets.CustomBusinessMonthBegin(calendar=hcal), offsets.CustomBusinessMonthEnd(calendar=hcal), offsets.CustomBusinessMonthEnd(calendar=hcal)]
other_offsets = [offsets.YearEnd(), offsets.YearBegin(), offsets.QuarterEnd(), offsets.QuarterBegin(), offsets.MonthEnd(), offsets.MonthBegin(), offsets.DateOffset(months=2, days=2), offsets.BusinessDay(), offsets.SemiMonthEnd(), offsets.SemiMonthBegin()]
offset_objs = non_apply + other_offsets

class OnOffset:
    params = offset_objs
    param_names = ['offset']

    def setup(self, offset):
        if False:
            while True:
                i = 10
        self.dates = [datetime(2016, m, d) for m in [10, 11, 12] for d in [1, 2, 3, 28, 29, 30, 31] if not (m == 11 and d == 31)]

    def time_on_offset(self, offset):
        if False:
            while True:
                i = 10
        for date in self.dates:
            offset.is_on_offset(date)

class OffestDatetimeArithmetic:
    params = offset_objs
    param_names = ['offset']

    def setup(self, offset):
        if False:
            for i in range(10):
                print('nop')
        self.date = datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')

    def time_add_np_dt64(self, offset):
        if False:
            for i in range(10):
                print('nop')
        offset + self.dt64

    def time_add(self, offset):
        if False:
            while True:
                i = 10
        self.date + offset

    def time_add_10(self, offset):
        if False:
            return 10
        self.date + 10 * offset

    def time_subtract(self, offset):
        if False:
            print('Hello World!')
        self.date - offset

    def time_subtract_10(self, offset):
        if False:
            i = 10
            return i + 15
        self.date - 10 * offset