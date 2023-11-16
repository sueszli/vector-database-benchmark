from __future__ import absolute_import, division, print_function, unicode_literals
import datetime

class WeekDaysFiller(object):
    """Bar Filler to add missing calendar days to trading days"""
    ONEDAY = datetime.timedelta(days=1)
    lastdt = datetime.date.max - ONEDAY

    def __init__(self, data, fillclose=False):
        if False:
            print('Hello World!')
        self.fillclose = fillclose
        self.voidbar = [float('Nan')] * data.size()

    def __call__(self, data):
        if False:
            while True:
                i = 10
        'Empty bars (NaN) or with last close price are added for weekdays with no\n        data\n\n        Params:\n          - data: the data source to filter/process\n\n        Returns:\n          - True (always): bars are removed (even if put back on the stack)\n\n        '
        dt = data.datetime.date()
        lastdt = self.lastdt + self.ONEDAY
        while lastdt < dt:
            if lastdt.isoweekday() < 6:
                if self.fillclose:
                    self.voidbar = [self.lastclose] * data.size()
                dtime = datetime.datetime.combine(lastdt, data.p.sessionend)
                self.voidbar[-1] = data.date2num(dtime)
                data._add2stack(self.voidbar[:])
            lastdt += self.ONEDAY
        self.lastdt = dt
        self.lastclose = data.close[0]
        data._save2stack(erase=True)
        return True