from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import date, datetime, timedelta
from backtrader import TimeFrame
from backtrader.utils.py3 import with_metaclass
from .. import metabase

class CalendarDays(with_metaclass(metabase.MetaParams, object)):
    """
    Bar Filler to add missing calendar days to trading days

    Params:

      - fill_price (def: None):

        > 0: The given value to fill
        0 or None: Use the last known closing price
        -1: Use the midpoint of the last bar (High-Low average)

      - fill_vol (def: float('NaN')):

        Value to use to fill the missing volume

      - fill_oi (def: float('NaN')):

        Value to use to fill the missing Open Interest
    """
    params = (('fill_price', None), ('fill_vol', float('NaN')), ('fill_oi', float('NaN')))
    ONEDAY = timedelta(days=1)
    lastdt = date.max

    def __init__(self, data):
        if False:
            print('Hello World!')
        pass

    def __call__(self, data):
        if False:
            i = 10
            return i + 15
        '\n        If the data has a gap larger than 1 day amongst bars, the missing bars\n        are added to the stream.\n\n        Params:\n          - data: the data source to filter/process\n\n        Returns:\n          - False (always): this filter does not remove bars from the stream\n\n        '
        dt = data.datetime.date()
        if dt - self.lastdt > self.ONEDAY:
            self._fillbars(data, dt, self.lastdt)
        self.lastdt = dt
        return False

    def _fillbars(self, data, dt, lastdt):
        if False:
            while True:
                i = 10
        '\n        Fills one by one bars as needed from time_start to time_end\n\n        Invalidates the control dtime_prev if requested\n        '
        tm = data.datetime.time(0)
        if self.p.fill_price > 0:
            price = self.p.fill_price
        elif not self.p.fill_price:
            price = data.close[-1]
        elif self.p.fill_price == -1:
            price = (data.high[-1] + data.low[-1]) / 2.0
        while lastdt < dt:
            lastdt += self.ONEDAY
            bar = [float('Nan')] * data.size()
            bar[data.DateTime] = data.date2num(datetime.combine(lastdt, tm))
            for pricetype in [data.Open, data.High, data.Low, data.Close]:
                bar[pricetype] = price
            bar[data.Volume] = self.p.fill_vol
            bar[data.OpenInterest] = self.p.fill_oi
            for i in range(data.DateTime + 1, data.size()):
                bar[i] = data.lines[i][0]
            data._add2stack(bar)
        data._save2stack(erase=True)