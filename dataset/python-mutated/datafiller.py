from __future__ import absolute_import, division, print_function, unicode_literals
import collections
from datetime import datetime, timedelta
from backtrader import AbstractDataBase, TimeFrame

class DataFiller(AbstractDataBase):
    """This class will fill gaps in the source data using the following
    information bits from the underlying data source

      - timeframe and compression to dimension the output bars

      - sessionstart and sessionend

    If a data feed has missing bars in between 10:31 and 10:34 and the
    timeframe is minutes, the output will be filled with bars for minutes
    10:32 and 10:33 using the closing price of the last bar (10:31)

    Bars can be missinga amongst other things because

    Params:
      - ``fill_price`` (def: None): if None (or evaluates to False),the
        closing price will be used, else the passed value (which can be
        for example 'NaN' to have a missing bar in terms of evaluation but
        present in terms of time

      - ``fill_vol`` (def: NaN): used to fill the volume of missing bars

      - ``fill_oi`` (def: NaN): used to fill the openinterest of missing bars
    """
    params = (('fill_price', None), ('fill_vol', float('NaN')), ('fill_oi', float('NaN')))

    def start(self):
        if False:
            while True:
                i = 10
        super(DataFiller, self).start()
        self._fillbars = collections.deque()
        self._dbar = False

    def preload(self):
        if False:
            i = 10
            return i + 15
        if len(self.p.dataname) == self.p.dataname.buflen():
            self.p.dataname.start()
            self.p.dataname.preload()
            self.p.dataname.home()
        self.p.timeframe = self._timeframe = self.p.dataname._timeframe
        self.p.compression = self._compression = self.p.dataname._compression
        super(DataFiller, self).preload()

    def _copyfromdata(self):
        if False:
            i = 10
            return i + 15
        for i in range(self.p.dataname.size()):
            self.lines[i][0] = self.p.dataname.lines[i][0]
        self._dbar = False
        return True

    def _frombars(self):
        if False:
            print('Hello World!')
        (dtime, price) = self._fillbars.popleft()
        price = self.p.fill_price or price
        self.lines.datetime[0] = self.p.dataname.date2num(dtime)
        self.lines.open[0] = price
        self.lines.high[0] = price
        self.lines.low[0] = price
        self.lines.close[0] = price
        self.lines.volume[0] = self.p.fill_vol
        self.lines.openinterest[0] = self.p.fill_oi
        return True
    _tdeltas = {TimeFrame.Minutes: timedelta(seconds=60), TimeFrame.Seconds: timedelta(seconds=1), TimeFrame.MicroSeconds: timedelta(microseconds=1)}

    def _load(self):
        if False:
            print('Hello World!')
        if not len(self.p.dataname):
            self.p.dataname.start()
            self._timeframe = self.p.dataname._timeframe
            self._compression = self.p.dataname._compression
            self.p.timeframe = self._timeframe
            self.p.compression = self._compression
            self._tdunit = self._tdeltas[self._timeframe]
            self._tdunit *= self._compression
        if self._fillbars:
            return self._frombars()
        self._dbar = self._dbar or self.p.dataname.next()
        if not self._dbar:
            return False
        if len(self) == 1:
            return self._copyfromdata()
        pclose = self.lines.close[-1]
        dtime_prev = self.lines.datetime.datetime(-1)
        dtime_cur = self.p.dataname.datetime.datetime(0)
        send = datetime.combine(dtime_prev.date(), self.p.dataname.sessionend)
        if dtime_cur > send:
            dtime_prev += self._tdunit
            while dtime_prev < send:
                self._fillbars.append((dtime_prev, pclose))
                dtime_prev += self._tdunit
            sstart = datetime.combine(dtime_cur.date(), self.p.dataname.sessionstart)
            while sstart < dtime_cur:
                self._fillbars.append((sstart, pclose))
                sstart += self._tdunit
        else:
            dtime_prev += self._tdunit
            while dtime_prev < dtime_cur:
                self._fillbars.append((dtime_prev, pclose))
                dtime_prev += self._tdunit
        if self._fillbars:
            self._dbar = True
            return self._frombars()
        return self._copyfromdata()