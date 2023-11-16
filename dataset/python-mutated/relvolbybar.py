from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import datetime
import math
import backtrader as bt

class RelativeVolumeByBar(bt.Indicator):
    alias = ('RVBB',)
    lines = ('rvbb',)
    params = (('prestart', datetime.time(8, 0)), ('start', datetime.time(9, 10)), ('end', datetime.time(17, 15)))

    def _plotlabel(self):
        if False:
            i = 10
            return i + 15
        plabels = []
        for (name, value) in self.params._getitems():
            plabels.append('%s: %s' % (name, value.strftime('%H:%M')))
        return plabels

    def __init__(self):
        if False:
            print('Hello World!')
        minbuffer = self._calcbuffer()
        self.addminperiod(minbuffer)
        self.pvol = dict()
        self.vcount = collections.defaultdict(int)
        self.days = 0
        self.dtlast = datetime.date.min
        super(RelativeVolumeByBar, self).__init__()

    def _barisvalid(self, tm):
        if False:
            for i in range(10):
                print('nop')
        return self.p.start <= tm <= self.p.end

    def _daycount(self):
        if False:
            print('Hello World!')
        dt = self.data.datetime.date()
        if dt > self.dtlast:
            self.days += 1
            self.dtlast = dt

    def prenext(self):
        if False:
            while True:
                i = 10
        self._daycount()
        tm = self.data.datetime.time()
        if self._barisvalid(tm):
            self.pvol[tm] = self.data.volume[0]
            self.vcount[tm] += 1

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        self._daycount()
        tm = self.data.datetime.time()
        if not self._barisvalid(tm):
            return
        self.vcount[tm] += 1
        vol = self.data.volume[0]
        if self.vcount[tm] == self.days:
            self.lines.rvbb[0] = vol / self.pvol[tm]
        self.vcount[tm] = self.days
        self.pvol[tm] = vol

    def _calcbuffer(self):
        if False:
            return 10
        minend = self.p.end.hour * 60 + self.p.end.minute
        minstart = self.p.prestart.hour * 60 + self.p.prestart.minute
        minbuffer = minend - minstart
        tframe = self.data._timeframe
        tcomp = self.data._compression
        if tframe == bt.TimeFrame.Seconds:
            minbuffer = minperiod * 60
        minbuffer = minbuffer // tcomp + tcomp
        return minbuffer