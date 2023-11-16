from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime
from struct import unpack
import os.path
import backtrader as bt
from backtrader import date2num

class MetaVChartFile(bt.DataBase.__class__):

    def __init__(cls, name, bases, dct):
        if False:
            return 10
        'Class has already been created ... register'
        super(MetaVChartFile, cls).__init__(name, bases, dct)
        bt.stores.VChartFile.DataCls = cls

class VChartFile(bt.with_metaclass(MetaVChartFile, bt.DataBase)):
    """
    Support for `Visual Chart <www.visualchart.com>`_ binary on-disk files for
    both daily and intradaily formats.

    Note:

      - ``dataname``: Market code displayed by Visual Chart. Example: 015ES for
        EuroStoxx 50 continuous future
    """

    def start(self):
        if False:
            i = 10
            return i + 15
        super(VChartFile, self).start()
        if self._store is None:
            self._store = bt.stores.VChartFile()
            self._store.start()
        self._store.start(data=self)
        if self.p.timeframe < bt.TimeFrame.Minutes:
            ext = '.tck'
        elif self.p.timeframe < bt.TimeFrame.Days:
            ext = '.min'
            self._dtsize = 2
            self._barsize = 32
            self._barfmt = 'IIffffII'
        else:
            ext = '.fd'
            self._barsize = 28
            self._dtsize = 1
            self._barfmt = 'IffffII'
        basepath = self._store.get_datapath()
        dataname = '01' + '0' + self.p.dataname + ext
        mktcode = '0' + self.p.dataname[0:3]
        path = os.path.join(basepath, mktcode, dataname)
        try:
            self.f = open(path, 'rb')
        except IOError:
            self.f = None

    def stop(self):
        if False:
            while True:
                i = 10
        if self.f is not None:
            self.f.close()
            self.f = None

    def _load(self):
        if False:
            i = 10
            return i + 15
        if self.f is None:
            return False
        try:
            bardata = self.f.read(self._barsize)
        except IOError:
            self.f = None
            return False
        if not bardata or len(bardata) < self._barsize:
            self.f = None
            return False
        try:
            bdata = unpack(self._barfmt, bardata)
        except:
            self.f = None
            return False
        (y, md) = divmod(bdata[0], 500)
        (m, d) = divmod(md, 32)
        dt = datetime(y, m, d)
        if self._dtsize > 1:
            (hhmm, ss) = divmod(bdata[1], 60)
            (hh, mm) = divmod(hhmm, 60)
            dt = dt.replace(hour=hh, minute=mm, second=ss)
        else:
            dt = datetime.combine(dt, self.p.sessionend)
        self.lines.datetime[0] = date2num(dt)
        (o, h, l, c, v, oi) = bdata[self._dtsize:]
        self.lines.open[0] = o
        self.lines.high[0] = h
        self.lines.low[0] = l
        self.lines.close[0] = c
        self.lines.volume[0] = v
        self.lines.openinterest[0] = oi
        return True