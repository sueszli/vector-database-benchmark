from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import struct
import os.path
from .. import feed
from .. import TimeFrame
from ..utils import date2num

class VChartData(feed.DataBase):
    """
    Support for `Visual Chart <www.visualchart.com>`_ binary on-disk files for
    both daily and intradaily formats.

    Note:

      - ``dataname``: to file or open file-like object

        If a file-like object is passed, the ``timeframe`` parameter will be
        used to determine which is the actual timeframe.

        Else the file extension (``.fd`` for daily and ``.min`` for intraday)
        will be used.
    """

    def start(self):
        if False:
            while True:
                i = 10
        super(VChartData, self).start()
        self.ext = ''
        if not hasattr(self.p.dataname, 'read'):
            if self.p.dataname.endswith('.fd'):
                self.p.timeframe = TimeFrame.Days
            elif self.p.dataname.endswith('.min'):
                self.p.timeframe = TimeFrame.Minutes
            elif self.p.timeframe == TimeFrame.Days:
                self.ext = '.fd'
            else:
                self.ext = '.min'
        if self.p.timeframe >= TimeFrame.Days:
            self.barsize = 28
            self.dtsize = 1
            self.barfmt = 'IffffII'
        else:
            self.dtsize = 2
            self.barsize = 32
            self.barfmt = 'IIffffII'
        self.f = None
        if hasattr(self.p.dataname, 'read'):
            self.f = self.p.dataname
        else:
            dataname = self.p.dataname + self.ext
            self.f = open(dataname, 'rb')

    def stop(self):
        if False:
            return 10
        if self.f is not None:
            self.f.close()
            self.f = None

    def _load(self):
        if False:
            while True:
                i = 10
        if self.f is None:
            return False
        bardata = self.f.read(self.barsize)
        if not bardata:
            return False
        bdata = struct.unpack(self.barfmt, bardata)
        (y, md) = divmod(bdata[0], 500)
        (m, d) = divmod(md, 32)
        dt = datetime.datetime(y, m, d)
        if self.dtsize > 1:
            (hhmm, ss) = divmod(bdata[1], 60)
            (hh, mm) = divmod(hhmm, 60)
            dt = dt.replace(hour=hh, minute=mm, second=ss)
        self.lines.datetime[0] = date2num(dt)
        (o, h, l, c, v, oi) = bdata[self.dtsize:]
        self.lines.open[0] = o
        self.lines.high[0] = h
        self.lines.low[0] = l
        self.lines.close[0] = c
        self.lines.volume[0] = v
        self.lines.openinterest[0] = oi
        return True

class VChartFeed(feed.FeedBase):
    DataCls = VChartData
    params = (('basepath', ''),) + DataCls.params._gettuple()

    def _getdata(self, dataname, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        maincode = dataname[0:2]
        subcode = dataname[2:6]
        datapath = os.path.join(self.p.basepath, 'RealServer', 'Data', maincode, subcode, dataname)
        newkwargs = self.p._getkwargs()
        newkwargs.update(kwargs)
        kwargs['dataname'] = datapath
        return self.DataCls(**kwargs)