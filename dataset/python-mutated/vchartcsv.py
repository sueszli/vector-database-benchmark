from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
from .. import feed
from .. import TimeFrame
from ..utils import date2num

class VChartCSVData(feed.CSVDataBase):
    """
    Parses a `VisualChart <http://www.visualchart.com>`_ CSV exported file.

    Specific parameters (or specific meaning):

      - ``dataname``: The filename to parse or a file-like object
    """
    vctframes = dict(I=TimeFrame.Minutes, D=TimeFrame.Days, W=TimeFrame.Weeks, M=TimeFrame.Months)

    def _loadline(self, linetokens):
        if False:
            print('Hello World!')
        itokens = iter(linetokens)
        ticker = next(itokens)
        if not self._name:
            self._name = ticker
        timeframe = next(itokens)
        self._timeframe = self.vctframes[timeframe]
        dttxt = next(itokens)
        (y, m, d) = (int(dttxt[0:4]), int(dttxt[4:6]), int(dttxt[6:8]))
        tmtxt = next(itokens)
        if timeframe == 'I':
            (hh, mmss) = divmod(int(tmtxt), 10000)
            (mm, ss) = divmod(mmss, 100)
        else:
            hh = self.p.sessionend.hour
            mm = self.p.sessionend.minute
            ss = self.p.sessionend.second
        dtnum = date2num(datetime.datetime(y, m, d, hh, mm, ss))
        self.lines.datetime[0] = dtnum
        self.lines.open[0] = float(next(itokens))
        self.lines.high[0] = float(next(itokens))
        self.lines.low[0] = float(next(itokens))
        self.lines.close[0] = float(next(itokens))
        self.lines.volume[0] = float(next(itokens))
        self.lines.openinterest[0] = float(next(itokens))
        return True

class VChartCSV(feed.CSVFeedBase):
    DataCls = VChartCSVData