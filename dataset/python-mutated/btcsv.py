from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import date, datetime, time
from .. import feed
from ..utils import date2num

class BacktraderCSVData(feed.CSVDataBase):
    """
    Parses a self-defined CSV Data used for testing.

    Specific parameters:

      - ``dataname``: The filename to parse or a file-like object
    """

    def _loadline(self, linetokens):
        if False:
            for i in range(10):
                print('nop')
        itoken = iter(linetokens)
        dttxt = next(itoken)
        dt = date(int(dttxt[0:4]), int(dttxt[5:7]), int(dttxt[8:10]))
        if len(linetokens) == 8:
            tmtxt = next(itoken)
            tm = time(int(tmtxt[0:2]), int(tmtxt[3:5]), int(tmtxt[6:8]))
        else:
            tm = self.p.sessionend
        self.lines.datetime[0] = date2num(datetime.combine(dt, tm))
        self.lines.open[0] = float(next(itoken))
        self.lines.high[0] = float(next(itoken))
        self.lines.low[0] = float(next(itoken))
        self.lines.close[0] = float(next(itoken))
        self.lines.volume[0] = float(next(itoken))
        self.lines.openinterest[0] = float(next(itoken))
        return True

class BacktraderCSV(feed.CSVFeedBase):
    DataCls = BacktraderCSVData