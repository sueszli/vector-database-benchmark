from __future__ import absolute_import, division, print_function, unicode_literals
from backtrader import date2num
import backtrader.feed as feed

class BlazeData(feed.DataBase):
    """
    Support for `Blaze <blaze.pydata.org>`_ ``Data`` objects.

    Only numeric indices to columns are supported.

    Note:

      - The ``dataname`` parameter is a blaze ``Data`` object

      - A negative value in any of the parameters for the Data lines
        indicates it's not present in the DataFrame
        it is
    """
    params = (('datetime', 0), ('open', 1), ('high', 2), ('low', 3), ('close', 4), ('volume', 5), ('openinterest', 6))
    datafields = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest']

    def start(self):
        if False:
            return 10
        super(BlazeData, self).start()
        self._rows = iter(self.p.dataname)

    def _load(self):
        if False:
            while True:
                i = 10
        try:
            row = next(self._rows)
        except StopIteration:
            return False
        for datafield in self.datafields[1:]:
            colidx = getattr(self.params, datafield)
            if colidx < 0:
                continue
            line = getattr(self.lines, datafield)
            line[0] = row[colidx]
        colidx = getattr(self.params, self.datafields[0])
        dt = row[colidx]
        dtnum = date2num(dt)
        line = getattr(self.lines, self.datafields[0])
        line[0] = dtnum
        return True