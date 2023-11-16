from __future__ import absolute_import, division, print_function, unicode_literals
import backtrader as bt

class DataFilter(bt.AbstractDataBase):
    """
    This class filters out bars from a given data source. In addition to the
    standard parameters of a DataBase it takes a ``funcfilter`` parameter which
    can be any callable

    Logic:

      - ``funcfilter`` will be called with the underlying data source

        It can be any callable

        - Return value ``True``: current data source bar values will used
        - Return value ``False``: current data source bar values will discarded
    """
    params = (('funcfilter', None),)

    def preload(self):
        if False:
            return 10
        if len(self.p.dataname) == self.p.dataname.buflen():
            self.p.dataname.start()
            self.p.dataname.preload()
            self.p.dataname.home()
        self.p.timeframe = self._timeframe = self.p.dataname._timeframe
        self.p.compression = self._compression = self.p.dataname._compression
        super(DataFilter, self).preload()

    def _load(self):
        if False:
            i = 10
            return i + 15
        if not len(self.p.dataname):
            self.p.dataname.start()
        while self.p.dataname.next():
            if not self.p.funcfilter(self.p.dataname):
                continue
            for i in range(self.p.dataname.size()):
                self.lines[i][0] = self.p.dataname.lines[i][0]
            return True
        return False