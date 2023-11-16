from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from backtrader.utils.py3 import range
from backtrader import Analyzer

class AnnualReturn(Analyzer):
    """
    This analyzer calculates the AnnualReturns by looking at the beginning
    and end of the year

    Params:

      - (None)

    Member Attributes:

      - ``rets``: list of calculated annual returns

      - ``ret``: dictionary (key: year) of annual returns

    **get_analysis**:

      - Returns a dictionary of annual returns (key: year)
    """

    def stop(self):
        if False:
            return 10
        cur_year = -1
        value_start = 0.0
        value_cur = 0.0
        value_end = 0.0
        self.rets = list()
        self.ret = OrderedDict()
        for i in range(len(self.data) - 1, -1, -1):
            dt = self.data.datetime.date(-i)
            value_cur = self.strategy.stats.broker.value[-i]
            if dt.year > cur_year:
                if cur_year >= 0:
                    annualret = value_end / value_start - 1.0
                    self.rets.append(annualret)
                    self.ret[cur_year] = annualret
                    value_start = value_end
                else:
                    value_start = value_cur
                cur_year = dt.year
            value_end = value_cur
        if cur_year not in self.ret:
            annualret = value_end / value_start - 1.0
            self.rets.append(annualret)
            self.ret[cur_year] = annualret

    def get_analysis(self):
        if False:
            print('Hello World!')
        return self.ret