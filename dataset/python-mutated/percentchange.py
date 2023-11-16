from __future__ import absolute_import, division, print_function, unicode_literals
from . import Indicator
__all__ = ['PercentChange', 'PctChange']

class PercentChange(Indicator):
    """
      Measures the perccentage change of the current value with respect to that
      of period bars ago
    """
    alias = ('PctChange',)
    lines = ('pctchange',)
    plotlines = dict(pctchange=dict(_name='%change'))
    params = (('period', 30),)

    def __init__(self):
        if False:
            return 10
        self.lines.pctchange = self.data / self.data(-self.p.period) - 1.0
        super(PercentChange, self).__init__()