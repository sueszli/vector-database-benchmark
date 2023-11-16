from __future__ import absolute_import, division, print_function, unicode_literals
from . import MovingAverageBase, Average

class MovingAverageSimple(MovingAverageBase):
    """
    Non-weighted average of the last n periods

    Formula:
      - movav = Sum(data, period) / period

    See also:
      - http://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
    """
    alias = ('SMA', 'SimpleMovingAverage')
    lines = ('sma',)

    def __init__(self):
        if False:
            return 10
        self.lines[0] = Average(self.data, period=self.p.period)
        super(MovingAverageSimple, self).__init__()