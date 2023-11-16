from __future__ import absolute_import, division, print_function, unicode_literals
from backtrader import TimeFrameAnalyzerBase

class TimeReturn(TimeFrameAnalyzerBase):
    """This analyzer calculates the Returns by looking at the beginning
    and end of the timeframe

    Params:

      - ``timeframe`` (default: ``None``)
        If ``None`` the ``timeframe`` of the 1st data in the system will be
        used

        Pass ``TimeFrame.NoTimeFrame`` to consider the entire dataset with no
        time constraints

      - ``compression`` (default: ``None``)

        Only used for sub-day timeframes to for example work on an hourly
        timeframe by specifying "TimeFrame.Minutes" and 60 as compression

        If ``None`` then the compression of the 1st data of the system will be
        used

      - ``data`` (default: ``None``)

        Reference asset to track instead of the portfolio value.

        .. note:: this data must have been added to a ``cerebro`` instance with
                  ``addata``, ``resampledata`` or ``replaydata``

      - ``firstopen`` (default: ``True``)

        When tracking the returns of a ``data`` the following is done when
        crossing a timeframe boundary, for example ``Years``:

          - Last ``close`` of previous year is used as the reference price to
            see the return in the current year

        The problem is the 1st calculation, because the data has** no
        previous** closing price. As such and when this parameter is ``True``
        the *opening* price will be used for the 1st calculation.

        This requires the data feed to have an ``open`` price (for ``close``
        the standard [0] notation will be used without reference to a field
        price)

        Else the initial close will be used.

      - ``fund`` (default: ``None``)

        If ``None`` the actual mode of the broker (fundmode - True/False) will
        be autodetected to decide if the returns are based on the total net
        asset value or on the fund value. See ``set_fundmode`` in the broker
        documentation

        Set it to ``True`` or ``False`` for a specific behavior

    Methods:

      - get_analysis

        Returns a dictionary with returns as values and the datetime points for
        each return as keys
    """
    params = (('data', None), ('firstopen', True), ('fund', None))

    def start(self):
        if False:
            return 10
        super(TimeReturn, self).start()
        if self.p.fund is None:
            self._fundmode = self.strategy.broker.fundmode
        else:
            self._fundmode = self.p.fund
        self._value_start = 0.0
        self._lastvalue = None
        if self.p.data is None:
            if not self._fundmode:
                self._lastvalue = self.strategy.broker.getvalue()
            else:
                self._lastvalue = self.strategy.broker.fundvalue

    def notify_fund(self, cash, value, fundvalue, shares):
        if False:
            print('Hello World!')
        if not self._fundmode:
            if self.p.data is None:
                self._value = value
            else:
                self._value = self.p.data[0]
        elif self.p.data is None:
            self._value = fundvalue
        else:
            self._value = self.p.data[0]

    def on_dt_over(self):
        if False:
            return 10
        if self.p.data is None or self._lastvalue is not None:
            self._value_start = self._lastvalue
        elif self.p.firstopen:
            self._value_start = self.p.data.open[0]
        else:
            self._value_start = self.p.data[0]

    def next(self):
        if False:
            while True:
                i = 10
        super(TimeReturn, self).next()
        self.rets[self.dtkey] = self._value / self._value_start - 1.0
        self._lastvalue = self._value