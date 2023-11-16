from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import backtrader as bt

class DaySplitter_Close(bt.with_metaclass(bt.MetaParams, object)):
    """
    Splits a daily bar in two parts simulating 2 ticks which will be used to
    replay the data:

      - First tick: ``OHLX``

        The ``Close`` will be replaced by the *average* of ``Open``, ``High``
        and ``Low``

        The session opening time is used for this tick

      and

      - Second tick: ``CCCC``

        The ``Close`` price will be used for the four components of the price

        The session closing time is used for this tick

    The volume will be split amongst the 2 ticks using the parameters:

      - ``closevol`` (default: ``0.5``) The value indicate which percentage, in
        absolute terms from 0.0 to 1.0, has to be assigned to the *closing*
        tick. The rest will be assigned to the ``OHLX`` tick.

    **This filter is meant to be used together with** ``cerebro.replaydata``

    """
    params = (('closevol', 0.5),)

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self.lastdt = None

    def __call__(self, data):
        if False:
            return 10
        datadt = data.datetime.date()
        if self.lastdt == datadt:
            return False
        self.lastdt = datadt
        ohlbar = [data.lines[i][0] for i in range(data.size())]
        closebar = ohlbar[:]
        ohlprice = ohlbar[data.Open] + ohlbar[data.High] + ohlbar[data.Low]
        ohlbar[data.Close] = ohlprice / 3.0
        vol = ohlbar[data.Volume]
        ohlbar[data.Volume] = vohl = int(vol * (1.0 - self.p.closevol))
        oi = ohlbar[data.OpenInterest]
        ohlbar[data.OpenInterest] = 0
        dt = datetime.datetime.combine(datadt, data.p.sessionstart)
        ohlbar[data.DateTime] = data.date2num(dt)
        closebar[data.Open] = cprice = closebar[data.Close]
        closebar[data.High] = cprice
        closebar[data.Low] = cprice
        closebar[data.Volume] = vol - vohl
        ohlbar[data.OpenInterest] = oi
        dt = datetime.datetime.combine(datadt, data.p.sessionend)
        closebar[data.DateTime] = data.date2num(dt)
        data.backwards(force=True)
        data._add2stack(ohlbar)
        data._add2stack(closebar, stash=True)
        return False