from __future__ import absolute_import, division, print_function, unicode_literals
import backtrader as bt
from backtrader.utils.py3 import MAXINT
from . import MovingAverageBase, MovAv

class ZeroLagIndicator(MovingAverageBase):
    """By John Ehlers and Ric Way

    The zero-lag indicator (ZLIndicator) is a variation of the EMA
    which modifies the EMA by trying to minimize the error (distance price -
    error correction) and thus reduce the lag

    Formula:
      - EMA(data, period)

      - For each iteration calculate a best-error-correction of the ema (see
        the paper and/or the code) iterating over ``-bestgain`` ->
        ``+bestgain`` for the error correction factor (both incl.)

      - The default moving average is EMA, but can be changed with the
        parameter ``_movav``

        .. note:: the passed moving average must calculate alpha (and 1 -
                  alpha) and make them available as attributes ``alpha`` and
                  ``alpha1`` in the instance

    See also:
      - http://www.mesasoftware.com/papers/ZeroLag.pdf

    """
    alias = ('ZLIndicator', 'ZLInd', 'EC', 'ErrorCorrecting')
    lines = ('ec',)
    params = (('gainlimit', 50), ('_movav', MovAv.EMA))

    def _plotlabel(self):
        if False:
            print('Hello World!')
        plabels = [self.p.period, self.p.gainlimit]
        plabels += [self.p._movav] * self.p.notdefault('_movav')
        return plabels

    def __init__(self):
        if False:
            return 10
        self.ema = MovAv.EMA(period=self.p.period)
        self.limits = [-self.p.gainlimit, self.p.gainlimit + 1]
        super(ZeroLagIndicator, self).__init__()

    def next(self):
        if False:
            print('Hello World!')
        leasterror = MAXINT
        bestec = ema = self.ema[0]
        price = self.data[0]
        ec1 = self.lines.ec[-1]
        (alpha, alpha1) = (self.ema.alpha, self.ema.alpha1)
        for value1 in range(*self.limits):
            gain = value1 / 10
            ec = alpha * (ema + gain * (price - ec1)) + alpha1 * ec1
            error = abs(price - ec)
            if error < leasterror:
                leasterror = error
                bestec = ec
        self.lines.ec[0] = bestec