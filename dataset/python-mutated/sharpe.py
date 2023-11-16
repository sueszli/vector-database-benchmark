from __future__ import absolute_import, division, print_function, unicode_literals
import math
from backtrader.utils.py3 import itervalues
from backtrader import Analyzer, TimeFrame
from backtrader.mathsupport import average, standarddev
from backtrader.analyzers import TimeReturn, AnnualReturn

class SharpeRatio(Analyzer):
    """This analyzer calculates the SharpeRatio of a strategy using a risk free
    asset which is simply an interest rate

    See also:

      - https://en.wikipedia.org/wiki/Sharpe_ratio

    Params:

      - ``timeframe``: (default: ``TimeFrame.Years``)

      - ``compression`` (default: ``1``)

        Only used for sub-day timeframes to for example work on an hourly
        timeframe by specifying "TimeFrame.Minutes" and 60 as compression

      - ``riskfreerate`` (default: 0.01 -> 1%)

        Expressed in annual terms (see ``convertrate`` below)

      - ``convertrate`` (default: ``True``)

        Convert the ``riskfreerate`` from annual to monthly, weekly or daily
        rate. Sub-day conversions are not supported

      - ``factor`` (default: ``None``)

        If ``None``, the conversion factor for the riskfree rate from *annual*
        to the chosen timeframe will be chosen from a predefined table

          Days: 252, Weeks: 52, Months: 12, Years: 1

        Else the specified value will be used

      - ``annualize`` (default: ``False``)

        If ``convertrate`` is ``True``, the *SharpeRatio* will be delivered in
        the ``timeframe`` of choice.

        In most occasions the SharpeRatio is delivered in annualized form.
        Convert the ``riskfreerate`` from annual to monthly, weekly or daily
        rate. Sub-day conversions are not supported

      - ``stddev_sample`` (default: ``False``)

        If this is set to ``True`` the *standard deviation* will be calculated
        decreasing the denominator in the mean by ``1``. This is used when
        calculating the *standard deviation* if it's considered that not all
        samples are used for the calculation. This is known as the *Bessels'
        correction*

      - ``daysfactor`` (default: ``None``)

        Old naming for ``factor``. If set to anything else than ``None`` and
        the ``timeframe`` is ``TimeFrame.Days`` it will be assumed this is old
        code and the value will be used

      - ``legacyannual`` (default: ``False``)

        Use the ``AnnualReturn`` return analyzer, which as the name implies
        only works on years

      - ``fund`` (default: ``None``)

        If ``None`` the actual mode of the broker (fundmode - True/False) will
        be autodetected to decide if the returns are based on the total net
        asset value or on the fund value. See ``set_fundmode`` in the broker
        documentation

        Set it to ``True`` or ``False`` for a specific behavior

    Methods:

      - get_analysis

        Returns a dictionary with key "sharperatio" holding the ratio

    """
    params = (('timeframe', TimeFrame.Years), ('compression', 1), ('riskfreerate', 0.01), ('factor', None), ('convertrate', True), ('annualize', False), ('stddev_sample', False), ('daysfactor', None), ('legacyannual', False), ('fund', None))
    RATEFACTORS = {TimeFrame.Days: 252, TimeFrame.Weeks: 52, TimeFrame.Months: 12, TimeFrame.Years: 1}

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.p.legacyannual:
            self.anret = AnnualReturn()
        else:
            self.timereturn = TimeReturn(timeframe=self.p.timeframe, compression=self.p.compression, fund=self.p.fund)

    def stop(self):
        if False:
            print('Hello World!')
        super(SharpeRatio, self).stop()
        if self.p.legacyannual:
            rate = self.p.riskfreerate
            retavg = average([r - rate for r in self.anret.rets])
            retdev = standarddev(self.anret.rets)
            self.ratio = retavg / retdev
        else:
            returns = list(itervalues(self.timereturn.get_analysis()))
            rate = self.p.riskfreerate
            factor = None
            if self.p.timeframe == TimeFrame.Days and self.p.daysfactor is not None:
                factor = self.p.daysfactor
            elif self.p.factor is not None:
                factor = self.p.factor
            elif self.p.timeframe in self.RATEFACTORS:
                factor = self.RATEFACTORS[self.p.timeframe]
            if factor is not None:
                if self.p.convertrate:
                    rate = pow(1.0 + rate, 1.0 / factor) - 1.0
                else:
                    returns = [pow(1.0 + x, factor) - 1.0 for x in returns]
            lrets = len(returns) - self.p.stddev_sample
            if lrets:
                ret_free = [r - rate for r in returns]
                ret_free_avg = average(ret_free)
                retdev = standarddev(ret_free, avgx=ret_free_avg, bessel=self.p.stddev_sample)
                try:
                    ratio = ret_free_avg / retdev
                    if factor is not None and self.p.convertrate and self.p.annualize:
                        ratio = math.sqrt(factor) * ratio
                except (ValueError, TypeError, ZeroDivisionError):
                    ratio = None
            else:
                ratio = None
            self.ratio = ratio
        self.rets['sharperatio'] = self.ratio

class SharpeRatio_A(SharpeRatio):
    """Extension of the SharpeRatio which returns the Sharpe Ratio directly in
    annualized form

    The following param has been changed from ``SharpeRatio``

      - ``annualize`` (default: ``True``)

    """
    params = (('annualize', True),)