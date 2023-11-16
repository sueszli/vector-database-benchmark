from __future__ import absolute_import, division, print_function, unicode_literals
from backtrader.utils.py3 import MAXINT, with_metaclass
from backtrader.metabase import MetaParams

class FixedSize(with_metaclass(MetaParams, object)):
    """Returns the execution size for a given order using a *percentage* of the
    volume in a bar.

    This percentage is set with the parameter ``perc``

    Params:

      - ``size`` (default: ``None``)  maximum size to be executed. The actual
        volume of the bar at execution time is also a limit if smaller than the
        size

        If the value of this parameter evaluates to False, the entire volume
        of the bar will be used to match the order
    """
    params = (('size', None),)

    def __call__(self, order, price, ago):
        if False:
            for i in range(10):
                print('nop')
        size = self.p.size or MAXINT
        return min((order.data.volume[ago], abs(order.executed.remsize), size))

class FixedBarPerc(with_metaclass(MetaParams, object)):
    """Returns the execution size for a given order using a *percentage* of the
    volume in a bar.

    This percentage is set with the parameter ``perc``

    Params:

      - ``perc`` (default: ``100.0``) (valied values: ``0.0 - 100.0``)

        Percentage of the volume bar to use to execute an order
    """
    params = (('perc', 100.0),)

    def __call__(self, order, price, ago):
        if False:
            return 10
        maxsize = order.data.volume[ago] * self.p.perc // 100
        return min(maxsize, abs(order.executed.remsize))

class BarPointPerc(with_metaclass(MetaParams, object)):
    """Returns the execution size for a given order. The volume will be
    distributed uniformly in the range *high*-*low* using ``minmov`` to
    partition.

    From the allocated volume for the given price, the ``perc`` percentage will
    be used

    Params:

      - ``minmov`` (default: ``0.01``)

        Minimum price movement. Used to partition the range *high*-*low* to
        proportionally distribute the volume amongst possible prices

      - ``perc`` (default: ``100.0``) (valied values: ``0.0 - 100.0``)

        Percentage of the volume allocated to the order execution price to use
        for matching

    """
    params = (('minmov', None), ('perc', 100.0))

    def __call__(self, order, price, ago):
        if False:
            print('Hello World!')
        data = order.data
        minmov = self.p.minmov
        parts = 1
        if minmov:
            parts = (data.high[ago] - data.low[ago] + minmov) // minmov
        alloc_vol = data.volume[ago] / parts * self.p.perc // 100.0
        return min(alloc_vol, abs(order.executed.remsize))