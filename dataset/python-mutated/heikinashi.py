from __future__ import absolute_import, division, print_function, unicode_literals
__all__ = ['HeikinAshi']

class HeikinAshi(object):
    """
    The filter remodels the open, high, low, close to make HeikinAshi
    candlesticks

    See:
      - https://en.wikipedia.org/wiki/Candlestick_chart#Heikin_Ashi_candlesticks
      - http://stockcharts.com/school/doku.php?id=chart_school:chart_analysis:heikin_ashi

    """

    def __init__(self, data):
        if False:
            for i in range(10):
                print('nop')
        pass

    def __call__(self, data):
        if False:
            print('Hello World!')
        (o, h, l, c) = (data.open[0], data.high[0], data.low[0], data.close[0])
        data.close[0] = ha_close0 = (o + h + l + c) / 4.0
        if len(data) > 1:
            data.open[0] = ha_open0 = (data.open[-1] + data.close[-1]) / 2.0
            data.high[0] = max(ha_open0, ha_close0, h)
            data.low[0] = min(ha_open0, ha_close0, l)
        else:
            data.open[0] = ha_open0 = (o + c) / 2.0
        return False