from __future__ import absolute_import, division, print_function, unicode_literals
import math
import backtrader as bt

class MTradeObserver(bt.observer.Observer):
    lines = ('Id_0', 'Id_1', 'Id_2')
    plotinfo = dict(plot=True, subplot=True, plotlinelabels=True)
    plotlines = dict(Id_0=dict(marker='*', markersize=8.0, color='lime', fillstyle='full'), Id_1=dict(marker='o', markersize=8.0, color='red', fillstyle='full'), Id_2=dict(marker='s', markersize=8.0, color='blue', fillstyle='full'))

    def next(self):
        if False:
            return 10
        for trade in self._owner._tradespending:
            if trade.data is not self.data:
                continue
            if not trade.isclosed:
                continue
            self.lines[trade.tradeid][0] = trade.pnlcomm