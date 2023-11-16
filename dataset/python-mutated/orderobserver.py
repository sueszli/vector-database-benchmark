from __future__ import absolute_import, division, print_function, unicode_literals
import math
import backtrader as bt

class OrderObserver(bt.observer.Observer):
    lines = ('created', 'expired')
    plotinfo = dict(plot=True, subplot=True, plotlinelabels=True)
    plotlines = dict(created=dict(marker='*', markersize=8.0, color='lime', fillstyle='full'), expired=dict(marker='s', markersize=8.0, color='red', fillstyle='full'))

    def next(self):
        if False:
            print('Hello World!')
        for order in self._owner._orderspending:
            if order.data is not self.data:
                continue
            if not order.isbuy():
                continue
            if order.status in [bt.Order.Accepted, bt.Order.Submitted]:
                self.lines.created[0] = order.created.price
            elif order.status in [bt.Order.Expired]:
                self.lines.expired[0] = order.created.price