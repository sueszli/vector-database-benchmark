from __future__ import absolute_import, division, print_function
import backtrader as bt

class PivotPoint1(bt.Indicator):
    lines = ('p', 's1', 's2', 'r1', 'r2')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        h = self.data.high(-1)
        l = self.data.low(-1)
        c = self.data.close(-1)
        self.lines.p = p = (h + l + c) / 3.0
        p2 = p * 2.0
        self.lines.s1 = p2 - h
        self.lines.r1 = p2 - l
        hilo = h - l
        self.lines.s2 = p - hilo
        self.lines.r2 = p + hilo

class PivotPoint(bt.Indicator):
    lines = ('p', 's1', 's2', 'r1', 'r2')
    plotinfo = dict(subplot=False)

    def __init__(self):
        if False:
            return 10
        h = self.data.high
        l = self.data.low
        c = self.data.close
        self.lines.p = p = (h + l + c) / 3.0
        p2 = p * 2.0
        self.lines.s1 = p2 - h
        self.lines.r1 = p2 - l
        hilo = h - l
        self.lines.s2 = p - hilo
        self.lines.r2 = p + hilo