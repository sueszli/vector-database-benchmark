if __name__ == '__main__':
    import os
    import sys
    path = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(path, '..', '..'))
from .. import functions as fn
from ..Qt import QtGui, QtWidgets
from .UIGraphicsItem import UIGraphicsItem
__all__ = ['VTickGroup']

class VTickGroup(UIGraphicsItem):
    """
    **Bases:** :class:`UIGraphicsItem <pyqtgraph.UIGraphicsItem>`
    
    Draws a set of tick marks which always occupy the same vertical range of the view,
    but have x coordinates relative to the data within the view.
    
    """

    def __init__(self, xvals=None, yrange=None, pen=None):
        if False:
            while True:
                i = 10
        '\n        ==============  ===================================================================\n        **Arguments:**\n        xvals           A list of x values (in data coordinates) at which to draw ticks.\n        yrange          A list of [low, high] limits for the tick. 0 is the bottom of\n                        the view, 1 is the top. [0.8, 1] would draw ticks in the top\n                        fifth of the view.\n        pen             The pen to use for drawing ticks. Default is grey. Can be specified\n                        as any argument valid for :func:`mkPen<pyqtgraph.mkPen>`\n        ==============  ===================================================================\n        '
        if yrange is None:
            yrange = [0, 1]
        if xvals is None:
            xvals = []
        UIGraphicsItem.__init__(self)
        if pen is None:
            pen = (200, 200, 200)
        self.path = QtWidgets.QGraphicsPathItem()
        self.ticks = []
        self.xvals = []
        self.yrange = [0, 1]
        self.setPen(pen)
        self.setYRange(yrange)
        self.setXVals(xvals)

    def setPen(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Set the pen to use for drawing ticks. Can be specified as any arguments valid\n        for :func:`mkPen<pyqtgraph.mkPen>`'
        self.pen = fn.mkPen(*args, **kwargs)

    def setXVals(self, vals):
        if False:
            while True:
                i = 10
        'Set the x values for the ticks. \n        \n        ==============   =====================================================================\n        **Arguments:**\n        vals             A list of x values (in data/plot coordinates) at which to draw ticks.\n        ==============   =====================================================================\n        '
        self.xvals = vals
        self.rebuildTicks()

    def setYRange(self, vals):
        if False:
            print('Hello World!')
        'Set the y range [low, high] that the ticks are drawn on. 0 is the bottom of \n        the view, 1 is the top.'
        self.yrange = vals
        self.rebuildTicks()

    def dataBounds(self, *args, **kargs):
        if False:
            i = 10
            return i + 15
        return None

    def yRange(self):
        if False:
            print('Hello World!')
        return self.yrange

    def rebuildTicks(self):
        if False:
            print('Hello World!')
        self.path = QtGui.QPainterPath()
        for x in self.xvals:
            self.path.moveTo(x, 0.0)
            self.path.lineTo(x, 1.0)

    def paint(self, p, *args):
        if False:
            print('Hello World!')
        UIGraphicsItem.paint(self, p, *args)
        br = self.boundingRect()
        h = br.height()
        br.setY(br.y() + self.yrange[0] * h)
        br.setHeight((self.yrange[1] - self.yrange[0]) * h)
        p.translate(0, br.y())
        p.scale(1.0, br.height())
        p.setPen(self.pen)
        p.drawPath(self.path)