"""
MultiPlotWidget.py -  Convenience class--GraphicsView widget displaying a MultiPlotItem
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.
"""
from ..graphicsItems import MultiPlotItem as MultiPlotItem
from ..Qt import QtCore
from .GraphicsView import GraphicsView
__all__ = ['MultiPlotWidget']

class MultiPlotWidget(GraphicsView):
    """Widget implementing a :class:`~pyqtgraph.GraphicsView` with a single
    :class:`~pyqtgraph.MultiPlotItem` inside."""

    def __init__(self, parent=None):
        if False:
            return 10
        self.minPlotHeight = 50
        self.mPlotItem = MultiPlotItem.MultiPlotItem()
        GraphicsView.__init__(self, parent)
        self.enableMouse(False)
        self.setCentralItem(self.mPlotItem)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self.mPlotItem, attr):
            m = getattr(self.mPlotItem, attr)
            if hasattr(m, '__call__'):
                return m
        raise AttributeError(attr)

    def setMinimumPlotHeight(self, min):
        if False:
            while True:
                i = 10
        'Set the minimum height for each sub-plot displayed. \n        \n        If the total height of all plots is greater than the height of the \n        widget, then a scroll bar will appear to provide access to the entire\n        set of plots.\n        \n        Added in version 0.9.9\n        '
        self.minPlotHeight = min
        self.resizeEvent(None)

    def widgetGroupInterface(self):
        if False:
            return 10
        return (None, MultiPlotWidget.saveState, MultiPlotWidget.restoreState)

    def saveState(self):
        if False:
            i = 10
            return i + 15
        return {}

    def restoreState(self, state):
        if False:
            while True:
                i = 10
        pass

    def close(self):
        if False:
            i = 10
            return i + 15
        self.mPlotItem.close()
        self.mPlotItem = None
        self.setParent(None)
        GraphicsView.close(self)

    def setRange(self, *args, **kwds):
        if False:
            print('Hello World!')
        GraphicsView.setRange(self, *args, **kwds)
        if self.centralWidget is not None:
            r = self.range
            minHeight = len(self.mPlotItem.plots) * self.minPlotHeight
            if r.height() < minHeight:
                r.setHeight(minHeight)
                r.setWidth(r.width() - self.verticalScrollBar().width())
            self.centralWidget.setGeometry(r)

    def resizeEvent(self, ev):
        if False:
            while True:
                i = 10
        if self.closed:
            return
        if self.autoPixelRange:
            self.range = QtCore.QRectF(0, 0, self.size().width(), self.size().height())
        MultiPlotWidget.setRange(self, self.range, padding=0, disableAutoPixel=False)
        self.updateMatrix()