import pyqtgraph as pg
from PyQt5 import QtCore

class CustomViewBox(pg.ViewBox):

    def __init__(self, *args, **kwds):
        if False:
            return 10
        kwds['enableMenu'] = False
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)

    def mouseClickEvent(self, ev):
        if False:
            while True:
                i = 10
        if ev.button() == QtCore.Qt.RightButton:
            self.autoRange()