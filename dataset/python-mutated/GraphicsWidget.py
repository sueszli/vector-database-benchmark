from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsItem import GraphicsItem
__all__ = ['GraphicsWidget']

class GraphicsWidget(GraphicsItem, QtWidgets.QGraphicsWidget):
    _qtBaseClass = QtWidgets.QGraphicsWidget

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        '\n        **Bases:** :class:`GraphicsItem <pyqtgraph.GraphicsItem>`, :class:`QtWidgets.QGraphicsWidget`\n        \n        Extends QGraphicsWidget with several helpful methods and workarounds for PyQt bugs. \n        Most of the extra functionality is inherited from :class:`GraphicsItem <pyqtgraph.GraphicsItem>`.\n        '
        QtWidgets.QGraphicsWidget.__init__(self, *args, **kwargs)
        GraphicsItem.__init__(self)
        self._boundingRectCache = self._previousGeometry = None
        self._painterPathCache = None
        self.geometryChanged.connect(self._resetCachedProperties)

    def _resetCachedProperties(self):
        if False:
            i = 10
            return i + 15
        self._boundingRectCache = self._previousGeometry = None
        self._painterPathCache = None

    def setFixedHeight(self, h):
        if False:
            for i in range(10):
                print('nop')
        self.setMaximumHeight(h)
        self.setMinimumHeight(h)

    def setFixedWidth(self, h):
        if False:
            for i in range(10):
                print('nop')
        self.setMaximumWidth(h)
        self.setMinimumWidth(h)

    def height(self):
        if False:
            return 10
        return self.geometry().height()

    def width(self):
        if False:
            for i in range(10):
                print('nop')
        return self.geometry().width()

    def boundingRect(self):
        if False:
            print('Hello World!')
        geometry = self.geometry()
        if geometry != self._previousGeometry:
            self._painterPathCache = None
            br = self.mapRectFromParent(geometry).normalized()
            self._boundingRectCache = br
            self._previousGeometry = geometry
        else:
            br = self._boundingRectCache
        return QtCore.QRectF(br)

    def shape(self):
        if False:
            return 10
        p = self._painterPathCache
        if p is None:
            self._painterPathCache = p = QtGui.QPainterPath()
            p.addRect(self.boundingRect())
        return p