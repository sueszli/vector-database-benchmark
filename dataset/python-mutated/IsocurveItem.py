from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
__all__ = ['IsocurveItem']

class IsocurveItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    
    Item displaying an isocurve of a 2D array. To align this item correctly with an 
    ImageItem, call ``isocurve.setParentItem(image)``.
    """

    def __init__(self, data=None, level=0, pen='w', axisOrder=None):
        if False:
            i = 10
            return i + 15
        "\n        Create a new isocurve item. \n        \n        ==============  ===============================================================\n        **Arguments:**\n        data            A 2-dimensional ndarray. Can be initialized as None, and set\n                        later using :func:`setData <pyqtgraph.IsocurveItem.setData>`\n        level           The cutoff value at which to draw the isocurve.\n        pen             The color of the curve item. Can be anything valid for\n                        :func:`mkPen <pyqtgraph.mkPen>`\n        axisOrder       May be either 'row-major' or 'col-major'. By default this uses\n                        the ``imageAxisOrder``\n                        :ref:`global configuration option <apiref_config>`.\n        ==============  ===============================================================\n        "
        GraphicsObject.__init__(self)
        self.level = level
        self.data = None
        self.path = None
        self.axisOrder = getConfigOption('imageAxisOrder') if axisOrder is None else axisOrder
        self.setPen(pen)
        self.setData(data, level)

    def setData(self, data, level=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the data/image to draw isocurves for.\n        \n        ==============  ========================================================================\n        **Arguments:**\n        data            A 2-dimensional ndarray.\n        level           The cutoff value at which to draw the curve. If level is not specified,\n                        the previously set level is used.\n        ==============  ========================================================================\n        '
        if level is None:
            level = self.level
        self.level = level
        self.data = data
        self.path = None
        self.prepareGeometryChange()
        self.update()

    def setLevel(self, level):
        if False:
            for i in range(10):
                print('nop')
        'Set the level at which the isocurve is drawn.'
        self.level = level
        self.path = None
        self.prepareGeometryChange()
        self.update()

    def setPen(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Set the pen used to draw the isocurve. Arguments can be any that are valid \n        for :func:`mkPen <pyqtgraph.mkPen>`'
        self.pen = fn.mkPen(*args, **kwargs)
        self.update()

    def setBrush(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Set the brush used to draw the isocurve. Arguments can be any that are valid \n        for :func:`mkBrush <pyqtgraph.mkBrush>`'
        self.brush = fn.mkBrush(*args, **kwargs)
        self.update()

    def updateLines(self, data, level):
        if False:
            print('Hello World!')
        self.setData(data, level)

    def boundingRect(self):
        if False:
            while True:
                i = 10
        if self.data is None:
            return QtCore.QRectF()
        if self.path is None:
            self.generatePath()
        return self.path.boundingRect()

    def generatePath(self):
        if False:
            for i in range(10):
                print('nop')
        if self.data is None:
            self.path = None
            return
        if self.axisOrder == 'row-major':
            data = self.data.T
        else:
            data = self.data
        lines = fn.isocurve(data, self.level, connected=True, extendToEdge=True)
        self.path = QtGui.QPainterPath()
        for line in lines:
            self.path.moveTo(*line[0])
            for p in line[1:]:
                self.path.lineTo(*p)

    def paint(self, p, *args):
        if False:
            i = 10
            return i + 15
        if self.data is None:
            return
        if self.path is None:
            self.generatePath()
        p.setPen(self.pen)
        p.drawPath(self.path)