from .. import functions as fn
from ..Qt import QtCore, QtGui
from .UIGraphicsItem import UIGraphicsItem
__all__ = ['GradientLegend']

class GradientLegend(UIGraphicsItem):
    """
    Draws a color gradient rectangle along with text labels denoting the value at specific
    points along the gradient.
    """

    def __init__(self, size, offset):
        if False:
            for i in range(10):
                print('nop')
        self.size = size
        self.offset = offset
        UIGraphicsItem.__init__(self)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 100))
        self.pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        self.textPen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        self.labels = {'max': 1, 'min': 0}
        self.gradient = QtGui.QLinearGradient()
        self.gradient.setColorAt(0, QtGui.QColor(0, 0, 0))
        self.gradient.setColorAt(1, QtGui.QColor(255, 0, 0))
        self.setZValue(100)

    def setGradient(self, g):
        if False:
            print('Hello World!')
        self.gradient = g
        self.update()

    def setColorMap(self, colormap):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set displayed gradient from a :class:`~pyqtgraph.ColorMap` object.\n        '
        self.gradient = colormap.getGradient()

    def setIntColorScale(self, minVal, maxVal, *args, **kargs):
        if False:
            print('Hello World!')
        colors = [fn.intColor(i, maxVal - minVal, *args, **kargs) for i in range(minVal, maxVal)]
        g = QtGui.QLinearGradient()
        for i in range(len(colors)):
            x = float(i) / len(colors)
            g.setColorAt(x, colors[i])
        self.setGradient(g)
        if 'labels' not in kargs:
            self.setLabels({str(minVal): 0, str(maxVal): 1})
        else:
            self.setLabels({kargs['labels'][0]: 0, kargs['labels'][1]: 1})

    def setLabels(self, l):
        if False:
            for i in range(10):
                print('nop')
        'Defines labels to appear next to the color scale. Accepts a dict of {text: value} pairs'
        self.labels = l
        self.update()

    def paint(self, p, opt, widget):
        if False:
            for i in range(10):
                print('nop')
        UIGraphicsItem.paint(self, p, opt, widget)
        view = self.getViewBox()
        if view is None:
            return
        p.save()
        trans = view.sceneTransform()
        p.setTransform(trans)
        rect = view.rect()
        labelWidth = 0
        labelHeight = 0
        for k in self.labels:
            b = p.boundingRect(QtCore.QRectF(0, 0, 0, 0), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, str(k))
            labelWidth = max(labelWidth, b.width())
            labelHeight = max(labelHeight, b.height())
        textPadding = 2
        xR = rect.right()
        xL = rect.left()
        yT = rect.top()
        yB = rect.bottom()
        if self.offset[0] < 0:
            x3 = xR + self.offset[0]
            x2 = x3 - labelWidth - 2 * textPadding
            x1 = x2 - self.size[0]
        else:
            x1 = xL + self.offset[0]
            x2 = x1 + self.size[0]
            x3 = x2 + labelWidth + 2 * textPadding
        if self.offset[1] < 0:
            y2 = yB + self.offset[1]
            y1 = y2 - self.size[1]
        else:
            y1 = yT + self.offset[1]
            y2 = y1 + self.size[1]
        self.b = [x1, x2, x3, y1, y2, labelWidth]
        p.setPen(self.pen)
        p.setBrush(self.brush)
        rect = QtCore.QRectF(QtCore.QPointF(x1 - textPadding, y1 - labelHeight / 2 - textPadding), QtCore.QPointF(x3 + textPadding, y2 + labelHeight / 2 + textPadding))
        p.drawRect(rect)
        self.gradient.setStart(0, y2)
        self.gradient.setFinalStop(0, y1)
        p.setBrush(self.gradient)
        rect = QtCore.QRectF(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))
        p.drawRect(rect)
        p.setPen(self.textPen)
        tx = x2 + 2 * textPadding
        lh = labelHeight
        lw = labelWidth
        for k in self.labels:
            y = y2 - self.labels[k] * (y2 - y1)
            p.drawText(QtCore.QRectF(tx, y - lh / 2, lw, lh), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, str(k))
        p.restore()