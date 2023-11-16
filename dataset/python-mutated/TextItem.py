from math import atan2, degrees
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsObject import GraphicsObject
__all__ = ['TextItem']

class TextItem(GraphicsObject):
    """
    GraphicsItem displaying unscaled text (the text will always appear normal even inside a scaled ViewBox). 
    """

    def __init__(self, text='', color=(200, 200, 200), html=None, anchor=(0, 0), border=None, fill=None, angle=0, rotateAxis=None, ensureInBounds=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        ================  =================================================================================\n        **Arguments:**\n        *text*            The text to display\n        *color*           The color of the text (any format accepted by pg.mkColor)\n        *html*            If specified, this overrides both *text* and *color*\n        *anchor*          A QPointF or (x,y) sequence indicating what region of the text box will\n                          be anchored to the item\'s position. A value of (0,0) sets the upper-left corner\n                          of the text box to be at the position specified by setPos(), while a value of (1,1)\n                          sets the lower-right corner.\n        *border*          A pen to use when drawing the border\n        *fill*            A brush to use when filling within the border\n        *angle*           Angle in degrees to rotate text. Default is 0; text will be displayed upright.\n        *rotateAxis*      If None, then a text angle of 0 always points along the +x axis of the scene.\n                          If a QPointF or (x,y) sequence is given, then it represents a vector direction\n                          in the parent\'s coordinate system that the 0-degree line will be aligned to. This\n                          Allows text to follow both the position and orientation of its parent while still\n                          discarding any scale and shear factors.\n        *ensureInBounds*  Ensures that the entire TextItem will be visible when using autorange, but may\n                          produce runaway scaling in certain circumstances (See issue #2642). Setting to \n                          "True" retains legacy behavior.\n        ================  =================================================================================\n\n\n        The effects of the `rotateAxis` and `angle` arguments are added independently. So for example:\n\n          * rotateAxis=None, angle=0 -> normal horizontal text\n          * rotateAxis=None, angle=90 -> normal vertical text\n          * rotateAxis=(1, 0), angle=0 -> text aligned with x axis of its parent\n          * rotateAxis=(0, 1), angle=0 -> text aligned with y axis of its parent\n          * rotateAxis=(1, 0), angle=90 -> text orthogonal to x axis of its parent\n        '
        self.anchor = Point(anchor)
        self.rotateAxis = None if rotateAxis is None else Point(rotateAxis)
        GraphicsObject.__init__(self)
        self.textItem = QtWidgets.QGraphicsTextItem()
        self.textItem.setParentItem(self)
        self._lastTransform = None
        self._lastScene = None
        if ensureInBounds:
            self.dataBounds = None
        self._bounds = QtCore.QRectF()
        if html is None:
            self.setColor(color)
            self.setText(text)
        else:
            self.setHtml(html)
        self.fill = fn.mkBrush(fill)
        self.border = fn.mkPen(border)
        self.setAngle(angle)

    def setText(self, text, color=None):
        if False:
            while True:
                i = 10
        '\n        Set the text of this item. \n        \n        This method sets the plain text of the item; see also setHtml().\n        '
        if color is not None:
            self.setColor(color)
        self.setPlainText(text)

    def setPlainText(self, text):
        if False:
            return 10
        '\n        Set the plain text to be rendered by this item. \n        \n        See QtWidgets.QGraphicsTextItem.setPlainText().\n        '
        if text != self.toPlainText():
            self.textItem.setPlainText(text)
            self.updateTextPos()

    def toPlainText(self):
        if False:
            for i in range(10):
                print('nop')
        return self.textItem.toPlainText()

    def setHtml(self, html):
        if False:
            while True:
                i = 10
        '\n        Set the HTML code to be rendered by this item. \n        \n        See QtWidgets.QGraphicsTextItem.setHtml().\n        '
        if self.toHtml() != html:
            self.textItem.setHtml(html)
            self.updateTextPos()

    def toHtml(self):
        if False:
            while True:
                i = 10
        return self.textItem.toHtml()

    def setTextWidth(self, *args):
        if False:
            i = 10
            return i + 15
        '\n        Set the width of the text.\n        \n        If the text requires more space than the width limit, then it will be\n        wrapped into multiple lines.\n        \n        See QtWidgets.QGraphicsTextItem.setTextWidth().\n        '
        self.textItem.setTextWidth(*args)
        self.updateTextPos()

    def setFont(self, *args):
        if False:
            print('Hello World!')
        '\n        Set the font for this text. \n        \n        See QtWidgets.QGraphicsTextItem.setFont().\n        '
        self.textItem.setFont(*args)
        self.updateTextPos()

    def setAngle(self, angle):
        if False:
            return 10
        '\n        Set the angle of the text in degrees.\n\n        This sets the rotation angle of the text as a whole, measured\n        counter-clockwise from the x axis of the parent. Note that this rotation\n        angle does not depend on horizontal/vertical scaling of the parent.\n        '
        self.angle = angle
        self.updateTransform(force=True)

    def setAnchor(self, anchor):
        if False:
            while True:
                i = 10
        self.anchor = Point(anchor)
        self.updateTextPos()

    def setColor(self, color):
        if False:
            return 10
        '\n        Set the color for this text.\n        \n        See QtWidgets.QGraphicsItem.setDefaultTextColor().\n        '
        self.color = fn.mkColor(color)
        self.textItem.setDefaultTextColor(self.color)

    def updateTextPos(self):
        if False:
            i = 10
            return i + 15
        r = self.textItem.boundingRect()
        tl = self.textItem.mapToParent(r.topLeft())
        br = self.textItem.mapToParent(r.bottomRight())
        offset = (br - tl) * self.anchor
        self.textItem.setPos(-offset)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if False:
            while True:
                i = 10
        '\n        Returns only the anchor point for when calulating view ranges.\n        \n        Sacrifices some visual polish for fixing issue #2642.\n        '
        if orthoRange:
            (range_min, range_max) = (orthoRange[0], orthoRange[1])
            if not range_min <= self.anchor[ax] <= range_max:
                return [None, None]
        return [self.anchor[ax], self.anchor[ax]]

    def boundingRect(self):
        if False:
            while True:
                i = 10
        return self.textItem.mapRectToParent(self.textItem.boundingRect())

    def viewTransformChanged(self):
        if False:
            print('Hello World!')
        self.updateTransform()

    def paint(self, p, *args):
        if False:
            i = 10
            return i + 15
        s = self.scene()
        ls = self._lastScene
        if s is not ls:
            if ls is not None:
                ls.sigPrepareForPaint.disconnect(self.updateTransform)
            self._lastScene = s
            if s is not None:
                s.sigPrepareForPaint.connect(self.updateTransform)
            self.updateTransform()
            p.setTransform(self.sceneTransform())
        if self.border.style() != QtCore.Qt.PenStyle.NoPen or self.fill.style() != QtCore.Qt.BrushStyle.NoBrush:
            p.setPen(self.border)
            p.setBrush(self.fill)
            p.setRenderHint(p.RenderHint.Antialiasing, True)
            p.drawPolygon(self.textItem.mapToParent(self.textItem.boundingRect()))

    def setVisible(self, v):
        if False:
            print('Hello World!')
        GraphicsObject.setVisible(self, v)
        if v:
            self.updateTransform()

    def updateTransform(self, force=False):
        if False:
            while True:
                i = 10
        if not self.isVisible():
            return
        p = self.parentItem()
        if p is None:
            pt = QtGui.QTransform()
        else:
            pt = p.sceneTransform()
        if not force and pt == self._lastTransform:
            return
        t = fn.invertQTransform(pt)
        t.setMatrix(t.m11(), t.m12(), t.m13(), t.m21(), t.m22(), t.m23(), 0, 0, t.m33())
        angle = -self.angle
        if self.rotateAxis is not None:
            d = pt.map(self.rotateAxis) - pt.map(Point(0, 0))
            a = degrees(atan2(d.y(), d.x()))
            angle += a
        t.rotate(angle)
        self.setTransform(t)
        self._lastTransform = pt
        self.updateTextPos()