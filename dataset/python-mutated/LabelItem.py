from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtWidgets, QtGui
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
__all__ = ['LabelItem']

class LabelItem(GraphicsWidgetAnchor, GraphicsWidget):
    """
    GraphicsWidget displaying text.
    Used mainly as axis labels, titles, etc.
    
    Note: To display text inside a scaled view (ViewBox, PlotWidget, etc) use TextItem
    """

    def __init__(self, text=' ', parent=None, angle=0, **args):
        if False:
            return 10
        GraphicsWidget.__init__(self, parent)
        GraphicsWidgetAnchor.__init__(self)
        self.item = QtWidgets.QGraphicsTextItem(self)
        self.opts = {'color': None, 'justify': 'center'}
        self.opts.update(args)
        self._sizeHint = {}
        self.setText(text)
        self.setAngle(angle)

    def setAttr(self, attr, value):
        if False:
            print('Hello World!')
        'Set default text properties. See setText() for accepted parameters.'
        self.opts[attr] = value

    def setText(self, text, **args):
        if False:
            while True:
                i = 10
        "Set the text and text properties in the label. Accepts optional arguments for auto-generating\n        a CSS style string:\n\n        ==================== ==============================\n        **Style Arguments:**\n        color                (str) example: '#CCFF00'\n        size                 (str) example: '8pt'\n        bold                 (bool)\n        italic               (bool)\n        ==================== ==============================\n        "
        self.text = text
        opts = self.opts
        for k in args:
            opts[k] = args[k]
        optlist = []
        color = self.opts['color']
        if color is None:
            color = getConfigOption('foreground')
        color = fn.mkColor(color)
        optlist.append('color: ' + color.name(QtGui.QColor.NameFormat.HexArgb))
        if 'size' in opts:
            optlist.append('font-size: ' + opts['size'])
        if 'bold' in opts and opts['bold'] in [True, False]:
            optlist.append('font-weight: ' + {True: 'bold', False: 'normal'}[opts['bold']])
        if 'italic' in opts and opts['italic'] in [True, False]:
            optlist.append('font-style: ' + {True: 'italic', False: 'normal'}[opts['italic']])
        full = "<span style='%s'>%s</span>" % ('; '.join(optlist), text)
        self.item.setHtml(full)
        self.updateMin()
        self.resizeEvent(None)
        self.updateGeometry()

    def resizeEvent(self, ev):
        if False:
            i = 10
            return i + 15
        self.item.setPos(0, 0)
        bounds = self.itemRect()
        left = self.mapFromItem(self.item, QtCore.QPointF(0, 0)) - self.mapFromItem(self.item, QtCore.QPointF(1, 0))
        rect = self.rect()
        if self.opts['justify'] == 'left':
            if left.x() != 0:
                bounds.moveLeft(rect.left())
            if left.y() < 0:
                bounds.moveTop(rect.top())
            elif left.y() > 0:
                bounds.moveBottom(rect.bottom())
        elif self.opts['justify'] == 'center':
            bounds.moveCenter(rect.center())
        elif self.opts['justify'] == 'right':
            if left.x() != 0:
                bounds.moveRight(rect.right())
            if left.y() < 0:
                bounds.moveBottom(rect.bottom())
            elif left.y() > 0:
                bounds.moveTop(rect.top())
        self.item.setPos(bounds.topLeft() - self.itemRect().topLeft())
        self.updateMin()

    def setAngle(self, angle):
        if False:
            return 10
        self.angle = angle
        self.item.resetTransform()
        self.item.setRotation(angle)
        self.updateMin()

    def updateMin(self):
        if False:
            return 10
        bounds = self.itemRect()
        self.setMinimumWidth(bounds.width())
        self.setMinimumHeight(bounds.height())
        self._sizeHint = {QtCore.Qt.SizeHint.MinimumSize: (bounds.width(), bounds.height()), QtCore.Qt.SizeHint.PreferredSize: (bounds.width(), bounds.height()), QtCore.Qt.SizeHint.MaximumSize: (-1, -1), QtCore.Qt.SizeHint.MinimumDescent: (0, 0)}
        self.updateGeometry()

    def sizeHint(self, hint, constraint):
        if False:
            i = 10
            return i + 15
        if hint not in self._sizeHint:
            return QtCore.QSizeF(0, 0)
        return QtCore.QSizeF(*self._sizeHint[hint])

    def itemRect(self):
        if False:
            i = 10
            return i + 15
        return self.item.mapRectToParent(self.item.boundingRect())