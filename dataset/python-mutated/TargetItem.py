import string
from math import atan2
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .ScatterPlotItem import Symbols
from .TextItem import TextItem
from .UIGraphicsItem import UIGraphicsItem
from .ViewBox import ViewBox
__all__ = ['TargetItem', 'TargetLabel']

class TargetItem(UIGraphicsItem):
    """Draws a draggable target symbol (circle plus crosshair).

    The size of TargetItem will remain fixed on screen even as the view is zoomed.
    Includes an optional text label.
    """
    sigPositionChanged = QtCore.Signal(object)
    sigPositionChangeFinished = QtCore.Signal(object)

    def __init__(self, pos=None, size=10, symbol='crosshair', pen=None, hoverPen=None, brush=None, hoverBrush=None, movable=True, label=None, labelOpts=None):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        pos : list, tuple, QPointF, QPoint, Optional\n            Initial position of the symbol.  Default is (0, 0)\n        size : int\n            Size of the symbol in pixels.  Default is 10.\n        pen : QPen, tuple, list or str\n            Pen to use when drawing line. Can be any arguments that are valid\n            for :func:`~pyqtgraph.mkPen`. Default pen is transparent yellow.\n        brush : QBrush, tuple, list, or str\n            Defines the brush that fill the symbol. Can be any arguments that\n            is valid for :func:`~pyqtgraph.mkBrush`. Default is transparent\n            blue.\n        movable : bool\n            If True, the symbol can be dragged to a new position by the user.\n        hoverPen : QPen, tuple, list, or str\n            Pen to use when drawing symbol when hovering over it. Can be any\n            arguments that are valid for :func:`~pyqtgraph.mkPen`. Default pen\n            is red.\n        hoverBrush : QBrush, tuple, list or str\n            Brush to use to fill the symbol when hovering over it. Can be any\n            arguments that is valid for :func:`~pyqtgraph.mkBrush`. Default is\n            transparent blue.\n        symbol : QPainterPath or str\n            QPainterPath to use for drawing the target, should be centered at\n            ``(0, 0)`` with ``max(width, height) == 1.0``.  Alternatively a string\n            which can be any symbol accepted by\n            :func:`~pyqtgraph.ScatterPlotItem.setSymbol`\n        label : bool, str or callable, optional\n            Text to be displayed in a label attached to the symbol, or None to\n            show no label (default is None). May optionally include formatting\n            strings to display the symbol value, or a callable that accepts x\n            and y as inputs.  If True, the label is ``x = {: >.3n}\\ny = {: >.3n}``\n            False or None will result in no text being displayed\n        labelOpts : dict\n            A dict of keyword arguments to use when constructing the text\n            label. See :class:`TargetLabel` and :class:`~pyqtgraph.TextItem`\n        '
        super().__init__()
        self.movable = movable
        self.moving = False
        self._label = None
        self.mouseHovering = False
        if pen is None:
            pen = (255, 255, 0)
        self.setPen(pen)
        if hoverPen is None:
            hoverPen = (255, 0, 255)
        self.setHoverPen(hoverPen)
        if brush is None:
            brush = (0, 0, 255, 50)
        self.setBrush(brush)
        if hoverBrush is None:
            hoverBrush = (0, 255, 255, 100)
        self.setHoverBrush(hoverBrush)
        self.currentPen = self.pen
        self.currentBrush = self.brush
        self._shape = None
        self._pos = Point(0, 0)
        if pos is None:
            pos = Point(0, 0)
        self.setPos(pos)
        if isinstance(symbol, str):
            try:
                self._path = Symbols[symbol]
            except KeyError:
                raise KeyError('symbol name found in available Symbols')
        elif isinstance(symbol, QtGui.QPainterPath):
            self._path = symbol
        else:
            raise TypeError('Unknown type provided as symbol')
        self.scale = size
        self.setPath(self._path)
        self.setLabel(label, labelOpts)

    def setPos(self, *args):
        if False:
            return 10
        'Method to set the position to ``(x, y)`` within the plot view\n\n        Parameters\n        ----------\n        args : tuple or list or QtCore.QPointF or QtCore.QPoint or Point or float\n            Two float values or a container that specifies ``(x, y)`` position where the\n            TargetItem should be placed\n\n        Raises\n        ------\n        TypeError\n            If args cannot be used to instantiate a Point\n        '
        try:
            newPos = Point(*args)
        except TypeError:
            raise
        except Exception:
            raise TypeError(f'Could not make Point from arguments: {args!r}')
        if self._pos != newPos:
            self._pos = newPos
            super().setPos(self._pos)
            self.sigPositionChanged.emit(self)

    def setBrush(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Set the brush that fills the symbol. Allowable arguments are any that\n        are valid for :func:`~pyqtgraph.mkBrush`.\n        '
        self.brush = fn.mkBrush(*args, **kwargs)
        if not self.mouseHovering:
            self.currentBrush = self.brush
            self.update()

    def setHoverBrush(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Set the brush that fills the symbol when hovering over it. Allowable\n        arguments are any that are valid for :func:`~pyqtgraph.mkBrush`.\n        '
        self.hoverBrush = fn.mkBrush(*args, **kwargs)
        if self.mouseHovering:
            self.currentBrush = self.hoverBrush
            self.update()

    def setPen(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Set the pen for drawing the symbol. Allowable arguments are any that\n        are valid for :func:`~pyqtgraph.mkPen`.'
        self.pen = fn.mkPen(*args, **kwargs)
        if not self.mouseHovering:
            self.currentPen = self.pen
            self.update()

    def setHoverPen(self, *args, **kwargs):
        if False:
            return 10
        'Set the pen for drawing the symbol when hovering over it. Allowable\n        arguments are any that are valid for\n        :func:`~pyqtgraph.mkPen`.'
        self.hoverPen = fn.mkPen(*args, **kwargs)
        if self.mouseHovering:
            self.currentPen = self.hoverPen
            self.update()

    def boundingRect(self):
        if False:
            i = 10
            return i + 15
        return self.shape().boundingRect()

    def paint(self, p, *_):
        if False:
            print('Hello World!')
        p.setPen(self.currentPen)
        p.setBrush(self.currentBrush)
        p.drawPath(self.shape())

    def setPath(self, path):
        if False:
            return 10
        if path != self._path:
            self._path = path
            self._shape = None
        return None

    def shape(self):
        if False:
            print('Hello World!')
        if self._shape is None:
            s = self.generateShape()
            if s is None:
                return self._path
            self._shape = s
            self.prepareGeometryChange()
        return self._shape

    def generateShape(self):
        if False:
            return 10
        dt = self.deviceTransform()
        if dt is None:
            self._shape = self._path
            return None
        v = dt.map(QtCore.QPointF(1, 0)) - dt.map(QtCore.QPointF(0, 0))
        dti = fn.invertQTransform(dt)
        devPos = dt.map(QtCore.QPointF(0, 0))
        tr = QtGui.QTransform()
        tr.translate(devPos.x(), devPos.y())
        va = atan2(v.y(), v.x())
        tr.rotateRadians(va)
        tr.scale(self.scale, self.scale)
        return dti.map(tr.map(self._path))

    def mouseDragEvent(self, ev):
        if False:
            return 10
        if not self.movable or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ev.accept()
        if ev.isStart():
            self.symbolOffset = self.pos() - self.mapToView(ev.buttonDownPos())
            self.moving = True
        if not self.moving:
            return
        self.setPos(self.symbolOffset + self.mapToView(ev.pos()))
        if ev.isFinish():
            self.moving = False
            self.sigPositionChangeFinished.emit(self)

    def mouseClickEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        if self.moving and ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.moving = False
            self.sigPositionChanged.emit(self)
            self.sigPositionChangeFinished.emit(self)

    def setMouseHover(self, hover):
        if False:
            print('Hello World!')
        if self.mouseHovering is hover:
            return
        self.mouseHovering = hover
        if hover:
            self.currentBrush = self.hoverBrush
            self.currentPen = self.hoverPen
        else:
            self.currentBrush = self.brush
            self.currentPen = self.pen
        self.update()

    def hoverEvent(self, ev):
        if False:
            print('Hello World!')
        if self.movable and (not ev.isExit()) and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)

    def viewTransformChanged(self):
        if False:
            for i in range(10):
                print('nop')
        GraphicsObject.viewTransformChanged(self)
        self._shape = None
        self.update()

    def pos(self):
        if False:
            for i in range(10):
                print('nop')
        'Provides the current position of the TargetItem\n\n        Returns\n        -------\n        Point\n            pg.Point of the current position of the TargetItem\n        '
        return self._pos

    def label(self):
        if False:
            return 10
        'Provides the TargetLabel if it exists\n\n        Returns\n        -------\n        TargetLabel or None\n            If a TargetLabel exists for this TargetItem, return that, otherwise\n            return None\n        '
        return self._label

    def setLabel(self, text=None, labelOpts=None):
        if False:
            print('Hello World!')
        'Method to call to enable or disable the TargetLabel for displaying text\n\n        Parameters\n        ----------\n        text : Callable or str, optional\n            Details how to format the text, by default None\n            If None, do not show any text next to the TargetItem\n            If Callable, then the label will display the result of ``text(x, y)``\n            If a fromatted string, then the output of ``text.format(x, y)`` will be\n            displayed\n            If a non-formatted string, then the text label will display ``text``, by\n            default None\n        labelOpts : dict, optional\n            These arguments are passed on to :class:`~pyqtgraph.TextItem`\n        '
        if not text:
            if self._label is not None and self._label.scene() is not None:
                self._label.scene().removeItem(self._label)
            self._label = None
        else:
            if text is True:
                text = 'x = {: .3n}\ny = {: .3n}'
            labelOpts = {} if labelOpts is None else labelOpts
            if self._label is not None:
                self._label.scene().removeItem(self._label)
            self._label = TargetLabel(self, text=text, **labelOpts)

class TargetLabel(TextItem):
    """A TextItem that attaches itself to a TargetItem.

    This class extends TextItem with the following features :
      * Automatically positions adjacent to the symbol at a fixed position.
      * Automatically reformats text when the symbol location has changed.

    Parameters
    ----------
    target : TargetItem
        The TargetItem to which this label will be attached to.
    text : str or callable, Optional
        Governs the text displayed, can be a fixed string or a format string
        that accepts the x, and y position of the target item; or be a callable
        method that accepts a tuple (x, y) and returns a string to be displayed.
        If None, an empty string is used.  Default is None
    offset : tuple or list or QPointF or QPoint
        Position to set the anchor of the TargetLabel away from the center of
        the target in pixels, by default it is (20, 0).
    anchor : tuple or list or QPointF or QPoint
        Position to rotate the TargetLabel about, and position to set the
        offset value to see :class:`~pyqtgraph.TextItem` for more information.
    kwargs : dict 
        kwargs contains arguments that are passed onto
        :class:`~pyqtgraph.TextItem` constructor, excluding text parameter
    """

    def __init__(self, target, text='', offset=(20, 0), anchor=(0, 0.5), **kwargs):
        if False:
            i = 10
            return i + 15
        if isinstance(offset, Point):
            self.offset = offset
        elif isinstance(offset, (tuple, list)):
            self.offset = Point(*offset)
        elif isinstance(offset, (QtCore.QPoint, QtCore.QPointF)):
            self.offset = Point(offset.x(), offset.y())
        else:
            raise TypeError('Offset parameter is the wrong data type')
        super().__init__(anchor=anchor, **kwargs)
        self.setParentItem(target)
        self.target = target
        self.setFormat(text)
        self.target.sigPositionChanged.connect(self.valueChanged)
        self.valueChanged()

    def format(self):
        if False:
            return 10
        return self._format

    def setFormat(self, text):
        if False:
            return 10
        'Method to set how the TargetLabel should display the text.  This\n        method should be called from TargetItem.setLabel directly.\n\n        Parameters\n        ----------\n        text : Callable or str\n            Details how to format the text.\n            If Callable, then the label will display the result of ``text(x, y)``\n            If a fromatted string, then the output of ``text.format(x, y)`` will be\n            displayed\n            If a non-formatted string, then the text label will display ``text``\n        '
        if not callable(text):
            parsed = list(string.Formatter().parse(text))
            if parsed and parsed[0][1] is not None:
                self.setProperty('formattableText', True)
            else:
                self.setText(text)
                self.setProperty('formattableText', False)
        else:
            self.setProperty('formattableText', False)
        self._format = text
        self.valueChanged()

    def valueChanged(self):
        if False:
            i = 10
            return i + 15
        (x, y) = self.target.pos()
        if self.property('formattableText'):
            self.setText(self._format.format(float(x), float(y)))
        elif callable(self._format):
            self.setText(self._format(x, y))

    def viewTransformChanged(self):
        if False:
            return 10
        viewbox = self.getViewBox()
        if isinstance(viewbox, ViewBox):
            viewPixelSize = viewbox.viewPixelSize()
            scaledOffset = QtCore.QPointF(self.offset.x() * viewPixelSize[0], self.offset.y() * viewPixelSize[1])
            self.setPos(scaledOffset)
        return super().viewTransformChanged()

    def mouseClickEvent(self, ev):
        if False:
            return 10
        return self.parentItem().mouseClickEvent(ev)

    def mouseDragEvent(self, ev):
        if False:
            return 10
        targetItem = self.parentItem()
        if not targetItem.movable or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ev.accept()
        if ev.isStart():
            targetItem.symbolOffset = targetItem.pos() - self.mapToView(ev.buttonDownPos())
            targetItem.moving = True
        if not targetItem.moving:
            return
        targetItem.setPos(targetItem.symbolOffset + self.mapToView(ev.pos()))
        if ev.isFinish():
            targetItem.moving = False
            targetItem.sigPositionChangeFinished.emit(self)