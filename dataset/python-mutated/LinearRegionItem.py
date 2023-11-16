from .. import debug
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .InfiniteLine import InfiniteLine
__all__ = ['LinearRegionItem']

class LinearRegionItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    
    Used for marking a horizontal or vertical region in plots.
    The region can be dragged and is bounded by lines which can be dragged individually.
    
    ===============================  =============================================================================
    **Signals:**
    sigRegionChangeFinished(self)    Emitted when the user has finished dragging the region (or one of its lines)
                                     and when the region is changed programatically.
    sigRegionChanged(self)           Emitted while the user is dragging the region (or one of its lines)
                                     and when the region is changed programatically.
    ===============================  =============================================================================
    """
    sigRegionChangeFinished = QtCore.Signal(object)
    sigRegionChanged = QtCore.Signal(object)
    Vertical = 0
    Horizontal = 1
    _orientation_axis = {Vertical: 0, Horizontal: 1, 'vertical': 0, 'horizontal': 1}

    def __init__(self, values=(0, 1), orientation='vertical', brush=None, pen=None, hoverBrush=None, hoverPen=None, movable=True, bounds=None, span=(0, 1), swapMode='sort', clipItem=None):
        if False:
            while True:
                i = 10
        'Create a new LinearRegionItem.\n        \n        ==============  =====================================================================\n        **Arguments:**\n        values          A list of the positions of the lines in the region. These are not\n                        limits; limits can be set by specifying bounds.\n        orientation     Options are \'vertical\' or \'horizontal\'\n                        The default is \'vertical\', indicating that the region is bounded\n                        by vertical lines.\n        brush           Defines the brush that fills the region. Can be any arguments that\n                        are valid for :func:`mkBrush <pyqtgraph.mkBrush>`. Default is\n                        transparent blue.\n        pen             The pen to use when drawing the lines that bound the region.\n        hoverBrush      The brush to use when the mouse is hovering over the region.\n        hoverPen        The pen to use when the mouse is hovering over the region.\n        movable         If True, the region and individual lines are movable by the user; if\n                        False, they are static.\n        bounds          Optional [min, max] bounding values for the region\n        span            Optional [min, max] giving the range over the view to draw\n                        the region. For example, with a vertical line, use\n                        ``span=(0.5, 1)`` to draw only on the top half of the\n                        view.\n        swapMode        Sets the behavior of the region when the lines are moved such that\n                        their order reverses:\n\n                          * "block" means the user cannot drag one line past the other\n                          * "push" causes both lines to be moved if one would cross the other\n                          * "sort" means that lines may trade places, but the output of\n                            getRegion always gives the line positions in ascending order.\n                          * None means that no attempt is made to handle swapped line\n                            positions.\n\n                        The default is "sort".\n        clipItem        An item whose bounds will be used to limit the region bounds.\n                        This is useful when a LinearRegionItem is added on top of an\n                        :class:`~pyqtgraph.ImageItem` or\n                        :class:`~pyqtgraph.PlotDataItem` and the visual region should\n                        not extend beyond its range. This overrides ``bounds``.\n        ==============  =====================================================================\n        '
        GraphicsObject.__init__(self)
        self.orientation = orientation
        self.blockLineSignal = False
        self.moving = False
        self.mouseHovering = False
        self.span = span
        self.swapMode = swapMode
        self.clipItem = clipItem
        self._boundingRectCache = None
        self._clipItemBoundsCache = None
        lineKwds = dict(movable=movable, bounds=bounds, span=span, pen=pen, hoverPen=hoverPen)
        if orientation in ('horizontal', LinearRegionItem.Horizontal):
            self.lines = [InfiniteLine(QtCore.QPointF(0, values[0]), angle=0, **lineKwds), InfiniteLine(QtCore.QPointF(0, values[1]), angle=0, **lineKwds)]
            tr = QtGui.QTransform.fromScale(1, -1)
            self.lines[0].setTransform(tr, True)
            self.lines[1].setTransform(tr, True)
        elif orientation in ('vertical', LinearRegionItem.Vertical):
            self.lines = [InfiniteLine(QtCore.QPointF(values[0], 0), angle=90, **lineKwds), InfiniteLine(QtCore.QPointF(values[1], 0), angle=90, **lineKwds)]
        else:
            raise Exception("Orientation must be 'vertical' or 'horizontal'.")
        for l in self.lines:
            l.setParentItem(self)
            l.sigPositionChangeFinished.connect(self.lineMoveFinished)
        self.lines[0].sigPositionChanged.connect(self._line0Moved)
        self.lines[1].sigPositionChanged.connect(self._line1Moved)
        if brush is None:
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 255, 50))
        self.setBrush(brush)
        if hoverBrush is None:
            c = self.brush.color()
            c.setAlpha(min(c.alpha() * 2, 255))
            hoverBrush = fn.mkBrush(c)
        self.setHoverBrush(hoverBrush)
        self.setMovable(movable)

    def getRegion(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the values at the edges of the region.'
        r = (self.lines[0].value(), self.lines[1].value())
        if self.swapMode == 'sort':
            return (min(r), max(r))
        else:
            return r

    def setRegion(self, rgn):
        if False:
            i = 10
            return i + 15
        'Set the values for the edges of the region.\n        \n        ==============   ==============================================\n        **Arguments:**\n        rgn              A list or tuple of the lower and upper values.\n        ==============   ==============================================\n        '
        if self.lines[0].value() == rgn[0] and self.lines[1].value() == rgn[1]:
            return
        self.blockLineSignal = True
        self.lines[0].setValue(rgn[0])
        self.blockLineSignal = False
        self.lines[1].setValue(rgn[1])
        self.lineMoved(0)
        self.lineMoved(1)
        self.lineMoveFinished()

    def setBrush(self, *br, **kargs):
        if False:
            print('Hello World!')
        'Set the brush that fills the region. Can have any arguments that are valid\n        for :func:`mkBrush <pyqtgraph.mkBrush>`.\n        '
        self.brush = fn.mkBrush(*br, **kargs)
        self.currentBrush = self.brush

    def setHoverBrush(self, *br, **kargs):
        if False:
            while True:
                i = 10
        'Set the brush that fills the region when the mouse is hovering over.\n        Can have any arguments that are valid\n        for :func:`mkBrush <pyqtgraph.mkBrush>`.\n        '
        self.hoverBrush = fn.mkBrush(*br, **kargs)

    def setBounds(self, bounds):
        if False:
            return 10
        'Set ``(min, max)`` bounding values for the region.\n\n        The current position is only affected it is outside the new bounds. See\n        :func:`~pyqtgraph.LinearRegionItem.setRegion` to set the position of the region.\n\n        Use ``(None, None)`` to disable bounds.\n        '
        if self.clipItem is not None:
            self.setClipItem(None)
        self._setBounds(bounds)

    def _setBounds(self, bounds):
        if False:
            return 10
        for line in self.lines:
            line.setBounds(bounds)

    def setMovable(self, m=True):
        if False:
            i = 10
            return i + 15
        'Set lines to be movable by the user, or not. If lines are movable, they will \n        also accept HoverEvents.'
        for line in self.lines:
            line.setMovable(m)
        self.movable = m
        self.setAcceptHoverEvents(m)

    def setSpan(self, mn, mx):
        if False:
            return 10
        if self.span == (mn, mx):
            return
        self.span = (mn, mx)
        for line in self.lines:
            line.setSpan(mn, mx)
        self.update()

    def setClipItem(self, item=None):
        if False:
            while True:
                i = 10
        'Set an item to which the region is bounded.\n\n        If ``None``, bounds are disabled.\n        '
        self.clipItem = item
        self._clipItemBoundsCache = None
        if item is None:
            self._setBounds((None, None))
        if item is not None:
            self._updateClipItemBounds()

    def _updateClipItemBounds(self):
        if False:
            print('Hello World!')
        item_vb = self.clipItem.getViewBox()
        if item_vb is None:
            return
        item_bounds = item_vb.childrenBounds(items=(self.clipItem,))
        if item_bounds == self._clipItemBoundsCache or None in item_bounds:
            return
        self._clipItemBoundsCache = item_bounds
        if self.orientation in ('horizontal', LinearRegionItem.Horizontal):
            self._setBounds(item_bounds[1])
        else:
            self._setBounds(item_bounds[0])

    def boundingRect(self):
        if False:
            while True:
                i = 10
        br = QtCore.QRectF(self.viewRect())
        if self.clipItem is not None:
            self._updateClipItemBounds()
        rng = self.getRegion()
        if self.orientation in ('vertical', LinearRegionItem.Vertical):
            br.setLeft(rng[0])
            br.setRight(rng[1])
            length = br.height()
            br.setBottom(br.top() + length * self.span[1])
            br.setTop(br.top() + length * self.span[0])
        else:
            br.setTop(rng[0])
            br.setBottom(rng[1])
            length = br.width()
            br.setRight(br.left() + length * self.span[1])
            br.setLeft(br.left() + length * self.span[0])
        br = br.normalized()
        if self._boundingRectCache != br:
            self._boundingRectCache = br
            self.prepareGeometryChange()
        return br

    def paint(self, p, *args):
        if False:
            i = 10
            return i + 15
        profiler = debug.Profiler()
        p.setBrush(self.currentBrush)
        p.setPen(fn.mkPen(None))
        p.drawRect(self.boundingRect())

    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        if False:
            print('Hello World!')
        if axis == self._orientation_axis[self.orientation]:
            return self.getRegion()
        else:
            return None

    def lineMoved(self, i):
        if False:
            for i in range(10):
                print('nop')
        if self.blockLineSignal:
            return
        if self.lines[0].value() > self.lines[1].value():
            if self.swapMode == 'block':
                self.lines[i].setValue(self.lines[1 - i].value())
            elif self.swapMode == 'push':
                self.lines[1 - i].setValue(self.lines[i].value())
        self.prepareGeometryChange()
        self.sigRegionChanged.emit(self)

    def _line0Moved(self):
        if False:
            for i in range(10):
                print('nop')
        self.lineMoved(0)

    def _line1Moved(self):
        if False:
            return 10
        self.lineMoved(1)

    def lineMoveFinished(self):
        if False:
            return 10
        self.sigRegionChangeFinished.emit(self)

    def mouseDragEvent(self, ev):
        if False:
            i = 10
            return i + 15
        if not self.movable or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ev.accept()
        if ev.isStart():
            bdp = ev.buttonDownPos()
            self.cursorOffsets = [l.pos() - bdp for l in self.lines]
            self.startPositions = [l.pos() for l in self.lines]
            self.moving = True
        if not self.moving:
            return
        self.lines[0].blockSignals(True)
        for (i, l) in enumerate(self.lines):
            l.setPos(self.cursorOffsets[i] + ev.pos())
        self.lines[0].blockSignals(False)
        self.prepareGeometryChange()
        if ev.isFinish():
            self.moving = False
            self.sigRegionChangeFinished.emit(self)
        else:
            self.sigRegionChanged.emit(self)

    def mouseClickEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        if self.moving and ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            for (i, l) in enumerate(self.lines):
                l.setPos(self.startPositions[i])
            self.moving = False
            self.sigRegionChanged.emit(self)
            self.sigRegionChangeFinished.emit(self)

    def hoverEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        if self.movable and (not ev.isExit()) and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)

    def setMouseHover(self, hover):
        if False:
            return 10
        if self.mouseHovering == hover:
            return
        self.mouseHovering = hover
        if hover:
            self.currentBrush = self.hoverBrush
        else:
            self.currentBrush = self.brush
        self.update()