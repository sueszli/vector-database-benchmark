import operator
import weakref
import numpy as np
from .. import functions as fn
from .. import colormap
from ..colormap import ColorMap
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorMapButton import ColorMapMenu
from .GraphicsWidget import GraphicsWidget
from .GradientPresets import Gradients
translate = QtCore.QCoreApplication.translate
__all__ = ['TickSliderItem', 'GradientEditorItem', 'addGradientListToDocstring']

def addGradientListToDocstring():
    if False:
        print('Hello World!')
    'Decorator to add list of current pre-defined gradients to the end of a function docstring.'

    def dec(fn):
        if False:
            while True:
                i = 10
        if fn.__doc__ is not None:
            fn.__doc__ = fn.__doc__ + str(list(Gradients.keys())).strip('[').strip(']')
        return fn
    return dec

class TickSliderItem(GraphicsWidget):
    """**Bases:** :class:`GraphicsWidget <pyqtgraph.GraphicsWidget>`
    
    A rectangular item with tick marks along its length that can (optionally) be moved by the user."""
    sigTicksChanged = QtCore.Signal(object)
    sigTicksChangeFinished = QtCore.Signal(object)

    def __init__(self, orientation='bottom', allowAdd=True, allowRemove=True, **kargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        ==============  =================================================================================\n        **Arguments:**\n        orientation     Set the orientation of the gradient. Options are: 'left', 'right'\n                        'top', and 'bottom'.\n        allowAdd        Specifies whether the user can add ticks.\n        allowRemove     Specifies whether the user can remove new ticks.\n        tickPen         Default is white. Specifies the color of the outline of the ticks.\n                        Can be any of the valid arguments for :func:`mkPen <pyqtgraph.mkPen>`\n        ==============  =================================================================================\n        "
        GraphicsWidget.__init__(self)
        self.orientation = orientation
        self.length = 100
        self.tickSize = 15
        self.ticks = {}
        self.maxDim = 20
        self.allowAdd = allowAdd
        self.allowRemove = allowRemove
        if 'tickPen' in kargs:
            self.tickPen = fn.mkPen(kargs['tickPen'])
        else:
            self.tickPen = fn.mkPen('w')
        self.orientations = {'left': (90, 1, 1), 'right': (90, 1, 1), 'top': (0, 1, -1), 'bottom': (0, 1, 1)}
        self.setOrientation(orientation)

    def paint(self, p, opt, widget):
        if False:
            while True:
                i = 10
        return

    def keyPressEvent(self, ev):
        if False:
            while True:
                i = 10
        ev.ignore()

    def setMaxDim(self, mx=None):
        if False:
            while True:
                i = 10
        if mx is None:
            mx = self.maxDim
        else:
            self.maxDim = mx
        if self.orientation in ['bottom', 'top']:
            self.setFixedHeight(mx)
            self.setMaximumWidth(16777215)
        else:
            self.setFixedWidth(mx)
            self.setMaximumHeight(16777215)

    def setOrientation(self, orientation):
        if False:
            for i in range(10):
                print('nop')
        "Set the orientation of the TickSliderItem.\n        \n        ==============  ===================================================================\n        **Arguments:**\n        orientation     Options are: 'left', 'right', 'top', 'bottom'\n                        The orientation option specifies which side of the slider the\n                        ticks are on, as well as whether the slider is vertical ('right'\n                        and 'left') or horizontal ('top' and 'bottom').\n        ==============  ===================================================================\n        "
        self.orientation = orientation
        self.setMaxDim()
        self.resetTransform()
        ort = orientation
        if ort == 'top':
            transform = QtGui.QTransform.fromScale(1, -1)
            transform.translate(0, -self.height())
            self.setTransform(transform)
        elif ort == 'left':
            transform = QtGui.QTransform()
            transform.rotate(270)
            transform.scale(1, -1)
            transform.translate(-self.height(), -self.maxDim)
            self.setTransform(transform)
        elif ort == 'right':
            transform = QtGui.QTransform()
            transform.rotate(270)
            transform.translate(-self.height(), 0)
            self.setTransform(transform)
        elif ort != 'bottom':
            raise Exception("%s is not a valid orientation. Options are 'left', 'right', 'top', and 'bottom'" % str(ort))
        tr = QtGui.QTransform.fromTranslate(self.tickSize / 2.0, 0)
        self.setTransform(tr, True)

    def addTick(self, x, color=None, movable=True, finish=True):
        if False:
            while True:
                i = 10
        '\n        Add a tick to the item.\n        \n        ==============  ==================================================================\n        **Arguments:**\n        x               Position where tick should be added.\n        color           Color of added tick. If color is not specified, the color will be\n                        white.\n        movable         Specifies whether the tick is movable with the mouse.\n        ==============  ==================================================================\n        '
        if color is None:
            color = QtGui.QColor(255, 255, 255)
        tick = Tick([x * self.length, 0], color, movable, self.tickSize, pen=self.tickPen, removeAllowed=self.allowRemove)
        self.ticks[tick] = x
        tick.setParentItem(self)
        tick.sigMoving.connect(self.tickMoved)
        tick.sigMoved.connect(self.tickMoveFinished)
        tick.sigClicked.connect(self.tickClicked)
        self.sigTicksChanged.emit(self)
        if finish:
            self.sigTicksChangeFinished.emit(self)
        return tick

    def removeTick(self, tick, finish=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Removes the specified tick.\n        '
        del self.ticks[tick]
        tick.setParentItem(None)
        if self.scene() is not None:
            self.scene().removeItem(tick)
        self.sigTicksChanged.emit(self)
        if finish:
            self.sigTicksChangeFinished.emit(self)

    def tickMoved(self, tick, pos):
        if False:
            i = 10
            return i + 15
        newX = min(max(0, pos.x()), self.length)
        pos.setX(newX)
        tick.setPos(pos)
        self.ticks[tick] = float(newX) / self.length
        self.sigTicksChanged.emit(self)

    def tickMoveFinished(self, tick):
        if False:
            return 10
        self.sigTicksChangeFinished.emit(self)

    def tickClicked(self, tick, ev):
        if False:
            return 10
        if ev.button() == QtCore.Qt.MouseButton.RightButton and tick.removeAllowed:
            self.removeTick(tick)

    def widgetLength(self):
        if False:
            for i in range(10):
                print('nop')
        if self.orientation in ['bottom', 'top']:
            return self.width()
        else:
            return self.height()

    def resizeEvent(self, ev):
        if False:
            print('Hello World!')
        wlen = max(40, self.widgetLength())
        self.setLength(wlen - self.tickSize - 2)
        self.setOrientation(self.orientation)

    def setLength(self, newLen):
        if False:
            print('Hello World!')
        for (t, x) in list(self.ticks.items()):
            t.setPos(x * newLen + 1, t.pos().y())
        self.length = float(newLen)

    def mouseClickEvent(self, ev):
        if False:
            return 10
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and self.allowAdd:
            pos = ev.pos()
            if pos.x() < 0 or pos.x() > self.length:
                return
            if pos.y() < 0 or pos.y() > self.tickSize:
                return
            pos.setX(min(max(pos.x(), 0), self.length))
            self.addTick(pos.x() / self.length)
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.showMenu(ev)

    def hoverEvent(self, ev):
        if False:
            return 10
        if not ev.isExit() and ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton):
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)

    def showMenu(self, ev):
        if False:
            for i in range(10):
                print('nop')
        pass

    def setTickColor(self, tick, color):
        if False:
            return 10
        'Set the color of the specified tick.\n        \n        ==============  ==================================================================\n        **Arguments:**\n        tick            Can be either an integer corresponding to the index of the tick\n                        or a Tick object. Ex: if you had a slider with 3 ticks and you\n                        wanted to change the middle tick, the index would be 1.\n        color           The color to make the tick. Can be any argument that is valid for\n                        :func:`mkBrush <pyqtgraph.mkBrush>`\n        ==============  ==================================================================\n        '
        tick = self.getTick(tick)
        tick.color = color
        tick.update()
        self.sigTicksChanged.emit(self)
        self.sigTicksChangeFinished.emit(self)

    def setTickValue(self, tick, val):
        if False:
            return 10
        '\n        Set the position (along the slider) of the tick.\n        \n        ==============   ==================================================================\n        **Arguments:**\n        tick             Can be either an integer corresponding to the index of the tick\n                         or a Tick object. Ex: if you had a slider with 3 ticks and you\n                         wanted to change the middle tick, the index would be 1.\n        val              The desired position of the tick. If val is < 0, position will be\n                         set to 0. If val is > 1, position will be set to 1.\n        ==============   ==================================================================\n        '
        tick = self.getTick(tick)
        val = min(max(0.0, val), 1.0)
        x = val * self.length
        pos = tick.pos()
        pos.setX(x)
        tick.setPos(pos)
        self.ticks[tick] = val
        self.update()
        self.sigTicksChanged.emit(self)
        self.sigTicksChangeFinished.emit(self)

    def tickValue(self, tick):
        if False:
            i = 10
            return i + 15
        'Return the value (from 0.0 to 1.0) of the specified tick.\n        \n        ==============  ==================================================================\n        **Arguments:**\n        tick            Can be either an integer corresponding to the index of the tick\n                        or a Tick object. Ex: if you had a slider with 3 ticks and you\n                        wanted the value of the middle tick, the index would be 1.\n        ==============  ==================================================================\n        '
        tick = self.getTick(tick)
        return self.ticks[tick]

    def getTick(self, tick):
        if False:
            for i in range(10):
                print('nop')
        'Return the Tick object at the specified index.\n        \n        ==============  ==================================================================\n        **Arguments:**\n        tick            An integer corresponding to the index of the desired tick. If the\n                        argument is not an integer it will be returned unchanged.\n        ==============  ==================================================================\n        '
        if type(tick) is int:
            tick = self.listTicks()[tick][0]
        return tick

    def listTicks(self):
        if False:
            return 10
        'Return a sorted list of all the Tick objects on the slider.'
        ticks = sorted(self.ticks.items(), key=operator.itemgetter(1))
        return ticks

class GradientEditorItem(TickSliderItem):
    """
    **Bases:** :class:`TickSliderItem <pyqtgraph.TickSliderItem>`
    
    An item that can be used to define a color gradient. Implements common pre-defined gradients that are 
    customizable by the user. :class: `GradientWidget <pyqtgraph.GradientWidget>` provides a widget
    with a GradientEditorItem that can be added to a GUI. 
    
    ================================ ===========================================================
    **Signals:**
    sigGradientChanged(self)         Signal is emitted anytime the gradient changes. The signal 
                                     is emitted in real time while ticks are being dragged or 
                                     colors are being changed.
    sigGradientChangeFinished(self)  Signal is emitted when the gradient is finished changing.
    ================================ ===========================================================    
 
    """
    sigGradientChanged = QtCore.Signal(object)
    sigGradientChangeFinished = QtCore.Signal(object)

    def __init__(self, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Create a new GradientEditorItem. \n        All arguments are passed to :func:`TickSliderItem.__init__ <pyqtgraph.TickSliderItem.__init__>`\n        \n        ===============  =================================================================================\n        **Arguments:**\n        orientation      Set the orientation of the gradient. Options are: 'left', 'right'\n                         'top', and 'bottom'.\n        allowAdd         Default is True. Specifies whether ticks can be added to the item.\n        tickPen          Default is white. Specifies the color of the outline of the ticks.\n                         Can be any of the valid arguments for :func:`mkPen <pyqtgraph.mkPen>`\n        ===============  =================================================================================\n        "
        self.currentTick = None
        self.currentTickColor = None
        self.rectSize = 15
        self.gradRect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, self.rectSize, 100, self.rectSize))
        self.backgroundRect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, -self.rectSize, 100, self.rectSize))
        self.backgroundRect.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.DiagCrossPattern))
        self.colorMode = 'rgb'
        TickSliderItem.__init__(self, *args, **kargs)
        self.colorDialog = QtWidgets.QColorDialog()
        self.colorDialog.setOption(QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
        self.colorDialog.setOption(QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        self.colorDialog.currentColorChanged.connect(self.currentColorChanged)
        self.colorDialog.rejected.connect(self.currentColorRejected)
        self.colorDialog.accepted.connect(self.currentColorAccepted)
        self.backgroundRect.setParentItem(self)
        self.gradRect.setParentItem(self)
        self.setMaxDim(self.rectSize + self.tickSize)
        self.rgbAction = QtGui.QAction(translate('GradiantEditorItem', 'RGB'), self)
        self.rgbAction.setCheckable(True)
        self.rgbAction.triggered.connect(self._setColorModeToRGB)
        self.hsvAction = QtGui.QAction(translate('GradiantEditorItem', 'HSV'), self)
        self.hsvAction.setCheckable(True)
        self.hsvAction.triggered.connect(self._setColorModeToHSV)
        self.menu = ColorMapMenu(showGradientSubMenu=True)
        self.menu.triggered.connect(self.contextMenuClicked)
        self.menu.addSeparator()
        self.menu.addAction(self.rgbAction)
        self.menu.addAction(self.hsvAction)
        for t in list(self.ticks.keys()):
            self.removeTick(t)
        self.addTick(0, QtGui.QColor(0, 0, 0), True)
        self.addTick(1, QtGui.QColor(255, 0, 0), True)
        self.setColorMode('rgb')
        self.updateGradient()
        self.linkedGradients = {}
        self.sigTicksChanged.connect(self._updateGradientIgnoreArgs)
        self.sigTicksChangeFinished.connect(self.sigGradientChangeFinished)

    def showTicks(self, show=True):
        if False:
            print('Hello World!')
        for tick in self.ticks.keys():
            if show:
                tick.show()
                orig = getattr(self, '_allowAdd_backup', None)
                if orig:
                    self.allowAdd = orig
            else:
                self._allowAdd_backup = self.allowAdd
                self.allowAdd = False
                tick.hide()

    def setOrientation(self, orientation):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set the orientation of the GradientEditorItem. \n        \n        ==============  ===================================================================\n        **Arguments:**\n        orientation     Options are: 'left', 'right', 'top', 'bottom'\n                        The orientation option specifies which side of the gradient the\n                        ticks are on, as well as whether the gradient is vertical ('right'\n                        and 'left') or horizontal ('top' and 'bottom').\n        ==============  ===================================================================\n        "
        TickSliderItem.setOrientation(self, orientation)
        tr = QtGui.QTransform.fromTranslate(0, self.rectSize)
        self.setTransform(tr, True)

    def showMenu(self, ev):
        if False:
            print('Hello World!')
        self.menu.popup(ev.screenPos().toQPoint())

    def contextMenuClicked(self, action):
        if False:
            i = 10
            return i + 15
        if action in [self.rgbAction, self.hsvAction]:
            return
        (name, source) = action.data()
        if source == 'preset-gradient':
            self.loadPreset(name)
        else:
            if name is None:
                cmap = colormap.ColorMap(None, [0.0, 1.0])
            else:
                cmap = colormap.get(name, source=source)
            self.setColorMap(cmap)
            self.showTicks(False)

    @addGradientListToDocstring()
    def loadPreset(self, name):
        if False:
            print('Hello World!')
        '\n        Load a predefined gradient. Currently defined gradients are: \n        '
        self.restoreState(Gradients[name])

    def setColorMode(self, cm):
        if False:
            while True:
                i = 10
        "\n        Set the color mode for the gradient. Options are: 'hsv', 'rgb'\n        \n        "
        if cm not in ['rgb', 'hsv']:
            raise Exception("Unknown color mode %s. Options are 'rgb' and 'hsv'." % str(cm))
        try:
            self.rgbAction.blockSignals(True)
            self.hsvAction.blockSignals(True)
            self.rgbAction.setChecked(cm == 'rgb')
            self.hsvAction.setChecked(cm == 'hsv')
        finally:
            self.rgbAction.blockSignals(False)
            self.hsvAction.blockSignals(False)
        self.colorMode = cm
        self.sigTicksChanged.emit(self)
        self.sigGradientChangeFinished.emit(self)

    def _setColorModeToRGB(self):
        if False:
            i = 10
            return i + 15
        self.setColorMode('rgb')

    def _setColorModeToHSV(self):
        if False:
            for i in range(10):
                print('nop')
        self.setColorMode('hsv')

    def colorMap(self):
        if False:
            print('Hello World!')
        'Return a ColorMap object representing the current state of the editor.'
        if self.colorMode == 'hsv':
            raise NotImplementedError('hsv colormaps not yet supported')
        pos = []
        color = []
        for (t, x) in self.listTicks():
            pos.append(x)
            c = t.color
            color.append(c.getRgb())
        return ColorMap(np.array(pos), np.array(color, dtype=np.ubyte))

    def updateGradient(self):
        if False:
            print('Hello World!')
        self.gradient = self.getGradient()
        self.gradRect.setBrush(QtGui.QBrush(self.gradient))
        self.sigGradientChanged.emit(self)

    def _updateGradientIgnoreArgs(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.updateGradient()

    def setLength(self, newLen):
        if False:
            while True:
                i = 10
        TickSliderItem.setLength(self, newLen)
        self.backgroundRect.setRect(1, -self.rectSize, newLen, self.rectSize)
        self.gradRect.setRect(1, -self.rectSize, newLen, self.rectSize)
        self.sigTicksChanged.emit(self)

    def currentColorChanged(self, color):
        if False:
            print('Hello World!')
        if color.isValid() and self.currentTick is not None:
            self.setTickColor(self.currentTick, color)

    def currentColorRejected(self):
        if False:
            while True:
                i = 10
        self.setTickColor(self.currentTick, self.currentTickColor)

    def currentColorAccepted(self):
        if False:
            print('Hello World!')
        self.sigGradientChangeFinished.emit(self)

    def tickClicked(self, tick, ev):
        if False:
            print('Hello World!')
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.raiseColorDialog(tick)
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.raiseTickContextMenu(tick, ev)

    def raiseColorDialog(self, tick):
        if False:
            return 10
        if not tick.colorChangeAllowed:
            return
        self.currentTick = tick
        self.currentTickColor = tick.color
        self.colorDialog.setCurrentColor(tick.color)
        self.colorDialog.open()

    def raiseTickContextMenu(self, tick, ev):
        if False:
            return 10
        self.tickMenu = TickMenu(tick, self)
        self.tickMenu.popup(ev.screenPos().toQPoint())

    def tickMoveFinished(self, tick):
        if False:
            return 10
        self.sigGradientChangeFinished.emit(self)

    def getGradient(self):
        if False:
            return 10
        'Return a QLinearGradient object.'
        g = QtGui.QLinearGradient(QtCore.QPointF(0, 0), QtCore.QPointF(self.length, 0))
        if self.colorMode == 'rgb':
            ticks = self.listTicks()
            g.setStops([(x, QtGui.QColor(t.color)) for (t, x) in ticks])
        elif self.colorMode == 'hsv':
            ticks = self.listTicks()
            stops = []
            stops.append((ticks[0][1], ticks[0][0].color))
            for i in range(1, len(ticks)):
                x1 = ticks[i - 1][1]
                x2 = ticks[i][1]
                dx = (x2 - x1) / 10.0
                for j in range(1, 10):
                    x = x1 + dx * j
                    stops.append((x, self.getColor(x)))
                stops.append((x2, self.getColor(x2)))
            g.setStops(stops)
        return g

    def getColor(self, x, toQColor=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a color for a given value.\n        \n        ==============  ==================================================================\n        **Arguments:**\n        x               Value (position on gradient) of requested color.\n        toQColor        If true, returns a QColor object, else returns a (r,g,b,a) tuple.\n        ==============  ==================================================================\n        '
        ticks = self.listTicks()
        if x <= ticks[0][1]:
            c = ticks[0][0].color
            if toQColor:
                return QtGui.QColor(c)
            else:
                return c.getRgb()
        if x >= ticks[-1][1]:
            c = ticks[-1][0].color
            if toQColor:
                return QtGui.QColor(c)
            else:
                return c.getRgb()
        x2 = ticks[0][1]
        for i in range(1, len(ticks)):
            x1 = x2
            x2 = ticks[i][1]
            if x1 <= x and x2 >= x:
                break
        dx = x2 - x1
        if dx == 0:
            f = 0.0
        else:
            f = (x - x1) / dx
        c1 = ticks[i - 1][0].color
        c2 = ticks[i][0].color
        if self.colorMode == 'rgb':
            r = c1.red() * (1.0 - f) + c2.red() * f
            g = c1.green() * (1.0 - f) + c2.green() * f
            b = c1.blue() * (1.0 - f) + c2.blue() * f
            a = c1.alpha() * (1.0 - f) + c2.alpha() * f
            if toQColor:
                return QtGui.QColor(int(r), int(g), int(b), int(a))
            else:
                return (r, g, b, a)
        elif self.colorMode == 'hsv':
            (h1, s1, v1, _) = c1.getHsv()
            (h2, s2, v2, _) = c2.getHsv()
            h = h1 * (1.0 - f) + h2 * f
            s = s1 * (1.0 - f) + s2 * f
            v = v1 * (1.0 - f) + v2 * f
            c = QtGui.QColor.fromHsv(int(h), int(s), int(v))
            if toQColor:
                return c
            else:
                return c.getRgb()

    def getLookupTable(self, nPts, alpha=None):
        if False:
            while True:
                i = 10
        '\n        Return an RGB(A) lookup table (ndarray). \n        \n        ==============  ============================================================================\n        **Arguments:**\n        nPts            The number of points in the returned lookup table.\n        alpha           True, False, or None - Specifies whether or not alpha values are included\n                        in the table.If alpha is None, alpha will be automatically determined.\n        ==============  ============================================================================\n        '
        if alpha is None:
            alpha = self.usesAlpha()
        if alpha:
            table = np.empty((nPts, 4), dtype=np.ubyte)
        else:
            table = np.empty((nPts, 3), dtype=np.ubyte)
        for i in range(nPts):
            x = float(i) / (nPts - 1)
            color = self.getColor(x, toQColor=False)
            table[i] = color[:table.shape[1]]
        return table

    def usesAlpha(self):
        if False:
            for i in range(10):
                print('nop')
        'Return True if any ticks have an alpha < 255'
        ticks = self.listTicks()
        for t in ticks:
            if t[0].color.alpha() < 255:
                return True
        return False

    def isLookupTrivial(self):
        if False:
            while True:
                i = 10
        'Return True if the gradient has exactly two stops in it: black at 0.0 and white at 1.0'
        ticks = self.listTicks()
        if len(ticks) != 2:
            return False
        if ticks[0][1] != 0.0 or ticks[1][1] != 1.0:
            return False
        c1 = ticks[0][0].color.getRgb()
        c2 = ticks[1][0].color.getRgb()
        if c1 != (0, 0, 0, 255) or c2 != (255, 255, 255, 255):
            return False
        return True

    def addTick(self, x, color=None, movable=True, finish=True):
        if False:
            return 10
        '\n        Add a tick to the gradient. Return the tick.\n        \n        ==============  ==================================================================\n        **Arguments:**\n        x               Position where tick should be added.\n        color           Color of added tick. If color is not specified, the color will be\n                        the color of the gradient at the specified position.\n        movable         Specifies whether the tick is movable with the mouse.\n        ==============  ==================================================================\n        '
        if color is None:
            color = self.getColor(x)
        t = TickSliderItem.addTick(self, x, color=color, movable=movable, finish=finish)
        t.colorChangeAllowed = True
        return t

    def saveState(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a dictionary with parameters for rebuilding the gradient. Keys will include:\n        \n           - 'mode': hsv or rgb\n           - 'ticks': a list of tuples (pos, (r,g,b,a))\n        "
        ticks = []
        for t in self.ticks:
            c = t.color
            ticks.append((self.ticks[t], c.getRgb()))
        state = {'mode': self.colorMode, 'ticks': ticks, 'ticksVisible': next(iter(self.ticks)).isVisible()}
        return state

    def restoreState(self, state):
        if False:
            return 10
        "\n        Restore the gradient specified in state.\n        \n        ==============  ====================================================================\n        **Arguments:**\n        state           A dictionary with same structure as those returned by\n                        :func:`saveState <pyqtgraph.GradientEditorItem.saveState>`\n                      \n                        Keys must include:\n                      \n                            - 'mode': hsv or rgb\n                            - 'ticks': a list of tuples (pos, (r,g,b,a))\n        ==============  ====================================================================\n        "
        signalsBlocked = self.blockSignals(True)
        self.setColorMode(state['mode'])
        for t in list(self.ticks.keys()):
            self.removeTick(t, finish=False)
        for t in state['ticks']:
            c = QtGui.QColor(*t[1])
            self.addTick(t[0], c, finish=False)
        self.showTicks(state.get('ticksVisible', next(iter(self.ticks)).isVisible()))
        self.blockSignals(signalsBlocked)
        self.sigTicksChanged.emit(self)
        self.sigGradientChangeFinished.emit(self)

    def setColorMap(self, cm):
        if False:
            print('Hello World!')
        signalsBlocked = self.blockSignals(True)
        self.setColorMode('rgb')
        for t in list(self.ticks.keys()):
            self.removeTick(t, finish=False)
        colors = cm.getColors(mode='qcolor')
        for i in range(len(cm.pos)):
            x = cm.pos[i]
            c = colors[i]
            self.addTick(x, c, finish=False)
        self.blockSignals(signalsBlocked)
        self.sigTicksChanged.emit(self)
        self.sigGradientChangeFinished.emit(self)

    def linkGradient(self, slaveGradient, connect=True):
        if False:
            while True:
                i = 10
        if connect:
            fn = lambda g, slave=slaveGradient: slave.restoreState(g.saveState())
            self.linkedGradients[id(slaveGradient)] = fn
            self.sigGradientChanged.connect(fn)
            self.sigGradientChanged.emit(self)
        else:
            fn = self.linkedGradients.get(id(slaveGradient), None)
            if fn:
                self.sigGradientChanged.disconnect(fn)

class Tick(QtWidgets.QGraphicsWidget):
    sigMoving = QtCore.Signal(object, object)
    sigMoved = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object, object)

    def __init__(self, pos, color, movable=True, scale=10, pen='w', removeAllowed=True):
        if False:
            i = 10
            return i + 15
        self.movable = movable
        self.moving = False
        self.scale = scale
        self.color = color
        self.pen = fn.mkPen(pen)
        self.hoverPen = fn.mkPen(255, 255, 0)
        self.currentPen = self.pen
        self.removeAllowed = removeAllowed
        self.pg = QtGui.QPainterPath(QtCore.QPointF(0, 0))
        self.pg.lineTo(QtCore.QPointF(-scale / 3 ** 0.5, scale))
        self.pg.lineTo(QtCore.QPointF(scale / 3 ** 0.5, scale))
        self.pg.closeSubpath()
        QtWidgets.QGraphicsWidget.__init__(self)
        self.setPos(pos[0], pos[1])
        if self.movable:
            self.setZValue(1)
        else:
            self.setZValue(0)

    def boundingRect(self):
        if False:
            for i in range(10):
                print('nop')
        return self.pg.boundingRect()

    def shape(self):
        if False:
            print('Hello World!')
        return self.pg

    def paint(self, p, *args):
        if False:
            i = 10
            return i + 15
        p.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillPath(self.pg, fn.mkBrush(self.color))
        p.setPen(self.currentPen)
        p.drawPath(self.pg)

    def mouseDragEvent(self, ev):
        if False:
            while True:
                i = 10
        if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
            ev.accept()
            if not self.moving:
                return
            newPos = self.cursorOffset + self.mapToParent(ev.pos())
            newPos.setY(self.pos().y())
            self.setPos(newPos)
            self.sigMoving.emit(self, newPos)
            if ev.isFinish():
                self.moving = False
                self.sigMoved.emit(self)

    def mouseClickEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        ev.accept()
        if ev.button() == QtCore.Qt.MouseButton.RightButton and self.moving:
            self.setPos(self.startPosition)
            self.moving = False
            self.sigMoving.emit(self, self.startPosition)
            self.sigMoved.emit(self)
        else:
            self.sigClicked.emit(self, ev)

    def hoverEvent(self, ev):
        if False:
            return 10
        if not ev.isExit() and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
            ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton)
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)
            self.currentPen = self.hoverPen
        else:
            self.currentPen = self.pen
        self.update()

class TickMenu(QtWidgets.QMenu):

    def __init__(self, tick, sliderItem):
        if False:
            i = 10
            return i + 15
        QtWidgets.QMenu.__init__(self)
        self.tick = weakref.ref(tick)
        self.sliderItem = weakref.ref(sliderItem)
        self.removeAct = self.addAction(translate('GradientEditorItem', 'Remove Tick'), lambda : self.sliderItem().removeTick(tick))
        if not self.tick().removeAllowed or len(self.sliderItem().ticks) < 3:
            self.removeAct.setEnabled(False)
        positionMenu = self.addMenu(translate('GradientEditorItem', 'Set Position'))
        w = QtWidgets.QWidget()
        l = QtWidgets.QGridLayout()
        w.setLayout(l)
        value = sliderItem.tickValue(tick)
        self.fracPosSpin = SpinBox()
        self.fracPosSpin.setOpts(value=value, bounds=(0.0, 1.0), step=0.01, decimals=2)
        l.addWidget(QtWidgets.QLabel(f"{translate('GradiantEditorItem', 'Position')}:"), 0, 0)
        l.addWidget(self.fracPosSpin, 0, 1)
        a = QtWidgets.QWidgetAction(self)
        a.setDefaultWidget(w)
        positionMenu.addAction(a)
        self.fracPosSpin.sigValueChanging.connect(self.fractionalValueChanged)
        colorAct = self.addAction(translate('Context Menu', 'Set Color'), lambda : self.sliderItem().raiseColorDialog(self.tick()))
        if not self.tick().colorChangeAllowed:
            colorAct.setEnabled(False)

    def fractionalValueChanged(self, x):
        if False:
            while True:
                i = 10
        self.sliderItem().setTickValue(self.tick(), self.fracPosSpin.value())