__all__ = ['Terminal', 'TerminalGraphicsItem']
import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
translate = QtCore.QCoreApplication.translate

class Terminal(object):

    def __init__(self, node, name, io, optional=False, multi=False, pos=None, renamable=False, removable=False, multiable=False, bypass=None):
        if False:
            i = 10
            return i + 15
        "\n        Construct a new terminal. \n        \n        ==============  =================================================================================\n        **Arguments:**\n        node            the node to which this terminal belongs\n        name            string, the name of the terminal\n        io              'in' or 'out'\n        optional        bool, whether the node may process without connection to this terminal\n        multi           bool, for inputs: whether this terminal may make multiple connections\n                        for outputs: whether this terminal creates a different value for each connection\n        pos             [x, y], the position of the terminal within its node's boundaries\n        renamable       (bool) Whether the terminal can be renamed by the user\n        removable       (bool) Whether the terminal can be removed by the user\n        multiable       (bool) Whether the user may toggle the *multi* option for this terminal\n        bypass          (str) Name of the terminal from which this terminal's value is derived\n                        when the Node is in bypass mode.\n        ==============  =================================================================================\n        "
        self._io = io
        self._optional = optional
        self._multi = multi
        self._node = weakref.ref(node)
        self._name = name
        self._renamable = renamable
        self._removable = removable
        self._multiable = multiable
        self._connections = {}
        self._graphicsItem = TerminalGraphicsItem(self, parent=self._node().graphicsItem())
        self._bypass = bypass
        if multi:
            self._value = {}
        else:
            self._value = None
        self.valueOk = None
        self.recolor()

    def value(self, term=None):
        if False:
            for i in range(10):
                print('nop')
        'Return the value this terminal provides for the connected terminal'
        if term is None:
            return self._value
        if self.isMultiValue():
            return self._value.get(term, None)
        else:
            return self._value

    def bypassValue(self):
        if False:
            print('Hello World!')
        return self._bypass

    def setValue(self, val, process=True):
        if False:
            i = 10
            return i + 15
        'If this is a single-value terminal, val should be a single value.\n        If this is a multi-value terminal, val should be a dict of terminal:value pairs'
        if not self.isMultiValue():
            if fn.eq(val, self._value):
                return
            self._value = val
        else:
            if not isinstance(self._value, dict):
                self._value = {}
            if val is not None:
                self._value.update(val)
        self.setValueAcceptable(None)
        if self.isInput() and process:
            self.node().update()
        self.recolor()

    def setOpts(self, **opts):
        if False:
            return 10
        self._renamable = opts.get('renamable', self._renamable)
        self._removable = opts.get('removable', self._removable)
        self._multiable = opts.get('multiable', self._multiable)
        if 'multi' in opts:
            self.setMultiValue(opts['multi'])

    def connected(self, term):
        if False:
            i = 10
            return i + 15
        'Called whenever this terminal has been connected to another. (note--this function is called on both terminals)'
        if self.isInput() and term.isOutput():
            self.inputChanged(term)
        if self.isOutput() and self.isMultiValue():
            self.node().update()
        self.node().connected(self, term)

    def disconnected(self, term):
        if False:
            print('Hello World!')
        'Called whenever this terminal has been disconnected from another. (note--this function is called on both terminals)'
        if self.isMultiValue() and term in self._value:
            del self._value[term]
            self.node().update()
        elif self.isInput():
            self.setValue(None)
        self.node().disconnected(self, term)

    def inputChanged(self, term, process=True):
        if False:
            print('Hello World!')
        'Called whenever there is a change to the input value to this terminal.\n        It may often be useful to override this function.'
        if self.isMultiValue():
            self.setValue({term: term.value(self)}, process=process)
        else:
            self.setValue(term.value(self), process=process)

    def valueIsAcceptable(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns True->acceptable  None->unknown  False->Unacceptable'
        return self.valueOk

    def setValueAcceptable(self, v=True):
        if False:
            print('Hello World!')
        self.valueOk = v
        self.recolor()

    def connections(self):
        if False:
            while True:
                i = 10
        return self._connections

    def node(self):
        if False:
            for i in range(10):
                print('nop')
        return self._node()

    def isInput(self):
        if False:
            while True:
                i = 10
        return self._io == 'in'

    def isMultiValue(self):
        if False:
            return 10
        return self._multi

    def setMultiValue(self, multi):
        if False:
            return 10
        'Set whether this is a multi-value terminal.'
        self._multi = multi
        if not multi and len(self.inputTerminals()) > 1:
            self.disconnectAll()
        for term in self.inputTerminals():
            self.inputChanged(term)

    def isOutput(self):
        if False:
            while True:
                i = 10
        return self._io == 'out'

    def isRenamable(self):
        if False:
            for i in range(10):
                print('nop')
        return self._renamable

    def isRemovable(self):
        if False:
            for i in range(10):
                print('nop')
        return self._removable

    def isMultiable(self):
        if False:
            print('Hello World!')
        return self._multiable

    def name(self):
        if False:
            while True:
                i = 10
        return self._name

    def graphicsItem(self):
        if False:
            for i in range(10):
                print('nop')
        return self._graphicsItem

    def isConnected(self):
        if False:
            return 10
        return len(self.connections()) > 0

    def connectedTo(self, term):
        if False:
            i = 10
            return i + 15
        return term in self.connections()

    def hasInput(self):
        if False:
            while True:
                i = 10
        for t in self.connections():
            if t.isOutput():
                return True
        return False

    def inputTerminals(self):
        if False:
            while True:
                i = 10
        'Return the terminal(s) that give input to this one.'
        return [t for t in self.connections() if t.isOutput()]

    def dependentNodes(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the list of nodes which receive input from this terminal.'
        return set([t.node() for t in self.connections() if t.isInput()])

    def connectTo(self, term, connectionItem=None):
        if False:
            while True:
                i = 10
        try:
            if self.connectedTo(term):
                raise Exception('Already connected')
            if term is self:
                raise Exception('Not connecting terminal to self')
            if term.node() is self.node():
                raise Exception("Can't connect to terminal on same node.")
            for t in [self, term]:
                if t.isInput() and (not t._multi) and (len(t.connections()) > 0):
                    raise Exception('Cannot connect %s <-> %s: Terminal %s is already connected to %s (and does not allow multiple connections)' % (self, term, t, list(t.connections().keys())))
        except:
            if connectionItem is not None:
                connectionItem.close()
            raise
        if connectionItem is None:
            connectionItem = ConnectionItem(self.graphicsItem(), term.graphicsItem())
            self.graphicsItem().getViewBox().addItem(connectionItem)
        self._connections[term] = connectionItem
        term._connections[self] = connectionItem
        self.recolor()
        self.connected(term)
        term.connected(self)
        return connectionItem

    def disconnectFrom(self, term):
        if False:
            return 10
        if not self.connectedTo(term):
            return
        item = self._connections[term]
        item.close()
        del self._connections[term]
        del term._connections[self]
        self.recolor()
        term.recolor()
        self.disconnected(term)
        term.disconnected(self)

    def disconnectAll(self):
        if False:
            i = 10
            return i + 15
        for t in list(self._connections.keys()):
            self.disconnectFrom(t)

    def recolor(self, color=None, recurse=True):
        if False:
            i = 10
            return i + 15
        if color is None:
            if not self.isConnected():
                color = QtGui.QColor(0, 0, 0)
            elif self.isInput() and (not self.hasInput()):
                color = QtGui.QColor(200, 200, 0)
            elif self._value is None or fn.eq(self._value, {}):
                color = QtGui.QColor(255, 255, 255)
            elif self.valueIsAcceptable() is None:
                color = QtGui.QColor(200, 200, 0)
            elif self.valueIsAcceptable() is True:
                color = QtGui.QColor(0, 200, 0)
            else:
                color = QtGui.QColor(200, 0, 0)
        self.graphicsItem().setBrush(QtGui.QBrush(color))
        if recurse:
            for t in self.connections():
                t.recolor(color, recurse=False)

    def rename(self, name):
        if False:
            i = 10
            return i + 15
        oldName = self._name
        self._name = name
        self.node().terminalRenamed(self, oldName)
        self.graphicsItem().termRenamed(name)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<Terminal %s.%s>' % (str(self.node().name()), str(self.name()))

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return id(self)

    def close(self):
        if False:
            while True:
                i = 10
        self.disconnectAll()
        item = self.graphicsItem()
        if item.scene() is not None:
            item.scene().removeItem(item)

    def saveState(self):
        if False:
            print('Hello World!')
        return {'io': self._io, 'multi': self._multi, 'optional': self._optional, 'renamable': self._renamable, 'removable': self._removable, 'multiable': self._multiable}

    def __lt__(self, other):
        if False:
            return 10
        'When the terminal is multi value, the data passed to the DatTreeWidget for each input or output, is {Terminal: value}.\n        To make this sortable, we provide the < operator.\n        '
        return self._name < other._name

class TextItem(QtWidgets.QGraphicsTextItem):

    def __init__(self, text, parent, on_update):
        if False:
            return 10
        super().__init__(text, parent)
        self.on_update = on_update

    def focusOutEvent(self, ev):
        if False:
            i = 10
            return i + 15
        super().focusOutEvent(ev)
        if self.on_update is not None:
            self.on_update()

    def keyPressEvent(self, ev):
        if False:
            i = 10
            return i + 15
        if ev.key() == QtCore.Qt.Key.Key_Enter or ev.key() == QtCore.Qt.Key.Key_Return:
            if self.on_update is not None:
                self.on_update()
                return
        super().keyPressEvent(ev)

class TerminalGraphicsItem(GraphicsObject):

    def __init__(self, term, parent=None):
        if False:
            while True:
                i = 10
        self.term = term
        GraphicsObject.__init__(self, parent)
        self.brush = fn.mkBrush(0, 0, 0)
        self.box = QtWidgets.QGraphicsRectItem(0, 0, 10, 10, self)
        on_update = self.labelChanged if self.term.isRenamable() else None
        self.label = TextItem(self.term.name(), self, on_update)
        self.label.setScale(0.7)
        self.newConnection = None
        self.setFiltersChildEvents(True)
        if self.term.isRenamable():
            self.label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextEditorInteraction)
        self.setZValue(1)
        self.menu = None

    def labelChanged(self):
        if False:
            for i in range(10):
                print('nop')
        newName = self.label.toPlainText()
        if newName != self.term.name():
            self.term.rename(newName)

    def termRenamed(self, name):
        if False:
            while True:
                i = 10
        self.label.setPlainText(name)

    def setBrush(self, brush):
        if False:
            while True:
                i = 10
        self.brush = brush
        self.box.setBrush(brush)

    def disconnect(self, target):
        if False:
            i = 10
            return i + 15
        self.term.disconnectFrom(target.term)

    def boundingRect(self):
        if False:
            return 10
        br = self.box.mapRectToParent(self.box.boundingRect())
        lr = self.label.mapRectToParent(self.label.boundingRect())
        return br | lr

    def paint(self, p, *args):
        if False:
            while True:
                i = 10
        pass

    def setAnchor(self, x, y):
        if False:
            while True:
                i = 10
        pos = QtCore.QPointF(x, y)
        self.anchorPos = pos
        br = self.box.mapRectToParent(self.box.boundingRect())
        lr = self.label.mapRectToParent(self.label.boundingRect())
        if self.term.isInput():
            self.box.setPos(pos.x(), pos.y() - br.height() / 2.0)
            self.label.setPos(pos.x() + br.width(), pos.y() - lr.height() / 2.0)
        else:
            self.box.setPos(pos.x() - br.width(), pos.y() - br.height() / 2.0)
            self.label.setPos(pos.x() - br.width() - lr.width(), pos.y() - lr.height() / 2.0)
        self.updateConnections()

    def updateConnections(self):
        if False:
            for i in range(10):
                print('nop')
        for (t, c) in self.term.connections().items():
            c.updateLine()

    def mousePressEvent(self, ev):
        if False:
            while True:
                i = 10
        ev.ignore()

    def mouseClickEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.label.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.raiseContextMenu(ev)

    def raiseContextMenu(self, ev):
        if False:
            i = 10
            return i + 15
        menu = self.getMenu()
        menu = self.scene().addParentContextMenus(self, menu, ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))

    def getMenu(self):
        if False:
            return 10
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            self.menu.setTitle(translate('Context Menu', 'Terminal'))
            remAct = QtGui.QAction(translate('Context Menu', 'Remove terminal'), self.menu)
            remAct.triggered.connect(self.removeSelf)
            self.menu.addAction(remAct)
            self.menu.remAct = remAct
            if not self.term.isRemovable():
                remAct.setEnabled(False)
            multiAct = QtGui.QAction(translate('Context Menu', 'Multi-value'), self.menu)
            multiAct.setCheckable(True)
            multiAct.setChecked(self.term.isMultiValue())
            multiAct.setEnabled(self.term.isMultiable())
            multiAct.triggered.connect(self.toggleMulti)
            self.menu.addAction(multiAct)
            self.menu.multiAct = multiAct
            if self.term.isMultiable():
                multiAct.setEnabled = False
        return self.menu

    def toggleMulti(self):
        if False:
            return 10
        multi = self.menu.multiAct.isChecked()
        self.term.setMultiValue(multi)

    def removeSelf(self):
        if False:
            for i in range(10):
                print('nop')
        self.term.node().removeTerminal(self.term)

    def mouseDragEvent(self, ev):
        if False:
            i = 10
            return i + 15
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()
        if ev.isStart():
            if self.newConnection is None:
                self.newConnection = ConnectionItem(self)
                self.getViewBox().addItem(self.newConnection)
            self.newConnection.setTarget(self.mapToView(ev.pos()))
        elif ev.isFinish():
            if self.newConnection is not None:
                items = self.scene().items(ev.scenePos())
                gotTarget = False
                for i in items:
                    if isinstance(i, TerminalGraphicsItem):
                        self.newConnection.setTarget(i)
                        try:
                            self.term.connectTo(i.term, self.newConnection)
                            gotTarget = True
                        except:
                            self.scene().removeItem(self.newConnection)
                            self.newConnection = None
                            raise
                        break
                if not gotTarget:
                    self.newConnection.close()
                self.newConnection = None
        elif self.newConnection is not None:
            self.newConnection.setTarget(self.mapToView(ev.pos()))

    def hoverEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        if not ev.isExit() and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
            ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton)
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)
            self.box.setBrush(fn.mkBrush('w'))
        else:
            self.box.setBrush(self.brush)
        self.update()

    def connectPoint(self):
        if False:
            i = 10
            return i + 15
        return self.mapToView(self.mapFromItem(self.box, self.box.boundingRect().center()))

    def nodeMoved(self):
        if False:
            while True:
                i = 10
        for (t, item) in self.term.connections().items():
            item.updateLine()

class ConnectionItem(GraphicsObject):

    def __init__(self, source, target=None):
        if False:
            i = 10
            return i + 15
        GraphicsObject.__init__(self)
        self.setFlags(self.GraphicsItemFlag.ItemIsSelectable | self.GraphicsItemFlag.ItemIsFocusable)
        self.source = source
        self.target = target
        self.length = 0
        self.hovered = False
        self.path = None
        self.shapePath = None
        self.style = {'shape': 'line', 'color': (100, 100, 250), 'width': 1.0, 'hoverColor': (150, 150, 250), 'hoverWidth': 1.0, 'selectedColor': (200, 200, 0), 'selectedWidth': 3.0}
        self.source.getViewBox().addItem(self)
        self.updateLine()
        self.setZValue(0)

    def close(self):
        if False:
            while True:
                i = 10
        if self.scene() is not None:
            self.scene().removeItem(self)

    def setTarget(self, target):
        if False:
            return 10
        self.target = target
        self.updateLine()

    def setStyle(self, **kwds):
        if False:
            i = 10
            return i + 15
        self.style.update(kwds)
        if 'shape' in kwds:
            self.updateLine()
        else:
            self.update()

    def updateLine(self):
        if False:
            print('Hello World!')
        start = Point(self.source.connectPoint())
        if isinstance(self.target, TerminalGraphicsItem):
            stop = Point(self.target.connectPoint())
        elif isinstance(self.target, QtCore.QPointF):
            stop = Point(self.target)
        else:
            return
        self.prepareGeometryChange()
        self.path = self.generatePath(start, stop)
        self.shapePath = None
        self.update()

    def generatePath(self, start, stop):
        if False:
            for i in range(10):
                print('nop')
        path = QtGui.QPainterPath()
        path.moveTo(start)
        if self.style['shape'] == 'line':
            path.lineTo(stop)
        elif self.style['shape'] == 'cubic':
            path.cubicTo(Point(stop.x(), start.y()), Point(start.x(), stop.y()), Point(stop.x(), stop.y()))
        else:
            raise Exception('Invalid shape "%s"; options are "line" or "cubic"' % self.style['shape'])
        return path

    def keyPressEvent(self, ev):
        if False:
            while True:
                i = 10
        if not self.isSelected():
            ev.ignore()
            return
        if ev.key() == QtCore.Qt.Key.Key_Delete or ev.key() == QtCore.Qt.Key.Key_Backspace:
            self.source.disconnect(self.target)
            ev.accept()
        else:
            ev.ignore()

    def mousePressEvent(self, ev):
        if False:
            print('Hello World!')
        ev.ignore()

    def mouseClickEvent(self, ev):
        if False:
            while True:
                i = 10
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            sel = self.isSelected()
            self.setSelected(True)
            self.setFocus()
            if not sel and self.isSelected():
                self.update()

    def hoverEvent(self, ev):
        if False:
            i = 10
            return i + 15
        if not ev.isExit() and ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton):
            self.hovered = True
        else:
            self.hovered = False
        self.update()

    def boundingRect(self):
        if False:
            for i in range(10):
                print('nop')
        return self.shape().boundingRect()

    def viewRangeChanged(self):
        if False:
            return 10
        self.shapePath = None
        self.prepareGeometryChange()

    def shape(self):
        if False:
            i = 10
            return i + 15
        if self.shapePath is None:
            if self.path is None:
                return QtGui.QPainterPath()
            stroker = QtGui.QPainterPathStroker()
            px = self.pixelWidth()
            stroker.setWidth(px * 8)
            self.shapePath = stroker.createStroke(self.path)
        return self.shapePath

    def paint(self, p, *args):
        if False:
            while True:
                i = 10
        if self.isSelected():
            p.setPen(fn.mkPen(self.style['selectedColor'], width=self.style['selectedWidth']))
        elif self.hovered:
            p.setPen(fn.mkPen(self.style['hoverColor'], width=self.style['hoverWidth']))
        else:
            p.setPen(fn.mkPen(self.style['color'], width=self.style['width']))
        p.drawPath(self.path)