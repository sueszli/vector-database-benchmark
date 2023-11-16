import warnings
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop

class Dock(QtWidgets.QWidget):
    sigStretchChanged = QtCore.Signal()
    sigClosed = QtCore.Signal(object)

    def __init__(self, name, area=None, size=(10, 10), widget=None, hideTitle=False, autoOrientation=True, label=None, **kargs):
        if False:
            print('Hello World!')
        QtWidgets.QWidget.__init__(self)
        self.dockdrop = DockDrop(self)
        self._container = None
        self._name = name
        self.area = area
        self.label = label
        if self.label is None:
            self.label = DockLabel(name, **kargs)
        self.label.dock = self
        if self.label.isClosable():
            self.label.sigCloseClicked.connect(self.close)
        self.labelHidden = False
        self.moveLabel = True
        self.autoOrient = autoOrientation
        self.orientation = 'horizontal'
        self.topLayout = QtWidgets.QGridLayout()
        self.topLayout.setContentsMargins(0, 0, 0, 0)
        self.topLayout.setSpacing(0)
        self.setLayout(self.topLayout)
        self.topLayout.addWidget(self.label, 0, 1)
        self.widgetArea = QtWidgets.QWidget()
        self.topLayout.addWidget(self.widgetArea, 1, 1)
        self.layout = QtWidgets.QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.widgetArea.setLayout(self.layout)
        self.widgetArea.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.widgets = []
        self.currentRow = 0
        self.dockdrop.raiseOverlay()
        self.hStyle = '\n        Dock > QWidget {\n            border: 1px solid #000;\n            border-radius: 5px;\n            border-top-left-radius: 0px;\n            border-top-right-radius: 0px;\n            border-top-width: 0px;\n        }'
        self.vStyle = '\n        Dock > QWidget {\n            border: 1px solid #000;\n            border-radius: 5px;\n            border-top-left-radius: 0px;\n            border-bottom-left-radius: 0px;\n            border-left-width: 0px;\n        }'
        self.nStyle = '\n        Dock > QWidget {\n            border: 1px solid #000;\n            border-radius: 5px;\n        }'
        self.dragStyle = '\n        Dock > QWidget {\n            border: 4px solid #00F;\n            border-radius: 5px;\n        }'
        self.setAutoFillBackground(False)
        self.widgetArea.setStyleSheet(self.hStyle)
        self.setStretch(*size)
        if widget is not None:
            self.addWidget(widget)
        if hideTitle:
            self.hideTitleBar()

    def implements(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        if name is None:
            return ['dock']
        else:
            return name == 'dock'

    def setStretch(self, x=None, y=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set the 'target' size for this Dock.\n        The actual size will be determined by comparing this Dock's\n        stretch value to the rest of the docks it shares space with.\n        "
        if x is None:
            x = 0
        if y is None:
            y = 0
        self._stretch = (x, y)
        self.sigStretchChanged.emit()

    def stretch(self):
        if False:
            while True:
                i = 10
        return self._stretch

    def hideTitleBar(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Hide the title bar for this Dock.\n        This will prevent the Dock being moved by the user.\n        '
        self.label.hide()
        self.labelHidden = True
        self.dockdrop.removeAllowedArea('center')
        self.updateStyle()

    def showTitleBar(self):
        if False:
            print('Hello World!')
        '\n        Show the title bar for this Dock.\n        '
        self.label.show()
        self.labelHidden = False
        self.dockdrop.addAllowedArea('center')
        self.updateStyle()

    def title(self):
        if False:
            return 10
        '\n        Gets the text displayed in the title bar for this dock.\n        '
        return self.label.text()

    def setTitle(self, text):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the text displayed in title bar for this Dock.\n        '
        self.label.setText(text)

    def setOrientation(self, o='auto', force=False):
        if False:
            i = 10
            return i + 15
        "\n        Sets the orientation of the title bar for this Dock.\n        Must be one of 'auto', 'horizontal', or 'vertical'.\n        By default ('auto'), the orientation is determined\n        based on the aspect ratio of the Dock.\n        "
        if self.container() is None:
            return
        if o == 'auto' and self.autoOrient:
            if self.container().type() == 'tab':
                o = 'horizontal'
            elif self.width() > self.height() * 1.5:
                o = 'vertical'
            else:
                o = 'horizontal'
        if force or self.orientation != o:
            self.orientation = o
            self.label.setOrientation(o)
            self.updateStyle()

    def updateStyle(self):
        if False:
            i = 10
            return i + 15
        if self.labelHidden:
            self.widgetArea.setStyleSheet(self.nStyle)
        elif self.orientation == 'vertical':
            self.label.setOrientation('vertical')
            if self.moveLabel:
                self.topLayout.addWidget(self.label, 1, 0)
            self.widgetArea.setStyleSheet(self.vStyle)
        else:
            self.label.setOrientation('horizontal')
            if self.moveLabel:
                self.topLayout.addWidget(self.label, 0, 1)
            self.widgetArea.setStyleSheet(self.hStyle)

    def resizeEvent(self, ev):
        if False:
            return 10
        self.setOrientation()
        self.dockdrop.resizeOverlay(self.size())

    def name(self):
        if False:
            print('Hello World!')
        return self._name

    def addWidget(self, widget, row=None, col=0, rowspan=1, colspan=1):
        if False:
            while True:
                i = 10
        '\n        Add a new widget to the interior of this Dock.\n        Each Dock uses a QGridLayout to arrange widgets within.\n        '
        if row is None:
            row = self.currentRow
        self.currentRow = max(row + 1, self.currentRow)
        self.widgets.append(widget)
        self.layout.addWidget(widget, row, col, rowspan, colspan)
        self.dockdrop.raiseOverlay()

    def startDrag(self):
        if False:
            for i in range(10):
                print('nop')
        self.drag = QtGui.QDrag(self)
        mime = QtCore.QMimeData()
        self.drag.setMimeData(mime)
        self.widgetArea.setStyleSheet(self.dragStyle)
        self.update()
        action = self.drag.exec() if hasattr(self.drag, 'exec') else self.drag.exec_()
        self.updateStyle()

    def float(self):
        if False:
            for i in range(10):
                print('nop')
        self.area.floatDock(self)

    def container(self):
        if False:
            return 10
        return self._container

    def containerChanged(self, c):
        if False:
            print('Hello World!')
        if self._container is not None:
            self._container.apoptose()
        self._container = c
        if c is None:
            self.area = None
        else:
            self.area = c.area
            if c.type() != 'tab':
                self.moveLabel = True
                self.label.setDim(False)
            else:
                self.moveLabel = False
            self.setOrientation(force=True)

    def raiseDock(self):
        if False:
            while True:
                i = 10
        'If this Dock is stacked underneath others, raise it to the top.'
        self.container().raiseDock(self)

    def close(self):
        if False:
            print('Hello World!')
        'Remove this dock from the DockArea it lives inside.'
        if self._container is None:
            warnings.warn(f'Cannot close dock {self} because it is not open.', RuntimeWarning, stacklevel=2)
            return
        self.setParent(None)
        QtWidgets.QLabel.close(self.label)
        self.label.setParent(None)
        self._container.apoptose()
        self._container = None
        self.sigClosed.emit(self)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<Dock %s %s>' % (self.name(), self.stretch())

    def dragEnterEvent(self, *args):
        if False:
            i = 10
            return i + 15
        self.dockdrop.dragEnterEvent(*args)

    def dragMoveEvent(self, *args):
        if False:
            print('Hello World!')
        self.dockdrop.dragMoveEvent(*args)

    def dragLeaveEvent(self, *args):
        if False:
            return 10
        self.dockdrop.dragLeaveEvent(*args)

    def dropEvent(self, *args):
        if False:
            return 10
        self.dockdrop.dropEvent(*args)

class DockLabel(VerticalLabel):
    sigClicked = QtCore.Signal(object, object)
    sigCloseClicked = QtCore.Signal()

    def __init__(self, text, closable=False, fontSize='12px'):
        if False:
            return 10
        self.dim = False
        self.fixedWidth = False
        self.fontSize = fontSize
        VerticalLabel.__init__(self, text, orientation='horizontal', forceWidth=False)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.dock = None
        self.updateStyle()
        self.setAutoFillBackground(False)
        self.mouseMoved = False
        self.closeButton = None
        if closable:
            self.closeButton = QtWidgets.QToolButton(self)
            self.closeButton.clicked.connect(self.sigCloseClicked)
            self.closeButton.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarCloseButton))

    def updateStyle(self):
        if False:
            return 10
        r = '3px'
        if self.dim:
            fg = '#aaa'
            bg = '#44a'
            border = '#339'
        else:
            fg = '#fff'
            bg = '#66c'
            border = '#55B'
        if self.orientation == 'vertical':
            self.vStyle = 'DockLabel {\n                background-color : %s;\n                color : %s;\n                border-top-right-radius: 0px;\n                border-top-left-radius: %s;\n                border-bottom-right-radius: 0px;\n                border-bottom-left-radius: %s;\n                border-width: 0px;\n                border-right: 2px solid %s;\n                padding-top: 3px;\n                padding-bottom: 3px;\n                font-size: %s;\n            }' % (bg, fg, r, r, border, self.fontSize)
            self.setStyleSheet(self.vStyle)
        else:
            self.hStyle = 'DockLabel {\n                background-color : %s;\n                color : %s;\n                border-top-right-radius: %s;\n                border-top-left-radius: %s;\n                border-bottom-right-radius: 0px;\n                border-bottom-left-radius: 0px;\n                border-width: 0px;\n                border-bottom: 2px solid %s;\n                padding-left: 3px;\n                padding-right: 3px;\n                font-size: %s;\n            }' % (bg, fg, r, r, border, self.fontSize)
            self.setStyleSheet(self.hStyle)

    def setDim(self, d):
        if False:
            print('Hello World!')
        if self.dim != d:
            self.dim = d
            self.updateStyle()

    def setOrientation(self, o):
        if False:
            i = 10
            return i + 15
        VerticalLabel.setOrientation(self, o)
        self.updateStyle()

    def isClosable(self):
        if False:
            for i in range(10):
                print('nop')
        return self.closeButton is not None

    def mousePressEvent(self, ev):
        if False:
            print('Hello World!')
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.pressPos = lpos
        self.mouseMoved = False
        ev.accept()

    def mouseMoveEvent(self, ev):
        if False:
            return 10
        if not self.mouseMoved:
            lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
            self.mouseMoved = (lpos - self.pressPos).manhattanLength() > QtWidgets.QApplication.startDragDistance()
        if self.mouseMoved and ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            self.dock.startDrag()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if False:
            return 10
        ev.accept()
        if not self.mouseMoved:
            self.sigClicked.emit(self, ev)

    def mouseDoubleClickEvent(self, ev):
        if False:
            while True:
                i = 10
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.dock.float()

    def resizeEvent(self, ev):
        if False:
            i = 10
            return i + 15
        if self.closeButton:
            if self.orientation == 'vertical':
                size = ev.size().width()
                pos = QtCore.QPoint(0, 0)
            else:
                size = ev.size().height()
                pos = QtCore.QPoint(ev.size().width() - size, 0)
            self.closeButton.setFixedSize(QtCore.QSize(size, size))
            self.closeButton.move(pos)
        super(DockLabel, self).resizeEvent(ev)