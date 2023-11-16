from ..Qt import QtCore, QtGui, QtWidgets
from .PathButton import PathButton
__all__ = ['GroupBox']

class GroupBox(QtWidgets.QGroupBox):
    """Subclass of QGroupBox that implements collapse handle.
    """
    sigCollapseChanged = QtCore.Signal(object)

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        QtWidgets.QGroupBox.__init__(self, *args)
        self._collapsed = False
        self._lastSizePlocy = self.sizePolicy()
        self.closePath = QtGui.QPainterPath()
        self.closePath.moveTo(0, -1)
        self.closePath.lineTo(0, 1)
        self.closePath.lineTo(1, 0)
        self.closePath.lineTo(0, -1)
        self.openPath = QtGui.QPainterPath()
        self.openPath.moveTo(-1, 0)
        self.openPath.lineTo(1, 0)
        self.openPath.lineTo(0, 1)
        self.openPath.lineTo(-1, 0)
        self.collapseBtn = PathButton(path=self.openPath, size=(12, 12), margin=0)
        self.collapseBtn.setStyleSheet('\n            border: none;\n        ')
        self.collapseBtn.setPen('k')
        self.collapseBtn.setBrush('w')
        self.collapseBtn.setParent(self)
        self.collapseBtn.move(3, 3)
        self.collapseBtn.setFlat(True)
        self.collapseBtn.clicked.connect(self.toggleCollapsed)
        if len(args) > 0 and isinstance(args[0], str):
            self.setTitle(args[0])

    def toggleCollapsed(self):
        if False:
            i = 10
            return i + 15
        self.setCollapsed(not self._collapsed)

    def collapsed(self):
        if False:
            for i in range(10):
                print('nop')
        return self._collapsed

    def setCollapsed(self, c):
        if False:
            return 10
        if c == self._collapsed:
            return
        if c is True:
            self.collapseBtn.setPath(self.closePath)
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred, closing=True)
        elif c is False:
            self.collapseBtn.setPath(self.openPath)
            self.setSizePolicy(self._lastSizePolicy)
        else:
            raise TypeError('Invalid argument %r; must be bool.' % c)
        for ch in self.children():
            if isinstance(ch, QtWidgets.QWidget) and ch is not self.collapseBtn:
                ch.setVisible(not c)
        self._collapsed = c
        self.sigCollapseChanged.emit(c)

    def setSizePolicy(self, *args, **kwds):
        if False:
            while True:
                i = 10
        QtWidgets.QGroupBox.setSizePolicy(self, *args)
        if kwds.pop('closing', False) is True:
            self._lastSizePolicy = self.sizePolicy()

    def setHorizontalPolicy(self, *args):
        if False:
            i = 10
            return i + 15
        QtWidgets.QGroupBox.setHorizontalPolicy(self, *args)
        self._lastSizePolicy = self.sizePolicy()

    def setVerticalPolicy(self, *args):
        if False:
            for i in range(10):
                print('nop')
        QtWidgets.QGroupBox.setVerticalPolicy(self, *args)
        self._lastSizePolicy = self.sizePolicy()

    def setTitle(self, title):
        if False:
            for i in range(10):
                print('nop')
        QtWidgets.QGroupBox.setTitle(self, '   ' + title)

    def widgetGroupInterface(self):
        if False:
            return 10
        return (self.sigCollapseChanged, GroupBox.collapsed, GroupBox.setCollapsed, True)