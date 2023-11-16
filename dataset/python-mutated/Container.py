__all__ = ['Container', 'HContainer', 'VContainer', 'TContainer']
import weakref
from ..Qt import QtCore, QtWidgets
from .Dock import Dock

class Container(object):

    def __init__(self, area):
        if False:
            while True:
                i = 10
        object.__init__(self)
        self.area = area
        self._container = None
        self._stretch = (10, 10)
        self.stretches = weakref.WeakKeyDictionary()

    def container(self):
        if False:
            while True:
                i = 10
        return self._container

    def containerChanged(self, c):
        if False:
            print('Hello World!')
        self._container = c
        if c is None:
            self.area = None
        else:
            self.area = c.area

    def type(self):
        if False:
            i = 10
            return i + 15
        return None

    def insert(self, new, pos=None, neighbor=None):
        if False:
            while True:
                i = 10
        if not isinstance(new, list):
            new = [new]
        for n in new:
            n.setParent(None)
        if neighbor is None:
            if pos == 'before':
                index = 0
            else:
                index = self.count()
        else:
            index = self.indexOf(neighbor)
            if index == -1:
                index = 0
            if pos == 'after':
                index += 1
        for n in new:
            self._insertItem(n, index)
            n.containerChanged(self)
            index += 1
            n.sigStretchChanged.connect(self.childStretchChanged)
        self.updateStretch()

    def apoptose(self, propagate=True):
        if False:
            print('Hello World!')
        cont = self._container
        c = self.count()
        if c > 1:
            return
        if c == 1:
            ch = self.widget(0)
            if self.area is not None and self is self.area.topContainer and (not isinstance(ch, Container)) or self.container() is None:
                return
            self.container().insert(ch, 'before', self)
        self.close()
        if propagate and cont is not None:
            cont.apoptose()

    def close(self):
        if False:
            print('Hello World!')
        self.setParent(None)
        if self.area is not None and self.area.topContainer is self:
            self.area.topContainer = None
        self.containerChanged(None)

    def childEvent_(self, ev):
        if False:
            while True:
                i = 10
        ch = ev.child()
        if ev.removed() and hasattr(ch, 'sigStretchChanged'):
            try:
                ch.sigStretchChanged.disconnect(self.childStretchChanged)
            except:
                pass
            self.updateStretch()

    def childStretchChanged(self):
        if False:
            i = 10
            return i + 15
        self.updateStretch()

    def setStretch(self, x=None, y=None):
        if False:
            print('Hello World!')
        self._stretch = (x, y)
        self.sigStretchChanged.emit()

    def updateStretch(self):
        if False:
            i = 10
            return i + 15
        pass

    def stretch(self):
        if False:
            while True:
                i = 10
        'Return the stretch factors for this container'
        return self._stretch

class SplitContainer(Container, QtWidgets.QSplitter):
    """Horizontal or vertical splitter with some changes:
     - save/restore works correctly
    """
    sigStretchChanged = QtCore.Signal()

    def __init__(self, area, orientation):
        if False:
            i = 10
            return i + 15
        QtWidgets.QSplitter.__init__(self)
        self.setOrientation(orientation)
        Container.__init__(self, area)

    def _insertItem(self, item, index):
        if False:
            for i in range(10):
                print('nop')
        self.insertWidget(index, item)
        item.show()

    def saveState(self):
        if False:
            for i in range(10):
                print('nop')
        sizes = self.sizes()
        if all((x == 0 for x in sizes)):
            sizes = [10] * len(sizes)
        return {'sizes': sizes}

    def restoreState(self, state):
        if False:
            while True:
                i = 10
        sizes = state['sizes']
        self.setSizes(sizes)
        for i in range(len(sizes)):
            self.setStretchFactor(i, sizes[i])

    def childEvent(self, ev):
        if False:
            while True:
                i = 10
        super().childEvent(ev)
        Container.childEvent_(self, ev)

class HContainer(SplitContainer):

    def __init__(self, area):
        if False:
            i = 10
            return i + 15
        SplitContainer.__init__(self, area, QtCore.Qt.Orientation.Horizontal)

    def type(self):
        if False:
            return 10
        return 'horizontal'

    def updateStretch(self):
        if False:
            i = 10
            return i + 15
        x = 0
        y = 0
        sizes = []
        for i in range(self.count()):
            (wx, wy) = self.widget(i).stretch()
            x += wx
            y = max(y, wy)
            sizes.append(wx)
        self.setStretch(x, y)
        tot = float(sum(sizes))
        if tot == 0:
            scale = 1.0
        else:
            scale = self.width() / tot
        self.setSizes([int(s * scale) for s in sizes])

class VContainer(SplitContainer):

    def __init__(self, area):
        if False:
            i = 10
            return i + 15
        SplitContainer.__init__(self, area, QtCore.Qt.Orientation.Vertical)

    def type(self):
        if False:
            i = 10
            return i + 15
        return 'vertical'

    def updateStretch(self):
        if False:
            for i in range(10):
                print('nop')
        x = 0
        y = 0
        sizes = []
        for i in range(self.count()):
            (wx, wy) = self.widget(i).stretch()
            y += wy
            x = max(x, wx)
            sizes.append(wy)
        self.setStretch(x, y)
        tot = float(sum(sizes))
        if tot == 0:
            scale = 1.0
        else:
            scale = self.height() / tot
        self.setSizes([int(s * scale) for s in sizes])

class StackedWidget(QtWidgets.QStackedWidget):

    def __init__(self, *, container):
        if False:
            while True:
                i = 10
        super().__init__()
        self.container = container

    def childEvent(self, ev):
        if False:
            return 10
        super().childEvent(ev)
        self.container.childEvent_(ev)

class TContainer(Container, QtWidgets.QWidget):
    sigStretchChanged = QtCore.Signal()

    def __init__(self, area):
        if False:
            i = 10
            return i + 15
        QtWidgets.QWidget.__init__(self)
        Container.__init__(self, area)
        self.layout = QtWidgets.QGridLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.hTabLayout = QtWidgets.QHBoxLayout()
        self.hTabBox = QtWidgets.QWidget()
        self.hTabBox.setLayout(self.hTabLayout)
        self.hTabLayout.setSpacing(2)
        self.hTabLayout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.hTabBox, 0, 1)
        self.stack = StackedWidget(container=self)
        self.layout.addWidget(self.stack, 1, 1)
        self.setLayout(self.layout)
        for n in ['count', 'widget', 'indexOf']:
            setattr(self, n, getattr(self.stack, n))

    def _insertItem(self, item, index):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(item, Dock):
            raise Exception('Tab containers may hold only docks, not other containers.')
        self.stack.insertWidget(index, item)
        self.hTabLayout.insertWidget(index, item.label)
        item.label.sigClicked.connect(self.tabClicked)
        self.tabClicked(item.label)

    def tabClicked(self, tab, ev=None):
        if False:
            return 10
        if ev is None or ev.button() == QtCore.Qt.MouseButton.LeftButton:
            for i in range(self.count()):
                w = self.widget(i)
                if w is tab.dock:
                    w.label.setDim(False)
                    self.stack.setCurrentIndex(i)
                else:
                    w.label.setDim(True)

    def raiseDock(self, dock):
        if False:
            for i in range(10):
                print('nop')
        'Move *dock* to the top of the stack'
        self.stack.currentWidget().label.setDim(True)
        self.stack.setCurrentWidget(dock)
        dock.label.setDim(False)

    def type(self):
        if False:
            for i in range(10):
                print('nop')
        return 'tab'

    def saveState(self):
        if False:
            print('Hello World!')
        return {'index': self.stack.currentIndex()}

    def restoreState(self, state):
        if False:
            return 10
        self.stack.setCurrentIndex(state['index'])

    def updateStretch(self):
        if False:
            while True:
                i = 10
        x = 0
        y = 0
        for i in range(self.count()):
            (wx, wy) = self.widget(i).stretch()
            x = max(x, wx)
            y = max(y, wy)
        self.setStretch(x, y)