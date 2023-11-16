from typing import Iterable
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QEvent, QRectF
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QToolButton
from .scroll_area import SmoothScrollBar
from ...common.icon import FluentIcon, isDarkTheme

class ScrollButton(QToolButton):
    """ Scroll button """

    def __init__(self, icon: FluentIcon, parent=None):
        if False:
            return 10
        super().__init__(parent=parent)
        self._icon = icon
        self.isPressed = False
        self.installEventFilter(self)

    def eventFilter(self, obj, e: QEvent):
        if False:
            while True:
                i = 10
        if obj is self:
            if e.type() == QEvent.MouseButtonPress:
                self.isPressed = True
                self.update()
            elif e.type() == QEvent.MouseButtonRelease:
                self.isPressed = False
                self.update()
        return super().eventFilter(obj, e)

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        if not self.isPressed:
            (w, h) = (10, 10)
        else:
            (w, h) = (8, 8)
        x = (self.width() - w) / 2
        y = (self.height() - h) / 2
        if not isDarkTheme():
            self._icon.render(painter, QRectF(x, y, w, h), fill='#5e5e5e')
        else:
            self._icon.render(painter, QRectF(x, y, w, h))

class CycleListWidget(QListWidget):
    """ Cycle list widget """
    currentItemChanged = pyqtSignal(QListWidgetItem)

    def __init__(self, items: Iterable, itemSize: QSize, align=Qt.AlignCenter, parent=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        items: Iterable[Any]\n            the items to be added\n\n        itemSize: QSize\n            the size of item\n\n        align: Qt.AlignmentFlag\n            the text alignment of item\n\n        parent: QWidget\n            parent widget\n        '
        super().__init__(parent=parent)
        self.itemSize = itemSize
        self.align = align
        self.upButton = ScrollButton(FluentIcon.CARE_UP_SOLID, self)
        self.downButton = ScrollButton(FluentIcon.CARE_DOWN_SOLID, self)
        self.scrollDuration = 250
        self.originItems = list(items)
        self.vScrollBar = SmoothScrollBar(Qt.Vertical, self)
        self.visibleNumber = 9
        self.setItems(items)
        self.setVerticalScrollMode(self.ScrollPerPixel)
        self.vScrollBar.setScrollAnimation(self.scrollDuration)
        self.vScrollBar.setForceHidden(True)
        self.setViewportMargins(0, 0, 0, 0)
        self.setFixedSize(itemSize.width() + 8, itemSize.height() * self.visibleNumber)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.upButton.hide()
        self.downButton.hide()
        self.upButton.clicked.connect(self.scrollUp)
        self.downButton.clicked.connect(self.scrollDown)
        self.itemClicked.connect(self._onItemClicked)
        self.installEventFilter(self)

    def setItems(self, items: list):
        if False:
            return 10
        ' set items in the list\n\n        Parameters\n        ----------\n        items: Iterable[Any]\n            the items to be added\n\n        itemSize: QSize\n            the size of item\n\n        align: Qt.AlignmentFlag\n            the text alignment of item\n        '
        self.clear()
        self._createItems(items)

    def _createItems(self, items: list):
        if False:
            while True:
                i = 10
        N = len(items)
        self.isCycle = N > self.visibleNumber
        if self.isCycle:
            for _ in range(2):
                self._addColumnItems(items)
            self._currentIndex = len(items)
            super().scrollToItem(self.item(self.currentIndex() - self.visibleNumber // 2), QListWidget.PositionAtTop)
        else:
            n = self.visibleNumber // 2
            self._addColumnItems([''] * n, True)
            self._addColumnItems(items)
            self._addColumnItems([''] * n, True)
            self._currentIndex = n

    def _addColumnItems(self, items, disabled=False):
        if False:
            return 10
        for i in items:
            item = QListWidgetItem(str(i), self)
            item.setSizeHint(self.itemSize)
            item.setTextAlignment(self.align | Qt.AlignVCenter)
            if disabled:
                item.setFlags(Qt.NoItemFlags)
            self.addItem(item)

    def _onItemClicked(self, item):
        if False:
            while True:
                i = 10
        self.setCurrentIndex(self.row(item))
        self.scrollToItem(self.currentItem())

    def setSelectedItem(self, text: str):
        if False:
            return 10
        ' set the selected item '
        if text is None:
            return
        items = self.findItems(str(text), Qt.MatchExactly)
        if not items:
            return
        if len(items) >= 2:
            self.setCurrentIndex(self.row(items[1]))
        else:
            self.setCurrentIndex(self.row(items[0]))
        super().scrollToItem(self.currentItem(), QListWidget.PositionAtCenter)

    def scrollToItem(self, item: QListWidgetItem, hint=QListWidget.PositionAtCenter):
        if False:
            print('Hello World!')
        ' scroll to item '
        index = self.row(item)
        y = item.sizeHint().height() * (index - self.visibleNumber // 2)
        self.vScrollBar.scrollTo(y)
        self.clearSelection()
        item.setSelected(False)
        self.currentItemChanged.emit(item)

    def wheelEvent(self, e):
        if False:
            while True:
                i = 10
        if e.angleDelta().y() < 0:
            self.scrollDown()
        else:
            self.scrollUp()

    def scrollDown(self):
        if False:
            return 10
        ' scroll down an item '
        self.setCurrentIndex(self.currentIndex() + 1)
        self.scrollToItem(self.currentItem())

    def scrollUp(self):
        if False:
            i = 10
            return i + 15
        ' scroll up an item '
        self.setCurrentIndex(self.currentIndex() - 1)
        self.scrollToItem(self.currentItem())

    def enterEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.upButton.show()
        self.downButton.show()

    def leaveEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.upButton.hide()
        self.downButton.hide()

    def resizeEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.upButton.resize(self.width(), 34)
        self.downButton.resize(self.width(), 34)
        self.downButton.move(0, self.height() - 34)

    def eventFilter(self, obj, e: QEvent):
        if False:
            print('Hello World!')
        if obj is not self or e.type() != QEvent.KeyPress:
            return super().eventFilter(obj, e)
        if e.key() == Qt.Key_Down:
            self.scrollDown()
            return True
        elif e.key() == Qt.Key_Up:
            self.scrollUp()
            return True
        return super().eventFilter(obj, e)

    def currentItem(self):
        if False:
            for i in range(10):
                print('nop')
        return self.item(self.currentIndex())

    def currentIndex(self):
        if False:
            return 10
        return self._currentIndex

    def setCurrentIndex(self, index: int):
        if False:
            for i in range(10):
                print('nop')
        if not self.isCycle:
            n = self.visibleNumber // 2
            self._currentIndex = max(n, min(n + len(self.originItems) - 1, index))
        else:
            N = self.count() // 2
            m = (self.visibleNumber + 1) // 2
            self._currentIndex = index
            if index >= self.count() - m:
                self._currentIndex = N + index - self.count()
                super().scrollToItem(self.item(self.currentIndex() - 1), self.PositionAtCenter)
            elif index <= m - 1:
                self._currentIndex = N + index
                super().scrollToItem(self.item(N + index + 1), self.PositionAtCenter)