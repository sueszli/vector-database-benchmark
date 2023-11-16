import logging
from PyQt5.QtCore import QAbstractListModel, Qt, QModelIndex, QPoint, QSize, QTimer, QVariant
from PyQt5.QtGui import QPainter, QFontMetrics
from PyQt5.QtWidgets import QListView, QStyledItemDelegate
logger = logging.getLogger(__name__)

class TextlistModel(QAbstractListModel):
    """

    Public methods:

    - add: add item to list
    - remove: remove item from list
    - clear: clear list
    - __len__: for truth judgement
    """

    def __init__(self, parent):
        if False:
            return 10
        super().__init__(parent)
        self._items = []

    @property
    def items(self):
        if False:
            i = 10
            return i + 15
        return self._items

    def add(self, item):
        if False:
            while True:
                i = 10
        start = len(self._items)
        end = start + 1
        self.beginInsertRows(QModelIndex(), start, end)
        self._items.append(item)
        self.endInsertRows()

    def remove(self, item):
        if False:
            print('Hello World!')
        row = self._items.index(item)
        self.beginRemoveRows(QModelIndex(), row, row + 1)
        self._items.remove(item)
        self.endRemoveRows()

    def clear(self):
        if False:
            print('Hello World!')
        total_length = len(self.items)
        self.beginRemoveRows(QModelIndex(), 0, total_length - 1)
        self._items = []
        self.endRemoveRows()

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self._items)

    def rowCount(self, _=QModelIndex()):
        if False:
            while True:
                i = 10
        return len(self.items)

    def flags(self, index):
        if False:
            while True:
                i = 10
        if not index.isValid():
            return 0
        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        return flags

    def data(self, index, role=Qt.DisplayRole):
        if False:
            return 10
        if not index.isValid():
            return QVariant()
        row = index.row()
        item = self.items[row]
        if role == Qt.UserRole:
            return item
        return QVariant()

class TextlistDelegate(QStyledItemDelegate):

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent=parent)

    def paint(self, painter, option, index):
        if False:
            while True:
                i = 10
        return super().paint(painter, option, index)

    def sizeHint(self, option, index):
        if False:
            print('Hello World!')
        size = super().sizeHint(option, index)
        if index.isValid():
            return QSize(size.width(), 25)
        return size

class TextlistView(QListView):
    """
    .. versiondeprecated:: 3.9
        A TextlistView should have scrollbar by default.
    """

    def __init__(self, parent=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.delegate = TextlistDelegate(self)
        self.setItemDelegate(self.delegate)
        self._result_timer = QTimer(self)
        self._result_timer.timeout.connect(self.__on_timeout)
        self._results = {}
        self.setAttribute(Qt.WA_MacShowFocusRect, False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def __on_timeout(self):
        if False:
            while True:
                i = 10
        self._result_timer.stop()
        self._results.clear()
        self.viewport().update()

    def wheelEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        e.ignore()

    def sizeHint(self):
        if False:
            print('Hello World!')
        height = 10
        for i in range(0, self.model().rowCount()):
            height += self.sizeHintForRow(i)
        return QSize(self.width(), height)

    def rowsInserted(self, parent, start, end):
        if False:
            i = 10
            return i + 15
        super().rowsInserted(parent, start, end)
        self.setFixedHeight(self.sizeHint().height())

    def rowsAboutToBeRemoved(self, parent, start, end):
        if False:
            i = 10
            return i + 15
        super().rowsAboutToBeRemoved(parent, start, end)
        self.setFixedHeight(self.sizeHint().height())

    def paintEvent(self, e):
        if False:
            print('Hello World!')
        super().paintEvent(e)
        if not self._results:
            return
        painter = QPainter(self.viewport())
        option = self.viewOptions()
        painter.setRenderHint(QPainter.Antialiasing)
        fm = QFontMetrics(option.font)
        for (_, result) in self._results.items():
            (index, state) = result
            rect = self.rectForIndex(index)
            if state is None:
                text = '😶'
            elif state is True:
                text = '👋'
            else:
                text = '🙁'
            x = rect.width() - 20 + rect.x()
            y = (rect.height() + fm.ascent() - fm.descent()) // 2 + rect.y()
            topleft = QPoint(x, y)
            painter.drawText(topleft, text)