from typing import List, Union
from PyQt5.QtCore import Qt, pyqtSignal, QModelIndex, QItemSelectionModel, pyqtProperty
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QStyleOptionViewItem, QListView, QListWidgetItem, QListView, QListWidget, QWidget
from .scroll_bar import SmoothScrollDelegate
from .table_view import TableItemDelegate
from ...common.style_sheet import FluentStyleSheet, themeColor

class ListItemDelegate(TableItemDelegate):
    """ List item delegate """

    def __init__(self, parent: QListView):
        if False:
            print('Hello World!')
        super().__init__(parent)

    def _drawBackground(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        if False:
            while True:
                i = 10
        painter.drawRoundedRect(option.rect, 5, 5)

    def _drawIndicator(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        if False:
            i = 10
            return i + 15
        (y, h) = (option.rect.y(), option.rect.height())
        ph = round(0.35 * h if self.pressedRow == index.row() else 0.257 * h)
        painter.setBrush(themeColor())
        painter.drawRoundedRect(0, ph + y, 3, h - 2 * ph, 1.5, 1.5)

class ListBase:

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.delegate = ListItemDelegate(self)
        self.scrollDelegate = SmoothScrollDelegate(self)
        self._isSelectRightClickedRow = False
        FluentStyleSheet.LIST_VIEW.apply(self)
        self.setItemDelegate(self.delegate)
        self.setMouseTracking(True)
        self.entered.connect(lambda i: self._setHoverRow(i.row()))
        self.pressed.connect(lambda i: self._setPressedRow(i.row()))

    def _setHoverRow(self, row: int):
        if False:
            print('Hello World!')
        ' set hovered row '
        self.delegate.setHoverRow(row)
        self.viewport().update()

    def _setPressedRow(self, row: int):
        if False:
            return 10
        ' set pressed row '
        self.delegate.setPressedRow(row)
        self.viewport().update()

    def _setSelectedRows(self, indexes: List[QModelIndex]):
        if False:
            print('Hello World!')
        self.delegate.setSelectedRows(indexes)
        self.viewport().update()

    def leaveEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        QListView.leaveEvent(self, e)
        self._setHoverRow(-1)

    def resizeEvent(self, e):
        if False:
            print('Hello World!')
        QListView.resizeEvent(self, e)
        self.viewport().update()

    def keyPressEvent(self, e):
        if False:
            print('Hello World!')
        QListView.keyPressEvent(self, e)
        self.updateSelectedRows()

    def mousePressEvent(self, e):
        if False:
            i = 10
            return i + 15
        if e.button() == Qt.LeftButton or self._isSelectRightClickedRow:
            return QListView.mousePressEvent(self, e)
        index = self.indexAt(e.pos())
        if index.isValid():
            self._setPressedRow(index.row())
        QWidget.mousePressEvent(self, e)

    def mouseReleaseEvent(self, e):
        if False:
            i = 10
            return i + 15
        QListView.mouseReleaseEvent(self, e)
        self.updateSelectedRows()
        if self.indexAt(e.pos()).row() < 0 or e.button() == Qt.RightButton:
            self._setPressedRow(-1)

    def setItemDelegate(self, delegate: ListItemDelegate):
        if False:
            for i in range(10):
                print('nop')
        self.delegate = delegate
        super().setItemDelegate(delegate)

    def clearSelection(self):
        if False:
            for i in range(10):
                print('nop')
        QListView.clearSelection(self)
        self.updateSelectedRows()

    def setCurrentIndex(self, index: QModelIndex):
        if False:
            for i in range(10):
                print('nop')
        QListView.setCurrentIndex(self, index)
        self.updateSelectedRows()

    def updateSelectedRows(self):
        if False:
            for i in range(10):
                print('nop')
        self._setSelectedRows(self.selectedIndexes())

class ListWidget(ListBase, QListWidget):
    """ List widget """

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent)

    def setCurrentItem(self, item: QListWidgetItem, command: Union[QItemSelectionModel.SelectionFlag, QItemSelectionModel.SelectionFlags]=None):
        if False:
            while True:
                i = 10
        self.setCurrentRow(self.row(item), command)

    def setCurrentRow(self, row: int, command: Union[QItemSelectionModel.SelectionFlag, QItemSelectionModel.SelectionFlags]=None):
        if False:
            print('Hello World!')
        if not command:
            super().setCurrentRow(row)
        else:
            super().setCurrentRow(row, command)
        self.updateSelectedRows()

    def isSelectRightClickedRow(self):
        if False:
            i = 10
            return i + 15
        return self._isSelectRightClickedRow

    def setSelectRightClickedRow(self, isSelect: bool):
        if False:
            return 10
        self._isSelectRightClickedRow = isSelect
    selectRightClickedRow = pyqtProperty(bool, isSelectRightClickedRow, setSelectRightClickedRow)

class ListView(ListBase, QListView):
    """ List view """

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent)

    def isSelectRightClickedRow(self):
        if False:
            i = 10
            return i + 15
        return self._isSelectRightClickedRow

    def setSelectRightClickedRow(self, isSelect: bool):
        if False:
            print('Hello World!')
        self._isSelectRightClickedRow = isSelect
    selectRightClickedRow = pyqtProperty(bool, isSelectRightClickedRow, setSelectRightClickedRow)