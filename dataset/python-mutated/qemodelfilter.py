from PyQt6.QtCore import pyqtSignal, pyqtProperty, QSortFilterProxyModel, QModelIndex, pyqtSlot
from electrum.logging import get_logger

class QEFilterProxyModel(QSortFilterProxyModel):
    _logger = get_logger(__name__)

    def __init__(self, parent_model, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._filter_value = None
        self.setSourceModel(parent_model)
    countChanged = pyqtSignal()

    @pyqtProperty(int, notify=countChanged)
    def count(self):
        if False:
            return 10
        return self.rowCount(QModelIndex())

    def isCustomFilter(self):
        if False:
            print('Hello World!')
        return self._filter_value is not None

    @pyqtSlot(str)
    def setFilterValue(self, filter_value):
        if False:
            i = 10
            return i + 15
        self._filter_value = filter_value
        self.invalidate()

    def filterAcceptsRow(self, s_row, s_parent):
        if False:
            i = 10
            return i + 15
        if not self.isCustomFilter:
            return super().filterAcceptsRow(s_row, s_parent)
        parent_model = self.sourceModel()
        d = parent_model.data(parent_model.index(s_row, 0, s_parent), self.filterRole())
        return True if self._filter_value is None else d == self._filter_value