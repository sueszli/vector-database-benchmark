"""A model that proxies access to one or more completion categories."""
from typing import MutableSequence
from qutebrowser.qt.core import Qt, QModelIndex, QAbstractItemModel
from qutebrowser.utils import log, qtutils, utils
from qutebrowser.api import cmdutils

class CompletionModel(QAbstractItemModel):
    """A model that proxies access to one or more completion categories.

    Top level indices represent categories.
    Child indices represent rows of those tables.

    Attributes:
        column_widths: The width percentages of the columns used in the
                       completion view.
        _categories: The sub-categories.
    """

    def __init__(self, *, column_widths=(30, 70, 0), parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.column_widths = column_widths
        self._categories: MutableSequence[QAbstractItemModel] = []

    def _cat_from_idx(self, index):
        if False:
            for i in range(10):
                print('nop')
        'Return the category pointed to by the given index.\n\n        Args:\n            idx: A QModelIndex\n        Returns:\n            A category if the index points at one, else None\n        '
        if index.isValid() and (not index.internalPointer()):
            return self._categories[index.row()]
        return None

    def add_category(self, cat):
        if False:
            i = 10
            return i + 15
        'Add a completion category to the model.'
        self._categories.append(cat)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if False:
            while True:
                i = 10
        'Return the item data for index.\n\n        Override QAbstractItemModel::data.\n\n        Args:\n            index: The QModelIndex to get item flags for.\n            role: The Qt ItemRole to get the data for.\n\n        Return: The item data, or None on an invalid index.\n        '
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        cat = self._cat_from_idx(index)
        if cat:
            if index.column() == 0:
                return self._categories[index.row()].name
            return None
        cat = self._cat_from_idx(index.parent())
        if not cat:
            return None
        idx = cat.index(index.row(), index.column())
        return cat.data(idx)

    def flags(self, index):
        if False:
            return 10
        'Return the item flags for index.\n\n        Override QAbstractItemModel::flags.\n\n        Return: The item flags, or Qt.ItemFlag.NoItemFlags on error.\n        '
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        if index.parent().isValid():
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemNeverHasChildren
        else:
            return Qt.ItemFlag.NoItemFlags

    def index(self, row, col, parent=QModelIndex()):
        if False:
            while True:
                i = 10
        'Get an index into the model.\n\n        Override QAbstractItemModel::index.\n\n        Return: A QModelIndex.\n        '
        if row < 0 or row >= self.rowCount(parent) or col < 0 or (col >= self.columnCount(parent)):
            return QModelIndex()
        if parent.isValid():
            if parent.column() != 0:
                return QModelIndex()
            return self.createIndex(row, col, self._categories[parent.row()])
        return self.createIndex(row, col, None)

    def parent(self, index):
        if False:
            for i in range(10):
                print('nop')
        'Get an index to the parent of the given index.\n\n        Override QAbstractItemModel::parent.\n\n        Args:\n            index: The QModelIndex to get the parent index for.\n        '
        parent_cat = index.internalPointer()
        if not parent_cat:
            return QModelIndex()
        row = self._categories.index(parent_cat)
        return self.createIndex(row, 0, None)

    def rowCount(self, parent=QModelIndex()):
        if False:
            return 10
        'Override QAbstractItemModel::rowCount.'
        if not parent.isValid():
            return len(self._categories)
        cat = self._cat_from_idx(parent)
        if not cat or parent.column() != 0:
            return 0
        else:
            return cat.rowCount()

    def columnCount(self, parent=QModelIndex()):
        if False:
            while True:
                i = 10
        'Override QAbstractItemModel::columnCount.'
        utils.unused(parent)
        return len(self.column_widths)

    def canFetchMore(self, parent):
        if False:
            print('Hello World!')
        'Override to forward the call to the categories.'
        cat = self._cat_from_idx(parent)
        if cat:
            return cat.canFetchMore(QModelIndex())
        return False

    def fetchMore(self, parent):
        if False:
            print('Hello World!')
        'Override to forward the call to the categories.'
        cat = self._cat_from_idx(parent)
        if cat:
            cat.fetchMore(QModelIndex())

    def count(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the count of non-category items.'
        return sum((t.rowCount() for t in self._categories))

    def set_pattern(self, pattern):
        if False:
            return 10
        'Set the filter pattern for all categories.\n\n        Args:\n            pattern: The filter pattern to set.\n        '
        log.completion.debug("Setting completion pattern '{}'".format(pattern))
        self.layoutAboutToBeChanged.emit()
        for cat in self._categories:
            cat.set_pattern(pattern)
        self.layoutChanged.emit()

    def first_item(self):
        if False:
            while True:
                i = 10
        'Return the index of the first child (non-category) in the model.'
        for (row, cat) in enumerate(self._categories):
            if cat.rowCount() > 0:
                parent = self.index(row, 0)
                index = self.index(0, 0, parent)
                qtutils.ensure_valid(index)
                return index
        return QModelIndex()

    def last_item(self):
        if False:
            print('Hello World!')
        'Return the index of the last child (non-category) in the model.'
        for (row, cat) in reversed(list(enumerate(self._categories))):
            childcount = cat.rowCount()
            if childcount > 0:
                parent = self.index(row, 0)
                index = self.index(childcount - 1, 0, parent)
                qtutils.ensure_valid(index)
                return index
        return QModelIndex()

    def columns_to_filter(self, index):
        if False:
            return 10
        'Return the column indices the filter pattern applies to.\n\n        Args:\n            index: index of the item to check.\n\n        Return: A list of integers.\n        '
        cat = self._cat_from_idx(index.parent())
        return cat.columns_to_filter if cat else []

    def delete_cur_item(self, index):
        if False:
            return 10
        'Delete the row at the given index.'
        qtutils.ensure_valid(index)
        parent = index.parent()
        cat = self._cat_from_idx(parent)
        assert cat, 'CompletionView sent invalid index for deletion'
        if not cat.delete_func:
            raise cmdutils.CommandError('Cannot delete this item.')
        data = [cat.data(cat.index(index.row(), i)) for i in range(cat.columnCount())]
        cat.delete_func(data)
        self.beginRemoveRows(parent, index.row(), index.row())
        cat.removeRow(index.row(), QModelIndex())
        self.endRemoveRows()