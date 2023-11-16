"""Completion category that uses a list of tuples as a data source."""
import re
from typing import Iterable, Tuple
from qutebrowser.qt.core import QSortFilterProxyModel, QRegularExpression
from qutebrowser.qt.gui import QStandardItem, QStandardItemModel
from qutebrowser.qt.widgets import QWidget
from qutebrowser.completion.models import util
from qutebrowser.utils import qtutils, log

class ListCategory(QSortFilterProxyModel):
    """Expose a list of items as a category for the CompletionModel."""

    def __init__(self, name: str, items: Iterable[Tuple[str, ...]], sort: bool=True, delete_func: util.DeleteFuncType=None, parent: QWidget=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.name = name
        self.srcmodel = QStandardItemModel(parent=self)
        self._pattern = ''
        self.columns_to_filter = [0, 1, 2]
        self.setFilterKeyColumn(-1)
        for item in items:
            self.srcmodel.appendRow([QStandardItem(x) for x in item])
        self.setSourceModel(self.srcmodel)
        self.delete_func = delete_func
        self._sort = sort

    def set_pattern(self, val):
        if False:
            for i in range(10):
                print('nop')
        'Setter for pattern.\n\n        Args:\n            val: The value to set.\n        '
        if len(val) > 5000:
            log.completion.warning(f'Trimming {len(val)}-char pattern to 5000')
            val = val[:5000]
        self._pattern = val
        val = re.sub(' +', ' ', val)
        val = re.escape(val)
        val = val.replace('\\ ', '.*')
        rx = QRegularExpression(val, QRegularExpression.PatternOption.CaseInsensitiveOption)
        qtutils.ensure_valid(rx)
        self.setFilterRegularExpression(rx)
        self.invalidate()
        sortcol = 0
        self.sort(sortcol)

    def lessThan(self, lindex, rindex):
        if False:
            i = 10
            return i + 15
        'Custom sorting implementation.\n\n        Prefers all items which start with self._pattern. Other than that, uses\n        normal Python string sorting.\n\n        Args:\n            lindex: The QModelIndex of the left item (*left* < right)\n            rindex: The QModelIndex of the right item (left < *right*)\n\n        Return:\n            True if left < right, else False\n        '
        qtutils.ensure_valid(lindex)
        qtutils.ensure_valid(rindex)
        left = self.srcmodel.data(lindex)
        right = self.srcmodel.data(rindex)
        if left is None or right is None:
            log.completion.warning('Got unexpected None value, left={!r} right={!r} lindex={!r} rindex={!r}'.format(left, right, lindex, rindex))
            return False
        leftstart = left.startswith(self._pattern)
        rightstart = right.startswith(self._pattern)
        if leftstart and (not rightstart):
            return True
        elif rightstart and (not leftstart):
            return False
        elif self._sort:
            return left < right
        else:
            return False