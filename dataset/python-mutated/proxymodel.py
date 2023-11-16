"""Switcher Proxy Model."""
from qtpy.QtCore import QSortFilterProxyModel, Qt

class SwitcherProxyModel(QSortFilterProxyModel):
    """A proxy model to perform sorting on the scored items."""

    def __init__(self, parent=None):
        if False:
            return 10
        'Proxy model to perform sorting on the scored items.'
        super(SwitcherProxyModel, self).__init__(parent)
        self.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.setSortCaseSensitivity(Qt.CaseInsensitive)
        self.setDynamicSortFilter(True)
        self.__filter_by_score = False

    def set_filter_by_score(self, value):
        if False:
            return 10
        '\n        Set whether the items should be filtered by their score result.\n\n        Parameters\n        ----------\n        value : bool\n           Indicates whether the items should be filtered by their\n           score result.\n        '
        self.__filter_by_score = value
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        if False:
            while True:
                i = 10
        'Override Qt method to filter items by their score result.'
        item = self.sourceModel().item(source_row)
        if self.__filter_by_score is False or item.is_action_item():
            return True
        else:
            return not item.get_score() == -1

    def sortBy(self, attr):
        if False:
            return 10
        'Override Qt method.'
        self.__sort_by = attr
        self.invalidate()
        self.sort(0, Qt.AscendingOrder)

    def lessThan(self, left, right):
        if False:
            for i in range(10):
                print('nop')
        'Override Qt method.'
        left_item = self.sourceModel().itemFromIndex(left)
        right_item = self.sourceModel().itemFromIndex(right)
        if hasattr(left_item, self.__sort_by):
            left_data = getattr(left_item, self.__sort_by)
            right_data = getattr(right_item, self.__sort_by)
            return left_data < right_data