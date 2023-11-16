from PyQt6 import QtCore, QtWidgets

class TristateSortHeaderView(QtWidgets.QHeaderView):
    """A QHeaderView implementation supporting tristate sorting.

    A column can either be sorted ascending, descending or not sorted. The view
    toggles through these states by clicking on a section header.
    """
    STATE_NONE = 0
    STATE_SECTION_MOVED_OR_RESIZED = 1

    def __init__(self, orientation, parent=None):
        if False:
            print('Hello World!')
        super().__init__(orientation, parent)
        self._section_moved_or_resized = False
        self.lock(False)

        def update_state(i, o, n):
            if False:
                return 10
            self._section_moved_or_resized = True
        self.sectionResized.connect(update_state)
        self.sectionMoved.connect(update_state)

    def mouseReleaseEvent(self, event):
        if False:
            while True:
                i = 10
        if self.is_locked:
            tooltip = _("The table is locked. To enable sorting and column resizing\nunlock the table in the table header's context menu.")
            QtWidgets.QToolTip.showText(event.globalPos(), tooltip, self)
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            index = self.logicalIndexAt(event.pos())
            if index != -1 and index == self.sortIndicatorSection() and (self.sortIndicatorOrder() == QtCore.Qt.SortOrder.DescendingOrder):
                self.setSectionsClickable(False)
                self._section_moved_or_resized = False
                super().mouseReleaseEvent(event)
                self.setSectionsClickable(True)
                if not self._section_moved_or_resized:
                    self.setSortIndicator(-1, self.sortIndicatorOrder())
                return
        super().mouseReleaseEvent(event)

    def lock(self, is_locked):
        if False:
            return 10
        self.is_locked = is_locked
        self.setSectionsClickable(not is_locked)
        self.setSectionsMovable(not is_locked)
        if is_locked:
            resize_mode = QtWidgets.QHeaderView.ResizeMode.Fixed
        else:
            resize_mode = QtWidgets.QHeaderView.ResizeMode.Interactive
        self.setSectionResizeMode(resize_mode)