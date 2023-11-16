import numpy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QFontMetrics
from PyQt5.QtWidgets import QTableView, qApp

class FuzzingTableView(QTableView):
    deletion_wanted = pyqtSignal(int, int)

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.horizontalHeader().setMinimumSectionSize(0)

    def resize_me(self):
        if False:
            for i in range(10):
                print('nop')
        qApp.setOverrideCursor(Qt.WaitCursor)
        w = QFontMetrics(self.font()).widthChar('0') + 2
        for i in range(10):
            self.setColumnWidth(i, 3 * w)
        for i in range(10, self.model().col_count):
            self.setColumnWidth(i, w * (len(str(i + 1)) + 1))
        qApp.restoreOverrideCursor()

    def selection_range(self):
        if False:
            print('Hello World!')
        '\n        :rtype: int, int, int, int\n        '
        selected = self.selectionModel().selection()
        ':type: QItemSelection '
        if selected.isEmpty():
            return (-1, -1, -1, -1)
        min_row = numpy.min([rng.top() for rng in selected])
        max_row = numpy.max([rng.bottom() for rng in selected])
        start = numpy.min([rng.left() for rng in selected])
        end = numpy.max([rng.right() for rng in selected]) + 1
        return (min_row, max_row, start, end)

    def keyPressEvent(self, event: QKeyEvent):
        if False:
            for i in range(10):
                print('nop')
        if event.key() == Qt.Key_Delete:
            selected = self.selectionModel().selection()
            ':type: QtGui.QItemSelection '
            if selected.isEmpty():
                return
            min_row = numpy.min([rng.top() for rng in selected])
            max_row = numpy.max([rng.bottom() for rng in selected])
            self.deletion_wanted.emit(min_row, max_row)
        else:
            super().keyPressEvent(event)