from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDropEvent, QDragEnterEvent, QWheelEvent
from PyQt5.QtWidgets import QScrollArea

class ScrollArea(QScrollArea):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dropEvent(self, event: QDropEvent):
        if False:
            i = 10
            return i + 15
        self.files_dropped.emit(event.mimeData().urls())

    def dragEnterEvent(self, event: QDragEnterEvent):
        if False:
            for i in range(10):
                print('nop')
        event.accept()

    def wheelEvent(self, event: QWheelEvent):
        if False:
            i = 10
            return i + 15
        event.ignore()