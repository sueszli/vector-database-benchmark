from qt.core import Qt, QTableWidget, pyqtSignal

class TleTableWidget(QTableWidget):
    delete_pressed = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        QTableWidget.__init__(self, parent=parent)

    def keyPressEvent(self, event):
        if False:
            i = 10
            return i + 15
        if event.key() == Qt.Key.Key_Delete:
            self.delete_pressed.emit()
            event.accept()
            return
        return QTableWidget.keyPressEvent(self, event)