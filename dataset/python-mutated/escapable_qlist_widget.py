from qtpy.QtCore import Qt
from qtpy import QtWidgets

class EscapableQListWidget(QtWidgets.QListWidget):

    def keyPressEvent(self, event):
        if False:
            print('Hello World!')
        super(EscapableQListWidget, self).keyPressEvent(event)
        if event.key() == Qt.Key_Escape:
            self.clearSelection()