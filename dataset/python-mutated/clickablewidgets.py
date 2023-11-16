from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QLabel

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if False:
            i = 10
            return i + 15
        self.clicked.emit()
        QLabel.mousePressEvent(self, event)