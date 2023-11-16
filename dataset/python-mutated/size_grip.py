from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtWidgets import QSizeGrip
from PyQt5.QtGui import QTextOption, QPainter

class SizeGrip(QSizeGrip):

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        painter = QPainter(self)
        option = QTextOption()
        option.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        painter.drawText(QRectF(self.rect()), '●', option)