"""
Created on 2017年3月30日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: AwesomeFont
@description: 
"""
import sys
try:
    from PyQt5.QtGui import QFontDatabase, QFont
    from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QScrollArea, QPushButton
except ImportError:
    from PySide2.QtGui import QFontDatabase, QFont
    from PySide2.QtWidgets import QApplication, QWidget, QGridLayout, QScrollArea, QPushButton
from Lib.FontAwesome import FontAwesomes

class ScrollArea(QScrollArea):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(ScrollArea, self).__init__()
        self.setWindowTitle('FontAwesome Fonts')
        self.window = QWidget(self)
        self.setWidget(self.window)
        layout = QGridLayout(self.window)
        fonts = list(FontAwesomes.alls().items())
        print(fonts)
        for row in range(131):
            for col in range(6):
                layout.addWidget(QPushButton(': '.join(fonts[row * 6 + col]), self.window, minimumHeight=33, font=QFont('FontAwesome', 14)), row, col, 1, 1)
        self.showMaximized()

    def resizeEvent(self, event):
        if False:
            while True:
                i = 10
        super(ScrollArea, self).resizeEvent(event)
        self.window.resize(self.width(), self.height() * 4)
app = QApplication(sys.argv)
app.setStyleSheet('QPushButton:hover {\n    color: green;\n}\nQPushButton:pressed {\n    color: red;\n}\n')
QFontDatabase.addApplicationFont('Data/Fonts/fontawesome-webfont.ttf')
window = ScrollArea()
window.show()
sys.exit(app.exec_())