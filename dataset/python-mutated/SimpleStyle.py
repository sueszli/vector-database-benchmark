"""
Created on 2018年1月30日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: SimpleStyle
@description: 
"""
import sys
from random import randint
try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QProgressBar
except ImportError:
    from PySide2.QtCore import QTimer
    from PySide2.QtWidgets import QWidget, QApplication, QVBoxLayout, QProgressBar
StyleSheet = '\n/*设置红色进度条*/\n#RedProgressBar {\n    text-align: center; /*进度值居中*/\n}\n#RedProgressBar::chunk {\n    background-color: #F44336;\n}\n\n\n#GreenProgressBar {\n    min-height: 12px;\n    max-height: 12px;\n    border-radius: 6px;\n}\n#GreenProgressBar::chunk {\n    border-radius: 6px;\n    background-color: #009688;\n}\n\n#BlueProgressBar {\n    border: 2px solid #2196F3;/*边框以及边框颜色*/\n    border-radius: 5px;\n    background-color: #E0E0E0;\n}\n#BlueProgressBar::chunk {\n    background-color: #2196F3;\n    width: 10px; /*区块宽度*/\n    margin: 0.5px;\n}\n'

class ProgressBar(QProgressBar):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(ProgressBar, self).__init__(*args, **kwargs)
        self.setValue(0)
        if self.minimum() != self.maximum():
            self.timer = QTimer(self, timeout=self.onTimeout)
            self.timer.start(randint(1, 3) * 1000)

    def onTimeout(self):
        if False:
            i = 10
            return i + 15
        if self.value() >= 100:
            self.timer.stop()
            self.timer.deleteLater()
            del self.timer
            return
        self.setValue(self.value() + 1)

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(Window, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        layout.addWidget(ProgressBar(self, minimum=0, maximum=100, objectName='RedProgressBar'))
        layout.addWidget(ProgressBar(self, minimum=0, maximum=0, objectName='RedProgressBar'))
        layout.addWidget(ProgressBar(self, minimum=0, maximum=100, textVisible=False, objectName='GreenProgressBar'))
        layout.addWidget(ProgressBar(self, minimum=0, maximum=0, textVisible=False, objectName='GreenProgressBar'))
        layout.addWidget(ProgressBar(self, minimum=0, maximum=100, textVisible=False, objectName='BlueProgressBar'))
        layout.addWidget(ProgressBar(self, minimum=0, maximum=0, textVisible=False, objectName='BlueProgressBar'))
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(StyleSheet)
    w = Window()
    w.show()
    sys.exit(app.exec_())