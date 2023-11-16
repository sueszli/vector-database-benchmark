"""
Created on 2017年4月12日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: 自定义属性测试
@description: 
"""
from random import randint
try:
    from PyQt5.QtCore import pyqtProperty, pyqtSignal
    from PyQt5.QtWidgets import QPushButton, QApplication
except ImportError:
    from PySide2.QtCore import Property as pyqtProperty, Signal as pyqtSignal
    from PySide2.QtWidgets import QPushButton, QApplication

class Window(QPushButton):
    bgChanged = pyqtSignal(str, str)

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(Window, self).__init__('QSS')
        self._textColor = ''
        self._backgroundColor = ''
        self.clicked.connect(self.onClick)
        self.bgChanged.connect(lambda old, new: print('old bg color', old, 'new bg color', new))

    def onClick(self):
        if False:
            return 10
        print('textColor', self._textColor)
        self.setStyleSheet('qproperty-backgroundColor: %s;' % randint(1, 1000))

    @pyqtProperty(str, notify=bgChanged)
    def backgroundColor(self):
        if False:
            while True:
                i = 10
        return self._backgroundColor

    @backgroundColor.setter
    def backgroundColor(self, color):
        if False:
            i = 10
            return i + 15
        self.bgChanged.emit(self._backgroundColor, color)
        self._backgroundColor = color

    def getTextColor(self):
        if False:
            i = 10
            return i + 15
        return self._textColor

    def setTextColor(self, c):
        if False:
            for i in range(10):
                print('nop')
        self._textColor = c
    textColor = pyqtProperty(str, getTextColor, setTextColor)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.setStyleSheet('qproperty-textColor: white;qproperty-backgroundColor: red;')
    w.show()
    sys.exit(app.exec_())