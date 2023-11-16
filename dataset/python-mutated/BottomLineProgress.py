"""
Created on 2018年2月1日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: PushButtonLine
@description: 
"""
import sys
from random import randint
try:
    from PyQt5.QtCore import QTimer, QThread, pyqtSignal
    from PyQt5.QtGui import QPainter, QColor, QPen
    from PyQt5.QtWidgets import QPushButton, QApplication, QWidget, QVBoxLayout
except ImportError:
    from PySide2.QtCore import QTimer, QThread, Signal as pyqtSignal
    from PySide2.QtGui import QPainter, QColor, QPen
    from PySide2.QtWidgets import QPushButton, QApplication, QWidget, QVBoxLayout
StyleSheet = '\nPushButtonLine {\n    color: white;\n    border: none;\n    min-height: 48px;\n    background-color: #90caf9;\n}\n'

class LoadingThread(QThread):
    valueChanged = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(LoadingThread, self).__init__(*args, **kwargs)
        self.totalValue = randint(100, 200)

    def run(self):
        if False:
            return 10
        for i in range(self.totalValue + 1):
            if self.isInterruptionRequested():
                break
            self.valueChanged.emit(i / self.totalValue)
            QThread.msleep(randint(50, 100))

class PushButtonLine(QPushButton):
    lineColor = QColor(0, 150, 136)

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._waitText = kwargs.pop('waitText', '等待中')
        super(PushButtonLine, self).__init__(*args, **kwargs)
        self._text = self.text()
        self._percent = 0
        self._timer = QTimer(self, timeout=self.update)
        self.clicked.connect(self.start)

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        self.stop()

    def paintEvent(self, event):
        if False:
            i = 10
            return i + 15
        super(PushButtonLine, self).paintEvent(event)
        if not self._timer.isActive():
            return
        painter = QPainter(self)
        pen = QPen(self.lineColor)
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawLine(0, self.height(), self.width() * self._percent, self.height())

    def start(self):
        if False:
            return 10
        if hasattr(self, 'loadingThread'):
            return self.stop()
        self.loadingThread = LoadingThread(self)
        self.loadingThread.valueChanged.connect(self.setPercent)
        self._timer.start(100)
        self.loadingThread.start()
        self.setText(self._waitText)

    def stop(self):
        if False:
            return 10
        try:
            if hasattr(self, 'loadingThread'):
                if self.loadingThread.isRunning():
                    self.loadingThread.requestInterruption()
                    self.loadingThread.quit()
                    self.loadingThread.wait(2000)
                del self.loadingThread
        except RuntimeError:
            pass
        try:
            self._percent = 0
            self._timer.stop()
            self.setText(self._text)
        except RuntimeError:
            pass

    def setPercent(self, v):
        if False:
            print('Hello World!')
        self._percent = v
        if v == 1:
            self.stop()
            self.update()

    def setLineColor(self, color):
        if False:
            while True:
                i = 10
        self.lineColor = QColor(color)
        return self

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(Window, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        layout.addWidget(PushButtonLine('点击加载'))
        layout.addWidget(PushButtonLine('点击加载').setLineColor('#ef5350'))
        layout.addWidget(PushButtonLine('点击加载').setLineColor('#ffc107'))
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(StyleSheet)
    w = Window()
    w.show()
    sys.exit(app.exec_())