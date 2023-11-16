"""
Created on 2018年10月30日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: ButtomZoom
@description: 
"""
try:
    from PyQt5.QtCore import QPropertyAnimation, QRect
    from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QHBoxLayout, QSpacerItem, QSizePolicy
except ImportError:
    from PySide2.QtCore import QPropertyAnimation, QRect
    from PySide2.QtWidgets import QApplication, QPushButton, QWidget, QHBoxLayout, QSpacerItem, QSizePolicy

class ZoomButton(QPushButton):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(ZoomButton, self).__init__(*args, **kwargs)
        self._animation = QPropertyAnimation(self, b'geometry', self, duration=200)

    def updatePos(self):
        if False:
            for i in range(10):
                print('nop')
        self._geometry = self.geometry()
        self._rect = QRect(self._geometry.x() - 6, self._geometry.y() - 2, self._geometry.width() + 12, self._geometry.height() + 4)

    def showEvent(self, event):
        if False:
            while True:
                i = 10
        super(ZoomButton, self).showEvent(event)
        self.updatePos()

    def enterEvent(self, event):
        if False:
            i = 10
            return i + 15
        super(ZoomButton, self).enterEvent(event)
        self._animation.stop()
        self._animation.setStartValue(self._geometry)
        self._animation.setEndValue(self._rect)
        self._animation.start()

    def leaveEvent(self, event):
        if False:
            print('Hello World!')
        super(ZoomButton, self).leaveEvent(event)
        self._animation.stop()
        self._animation.setStartValue(self._rect)
        self._animation.setEndValue(self._geometry)
        self._animation.start()

    def mousePressEvent(self, event):
        if False:
            while True:
                i = 10
        self._animation.stop()
        self._animation.setStartValue(self._rect)
        self._animation.setEndValue(self._geometry)
        self._animation.start()
        super(ZoomButton, self).mousePressEvent(event)

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Window, self).__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.button1 = ZoomButton('按钮1', self)
        layout.addWidget(self.button1)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.button2 = ZoomButton('按钮2', self)

    def showEvent(self, event):
        if False:
            print('Hello World!')
        super(Window, self).showEvent(event)
        self.button1.updatePos()
        self.button2.move(self.width() - self.button2.width() - 15, self.height() - self.button2.height() - 10)
        self.button2.updatePos()

    def resizeEvent(self, event):
        if False:
            i = 10
            return i + 15
        super(Window, self).resizeEvent(event)
        self.button1.updatePos()
        self.button2.move(self.width() - self.button2.width() - 15, self.height() - self.button2.height() - 10)
        self.button2.updatePos()
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet('QPushButton {\n    border: none;\n    font-weight: bold;\n    font-size: 16px;\n    border-radius: 18px;\n    min-width: 180px;\n    min-height: 40px;\n    background-color: white;\n    }')
    w = Window()
    w.show()
    sys.exit(app.exec_())