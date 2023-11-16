"""
Created on 2019年5月15日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: 翻转动画
@description: 
"""
try:
    from PyQt5.QtCore import Qt, pyqtSignal, QTimer
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication, QStackedWidget, QLabel
except ImportError:
    from PySide2.QtCore import Qt, Signal as pyqtSignal, QTimer
    from PySide2.QtGui import QPixmap
    from PySide2.QtWidgets import QApplication, QStackedWidget, QLabel
from Lib.FlipWidget import FlipWidget

class LoginWidget(QLabel):
    windowClosed = pyqtSignal()
    windowChanged = pyqtSignal()

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(LoginWidget, self).__init__(*args, **kwargs)
        self.setPixmap(QPixmap('Data/1.png'))

    def mousePressEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        super(LoginWidget, self).mousePressEvent(event)
        pos = event.pos()
        if pos.y() <= 40:
            if pos.x() > self.width() - 30:
                self.windowClosed.emit()
            elif self.width() - 90 <= pos.x() <= self.width() - 60:
                self.windowChanged.emit()

class SettingWidget(QLabel):
    windowClosed = pyqtSignal()
    windowChanged = pyqtSignal()

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(SettingWidget, self).__init__(*args, **kwargs)
        self.setPixmap(QPixmap('Data/2.png'))

    def mousePressEvent(self, event):
        if False:
            i = 10
            return i + 15
        super(SettingWidget, self).mousePressEvent(event)
        pos = event.pos()
        if pos.y() >= self.height() - 30:
            if self.width() - 95 <= pos.x() <= self.width() - 10:
                self.windowChanged.emit()
        elif pos.y() <= 40:
            if pos.x() > self.width() - 30:
                self.windowClosed.emit()

class Window(QStackedWidget):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(Window, self).__init__(*args, **kwargs)
        self.resize(428, 329)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.flipWidget = FlipWidget()
        self.flipWidget.finished.connect(self.showWidget)
        self.loginWidget = LoginWidget(self)
        self.loginWidget.windowClosed.connect(self.close)
        self.loginWidget.windowChanged.connect(self.jumpSettingWidget)
        self.addWidget(self.loginWidget)
        self.settingWidget = SettingWidget(self)
        self.settingWidget.windowClosed.connect(self.close)
        self.settingWidget.windowChanged.connect(self.jumpLoginWidget)
        self.addWidget(self.settingWidget)

    def showWidget(self):
        if False:
            print('Hello World!')
        self.setWindowOpacity(1)
        QTimer.singleShot(100, self.flipWidget.hide)

    def jumpLoginWidget(self):
        if False:
            i = 10
            return i + 15
        self.setWindowOpacity(0)
        self.setCurrentWidget(self.loginWidget)
        image1 = self.loginWidget.grab()
        image2 = self.settingWidget.grab()
        padding = 100
        self.flipWidget.setGeometry(self.geometry())
        self.flipWidget.updateImages(FlipWidget.Right, image2, image1)

    def jumpSettingWidget(self):
        if False:
            print('Hello World!')
        self.setWindowOpacity(0)
        self.setCurrentWidget(self.settingWidget)
        image1 = self.loginWidget.grab()
        image2 = self.settingWidget.grab()
        padding = 100
        self.flipWidget.setGeometry(self.geometry())
        self.flipWidget.updateImages(FlipWidget.Left, image1, image2)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())