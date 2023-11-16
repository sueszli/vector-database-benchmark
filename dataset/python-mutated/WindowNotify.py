"""
Created on 2017年3月30日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: WindowNotify
@description: 右下角弹窗
"""
import webbrowser
try:
    from PyQt5.QtCore import Qt, QPropertyAnimation, QPoint, QTimer, pyqtSignal
    from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QHBoxLayout
except ImportError:
    from PySide2.QtCore import Qt, QPropertyAnimation, QPoint, QTimer, Signal as pyqtSignal
    from PySide2.QtWidgets import QWidget, QPushButton, QApplication, QHBoxLayout
from Lib.UiNotify import Ui_NotifyForm

class WindowNotify(QWidget, Ui_NotifyForm):
    SignalClosed = pyqtSignal()

    def __init__(self, title='', content='', timeout=5000, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(WindowNotify, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setTitle(title).setContent(content)
        self._timeout = timeout
        self._init()

    def setTitle(self, title):
        if False:
            return 10
        if title:
            self.labelTitle.setText(title)
        return self

    def title(self):
        if False:
            i = 10
            return i + 15
        return self.labelTitle.text()

    def setContent(self, content):
        if False:
            for i in range(10):
                print('nop')
        if content:
            self.labelContent.setText(content)
        return self

    def content(self):
        if False:
            i = 10
            return i + 15
        return self.labelContent.text()

    def setTimeout(self, timeout):
        if False:
            while True:
                i = 10
        if isinstance(timeout, int):
            self._timeout = timeout
        return self

    def timeout(self):
        if False:
            while True:
                i = 10
        return self._timeout

    def onView(self):
        if False:
            return 10
        print('onView')
        webbrowser.open_new_tab('http://alyl.vip')

    def onClose(self):
        if False:
            print('Hello World!')
        print('onClose')
        self.isShow = False
        QTimer.singleShot(100, self.closeAnimation)

    def _init(self):
        if False:
            return 10
        self.setWindowFlags(Qt.Tool | Qt.X11BypassWindowManagerHint | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.buttonClose.clicked.connect(self.onClose)
        self.buttonView.clicked.connect(self.onView)
        self.isShow = True
        self._timeouted = False
        self._desktop = QApplication.instance().desktop()
        self._startPos = QPoint(self._desktop.screenGeometry().width() - self.width() - 5, self._desktop.screenGeometry().height())
        self._endPos = QPoint(self._desktop.screenGeometry().width() - self.width() - 5, self._desktop.availableGeometry().height() - self.height() - 5)
        self.move(self._startPos)
        self.animation = QPropertyAnimation(self, b'pos')
        self.animation.finished.connect(self.onAnimationEnd)
        self.animation.setDuration(1000)
        self._timer = QTimer(self, timeout=self.closeAnimation)

    def show(self, title='', content='', timeout=5000):
        if False:
            print('Hello World!')
        self._timer.stop()
        self.hide()
        self.move(self._startPos)
        super(WindowNotify, self).show()
        self.setTitle(title).setContent(content).setTimeout(timeout)
        return self

    def showAnimation(self):
        if False:
            i = 10
            return i + 15
        print('showAnimation isShow = True')
        self.isShow = True
        self.animation.stop()
        self.animation.setStartValue(self.pos())
        self.animation.setEndValue(self._endPos)
        self.animation.start()
        self._timer.start(self._timeout)

    def closeAnimation(self):
        if False:
            for i in range(10):
                print('nop')
        print('closeAnimation hasFocus', self.hasFocus())
        if self.hasFocus():
            self._timeouted = True
            return
        self.isShow = False
        self.animation.stop()
        self.animation.setStartValue(self.pos())
        self.animation.setEndValue(self._startPos)
        self.animation.start()

    def onAnimationEnd(self):
        if False:
            return 10
        print('onAnimationEnd isShow', self.isShow)
        if not self.isShow:
            print('onAnimationEnd close()')
            self.close()
            print('onAnimationEnd stop timer')
            self._timer.stop()
            print('onAnimationEnd close and emit signal')
            self.SignalClosed.emit()

    def enterEvent(self, event):
        if False:
            print('Hello World!')
        super(WindowNotify, self).enterEvent(event)
        print('enterEvent setFocus Qt.MouseFocusReason')
        self.setFocus(Qt.MouseFocusReason)

    def leaveEvent(self, event):
        if False:
            return 10
        super(WindowNotify, self).leaveEvent(event)
        print('leaveEvent clearFocus')
        self.clearFocus()
        if self._timeouted:
            QTimer.singleShot(1000, self.closeAnimation)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = QWidget()
    notify = WindowNotify(parent=window)
    layout = QHBoxLayout(window)
    b1 = QPushButton('弹窗1', window, clicked=lambda : notify.show(content=b1.text()).showAnimation())
    b2 = QPushButton('弹窗2', window, clicked=lambda : notify.show(content=b2.text()).showAnimation())
    layout.addWidget(b1)
    layout.addWidget(b2)
    window.show()
    sys.exit(app.exec_())