"""
Created on 2018年6月14日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: FadeInOut
@description:
"""
try:
    from PyQt5.QtCore import QPropertyAnimation
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
except ImportError:
    from PySide2.QtCore import QPropertyAnimation
    from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(Window, self).__init__(*args, **kwargs)
        self.resize(400, 400)
        layout = QVBoxLayout(self)
        layout.addWidget(QPushButton('退出', self, clicked=self.doClose))
        self.animation = QPropertyAnimation(self, b'windowOpacity')
        self.animation.setDuration(1000)
        self.doShow()

    def doShow(self):
        if False:
            return 10
        try:
            self.animation.finished.disconnect(self.close)
        except:
            pass
        self.animation.stop()
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.start()

    def doClose(self):
        if False:
            for i in range(10):
                print('nop')
        self.animation.stop()
        self.animation.finished.connect(self.close)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.start()
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())