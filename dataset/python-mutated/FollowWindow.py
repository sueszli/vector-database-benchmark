"""
Created on 2018年10月22日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: FollowWindow
@description: 跟随外部窗口
"""
import os
import win32gui
try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QApplication
except ImportError:
    from PySide2.QtCore import QTimer
    from PySide2.QtWidgets import QWidget, QVBoxLayout, QPushButton, QApplication

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(Window, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        layout.addWidget(QPushButton('test', self))
        self.tmpHwnd = None
        self.checkTimer = QTimer(self, timeout=self.checkWindow)
        self.checkTimer.start(10)

    def checkWindow(self):
        if False:
            while True:
                i = 10
        hwnd = win32gui.FindWindow('Notepad', None)
        if self.tmpHwnd and (not hwnd):
            self.checkTimer.stop()
            self.close()
            return
        if not hwnd:
            return
        self.tmpHwnd = hwnd
        rect = win32gui.GetWindowRect(hwnd)
        print(rect)
        self.move(rect[2], rect[1])
if __name__ == '__main__':
    import sys
    hwnd = win32gui.FindWindow('Notepad', None)
    print('hwnd', hwnd)
    if not hwnd:
        os.startfile('notepad')
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())