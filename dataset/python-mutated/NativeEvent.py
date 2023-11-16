"""Created on 2018年8月2日
author: Irony
site: https://pyqt.site , https://github.com/PyQt5
email: 892768447@qq.com
file: win无边框调整大小
description:
"""
import ctypes.wintypes
from ctypes.wintypes import POINT
import win32api
import win32con
import win32gui
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QCursor
    from PyQt5.QtWidgets import QApplication, QPushButton, QWidget
    from PyQt5.QtWinExtras import QtWin
except ImportError:
    from PySide2.QtCore import Qt
    from PySide2.QtGui import QCursor
    from PySide2.QtWidgets import QApplication, QPushButton, QWidget
    from PySide2.QtWinExtras import QtWin

class MINMAXINFO(ctypes.Structure):
    _fields_ = [('ptReserved', POINT), ('ptMaxSize', POINT), ('ptMaxPosition', POINT), ('ptMinTrackSize', POINT), ('ptMaxTrackSize', POINT)]

class Window(QWidget):
    BorderWidth = 5

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(Window, self).__init__(*args, **kwargs)
        self._rect = QApplication.instance().desktop().availableGeometry(self)
        self.resize(800, 600)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        style = win32gui.GetWindowLong(int(self.winId()), win32con.GWL_STYLE)
        win32gui.SetWindowLong(int(self.winId()), win32con.GWL_STYLE, style | win32con.WS_THICKFRAME)
        if QtWin.isCompositionEnabled():
            QtWin.extendFrameIntoClientArea(self, -1, -1, -1, -1)
        else:
            QtWin.resetExtendedFrame(self)

    def nativeEvent(self, eventType, message):
        if False:
            print('Hello World!')
        (retval, result) = super(Window, self).nativeEvent(eventType, message)
        if eventType == 'windows_generic_MSG':
            msg = ctypes.wintypes.MSG.from_address(message.__int__())
            pos = QCursor.pos()
            x = pos.x() - self.frameGeometry().x()
            y = pos.y() - self.frameGeometry().y()
            if self.childAt(x, y) != None:
                return (retval, result)
            if msg.message == win32con.WM_NCCALCSIZE:
                return (True, 0)
            if msg.message == win32con.WM_GETMINMAXINFO:
                info = ctypes.cast(msg.lParam, ctypes.POINTER(MINMAXINFO)).contents
                info.ptMaxSize.x = self._rect.width()
                info.ptMaxSize.y = self._rect.height()
                (info.ptMaxPosition.x, info.ptMaxPosition.y) = (0, 0)
            if msg.message == win32con.WM_NCHITTEST:
                (w, h) = (self.width(), self.height())
                lx = x < self.BorderWidth
                rx = x > w - self.BorderWidth
                ty = y < self.BorderWidth
                by = y > h - self.BorderWidth
                if lx and ty:
                    return (True, win32con.HTTOPLEFT)
                if rx and by:
                    return (True, win32con.HTBOTTOMRIGHT)
                if rx and ty:
                    return (True, win32con.HTTOPRIGHT)
                if lx and by:
                    return (True, win32con.HTBOTTOMLEFT)
                if ty:
                    return (True, win32con.HTTOP)
                if by:
                    return (True, win32con.HTBOTTOM)
                if lx:
                    return (True, win32con.HTLEFT)
                if rx:
                    return (True, win32con.HTRIGHT)
                return (True, win32con.HTCAPTION)
        return (retval, result)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    btn = QPushButton('exit', w, clicked=app.quit)
    btn.setGeometry(10, 10, 100, 40)
    w.show()
    sys.exit(app.exec_())