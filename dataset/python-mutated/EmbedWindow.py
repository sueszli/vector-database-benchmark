"""
Created on 2018年3月1日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: EmbedWindow
@description: 嵌入外部窗口
"""
import win32con
import win32gui
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QWindow
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QListWidget, QLabel, QApplication
except ImportError:
    from PySide2.QtCore import Qt
    from PySide2.QtGui import QWindow
    from PySide2.QtWidgets import QWidget, QVBoxLayout, QPushButton, QListWidget, QLabel, QApplication

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(Window, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.myhwnd = int(self.winId())
        layout.addWidget(QPushButton('获取所有可用、可视窗口', self, clicked=self._getWindowList, maximumHeight=30))
        layout.addWidget(QPushButton('释放窗口', clicked=self.releaseWidget, maximumHeight=30))
        layout.addWidget(QLabel('双击列表中的项目则进行嵌入目标窗口到下方\n格式为：句柄|父句柄|标题|类名', self, maximumHeight=30))
        self.windowList = QListWidget(self, itemDoubleClicked=self.onItemDoubleClicked, maximumHeight=200)
        layout.addWidget(self.windowList)

    def releaseWidget(self):
        if False:
            for i in range(10):
                print('nop')
        '释放窗口'
        if self.layout().count() == 5:
            self.restore()
            self._getWindowList()

    def closeEvent(self, event):
        if False:
            return 10
        '窗口关闭'
        self.releaseWidget()
        super(Window, self).closeEvent(event)

    def _getWindowList(self):
        if False:
            print('Hello World!')
        '清空原来的列表'
        self.windowList.clear()
        win32gui.EnumWindows(self._enumWindows, None)

    def onItemDoubleClicked(self, item):
        if False:
            return 10
        '列表双击选择事件'
        self.windowList.takeItem(self.windowList.indexFromItem(item).row())
        (hwnd, phwnd, _, _) = item.text().split('|')
        self.releaseWidget()
        (hwnd, phwnd) = (int(hwnd), int(phwnd))
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
        exstyle = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        wrect = win32gui.GetWindowRect(hwnd)[:2] + win32gui.GetClientRect(hwnd)[2:]
        print('save', hwnd, style, exstyle, wrect)
        widget = QWidget.createWindowContainer(QWindow.fromWinId(hwnd))
        widget.hwnd = hwnd
        widget.phwnd = phwnd
        widget.style = style
        widget.exstyle = exstyle
        widget.wrect = wrect
        self.layout().addWidget(widget)
        widget.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        win32gui.SetParent(hwnd, int(self.winId()))

    def restore(self):
        if False:
            for i in range(10):
                print('nop')
        '归还窗口'
        widget = self.layout().itemAt(4).widget()
        (hwnd, phwnd, style, exstyle, wrect) = (widget.hwnd, widget.phwnd, widget.style, widget.exstyle, widget.wrect)
        print('restore', hwnd, phwnd, style, exstyle, wrect)
        widget.close()
        self.layout().removeWidget(widget)
        widget.deleteLater()
        win32gui.SetParent(hwnd, phwnd)
        win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style | win32con.WS_VISIBLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, exstyle)
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        win32gui.SetWindowPos(hwnd, 0, wrect[0], wrect[1], wrect[2], wrect[3], win32con.SWP_NOACTIVATE)

    def _enumWindows(self, hwnd, _):
        if False:
            i = 10
            return i + 15
        '遍历回调函数'
        if hwnd == self.myhwnd:
            return
        if win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            phwnd = win32gui.GetParent(hwnd)
            title = win32gui.GetWindowText(hwnd)
            name = win32gui.GetClassName(hwnd)
            self.windowList.addItem('{0}|{1}|\t标题：{2}\t|\t类名：{3}'.format(hwnd, phwnd, title, name))
if __name__ == '__main__':
    import sys
    import cgitb
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())