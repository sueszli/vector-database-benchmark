"""
Created on 2017年12月11日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: HotKey
@description: 
"""
import sys
import keyboard
try:
    from PyQt5.QtCore import pyqtSignal, Qt
    from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QTextBrowser, QPushButton, QMessageBox
except ImportError:
    from PySide2.QtCore import Signal as pyqtSignal, Qt
    from PySide2.QtWidgets import QWidget, QApplication, QVBoxLayout, QTextBrowser, QPushButton, QMessageBox

class Window(QWidget):
    dialogShow = pyqtSignal()

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(Window, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.dialogShow.connect(self.onShowDialog, type=Qt.QueuedConnection)
        self.logView = QTextBrowser(self)
        self.logView.append('点击右上角关闭按钮会隐藏窗口,通过热键Alt+S来显示')
        self.logView.append('等待热键中')
        layout.addWidget(QPushButton('退出整个程序', self, clicked=self.onQuit))
        layout.addWidget(self.logView)
        keyboard.add_hotkey('alt+s', self.onShow, suppress=False)
        keyboard.add_hotkey('ctrl+s', self.onHide, suppress=False)
        keyboard.add_hotkey('shift+s', self.onQuit, suppress=False)
        keyboard.add_hotkey('win+s', lambda : self.logView.append('按下了win+s'), suppress=True)
        keyboard.add_hotkey('win+r', lambda : self.logView.append('按下了win+r'), suppress=True)
        keyboard.add_hotkey('ctrl+alt+del', lambda : self.logView.append('😏😏我知道你按了任务管理器😏😏'))

    def onShow(self):
        if False:
            while True:
                i = 10
        '显示'
        self.logView.append('按下alt+s')
        self.show()
        self.showNormal()
        self.dialogShow.emit()

    def onShowDialog(self):
        if False:
            print('Hello World!')
        QMessageBox.information(self, '对话框', '按下alt+s键')

    def onHide(self):
        if False:
            while True:
                i = 10
        '隐藏'
        self.logView.append('按下ctrl+s')
        self.hide()

    def onQuit(self):
        if False:
            for i in range(10):
                print('nop')
        '退出函数'
        keyboard.unhook_all_hotkeys()
        QApplication.instance().quit()

    def closeEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.hide()
        return event.ignore()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())