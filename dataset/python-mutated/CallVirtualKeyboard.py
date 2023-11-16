"""
Created on 2019年5月22日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: Demo.CallVirtualKeyboard
@description: 调用系统虚拟键盘
"""
import glob
try:
    from PyQt5.QtCore import QProcess, QSysInfo
    from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QVBoxLayout, QPushButton
except ImportError:
    from PySide2.QtCore import QProcess, QSysInfo
    from PySide2.QtWidgets import QApplication, QWidget, QTextEdit, QVBoxLayout, QPushButton

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(Window, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.resultEdit = QTextEdit(self)
        self.resultEdit.setReadOnly(True)
        layout.addWidget(self.resultEdit)
        layout.addWidget(QPushButton('打开虚拟键盘', self, clicked=self._onOpenKeyboard))

    def _onOpenKeyboard(self):
        if False:
            return 10
        kernelType = QSysInfo.kernelType()
        if kernelType == 'winnt':
            try:
                path = glob.glob('C:\\Windows\\WinSxS\\amd64_microsoft-windows-osk_*\\osk.exe')[0]
                ret = QProcess.startDetached(path)
                self.resultEdit.append('start 64 osk: %s' % ret)
            except Exception as e:
                self.resultEdit.append('start osk error: %s' % e)
            try:
                ret = QProcess.startDetached('C:\\Windows\\system32\\osk.exe')
                self.resultEdit.append('start 32 osk: %s' % ret)
            except Exception as e:
                self.resultEdit.append('start osk error: %s' % e)
        elif kernelType == 'darwin':
            pass
        else:
            ret = QProcess.startDetached('florence')
            self.resultEdit.append('start florence: %s' % ret)
            ret = QProcess.startDetached('onboard')
            self.resultEdit.append('start onboard: %s' % ret)
            ret = QProcess.startDetached('kvkbd')
            self.resultEdit.append('start kvkbd: %s' % ret)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())