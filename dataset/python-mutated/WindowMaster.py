"""
Created on 2019年8月7日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: QtRemoteObjects.SyncUi.WindowMaster
@description: 主窗口
"""
from PyQt5.QtCore import QUrl, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtRemoteObjects import QRemoteObjectHost
from PyQt5.QtWidgets import QWidget, QLineEdit, QVBoxLayout, QCheckBox, QProgressBar

class WindowMaster(QWidget):
    editValueChanged = pyqtSignal(str)
    checkToggled = pyqtSignal(bool)
    progressValueChanged = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(WindowMaster, self).__init__(*args, **kwargs)
        self.setupUi()
        host = QRemoteObjectHost(QUrl('local:WindowMaster'), parent=self)
        host.enableRemoting(self, 'WindowMaster')
        print('开启节点完成')
        self._value = 0
        self.utimer = QTimer(self, timeout=self.updateProgress)
        self.utimer.start(200)

    def setupUi(self):
        if False:
            while True:
                i = 10
        self.setWindowTitle('WindowMaster')
        self.resize(300, 400)
        layout = QVBoxLayout(self)
        self.lineEdit = QLineEdit(self)
        self.lineEdit.textChanged.connect(self.editValueChanged.emit)
        self.checkBox = QCheckBox('来勾我啊', self)
        self.checkBox.toggled.connect(self.checkToggled.emit)
        self.progressBar = QProgressBar(self)
        self.progressBar.valueChanged.connect(self.progressValueChanged.emit)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.checkBox)
        layout.addWidget(self.progressBar)

    def updateProgress(self):
        if False:
            print('Hello World!')
        self._value += 1
        if self._value > 100:
            self._value = 0
        self.progressBar.setValue(self._value)

    @pyqtSlot(str)
    def updateEdit(self, text):
        if False:
            i = 10
            return i + 15
        '更新输入框内容的槽函数\n        :param text:\n        '
        self.lineEdit.setText(text)

    @pyqtSlot(bool)
    def updateCheck(self, checked):
        if False:
            for i in range(10):
                print('nop')
        '更新勾选框的槽函数\n        :param checked:\n        '
        self.checkBox.setChecked(checked)
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = WindowMaster()
    w.show()
    sys.exit(app.exec_())