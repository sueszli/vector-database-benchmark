"""
Created on 2019年8月7日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: QtRemoteObjects.SyncUi.WindowSlave
@description: 备窗口
"""
from PyQt5.QtCore import QUrl
from PyQt5.QtRemoteObjects import QRemoteObjectNode, QRemoteObjectReplica
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QCheckBox, QProgressBar, QMessageBox

class WindowSlave(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(WindowSlave, self).__init__(*args, **kwargs)
        self.setupUi()
        node = QRemoteObjectNode(parent=self)
        node.connectToNode(QUrl('local:WindowMaster'))
        self.windowMaster = node.acquireDynamic('WindowMaster')
        self.windowMaster.initialized.connect(self.onInitialized)
        self.windowMaster.stateChanged.connect(self.onStateChanged)

    def setupUi(self):
        if False:
            i = 10
            return i + 15
        self.setWindowTitle('WindowSlave')
        self.resize(300, 400)
        layout = QVBoxLayout(self)
        self.lineEdit = QLineEdit(self)
        self.checkBox = QCheckBox('来勾我啊', self)
        self.progressBar = QProgressBar(self)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.checkBox)
        layout.addWidget(self.progressBar)

    def onStateChanged(self, newState, oldState):
        if False:
            i = 10
            return i + 15
        if newState == QRemoteObjectReplica.Suspect:
            QMessageBox.critical(self, '错误', '连接丢失')

    def onInitialized(self):
        if False:
            for i in range(10):
                print('nop')
        self.windowMaster.editValueChanged.connect(self.lineEdit.setText)
        self.lineEdit.textChanged.connect(self.windowMaster.updateEdit)
        self.windowMaster.checkToggled.connect(self.checkBox.setChecked)
        self.checkBox.toggled.connect(self.windowMaster.updateCheck)
        self.windowMaster.progressValueChanged.connect(self.progressBar.setValue)
        print('绑定信号槽完成')
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = WindowSlave()
    w.show()
    sys.exit(app.exec_())