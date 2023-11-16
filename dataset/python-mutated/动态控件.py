"""
Module implementing Dialog.
"""
from PyQt5 import QtGui, QtWidgets, QtCore, QtWinExtras
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Ui_动态控件 import Ui_Dialog

class Dialog(QDialog, Ui_Dialog):

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super(Dialog, self).__init__(parent)
        self.setupUi(self)
        self.dynamic1()
        self.dynamic2()

    def dynamic1(self):
        if False:
            return 10
        for i in range(5):
            self.pushButton = QtWidgets.QPushButton(self)
            self.pushButton.setText('pushButton%d' % i)
            self.pushButton.setObjectName('pushButton%d' % i)
            self.verticalLayout.addWidget(self.pushButton)
            self.pushButton.clicked.connect(self.pr)

    def dynamic2(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            txt = '\nself.pushButton_{i} = QtWidgets.QPushButton(self);\nself.pushButton_{i}.setText("pushButton{i}");\nself.pushButton_{i}.setObjectName("pushButton{i}");\nself.verticalLayout.addWidget(self.pushButton_{i});\nself.pushButton_{i}.clicked.connect(self.pr)\n                '.format(i=i)
            exec(txt)
        self.pushButton_1.clicked.connect(self.pr2)
        self.pushButton_2.clicked.connect(self.pr2)
        self.pushButton_3.clicked.connect(self.pr2)

    def pr(self):
        if False:
            print('Hello World!')
        "法一和法二都可用的调用\n        if self.sender().objectName=='XXX':\n            self.pr2()\n        "
        print(self.sender().text())
        print(self.sender().objectName())
        print(self.pushButton.text())

    def pr2(self):
        if False:
            while True:
                i = 10
        print(2)
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Dialog()
    ui.show()
    sys.exit(app.exec_())