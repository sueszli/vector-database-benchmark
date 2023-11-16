"""
Module implementing Dialog.
"""
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtQuickWidgets import *
from Ui_py_qml import Ui_Dialog

class Dialog(QDialog, Ui_Dialog):

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super(Dialog, self).__init__(parent)
        self.setupUi(self)
        self.m_quickWidget = QQuickWidget()
        self.m_quickWidget.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.m_quickWidget.setSource(QUrl('py_mqltest.qml'))
        self.verticalLayout.addWidget(self.m_quickWidget)
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = Dialog()
    Dialog.show()
    sys.exit(app.exec_())