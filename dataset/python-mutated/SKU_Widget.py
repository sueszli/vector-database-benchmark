"""
SKU分析菜单的界面。
"""
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from .Ui_SKU_Widget import Ui_Form

class Form(QWidget, Ui_Form):
    """
    Class documentation goes here.
    """

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructor\n        \n        @param parent reference to the parent widget\n        @type QWidget\n        '
        super(Form, self).__init__(parent)
        self.setupUi(self)
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = SKU_Form()
    ui.show()
    sys.exit(app.exec_())