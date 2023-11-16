"""
Module implementing GeographicAnalysis_Form.
"""
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from .Ui_GeographicAnalysis_Widget import Ui_Form

class Form(QWidget, Ui_Form):
    """
    Class documentation goes here.
    """

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        '\n        Constructor\n        \n        @param parent reference to the parent widget\n        @type QWidget\n        '
        super(Form, self).__init__(parent)
        self.setupUi(self)
        self.geographicAnalysis_table.horizontalHeader().setSectionResizeMode(1)
        self.geographicAnalysis_table.horizontalHeader().setStretchLastSection(True)
        self.geographicAnalysis_table.verticalHeader().setSectionResizeMode(1)
        self.geographicAnalysis_table.verticalHeader().setStretchLastSection(True)
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = GeographicAnalysis_Form()
    ui.show()
    sys.exit(app.exec_())