"""
@resource:none
@description: 1. exec()执行动态生成控件 //关闭程序时把model类型保存到ini文件中,打开时生成model对象.
@Created on 2018年3月17日
@email: 625781186@qq.com
"""
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QDialog
from PyQt5.QtSql import *
import re
from Ui_getModel import Ui_Dialog

class Dialog(QDialog, Ui_Dialog):
    """
    Class documentation goes here.
    """

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super(Dialog, self).__init__(parent)
        self.setupUi(self)
        self.setting = QSettings('./setting.ini', QSettings.IniFormat)
        self.getModel()
        self.tableView.setModel(self.qmodel)
        print('1:', self.tableView.model())

    def closeEvent(self, e):
        if False:
            i = 10
            return i + 15
        text = re.split('\\.| ', str(self.tableView.model()))
        if text != ['None']:
            i = [i for (i, x) in enumerate(text) if x.find('Model') != -1]
            self.setting.setValue('Model/model', text[i[-1]] + '()')

    def getModel(self):
        if False:
            print('Hello World!')
        if self.setting.contains('Model/model'):
            model = self.setting.value('Model/model')
            exec('self.qmodel=%s' % model)
            print('2:', self.qmodel)
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Dialog()
    ui.show()
    sys.exit(app.exec_())