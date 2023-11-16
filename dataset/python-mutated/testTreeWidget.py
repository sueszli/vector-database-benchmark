"""
Created on 2017年4月20日
@author: weike32
@site: https://pyqt.site , https://github.com/weike32
@email: 394967319@qq.com
@file: CopyContent
@description:
"""
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QApplication
from Lib.testTree import Ui_Form

class graphAnalysis(QDialog, Ui_Form):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(graphAnalysis, self).__init__()
        self.setupUi(self)
        self.treeWidget.itemChanged.connect(self.handleChanged)

    def handleChanged(self, item, column):
        if False:
            i = 10
            return i + 15
        count = item.childCount()
        if item.checkState(column) == Qt.Checked:
            for index in range(count):
                item.child(index).setCheckState(0, Qt.Checked)
        if item.checkState(column) == Qt.Unchecked:
            for index in range(count):
                item.child(index).setCheckState(0, Qt.Unchecked)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = graphAnalysis()
    w.show()
    sys.exit(app.exec_())