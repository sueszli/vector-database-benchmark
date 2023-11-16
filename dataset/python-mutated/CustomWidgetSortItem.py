"""
Created on 2018年8月4日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: QListView.显示自定义Widget并排序
@description:
"""
import string
from random import choice, randint
from time import time
try:
    from PyQt5.QtCore import QSortFilterProxyModel, Qt, QSize
    from PyQt5.QtGui import QStandardItem, QStandardItemModel
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QListView, QHBoxLayout, QLineEdit, QApplication
except ImportError:
    from PySide2.QtCore import QSortFilterProxyModel, Qt, QSize
    from PySide2.QtGui import QStandardItem, QStandardItemModel
    from PySide2.QtWidgets import QWidget, QVBoxLayout, QPushButton, QListView, QHBoxLayout, QLineEdit, QApplication

def randomChar(y):
    if False:
        return 10
    return ''.join((choice(string.ascii_letters) for _ in range(y)))

class CustomWidget(QWidget):

    def __init__(self, text, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(CustomWidget, self).__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLineEdit(text, self))
        layout.addWidget(QPushButton('x', self))

    def sizeHint(self):
        if False:
            for i in range(10):
                print('nop')
        return QSize(200, 40)

class SortFilterProxyModel(QSortFilterProxyModel):

    def lessThan(self, source_left, source_right):
        if False:
            return 10
        if not source_left.isValid() or not source_right.isValid():
            return False
        leftData = self.sourceModel().data(source_left)
        rightData = self.sourceModel().data(source_right)
        if self.sortOrder() == Qt.DescendingOrder:
            leftData = leftData.split('-')[-1]
            rightData = rightData.split('-')[-1]
            return leftData < rightData
        return super(SortFilterProxyModel, self).lessThan(source_left, source_right)

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Window, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        layout.addWidget(QPushButton('以名字升序', self, clicked=self.sortByName))
        layout.addWidget(QPushButton('以时间倒序', self, clicked=self.sortByTime))
        self.listView = QListView(self)
        layout.addWidget(self.listView)
        self.dmodel = QStandardItemModel(self.listView)
        self.fmodel = SortFilterProxyModel(self.listView)
        self.fmodel.setSourceModel(self.dmodel)
        self.listView.setModel(self.fmodel)
        for _ in range(50):
            name = randomChar(5)
            times = time() + randint(0, 30)
            value = '{}-{}'.format(name, times)
            item = QStandardItem(value)
            self.dmodel.appendRow(item)
            index = self.fmodel.mapFromSource(item.index())
            widget = CustomWidget(value, self)
            item.setSizeHint(widget.sizeHint())
            self.listView.setIndexWidget(index, widget)

    def sortByTime(self):
        if False:
            return 10
        self.fmodel.sort(0, Qt.DescendingOrder)

    def sortByName(self):
        if False:
            print('Hello World!')
        self.fmodel.sort(0, Qt.AscendingOrder)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())