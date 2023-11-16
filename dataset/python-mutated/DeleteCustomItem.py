"""
Created on 2018年11月4日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: 删除Item
@description: 
"""
try:
    from PyQt5.QtCore import QSize, pyqtSignal
    from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton, QListWidgetItem, QVBoxLayout, QListWidget, QApplication
except ImportError:
    from PySide2.QtCore import QSize, Signal as pyqtSignal
    from PySide2.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton, QListWidgetItem, QVBoxLayout, QListWidget, QApplication

class ItemWidget(QWidget):
    itemDeleted = pyqtSignal(QListWidgetItem)

    def __init__(self, text, item, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(ItemWidget, self).__init__(*args, **kwargs)
        self._item = item
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLineEdit(text, self))
        layout.addWidget(QPushButton('x', self, clicked=self.doDeleteItem))

    def doDeleteItem(self):
        if False:
            i = 10
            return i + 15
        self.itemDeleted.emit(self._item)

    def sizeHint(self):
        if False:
            while True:
                i = 10
        return QSize(200, 40)

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Window, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.listWidget = QListWidget(self)
        layout.addWidget(self.listWidget)
        self.clearBtn = QPushButton('清空', self, clicked=self.doClearItem)
        layout.addWidget(self.clearBtn)
        self.testData()

    def doDeleteItem(self, item):
        if False:
            while True:
                i = 10
        row = self.listWidget.indexFromItem(item).row()
        item = self.listWidget.takeItem(row)
        self.listWidget.removeItemWidget(item)
        del item

    def doClearItem(self):
        if False:
            while True:
                i = 10
        for _ in range(self.listWidget.count()):
            item = self.listWidget.takeItem(0)
            self.listWidget.removeItemWidget(item)
            del item

    def testData(self):
        if False:
            print('Hello World!')
        for i in range(100):
            item = QListWidgetItem(self.listWidget)
            widget = ItemWidget('item: {}'.format(i), item, self.listWidget)
            widget.itemDeleted.connect(self.doDeleteItem)
            self.listWidget.setItemWidget(item, widget)
if __name__ == '__main__':
    import sys
    import cgitb
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())