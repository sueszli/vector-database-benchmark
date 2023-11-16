"""
Created on 2018年8月4日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: QListView.显示自定义Widget
@description:
"""
try:
    from PyQt5.QtCore import QSize
    from PyQt5.QtGui import QStandardItemModel, QStandardItem
    from PyQt5.QtWidgets import QListView, QWidget, QHBoxLayout, QLineEdit, QPushButton, QApplication
except ImportError:
    from PySide2.QtCore import QSize
    from PySide2.QtGui import QStandardItemModel, QStandardItem
    from PySide2.QtWidgets import QListView, QWidget, QHBoxLayout, QLineEdit, QPushButton, QApplication

class CustomWidget(QWidget):

    def __init__(self, text, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(CustomWidget, self).__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLineEdit(text, self))
        layout.addWidget(QPushButton('x', self))

    def sizeHint(self):
        if False:
            return 10
        return QSize(200, 40)

class ListView(QListView):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(ListView, self).__init__(*args, **kwargs)
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        for i in range(10):
            item = QStandardItem()
            self._model.appendRow(item)
            index = self._model.indexFromItem(item)
            widget = CustomWidget(str(i))
            item.setSizeHint(widget.sizeHint())
            self.setIndexWidget(index, widget)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = ListView()
    w.show()
    sys.exit(app.exec_())