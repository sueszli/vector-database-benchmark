"""
Created on 2019年11月8日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: QTreeWidget.ParentNodeForbid
@description: 父节点不可选中
"""
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem, QStyledItemDelegate, QStyle
except ImportError:
    from PySide2.QtCore import Qt
    from PySide2.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem, QStyledItemDelegate, QStyle

class NoColorItemDelegate(QStyledItemDelegate):

    def paint(self, painter, option, index):
        if False:
            print('Hello World!')
        if option.state & QStyle.State_HasFocus:
            option.state = option.state & ~QStyle.State_HasFocus
        if option.state & QStyle.State_MouseOver and index.data(Qt.UserRole + 1):
            option.state = option.state & ~QStyle.State_MouseOver
        super(NoColorItemDelegate, self).paint(painter, option, index)

class Window(QTreeWidget):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(Window, self).__init__(*args, **kwargs)
        self.setItemDelegateForColumn(0, NoColorItemDelegate(self))
        pitem1 = QTreeWidgetItem(self, ['parent item 1'])
        pitem1.setFlags(pitem1.flags() & ~Qt.ItemIsSelectable)
        pitem1.setData(0, Qt.UserRole + 1, True)
        pitem2 = QTreeWidgetItem(self, ['parent item 2'])
        pitem2.setFlags(pitem2.flags() & ~Qt.ItemIsSelectable)
        pitem2.setData(0, Qt.UserRole + 1, True)
        citem1 = QTreeWidgetItem(pitem1, ['child item 1'])
        citem2 = QTreeWidgetItem(pitem2, ['child item 2'])
        self.expandAll()
        self.itemActivated.connect(self.onItemActivated)
        self.itemClicked.connect(self.onItemClicked)
        self.itemDoubleClicked.connect(self.onItemDoubleClicked)
        self.itemPressed.connect(self.onItemPressed)

    def mousePressEvent(self, event):
        if False:
            i = 10
            return i + 15
        item = self.itemAt(event.pos())
        if item and item.data(0, Qt.UserRole + 1):
            event.accept()
            return
        super(Window, self).mousePressEvent(event)

    def onItemActivated(self, item, column):
        if False:
            i = 10
            return i + 15
        print('Activated', item.text(0), item, column)

    def onItemClicked(self, item, column):
        if False:
            i = 10
            return i + 15
        print('Clicked', item.text(0), item, column)

    def onItemDoubleClicked(self, item, column):
        if False:
            while True:
                i = 10
        print('DoubleClicked', item.text(0), item, column)

    def onItemPressed(self, item, column):
        if False:
            while True:
                i = 10
        print('Pressed', item.text(0), item, column)
if __name__ == '__main__':
    import sys
    import cgitb
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())